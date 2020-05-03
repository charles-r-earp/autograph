use super::{AutographError, AutographResult, Element};
use std::{rc::{Rc, Weak}, cell::UnsafeCell, any::TypeId, borrow::Cow};
#[cfg(feature="cuda")]
use rustacuda::memory::{DeviceBuffer, CopyDestination};

pub mod cpu;

#[cfg(feature="cuda")]
pub mod cuda;

pub enum Device {
  Cpu(cpu::Cpu),
  #[cfg(feature="cuda")]
  Cuda(cuda::Gpu)
}

pub struct Backend {
  devices: Vec<Rc<Device>>
}

unsafe fn uninitialized_vec<T: Element>(len: usize) -> Vec<T> {
  let mut vec = Vec::with_capacity(len);
  vec.set_len(len);
  vec
}

pub enum Buffer<T: Element> {
  Cpu(UnsafeCell<Vec<T>>),
  #[cfg(feature="cuda")]
  Cuda(UnsafeCell<DeviceBuffer<T>>)
}

impl<T: Element> Buffer<T> {
  pub unsafe fn uninitialized(device: &Device, len: usize) -> AutographResult<Rc<Self>> {
    match &*device {
      Device::Cpu(_) => {
        let vec = uninitialized_vec(len);
        Ok(Rc::new(Buffer::Cpu(UnsafeCell::new(vec))))
      },
      #[cfg(feature="cuda")]
      Device::Cuda(device) => {
        device.set_current()?;
        device.sync()?;
        Ok(Rc::new(Buffer::Cuda(UnsafeCell::new(DeviceBuffer::uninitialized(len)?))))
      }
    }
  }
  pub fn from_vec<'a>(device: &Device, vec: impl Into<Cow<'a, [T]>>) -> AutographResult<Rc<Self>> {
    let vec = vec.into();
    match &*device {
      Device::Cpu(_) => Ok(Rc::new(Buffer::Cpu(UnsafeCell::new(vec.into_owned())))),
      #[cfg(feature="cuda")]
      Device::Cuda(device) => {
        device.set_current()?;
        device.sync()?;
        let buffer = UnsafeCell::new(DeviceBuffer::from_slice(vec.as_ref())?);
        Ok(Rc::new(Buffer::Cuda(buffer))) 
      }
    }    
  }
  pub unsafe fn cpu(&self) -> Option<&mut Vec<T>> { 
    match self {
      Buffer::Cpu(buffer) => Some(&mut *buffer.get()),
      _ => None
    }
  } 
  #[cfg(feature="cuda")]
  pub unsafe fn cuda(&self) -> Option<&mut DeviceBuffer<T>> {
    match self {
      Buffer::Cuda(buffer) => Some(&mut *buffer.get()),
      _ => None
    }
  }
}


pub enum StackVertex {
  U8(Vertex<u8>),
  F32(Vertex<f32>),
}

impl<T: Element> From<Vertex<T>> for StackVertex {
  fn from(vertex: Vertex<T>) -> Self {
    use std::mem::transmute;
    if TypeId::of::<T>() == TypeId::of::<u8>() {
      StackVertex::U8(unsafe { transmute(vertex) })
    }
    else if TypeId::of::<T>() == TypeId::of::<f32>() {
      StackVertex::F32(unsafe { transmute(vertex) })
    }
    else { unreachable!() }
  }
}

#[derive(Default)]
pub struct Stack {
  vertices: Vec<StackVertex>
}

impl Stack {
  pub fn push<T: Element>(&mut self, vertex: Vertex<T>) {
    self.vertices.push(vertex.into());
  }
  pub fn clear(&mut self) { self.vertices.clear() }
}

#[derive(Clone)]
pub struct Vertex<T: Element> {
  device: Weak<Device>,
  dims: Vec<usize>,
  is_t: bool,
  buffer: Rc<Buffer<T>>
}

impl<T: Element> Vertex<T> {
  pub unsafe fn uninitialized(device: &Rc<Device>, dims: impl AsRef<[usize]>) -> AutographResult<Self> {
    let dims = dims.as_ref().to_vec();
    let buffer = Buffer::uninitialized(device, dims.iter().product())?;
    Ok(Self {
      device: Rc::downgrade(device),
      dims,
      is_t: false,
      buffer
    })
  }
  pub fn from_dims_vec<'a>(device: &Rc<Device>, dims: impl AsRef<[usize]>, vec: impl Into<Cow<'a, [T]>>) -> AutographResult<Self> {
    let dims = dims.as_ref().to_vec();
    let buffer = Buffer::from_vec(device, vec)?;
    Ok(Self {
      device: Rc::downgrade(device),
      dims,
      is_t: false,
      buffer
    })
  }
  pub fn device(&self) -> &Weak<Device> { &self.device }
  pub fn dims(&self) -> &[usize] { &self.dims }
  pub fn is_t(&self) -> bool { self.is_t }
  pub fn buffer(&self) -> &Rc<Buffer<T>> { &self.buffer }
  pub fn t(&self) -> Self { 
    assert_eq!(self.dims.len(), 2);
    Vertex {
      device: self.device.clone(),
      dims: vec![self.dims[1], self.dims[0]],
      is_t: !self.is_t,
      buffer: self.buffer.clone()
    }
  }
  pub fn to_vec(&self) -> AutographResult<Vec<T>> {
    let device = Weak::upgrade(&self.device).unwrap();
    match &*device {
      Device::Cpu(device) => { 
        device.sync();
        Ok(unsafe { self.buffer().cpu().unwrap().clone() })
      },
      #[cfg(feature="cuda")]
      Device::Cuda(device) => {
        device.sync();
        let buffer = unsafe { self.buffer().cuda().unwrap() };
        let mut vec = unsafe { uninitialized_vec(buffer.len()) };
        buffer.copy_to(&mut vec)?;
        Ok(vec)
      }
    }
  }
}

pub fn gemm(alpha: f32, a: &Vertex<f32>, b: &Vertex<f32>, beta: f32, c: &Vertex<f32>) -> AutographResult<()> {
  let device = Weak::upgrade(&a.device).unwrap();
  match &*device {
    Device::Cpu(device) => { device.gemm(alpha, a, b, beta, c); },
    _ => unimplemented!()
  }
  Ok(())
}

impl Vertex<f32> {
  pub fn mm(&self, rhs: &Vertex<f32>) -> AutographResult<Self> {
    let device = Weak::upgrade(&self.device).unwrap();
    let output = unsafe { Vertex::uninitialized(&device, [self.dims[0], rhs.dims()[1]])? };
    gemm(1., self, rhs, 0., &output)?;
    Ok(output)
  }
}
