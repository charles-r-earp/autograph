#![allow(warnings)]
#![recursion_limit="1024"]
use std::{result::Result, rc::Rc, iter::Iterator, fmt::Debug, cell::Cell};
use num_traits::AsPrimitive;
use rc_cell::RcCell;


pub mod error;
use error::AutographError;
type AutographResult<T> = Result<T, AutographError>;

mod private {
  use super::Buffer;
  use std::{rc::Rc, cell::RefCell};
  use num_traits::Zero;
  
  #[cfg(feature="cuda")]
  use rustacuda::memory::DeviceCopy;
  
  #[cfg(not(feature="cuda"))]
  pub trait DeviceCopy {}
  
  #[cfg(not(feature="cuda"))]
  impl<T> DeviceCopy for T {}
  
  pub trait PrivateElement: 'static + Copy + Zero + DeviceCopy {
    #[doc(hidden)]
    type Grad;
  }
  
  impl PrivateElement for u8 {
    type Grad = ();
  }
  impl PrivateElement for f32 {
    type Grad = Option<Rc<Vec<Buffer<f32>>>>;
  }
}

pub trait Element: private::PrivateElement {}

impl<T> Element for T where T: private::PrivateElement {}

pub trait Unsigned: Element + AsPrimitive<usize> {}

impl Unsigned for u8 {}

pub mod backend;

use backend::{native, Buffer, DualBuffer, Vertex};

#[derive(Clone)]
pub enum Device {
  Native(Rc<native::Cpu>),
  #[cfg(feature="cuda")]
  Cuda(Rc<cuda::Gpu>)
}

pub mod init;

#[derive(Clone, Copy)]
pub enum TensorRepr {
  Batched,
  Broadcasted
}

#[derive(Clone)]
pub struct Tensor<T: Element> {
  devices: Vec<Device>,
  dims: Vec<usize>,
  repr: TensorRepr,
  vertices: Vec<Vertex<T>>
}

impl<T: Element> Tensor<T> {
  pub fn zeros(devices: impl AsRef<[Device]>, dims: impl AsRef<[usize]>, repr: TensorRepr) -> AutographResult<Self> {
    let devices = devices.as_ref().to_vec();
    let dims = dims.as_ref().to_vec();
    let mut vertices = Vec::with_capacity(dims.len());
    match repr {
      TensorRepr::Batched => {
        let mut batch_dims = dims.clone();
        batch_dims[0] = (dims[0] / devices.len()) + dims[0] % devices.len();
        vertices.push(Vertex::zeros(&devices[0], &batch_dims)?);
        batch_dims[0] = dims[0] / devices.len();
        for device in &devices[1..] {
          vertices.push(Vertex::zeros(device, &batch_dims)?);
        }
      },
      TensorRepr::Broadcasted => {
        for device in &devices {
          vertices.push(Vertex::zeros(device, &dims)?);
        }
      }
    }
    Ok(Self{devices, dims, repr, vertices})
  }
  pub fn devices(&self) -> &[Device] { &self.devices }
  pub fn dims(&self) -> &[usize] { &self.dims }
  pub fn len(&self) -> usize { self.dims.iter().product() }
  pub fn vertices(&self) -> &[Vertex<T>] { &self.vertices }
  pub fn buffers(&self) -> impl Iterator<Item=Buffer<T>> + '_ {
    self.vertices.iter()
      .map(|v| v.buffer())
  }
  pub fn write(&self, mut slice: &[T]) -> AutographResult<()> {
    debug_assert_eq!(slice.len(), self.dims.iter().product());
    for vertex in &self.vertices {
      match vertex {
        Vertex::Native(cpu, _, buffer) => {
          let len = buffer.len();
          buffer.write(cpu, &slice[..len]);
          slice = &slice[len..];
        }
      }
    }
    Ok(())
  } 
  pub fn read(&self, mut slice: &mut [T]) -> AutographResult<()> {
    debug_assert_eq!(slice.len(), self.dims.iter().product());
    for vertex in &self.vertices {
      match vertex {
        Vertex::Native(cpu, _, buffer) => {
          let len = buffer.len();
          buffer.read(cpu, &mut slice[..len]);
          slice = &mut slice[len..];
        }
      }
    }
    Ok(())
  }
  pub fn to_vec(&self) -> AutographResult<Vec<T>> {
    let mut vec = vec![T::zero(); self.len()];
    self.read(&mut vec)?;
    Ok(vec)
  }
}

#[derive(Clone)]
pub struct Variable {
  value: Tensor<f32>,
  grad: Option<Tensor<f32>>
}

impl Variable {
  pub fn zeros(devices: &[Device], dims: impl AsRef<[usize]>, requires_grad: bool) -> AutographResult<Self> {
    let dims = dims.as_ref();
    let value = Tensor::zeros(devices, &dims, TensorRepr::Batched)?;
    let grad = if requires_grad {
      let grad = Tensor::zeros(devices, &dims, TensorRepr::Batched)?;
      Some(grad)
    } else { None };
    Ok(Self{value, grad})
  }
  pub fn devices(&self) -> &[Device] { self.value.devices() }
  pub fn dims(&self) -> &[usize] { self.value.dims() }
  pub fn value(&self) -> &Tensor<f32> { &self.value }
  pub fn grad(&self) -> Option<&Tensor<f32>> { self.grad.as_ref() }
  pub fn duals(&self) -> impl Iterator<Item=DualBuffer> + '_ {
    let mut grads = self.grad.as_ref()
      .map(|g| {
      g.buffers()
    });
    let grads = std::iter::from_fn(move || {
      Some(grads.as_mut().map(|g| g.next().unwrap()))
    });
    self.value.buffers()
      .zip(grads)
      .map(|(x, dx)| DualBuffer::new(x, dx)) 
  }
}

#[derive(Clone)]
pub struct Parameter {
  value: Tensor<f32>,
  grad: Option<Tensor<f32>>,
  velocity: Option<Tensor<f32>>
}

impl Parameter {
  pub fn zeros(devices: &[Device], dims: impl AsRef<[usize]>, requires_grad: bool) -> AutographResult<Self> {
    let dims = dims.as_ref();
    let value = Tensor::zeros(devices, &dims, TensorRepr::Broadcasted)?;
    let (grad, velocity) = if requires_grad {
      let grad = Tensor::zeros(devices, &dims, TensorRepr::Broadcasted)?;
      let velocity = Tensor::zeros(devices, &dims, TensorRepr::Broadcasted)?;
      (Some(grad), Some(velocity))
    } else { (None, None) };
    Ok(Self{value, grad, velocity})
  }
  pub fn devices(&self) -> &[Device] { self.value.devices() }
  pub fn dims(&self) -> &[usize] { self.value.dims() }
  pub fn len(&self) -> usize { self.value.len() }
  pub fn value(&self) -> &Tensor<f32> { &self.value }
  pub fn grad(&self) -> Option<&Tensor<f32>> { self.grad.as_ref() }
  pub fn velocity(&self) -> Option<&Tensor<f32>> { self.velocity.as_ref() }
  pub fn duals(&self) -> impl Iterator<Item=DualBuffer> + '_ {
    let mut grads = self.grad.as_ref()
      .map(|g| {
      g.buffers()
    });
    let grads = std::iter::from_fn(move || {
      Some(grads.as_mut().map(|g| g.next().unwrap()))
    });
    self.value.buffers()
      .zip(grads)
      .map(|(x, dx)| DualBuffer::new(x, dx)) 
  }
}

pub trait Optimizer {
  fn step(&self, parameters: &[Parameter]);
}

#[derive(Clone, Copy)]
pub enum ForwardMode {
  Infer,
  Eval,
  Train
}

impl ForwardMode {
  pub fn backward_mode(&self) -> Option<BackwardMode> {
    match self {
      ForwardMode::Infer => None,
      ForwardMode::Eval => Some(BackwardMode::Eval),
      ForwardMode::Train => Some(BackwardMode::Train)
    }
  }
}

#[derive(Clone, Copy)]
pub enum BackwardMode {
  Eval, 
  Train
}

pub trait Autograd {
  fn forward(&self, mode: ForwardMode) -> AutographResult<()>;
  fn backward(&self) -> AutographResult<()>;
}

#[derive(Clone, Copy)]
pub enum Activation {
  Relu
}

pub mod layer;
pub mod loss;






