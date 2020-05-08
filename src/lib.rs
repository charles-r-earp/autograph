#![allow(warnings)]
#![recursion_limit="1024"]
use std::{fmt::{self, Debug}, sync::{Arc, Weak, Mutex, RwLock}, borrow::Cow, iter::Iterator};
use num_traits::{Zero, One, AsPrimitive};
use rand::Rng;
use rand_distr::{Distribution, Normal};


pub mod error;
use error::AutographError;

pub mod backend;
use backend::{Cpu, BackwardOp};

type Result<T, E = AutographError> = std::result::Result<T, E>; 

type ArcRwLock<T> = Arc<RwLock<T>>;

fn arc_rwlock<T>(t: T) -> ArcRwLock<T> { Arc::new(RwLock::new(t)) }

mod private {
  pub trait PrivateElement {}
  
  impl PrivateElement for u8 {}
  impl PrivateElement for f32 {}
}

pub trait Element: private::PrivateElement + Copy + Zero + One + 'static {}

impl Element for u8 {}
impl Element for f32 {}

pub trait Unsigned: Element + AsPrimitive<usize> {}

#[derive(Clone)]
pub enum Device {
  Native(&'static Cpu),
  #[cfg(feature="cuda")]
  Cuda(&'static Gpu)
}

impl Device {
  pub fn cpu() -> Self {
    Device::Native(Cpu::get())
  }
}

unsafe fn uninitialized_vec<T: Element>(len: usize) -> Vec<T> {
  let mut vec = Vec::with_capacity(len);
  vec.set_len(len);
  vec
}

pub enum Buffer<T: Element> {
  Native(Vec<T>),
  #[cfg(feature="cuda")]
  Cuda(DeviceBuffer<T>)
}

impl<T: Element> Buffer<T> {
  unsafe fn uninitialized(device: &Device, len: usize) -> Result<Self> {
    match device {
      Device::Native(_) => Ok(Buffer::Native(uninitialized_vec(len)))
    }
  }
  fn from_elem(device: &Device, elem: T, len: usize) -> Result<Self> {
    match device {
      Device::Native(_) => Ok(Buffer::Native(vec![elem; len]))
    }
  }
  fn from_cow<'a>(device: &Device, cow: Cow<'a, [T]>) -> Result<Self> {
    match device {
      Device::Native(_) => Ok(Buffer::Native(cow.into_owned()))
    } 
  }
  fn to_device(&self, device: &Device) -> Result<Self> {
    match self {
      Buffer::Native(slice) => match device {
        Device::Native(_) => unreachable!()
      }
    }
  }
  fn len(&self) -> usize {
    match self {
      Buffer::Native(slice) => slice.len()
    }
  }
  fn native_ptr(&self) -> *const T {
    match self {
      Buffer::Native(slice) => slice.as_ptr()
    }
  }
  fn native_mut_ptr(&mut self) -> *mut T {
    match self {
      Buffer::Native(slice) => slice.as_mut_ptr()
    }
  }
}

#[derive(Clone)]
pub struct Tensor<T: Element> {
  device: Device,
  dims: Vec<usize>,
  data: Arc<Buffer<T>>    
}

impl<T: Element> Tensor<T> {
  /*pub unsafe fn uninitialized(device: &Device, dims: impl AsRef<[usize]>) -> Result<Self> {
    let device = device.clone();
    let dims = dims.as_ref().to_vec();
    let data = Arc::new(Buffer::uninitialized(&device, dims.iter().product())?);
    Ok(Self{device, dims, data})
  }*/
  unsafe fn from_dims_buffer(device: &Device, dims: impl AsRef<[usize]>, buffer: Buffer<T>) -> Self {
    let device = device.clone();
    let dims = dims.as_ref().to_vec();
    debug_assert_eq!(buffer.len(), dims.iter().product());
    let data = Arc::new(buffer);
    Self{device, dims, data}
  }
  pub fn from_dims_cow<'a>(device: &Device, dims: impl AsRef<[usize]>, cow: impl Into<Cow<'a, [T]>>) -> Result<Self> {
    let device = device.clone();
    let cow = cow.into();
    let dims = dims.as_ref().to_vec();
    debug_assert_eq!(cow.len(), dims.iter().product());
    let data = Arc::new(Buffer::from_cow(&device, cow)?);
    Ok(Self{device, dims, data})
  }
  pub fn from_dims_iter(device: &Device, dims: impl AsRef<[usize]>, mut iter: impl Iterator<Item=T>) -> Result<Self> {
    let dims = dims.as_ref();
    let vec: Vec<T> = iter.take(dims.iter().product()).collect();
    Self::from_dims_cow(device, dims, vec)
  }
  pub fn from_dims_elem(device: &Device, dims: impl AsRef<[usize]>, elem: T) -> Result<Self> {
    let device = device.clone();
    let dims = dims.as_ref().to_vec();
    let data = Arc::new(Buffer::from_elem(&device, elem, dims.iter().product())?);
    Ok(Self{device, dims, data})
  }
  pub fn zeros(device: &Device, dims: impl AsRef<[usize]>) -> Result<Self> {
    Self::from_dims_elem(device, dims, T::zero())
  }
  pub fn random<D: AsRef<[usize]>, Dist: Distribution<T>, R: Rng>(device: &Device, dims: D, dist: Dist, rng: &mut R) -> Result<Self> {
    Self::from_dims_iter(device, dims, dist.sample_iter(rng))
  }
  pub fn dims(&self) -> &[usize] { &self.dims }
  pub fn to_device(&self, device: &Device) -> Result<Self> {
    match &self.device {
      Device::Native(_) => {
        match device {
          Device::Native(_) => Ok(Self {
            device: device.clone(), 
            dims: self.dims.clone(),
            data: self.data.clone()
          })
        }
      }
    }
  }
  pub fn as_slice(&self) -> Result<Cow<[T]>> {
    match &self.device {
      Device::Native(_) => {
        match &*self.data {
          Buffer::Native(slice) => Ok(Cow::from(slice))
        }
      } 
    }
  }
}

#[derive(Clone)]
pub struct Gradient {
  device: Device,
  dims: Vec<usize>,
  data: ArcRwLock<Buffer<f32>>
}

impl Gradient {
  pub fn zeros(device: &Device, dims: impl AsRef<[usize]>) -> Result<Self> {
    let device = device.clone();
    let dims = dims.as_ref().to_vec();
    let data = arc_rwlock(Buffer::from_elem(&device, 0., dims.iter().product())?);
    Ok(Self{device, dims, data})
  } 
}

pub struct Graph {
  ops: Mutex<Vec<BackwardOp>>
}

impl Graph {
  pub fn backward(self) -> Result<()> {
    let mut ops = self.ops.into_inner().unwrap();
    for op in ops.iter_mut().rev() {
      op.backward()?;
    }
    Ok(())
  }
  fn backward_op(&self, op: BackwardOp) {
    self.ops.lock()
      .unwrap()
      .push(op);
  }
}

impl Debug for Graph {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    f.debug_struct("Graph").finish()
  }
}

#[derive(Clone)]
pub struct Variable {
  graph: Weak<Graph>,
  value: Tensor<f32>,
  grad: Option<Gradient>
}

impl Variable {
  pub fn new(graph: Option<&Arc<Graph>>, value: Tensor<f32>) -> Self {
    let graph = graph.map_or(Weak::new(), |g| Arc::downgrade(g));
    let grad = None;
    Self{graph, value, grad}
  } 
  pub fn dims(&self) -> &[usize] { self.value.dims() }
  pub fn value(&self) -> &Tensor<f32> { &self.value }
} 

pub struct Parameter {
  value: Tensor<f32>,
  grad: Option<Gradient>
}

impl Parameter {
  pub fn to_device(&self, device: &Device) -> Result<Self> {
    Ok(Self::from(self.value.to_device(device)?))
  }
  pub fn dims(&self) -> &[usize] { self.value.dims() }
  pub fn zero_grad(&mut self) -> Result<()> {
    self.grad = Some(Gradient::zeros(&self.value.device, self.dims())?);
    Ok(())
  }
}

impl From<Tensor<f32>> for Parameter {
  fn from(tensor: Tensor<f32>) -> Self {
    Self{value: tensor, grad: None}
  }
}

pub trait Layer: Sized {
  fn forward(&self, input: &Variable) -> Result<Variable>;
  fn parameters_mut(&mut self) -> Vec<&mut Parameter>;
  fn to_device(&self, device: &Device) -> Result<Self>;
}

pub trait LayerBuilder {
  type Layer;
  fn input(self, dims: &[usize]) -> Self;
  fn output(&self) -> Vec<usize>; 
  fn build(self) -> Self::Layer;
}

pub struct Dense {
  weight: Parameter,
  bias: Option<Parameter>
}

impl Dense {
  pub fn builder() -> DenseBuilder {
    DenseBuilder::default()
  }
}

impl Layer for Dense {
  fn forward(&self, input: &Variable) -> Result<Variable> {
    backend::dense(input, &self.weight, self.bias.as_ref())
  }
  fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
    if let Some(bias) = &mut self.bias {
      vec![&mut self.weight, bias]
    }
    else {
      vec![&mut self.weight]
    }
  }
  fn to_device(&self, device: &Device) -> Result<Self> {
    let weight = self.weight.to_device(device)?;
    let bias = if let Some(bias) = &self.bias {
      Some(bias.to_device(device)?)
    } else { None };
    Ok(Self{weight, bias})
  } 
}

#[derive(Default)]
pub struct DenseBuilder {
  input_channels: usize,
  units: usize,
  use_bias: bool
}

impl DenseBuilder {
  pub fn units(mut self, units: usize) -> Self {
    self.units = units;
    self
  }
  pub fn bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
}

impl LayerBuilder for DenseBuilder {
  type Layer = Dense;
  fn input(mut self, dims: &[usize]) -> Self {
    self.input_channels = dims.iter().product();
    self
  }
  fn output(&self) -> Vec<usize> { 
    vec![self.units]
  }
  fn build(self) -> Dense {
    let device = Device::cpu();
    use std::f32::consts::SQRT_2;
    let normal = Normal::<f32>::new(0f32, SQRT_2 / AsPrimitive::<f32>::as_(self.units)).unwrap();
    let weight = Parameter::from(Tensor::random(&device, [self.units, self.input_channels], 
                                                normal, 
                                                backend::thread_rng()).unwrap());
    let bias = if self.use_bias {
      Some(Parameter::from(Tensor::zeros(&device, [1, self.units]).unwrap()))
    } else { None };
    Dense{weight, bias}
  }
}
/*
pub fn cross_entropy_loss<U: Unsigned>(input: &Variable, target: &Tensor<U>) -> f32 {
  unimplemented!()
}

pub fn accuracy<U: Unsigned>(input: &Tensor<f32>, target: &Tensor<U>) -> usize {
  unimplemented!()
}*/





