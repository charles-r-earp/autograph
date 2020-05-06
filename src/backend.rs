use super::{AutographResult, Element, Unsigned, Device, Activation, Autograd, ForwardMode, BackwardMode};
use std::rc::Rc;

pub mod native;

impl Device {
  pub fn cpu() -> Self {
    Device::Native(Rc::new(native::Cpu::new()))
  }
}

#[derive(Clone)]
pub enum Buffer<T> {
  Native(native::Buffer<T>),
  #[cfg(feature="cuda")]
  Cuda(RcCell<DeviceBuffer<T>>)
}

impl<T: Element> Buffer<T> {
  pub fn unwrap_native(self) -> native::Buffer<T> {
    match self {
      Buffer::Native(buffer) => buffer,
      _ => unreachable!()
    }
  }
}

pub struct DualBuffer {
  value: Buffer<f32>,
  grad: Option<Buffer<f32>>
}

impl DualBuffer {
  pub fn new(value: Buffer<f32>, grad: Option<Buffer<f32>>) -> Self {
    Self{value, grad}
  } 
  pub fn value(&self) -> &Buffer<f32> { &self.value }
  pub fn grad(&self) -> Option<&Buffer<f32>> { self.grad.as_ref() }
}

impl DualBuffer {
  pub fn unwrap_native(self) -> native::DualBuffer {
    native::DualBuffer::new(self.value.unwrap_native(), self.grad.map(|b| b.unwrap_native()))
  }
}

#[derive(Clone)]
pub enum Vertex<T> {
  Native(Rc<native::Cpu>, Vec<usize>, native::Buffer<T>),
  #[cfg(feature="cuda")]
  Cuda(Rc<cuda::Gpu>, Vec<usize>, RcCell<DeviceBuffer<T>>)
}

impl<T: Element> Vertex<T> {
  pub fn zeros(device: &Device, dims: impl AsRef<[usize]>) -> AutographResult<Self> {
    let device = device.clone();
    let dims = dims.as_ref().to_vec();
    let len = dims.iter().product();
    match device {
      Device::Native(cpu) => {
        let buffer = native::Buffer::zeros(&cpu, len);
        Ok(Vertex::Native(cpu, dims, buffer))
      } 
    }
  }
  pub fn dims(&self) -> &[usize] {
    match self {
      Vertex::Native(_, dims, _) => &dims
    }
  }
  pub fn buffer(&self) -> Buffer<T> {
    match self {
      Vertex::Native(_, _, buffer) => Buffer::Native(buffer.clone())
    }
  }
}

pub enum DenseOp {
  Native(native::DenseOp),
  #[cfg(feature="cuda")]
  Cuda(cuda::DenseOp)
}

impl DenseOp {
  pub(crate) fn new(device: &Device, mkn: [usize; 3], input: DualBuffer, weight: DualBuffer, bias: Option<DualBuffer>, act: Option<Activation>, output: DualBuffer) -> AutographResult<Self> {
    match device {
      Device::Native(cpu) => {
        Ok(DenseOp::Native(native::DenseOp::new(cpu, mkn, input.unwrap_native(), weight.unwrap_native(), bias.map(|b| b.unwrap_native()), act, output.unwrap_native())))
      }
    }
  } 
  pub(crate) fn forward(&self, mode: ForwardMode) -> AutographResult<()> {
    match self {
      DenseOp::Native(op) => { op.forward(mode); }
    }
    Ok(())
  }
  pub(crate) fn backward(&self, mode: BackwardMode) -> AutographResult<()> {
    match self {
      DenseOp::Native(op) => { op.backward(mode); }
    }
    Ok(())
  }
}

pub enum CrossEntropyOp<U: Unsigned> {
  Native(native::CrossEntropyOp<U>)
}

impl<U: Unsigned> CrossEntropyOp<U> {
  pub(crate) fn new(device: &Device, batch_size: usize, nclasses: usize, input: DualBuffer, target: Buffer<U>, output: Buffer<f32>) -> AutographResult<Self> {
    match device {
      Device::Native(cpu) => {
        Ok(CrossEntropyOp::Native(native::CrossEntropyOp::new(cpu, batch_size, nclasses, input.unwrap_native(), target.unwrap_native(), output.unwrap_native())))
      }
    }
  }
  pub(crate) fn forward(&self, mode: ForwardMode) -> AutographResult<()> {
    match self {
      CrossEntropyOp::Native(op) => { op.forward(mode); }
    }
    Ok(())
  }
  pub(crate) fn backward(&self, mode: BackwardMode) -> AutographResult<()> {
    match self {
      CrossEntropyOp::Native(op) => { op.backward(mode); }
    }
    Ok(())
  }
}
