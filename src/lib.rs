#![allow(warnings)]
#![recursion_limit="1024"]
use std::sync::{Arc, RwLock, LockResult, PoisonError, RwLockReadGuard, RwLockWriteGuard};
use std::borrow::Cow;
use num_traits::{Zero, One, ToPrimitive, Bounded};
use ndarray::{Array, ArrayView, CowArray, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
#[cfg(feature="cuda")]
use rustacuda::memory::DeviceCopy;

#[doc(hidden)]
pub mod cpu;
pub use cpu::{Cpu, CpuBuffer};

#[cfg(feature="cuda")]
pub mod cuda;
#[cfg(feature="cuda")]
pub use cuda::{CudaGpu, CudaBuffer};

pub mod autograd;

#[cfg(feature="datasets")]
pub mod datasets;

mod private_num {
  pub trait PrivateNum {}
}
use private_num::PrivateNum;

impl PrivateNum for u8 {}
impl PrivateNum for f32 {}

#[doc(hidden)]
#[cfg(not(feature="cuda"))] 
pub trait DeviceCopy {}

#[cfg(not(feature="cuda"))]
impl<T: PrivateNum> DeviceCopy for T {}

pub trait Num: 'static + Copy + DeviceCopy + Default + Zero + One + ToPrimitive + Bounded + PartialEq {}

impl Num for u8 {}
impl Num for f32 {}

pub trait Unsigned: Num {}

impl Unsigned for u8 {}

#[derive(Clone)]
pub enum Buffer<T: Num> {
  Cpu(CpuBuffer<T>),
  #[cfg(feature="cuda")]
  Cuda(CudaBuffer<T>)
}

impl<T: Num> From<CpuBuffer<T>> for Buffer<T> {
  fn from(cpu_buffer: CpuBuffer<T>) -> Self {
    Buffer::Cpu(cpu_buffer)
  }
}

#[cfg(feature="cuda")]
impl<T: Num> From<CudaBuffer<T>> for Buffer<T> {
  fn from(cuda_buffer: CudaBuffer<T>) -> Self {
    Buffer::Cuda(cuda_buffer)
  }
}

impl<T: Num> Buffer<T> {
  unsafe fn uninitialized(device: &Device, len: usize) -> Self {
    match device {
      Device::Cpu(_) => CpuBuffer::uninitialized(len).into(),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => CudaBuffer::uninitialized(cuda_gpu, len).into()
    }
  }
  fn from_vec<'a>(device: &Device, vec: impl Into<Cow<'a, [T]>>) -> Self {
    match device {
      Device::Cpu(_) => CpuBuffer::from_vec(vec).into(),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => {
        let slice = vec.into();
        let mut buffer = unsafe { CudaBuffer::uninitialized(cuda_gpu, slice.len()) };
        buffer.copy_from_slice(slice);
        buffer.into()
      }
    }
  } 
  fn zeros(device: &Device, len: usize) -> Self {
    match device {
      Device::Cpu(_) => CpuBuffer::from_vec(vec![T::zero(); len]).into(),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => {
        let mut buffer = unsafe { CudaBuffer::uninitialized(cuda_gpu, len) };
        buffer.fill(T::zero());
        buffer.into()
      }
    }
  }
  fn fill(&mut self, elem: T) {
    match self {
      Buffer::Cpu(cpu_buffer) => cpu_buffer.fill(elem),
      #[cfg(feature="cuda")]
      Buffer::Cuda(cuda_buffer) => cuda_buffer.fill(elem)
    }
  }
  fn as_slice(&self) -> Cow<[T]> {
    match self {
      Buffer::Cpu(cpu_buffer) => cpu_buffer.as_slice().into(),
      #[cfg(feature="cuda")]
      Buffer::Cuda(cuda_buffer) => cuda_buffer.to_vec().into()
    }
  }
  fn cpu(&self) -> Option<&CpuBuffer<T>> {
    match self {
      Buffer::Cpu(cpu_buffer) => Some(cpu_buffer),
      _ => None
    } 
  }
  fn cpu_mut(&mut self) -> Option<&mut CpuBuffer<T>> {
    match self {
      Buffer::Cpu(cpu_buffer) => Some(cpu_buffer),
      _ => None
    } 
  }
  #[cfg(feature="cuda")]
  fn cuda(&self) -> Option<&CudaBuffer<T>> {
    match self {
      Buffer::Cuda(cuda_buffer) => Some(cuda_buffer),
      _ => None
    }
  }
  #[cfg(feature="cuda")]
  fn cuda_mut(&mut self) -> Option<&mut CudaBuffer<T>> {
    match self {
      Buffer::Cuda(cuda_buffer) => Some(cuda_buffer),
      _ => None
    }
  }
}

#[derive(Clone, Debug)]
pub enum Device {
  Cpu(Arc<Cpu>),
  #[cfg(feature="cuda")]
  Cuda(Arc<CudaGpu>)
}

impl Device {
  fn cpu(&self) -> Option<&Arc<Cpu>> {
    match self {
      Device::Cpu(cpu) => Some(cpu),
      _ => None
    }
  }
  #[cfg(feature="cuda")]
  fn cuda(&self) -> Option<&Arc<CudaGpu>> {
    match self {
      Device::Cuda(cuda_gpu) => Some(cuda_gpu),
      _ => None
    }
  }
}

impl From<Arc<Cpu>> for Device {
  fn from(cpu: Arc<Cpu>) -> Self {
    Device::Cpu(cpu)
  }
}

#[cfg(feature="cuda")]
impl From<Arc<CudaGpu>> for Device {
  fn from(cuda_gpu: Arc<CudaGpu>) -> Self {
    Device::Cuda(cuda_gpu)
  }
} 

impl PartialEq for Device {
  fn eq(&self, other: &Self) -> bool {
    match self {
      Device::Cpu(cpu1) => {
        match other {
          Device::Cpu(cpu2) => Arc::ptr_eq(cpu1, cpu2),
          _ => false
        }
      },
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu1) => {
        match other {
          Device::Cuda(cuda_gpu2) => Arc::ptr_eq(cuda_gpu1, cuda_gpu2),
          _ => false
        }
      }
    }
  }
}

mod private_data {
  pub trait PrivateData {}
}
use private_data::PrivateData;

pub trait Data: PrivateData {
  type Elem: Num;
}

pub trait DataOwned: Data + Sized {
  fn from_buffer(buffer: Buffer<Self::Elem>) -> Self;
}

pub trait DataRef: Data {
  #[doc(hidden)]
  fn buffer(&self) -> &Buffer<Self::Elem>;
}

pub trait DataMut: DataRef {
  #[doc(hidden)]
  fn buffer_mut(&mut self) -> &mut Buffer<Self::Elem>;
}

#[doc(hidden)]
#[derive(Clone)]
pub struct OwnedRepr<T: Num> {
  buffer: Buffer<T>
}

impl<T: Num> PrivateData for OwnedRepr<T> {}

impl<T: Num> Data for OwnedRepr<T> {
  type Elem = T;
}

impl<T: Num> DataOwned for OwnedRepr<T> {
  fn from_buffer(buffer: Buffer<T>) -> Self {
    Self{buffer}
  }
}

impl<T: Num> DataRef for OwnedRepr<T> {
  fn buffer(&self) -> &Buffer<T> {
    &self.buffer
  }
}

impl<T: Num> DataMut for OwnedRepr<T> {
  fn buffer_mut(&mut self) -> &mut Buffer<T> {
    &mut self.buffer
  }
}

#[doc(hidden)]
pub struct ViewRepr<V> {
  buffer: V
}

impl<V> ViewRepr<V> {
  fn new(buffer: V) -> Self {
    Self{buffer}
  }
}

impl<V> PrivateData for ViewRepr<V> {}

impl<'a, T: Num> Data for ViewRepr<&'a Buffer<T>> {
  type Elem = T;
}

impl<'a, T: Num> DataRef for ViewRepr<&'a Buffer<T>> { 
  fn buffer(&self) -> &Buffer<T> {
    &*self.buffer
  }
}

impl<'a, T: Num> Data for ViewRepr<&'a mut Buffer<T>> {
  type Elem = T;
}

impl<'a, T: Num> DataRef for ViewRepr<&'a mut Buffer<T>> { 
  fn buffer(&self) -> &Buffer<T> {
    &*self.buffer
  }
}

impl<'a, T: Num> DataMut for ViewRepr<&'a mut Buffer<T>> { 
  fn buffer_mut(&mut self) -> &mut Buffer<T> {
    &mut *self.buffer
  }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct ArcRepr<T: Num> {
  buffer: Arc<Buffer<T>>
}

impl<T: Num> PrivateData for ArcRepr<T> {}

impl<T: Num> Data for ArcRepr<T> {
  type Elem = T;
}

impl<T: Num> DataOwned for ArcRepr<T> {
  fn from_buffer(buffer: Buffer<T>) -> Self {
    Self{buffer: Arc::new(buffer)}
  }
}

impl<T: Num> DataRef for ArcRepr<T> {
  fn buffer(&self) -> &Buffer<T> {
    &*self.buffer
  }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct RwRepr<T: Num> {
  buffer: Arc<RwLock<Buffer<T>>>
}

impl<T: Num> RwRepr<T> {
  fn read(&self) -> LockResult<RwReadRepr<T>> {
    self.buffer.read()
      .map(|buffer| RwReadRepr{buffer})
      .map_err(|e| PoisonError::new(RwReadRepr{buffer: e.into_inner()}))
  }
  fn write(&self) -> LockResult<RwWriteRepr<T>> {
     self.buffer.write()
      .map(|buffer| RwWriteRepr{buffer})
      .map_err(|e| PoisonError::new(RwWriteRepr{buffer: e.into_inner()}))
  }
}

impl<T: Num> PrivateData for RwRepr<T> {}

impl<T: Num> Data for RwRepr<T> {
  type Elem = T;
}

impl<T: Num> DataOwned for RwRepr<T> {
  fn from_buffer(buffer: Buffer<T>) -> Self {
    Self{buffer: Arc::new(RwLock::new(buffer))}
  }
}

#[doc(hidden)]
pub struct RwReadRepr<'a, T: Num> {
  buffer: RwLockReadGuard<'a, Buffer<T>>
}

impl<'a, T: Num> PrivateData for RwReadRepr<'a, T> {}

impl<'a, T: Num> Data for RwReadRepr<'a, T> {
  type Elem = T;
}

impl<'a, T: Num> DataRef for RwReadRepr<'a, T> {
  fn buffer(&self) -> &Buffer<T> {
    &*self.buffer
  }
}

#[doc(hidden)]
pub struct RwWriteRepr<'a, T: Num> {
  buffer: RwLockWriteGuard<'a, Buffer<T>>
}

impl<'a, T: Num> PrivateData for RwWriteRepr<'a, T> {}

impl<'a, T: Num> Data for RwWriteRepr<'a, T> {
  type Elem = T;
}

impl<'a, T: Num> DataRef for RwWriteRepr<'a, T> {
  fn buffer(&self) -> &Buffer<T> {
    &*self.buffer
  }
}

impl<'a, T: Num> DataMut for RwWriteRepr<'a, T> {
  fn buffer_mut(&mut self) -> &mut Buffer<T> {
    &mut *self.buffer
  }
}

#[derive(Clone)]
pub struct TensorBase<S: Data, D: Dimension> {
  device: Device,  
  dim: D,
  data: S
}

pub type Tensor<T, D> = TensorBase<OwnedRepr<T>, D>;
pub type Tensor0<T> = Tensor<T, Ix0>;
pub type Tensor2<T> = Tensor<T, Ix2>;

pub type TensorView<'a, T, D> = TensorBase<ViewRepr<&'a Buffer<T>>, D>;
pub type TensorView2<'a, T> = TensorView<'a, T, Ix2>;

pub type TensorViewMut<'a, T, D> = TensorBase<ViewRepr<&'a mut Buffer<T>>, D>;
pub type TensorViewMut2<'a, T> = TensorViewMut<'a, T, Ix2>;

pub type ArcTensor<T, D> = TensorBase<ArcRepr<T>, D>;
pub type ArcTensor2<T> = ArcTensor<T, Ix2>;

pub type RwTensor<T, D> = TensorBase<RwRepr<T>, D>;
pub type RwTensor0<T> = RwTensor<T, Ix0>;
pub type RwTensor2<T> = RwTensor<T, Ix2>;

pub type RwReadTensor<'a, T, D> = TensorBase<RwReadRepr<'a, T>, D>;
pub type RwWriteTensor<'a, T, D> = TensorBase<RwWriteRepr<'a, T>, D>;

impl<T: Num, S: DataOwned<Elem=T>, D: Dimension> TensorBase<S, D> {
  pub unsafe fn uninitialized(device: &Device, shape: impl IntoDimension<Dim=D>) -> Self {
    let device = device.clone();
    let dim = shape.into_dimension();
    let data = S::from_buffer(Buffer::uninitialized(&device, dim.size()));
    Self{device, dim, data}
  }
  pub fn from_shape_vec<'a>(device: &Device, shape: impl IntoDimension<Dim=D>, vec: impl Into<Cow<'a, [T]>>) -> Self {
    let device = device.clone();
    let dim = shape.into_dimension();
    let vec = vec.into();
    debug_assert_eq!(dim.size(), vec.len());
    let data = S::from_buffer(Buffer::from_vec(&device, vec));
    Self{device, dim, data}
  } 
  pub fn zeros(device: &Device, shape: impl IntoDimension<Dim=D>) -> Self {
    let device = device.clone();
    let dim = shape.into_dimension();
    let data = S::from_buffer(Buffer::zeros(&device, dim.size()));
    Self{device, dim, data}
  }
  pub fn ones(device: &Device, shape: impl IntoDimension<Dim=D>) -> Self {
    let device = device.clone();
    let dim = shape.into_dimension();
    let mut buffer = unsafe { Buffer::uninitialized(&device, dim.size()) };
    buffer.fill(T::one());
    let data = S::from_buffer(buffer);
    Self{device, dim, data}
  }
}

impl<T: Num, S: Data<Elem=T>, D: Dimension> TensorBase<S, D> {
  pub fn device(&self) -> &Device {
    &self.device
  }
  pub fn raw_dim(&self) -> D {
    self.dim.clone()
  }
  pub fn dim(&self) -> D::Pattern {
    self.dim.clone()
      .into_pattern()
  }
  pub fn len(&self) -> usize {
    self.dim.size()
  }
}

impl<T: Num, S: DataRef<Elem=T>, D: Dimension> TensorBase<S, D> {
  pub fn view(&self) -> TensorView<T, D> {
    let device = self.device.clone();
    let dim = self.dim.clone();
    let data = ViewRepr::new(self.data.buffer());
    TensorView{device, dim, data}
  }
  pub fn as_slice(&self) -> Cow<[T]> {
    self.data.buffer()
      .as_slice()
  }
  pub fn as_array(&self) -> CowArray<T, D> {
    let dim = self.dim.clone();
    match self.data.buffer().as_slice() {
      Cow::Owned(vec) => {
        unsafe { Array::from_shape_vec_unchecked(dim, vec) }.into()
      },
      Cow::Borrowed(slice) => {
        unsafe { ArrayView::from_shape_ptr(dim, slice.as_ptr()) }.into()
      }
    }
  }
  fn as_cpu_slice(&self) -> Option<&[T]> {
    self.data.buffer()
      .cpu()
      .map(|b| b.as_slice())
  }
  fn as_cpu_ptr(&self) -> Option<*const T> {
    self.data.buffer()
      .cpu()
      .map(|b| b.as_ptr())
  }
  #[cfg(feature="cuda")]
  fn as_cuda_ptr(&self) -> Option<*const T> {
    self.data.buffer()
      .cuda()
      .map(|b| b.as_ptr())
  }
}

impl<S: DataRef<Elem=f32>, D: Dimension> TensorBase<S, D> {
  pub fn sum(&self) -> Tensor0<f32> {
    let mut output = unsafe { Tensor::uninitialized(&self.device, ()) };
    match &self.device {
      Device::Cpu(cpu) => cpu::reduce_sum(self, &mut output),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => cuda::reduce_sum(self, &mut output)
    }
    output
  }
}

impl<S: DataMut<Elem=f32>, D: Dimension> TensorBase<S, D> {
  pub fn scaled_add<S2: DataRef<Elem=f32>>(&mut self, alpha: f32, rhs: &TensorBase<S2, D>) {
    debug_assert_eq!(&self.device, &rhs.device);
    debug_assert_eq!(&self.dim, &rhs.dim);
    match &self.device {
      Device::Cpu(cpu) => cpu::scaled_add(self, alpha, rhs),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => cuda::scaled_add(self, alpha, rhs)
    }
  }
}

impl<T: Unsigned, S: DataRef<Elem=T>, D: Dimension> TensorBase<S, D> {
  pub fn to_f32(&self) -> Tensor<f32, D> {
    let mut output = unsafe { Tensor::uninitialized(&self.device, self.dim.clone()) };
    match &self.device {
      Device::Cpu(cpu) => cpu::unsigned_to_f32(self, &mut output),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => cuda::unsigned_to_f32(self, &mut output)
    }
    output
  }
}

impl<T: Unsigned, S: DataRef<Elem=T>> TensorBase<S, Ix1> {
  pub fn to_one_hot_f32(&self, nclasses: usize) -> Tensor2<f32> {
    let mut output = Tensor2::zeros(&self.device, [self.len(), nclasses]);
    match &self.device {
      Device::Cpu(cpu) => cpu::unsigned_to_one_hot_f32(self, &mut output),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => cuda::unsigned_to_one_hot_f32(self, &mut output)
    }
    output
  }
}

impl<T: Num, S: DataMut<Elem=T>, D: Dimension> TensorBase<S, D> {
  pub fn view_mut(&mut self) -> TensorViewMut<T, D> {
    let device = self.device.clone();
    let dim = self.dim.clone();
    let data = ViewRepr::new(self.data.buffer_mut());
    TensorViewMut{device, dim, data}
  }
  fn as_mut_cpu_slice(&mut self) -> Option<&mut [T]> {
    self.data.buffer_mut()
      .cpu_mut()
      .map(|mut b| b.as_mut_slice())
  }
  fn as_mut_cpu_ptr(&mut self) -> Option<*mut T> {
    self.data.buffer_mut()
      .cpu_mut()
      .map(|mut b| b.as_mut_ptr())
  }
  #[cfg(feature="cuda")]
  fn as_mut_cuda_ptr(&mut self) -> Option<*mut T> {
    self.data.buffer_mut()
      .cuda_mut()
      .map(|mut b| b.as_mut_ptr())
  }
  
  pub fn fill(&mut self, elem: T) {
    self.data.buffer_mut()
      .fill(elem);
  }
}

impl<T: Num, D: Dimension> From<Tensor<T, D>> for ArcTensor<T, D> {
  fn from(tensor: Tensor<T, D>) -> Self {
    let Tensor{device, dim, data} = tensor;
    let data = ArcRepr::from_buffer(data.buffer);
    Self{device, dim, data}
  }
}

impl<T: Num, D: Dimension> RwTensor<T, D> {
  pub fn read(&self) -> LockResult<RwReadTensor<T, D>> {
    match self.data.read() {
      Ok(data) => {
        let device = self.device.clone();
        let dim = self.dim.clone();
        Ok(RwReadTensor{device, dim, data})
      }
      Err(poison_error) => {
        let data = poison_error.into_inner();
        let device = self.device.clone();
        let dim = self.dim.clone();
        Err(PoisonError::new(RwReadTensor{device, dim, data}))
      }
    }
  }
  pub fn write(&self) -> LockResult<RwWriteTensor<T, D>> {
    match self.data.write() {
      Ok(data) => {
        let device = self.device.clone();
        let dim = self.dim.clone();
        Ok(RwWriteTensor{device, dim, data})
      }
      Err(poison_error) => {
        let data = poison_error.into_inner();
        let device = self.device.clone();
        let dim = self.dim.clone();
        Err(PoisonError::new(RwWriteTensor{device, dim, data}))
      }
    }
  }
}

impl<T: Num, D: Dimension> From<Tensor<T, D>> for RwTensor<T, D> {
  fn from(tensor: Tensor<T, D>) -> Self {
    let Tensor{device, dim, data} = tensor;
    let data = RwRepr::from_buffer(data.buffer);
    Self{device, dim, data}
  }
}

fn broadcast<T: Num, D: Dimension, S1: DataRef<Elem=T>, S2: DataMut<Elem=T>>(input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D>) {
  debug_assert_eq!(input.device(), output.device());
  match input.device() {
    Device::Cpu(cpu) => cpu::broadcast(input, output),
    #[cfg(feature="cuda")]
    Device::Cuda(cuda_gpu) => cuda::broadcast(input, output)
  }
}

fn broadcast_backward<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, D: Dimension>
  (input_grad: &mut TensorBase<S1, D>, output_grad: &TensorBase<S2, D>) {
  debug_assert_eq!(input_grad.device(), output_grad.device());
  match input_grad.device() {
    Device::Cpu(cpu) => cpu::broadcast_backward(input_grad, output_grad),
    #[cfg(feature="cuda")]
    Device::Cuda(cuda_gpu) => cuda::broadcast_backward(input_grad, output_grad)
  }
}

#[derive(Clone, Copy, PartialEq)]
enum Transpose {
  No,
  Yes
}

fn gemm<S1: DataRef<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>>
  (alpha: f32, a: &TensorBase<S1, Ix2>, trans_a: Transpose, b: &TensorBase<S2, Ix2>, trans_b: Transpose, beta: f32, c: &mut TensorBase<S3, Ix2>) {
  debug_assert_eq!(&a.device, &b.device);
  debug_assert_eq!(&a.device, &c.device);
  match &a.device {
    Device::Cpu(_) => cpu::gemm(alpha, a, trans_a, b, trans_b, beta, c),
    #[cfg(feature="cuda")]
    Device::Cuda(_) => cuda::gemm(alpha, a, trans_a, b, trans_b, beta, c)
  }
} 

fn cross_entropy_backward<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, S3: DataRef<Elem=f32>, S4: DataRef<Elem=f32>>
  (input: &TensorBase<S1, Ix2>, input_grad: &mut TensorBase<S2, Ix2>,
   target: &TensorBase<S3, Ix2>, 
   output_grad: &TensorBase<S4, Ix0>) {
  let device = &input.device;
  debug_assert_eq!(device, &input_grad.device);
  debug_assert_eq!(device, &target.device);
  debug_assert_eq!(device, &output_grad.device);
  debug_assert_eq!(input.raw_dim(), input_grad.raw_dim());
  debug_assert_eq!(input.raw_dim(), target.raw_dim());
  match device {
    Device::Cpu(cpu) => cpu::cross_entropy_backward(input, input_grad, target, output_grad),
    #[cfg(feature="cuda")]
    Device::Cuda(cuda_gpu) => cuda::cross_entropy_backward(input, input_grad, target, output_grad)
  }
}

impl<S1: DataRef<Elem=f32>> TensorBase<S1, Ix2> {
  pub fn dense(&self, weight: &TensorView2<f32>, bias: Option<&TensorView2<f32>>) -> Tensor2<f32> {
    let (batch_size, inputs) = self.dim();
    let (outputs, inputs2) = weight.dim();
    debug_assert_eq!(inputs, inputs2);
    let mut output = unsafe { Tensor::uninitialized(&self.device, [batch_size, outputs]) };
    if let Some(bias) = bias {
      broadcast(bias, &mut output);
      gemm(1., &self, Transpose::No, &weight, Transpose::Yes, 1., &mut output);
    }
    else {
      gemm(1., &self, Transpose::No, &weight, Transpose::Yes, 0., &mut output);
    }
    output
  }
  pub fn cross_entropy_loss(&self, target: &TensorView2<f32>) -> Tensor0<f32> {
    debug_assert_eq!(&self.device, &target.device);
    debug_assert_eq!(self.raw_dim(), target.raw_dim());
    let mut output = unsafe { Tensor::uninitialized(&self.device, self.raw_dim()) };
    match &self.device {
      Device::Cpu(cpu) => cpu::cross_entropy(self, target, &mut output),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => cuda::cross_entropy_loss(self, target)
    }
    output.sum()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  
  fn test_dense(device: impl Into<Device>) {
    let device = device.into();
    let x = Tensor::from_shape_vec(&device, [2, 3], vec![1., 2., 3., 4., 5., 6.]);
    let w = Tensor::from_shape_vec(&device, [4, 3], vec![7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.]); 
    let y = x.dense(&w.view(), None);
    let x_arr = x.as_array();
    let w_arr = w.as_array();
    let y_arr = y.as_array();
    assert_eq!(x_arr.dot(&w_arr.t()), y_arr); 
  }
  #[test]
  fn test_dense_cpu() {
    test_dense(Cpu::new());
  } 
  #[cfg(feature="cuda")]
  #[test]
  fn test_dense_cuda() {
    test_dense(CudaGpu::new(0));
  }
}

