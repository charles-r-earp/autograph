#![allow(warnings)]
#![recursion_limit="1024"]
use std::sync::{Arc, RwLock, LockResult, PoisonError, RwLockReadGuard, RwLockWriteGuard};
use std::borrow::Cow;
use num_traits::{Zero, One, ToPrimitive, Bounded};
use ndarray::{Array, ArrayView, CowArray, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis};
#[cfg(feature="cuda")]
use rustacuda::memory::{DeviceCopy, DeviceSlice};

#[doc(hidden)]
pub mod cpu;
pub use cpu::{Cpu, CpuBuffer};

#[cfg(feature="cuda")]
pub mod cuda;
#[cfg(feature="cuda")]
pub use cuda::{CudaGpu, CudaBuffer};

pub mod autograd;

pub mod layer;

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
pub type Tensor1<T> = Tensor<T, Ix1>;
pub type Tensor2<T> = Tensor<T, Ix2>;
pub type Tensor4<T> = Tensor<T, Ix4>;

pub type TensorView<'a, T, D> = TensorBase<ViewRepr<&'a Buffer<T>>, D>;
pub type TensorView1<'a, T> = TensorView<'a, T, Ix1>;
pub type TensorView2<'a, T> = TensorView<'a, T, Ix2>;
pub type TensorView4<'a, T> = TensorView<'a, T, Ix4>;

pub type TensorViewMut<'a, T, D> = TensorBase<ViewRepr<&'a mut Buffer<T>>, D>;
pub type TensorViewMut2<'a, T> = TensorViewMut<'a, T, Ix2>;
pub type TensorViewMut4<'a, T> = TensorViewMut<'a, T, Ix4>;

pub type ArcTensor<T, D> = TensorBase<ArcRepr<T>, D>;
pub type ArcTensor2<T> = ArcTensor<T, Ix2>;

pub type RwTensor<T, D> = TensorBase<RwRepr<T>, D>;
pub type RwTensor0<T> = RwTensor<T, Ix0>;
pub type RwTensor1<T> = RwTensor<T, Ix1>;
pub type RwTensor2<T> = RwTensor<T, Ix2>;
pub type RwTensor3<T> = RwTensor<T, Ix3>;
pub type RwTensor4<T> = RwTensor<T, Ix4>;

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
  pub fn from_array<'a>(device: &Device, array: impl Into<CowArray<'a, T, D>>) -> Self {
    let array = array.into();
    if let Some(slice) = array.as_slice() {
      Self::from_shape_vec(&device, array.raw_dim(), slice)
    }
    else {
      let vec: Vec::<T> = array.iter()
        .copied()
        .collect();
      Self::from_shape_vec(&device, array.raw_dim(), vec)
    }
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
  pub fn into_dyn(self) -> TensorBase<S, IxDyn> {
    TensorBase {
      device: self.device,
      dim: self.dim.into_dyn(),
      data: self.data
    }
  }
  pub fn into_dimensionality<D2: Dimension>(self) -> Option<TensorBase<S, D2>> {
    D2::from_dimension(&self.dim)
      .map(|dim| {
        TensorBase {
          device: self.device,
          dim,
          data: self.data
        }
      })
  }
  pub fn into_shape<D2: Dimension>(self, shape: impl IntoDimension<Dim=D2>) -> Option<TensorBase<S, D2>> {
    let dim = shape.into_dimension();
    if self.dim.size() == dim.size() {
      Some(TensorBase {
        device: self.device,
        dim,
        data: self.data
      })
    } else { None }
  } 
  pub fn into_flatten(self) -> TensorBase<S, Ix2>
    where D: RemoveAxis {
    let batch_size = self.dim[0];
    let inputs = self.dim.slice()[1..].iter().product();
    self.into_shape([batch_size, inputs])
      .unwrap()
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
  #[cfg(feature="cuda")]
  fn as_cuda_slice(&self) -> Option<&DeviceSlice<T>> {
    self.data.buffer()
      .cuda()
      .map(|b| b.as_device_slice())
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
  pub fn relu(&self) -> Tensor<f32, D> {
    let mut output = unsafe { Tensor::uninitialized(&self.device, self.raw_dim()) };
    match &self.device {
      Device::Cpu(cpu) => cpu::relu(self, &mut output),
      #[cfg(feature="cuda")]
      Device::Cuda(cuda_gpu) => cuda::relu(self, &mut output),
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
  #[cfg(feature="cuda")]
  fn as_mut_cuda_slice(&mut self) -> Option<&mut DeviceSlice<T>> {
    self.data.buffer_mut()
      .cuda_mut()
      .map(|b| b.as_mut_device_slice())
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

fn broadcast<T: Num, S1: DataRef<Elem=T>, S2: DataMut<Elem=T>, D: Dimension>(input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D::Larger>) {
  debug_assert_eq!(input.device(), output.device());
  match input.device() {
    Device::Cpu(cpu) => cpu::broadcast(input, output),
    #[cfg(feature="cuda")]
    Device::Cuda(cuda_gpu) => cuda::broadcast(input, output)
  }
}

fn broadcast_backward<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, D: Dimension>
  (input_grad: &mut TensorBase<S1, D>, output_grad: &TensorBase<S2, D::Larger>) {
  debug_assert_eq!(input_grad.device(), output_grad.device());
  match input_grad.device() {
    Device::Cpu(cpu) => cpu::broadcast_backward(input_grad, output_grad),
    #[cfg(feature="cuda")]
    Device::Cuda(cuda_gpu) => cuda::broadcast_backward(input_grad, output_grad)
  }
}

fn relu_backward<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, S3: DataRef<Elem=f32>, D: Dimension>
  (input: &TensorBase<S1, D>, input_grad: &mut TensorBase<S2, D>, output_grad: &TensorBase<S3, D>) {
  debug_assert_eq!(input.device(), input_grad.device());
  debug_assert_eq!(input.device(), output_grad.device());
  debug_assert_eq!(input.raw_dim(), input_grad.raw_dim());
  debug_assert_eq!(input.raw_dim(), output_grad.raw_dim());
  match input.device() {
    Device::Cpu(cpu) => cpu::relu_backward(input, input_grad, output_grad),
    #[cfg(feature="cuda")]
    Device::Cuda(cuda_gpu) => cuda::relu_backward(input, input_grad, output_grad)
  } 
} 

#[derive(Clone, Copy, PartialEq)]
pub enum Transpose {
  No,
  Yes
}

pub fn gemm<S1: DataRef<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>>
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
  pub fn dense(&self, weight: &TensorView2<f32>, bias: Option<&TensorView1<f32>>) -> Tensor2<f32> {
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
      Device::Cuda(cuda_gpu) => cuda::cross_entropy(self, target, &mut output)
    }
    output.sum()
  }
}

pub trait Into2d {
  fn into_2d(self) -> [usize; 2];
}

impl Into2d for [usize; 2] {
  fn into_2d(self) -> Self {
    self
  }
}

impl Into2d for (usize, usize) {
  fn into_2d(self) -> [usize; 2] {
    [self.0, self.1]
  }
}

impl Into2d for usize {
  fn into_2d(self) -> [usize; 2] {
    [self, self]
  }
}

#[derive(Clone, Copy)]
pub struct Conv2dArgs {
  strides: [usize; 2],
  padding: [usize; 2]
}

impl Conv2dArgs {
  pub fn strides(mut self, strides: impl Into2d) -> Self {
    self.strides = strides.into_2d();
    self
  }
  pub fn padding(mut self, padding: impl Into2d) -> Self {
    self.padding = padding.into_2d();
    self
  }
}

impl Default for Conv2dArgs {
  fn default() -> Self {
    Self {
      strides: [1, 1],
      padding: [0, 0]
    }
  }
}

#[derive(Clone, Copy)]
pub struct Pool2dArgs {
  kernel: [usize; 2],
  strides: [usize; 2],
  padding: [usize; 2]
}

impl Default for Pool2dArgs {
  fn default() -> Self {
    Self {
      kernel: [2, 2],
      strides: [1, 1],
      padding: [0, 0]
    }
  }
}

impl Pool2dArgs {
  pub fn kernel(mut self, kernel: impl Into2d) -> Self {
    self.kernel = kernel.into_2d();
    self
  } 
  pub fn strides(mut self, strides: impl Into2d) -> Self {
    self.strides = strides.into_2d();
    self
  }
  pub fn padding(mut self, padding: impl Into2d) -> Self {
    self.padding = padding.into_2d();
    self
  }
}

impl<S1: DataRef<Elem=f32>> TensorBase<S1, Ix4> {
  pub fn conv2d(&self, weight: &TensorView4<f32>, bias: Option<&TensorView1<f32>>, args: &Conv2dArgs) -> Tensor4<f32> {
    let device = &self.device;
    let (batch_size, inputs, ih, iw) = self.dim();
    let (outputs, _, kh, kw) = weight.dim();
    debug_assert_eq!(device, &weight.device);
    debug_assert_eq!(weight.dim(), (outputs, inputs, kh, kw));
    #[cfg(debug_assertions)]
    {
      if let Some(bias) = &bias {
        assert_eq!(device, &bias.device);
        assert_eq!(bias.dim(), outputs);
      }
    }
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let oh = (ih - kh + 2 * ph) / sh + 1;
    let ow = (iw - kw + 2 * pw) / sw + 1;
    let mut output = if ph == 0 || pw == 0 { 
      unsafe { Tensor::uninitialized(&device, [batch_size, outputs, oh, ow]) }
    }
    else {
      Tensor::zeros(&device, [batch_size, outputs, oh, ow])
    };
    match device {
      Device::Cpu(_) => cpu::conv2d(self, weight, bias, args, &mut output),
      #[cfg(feature="cuda")]
      Device::Cuda(_) => cuda::conv2d(self, weight, bias, args, &mut output)
    } 
    output
  }
  pub fn max_pool2d(&self, args: &Pool2dArgs) -> Tensor4<f32> {
    let device = &self.device;
    let (batch_size, inputs, ih, iw) = self.dim();
    let [kh, kw] = args.kernel;
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let oh = (ih - (kh - 1) + 2 * ph - 1) / sh + 1;
    let ow = (iw - (kw - 1) + 2 * pw - 1) / sw + 1;
    let mut output = unsafe { Tensor::uninitialized(&device, [batch_size, inputs, oh, ow]) };
    match device {
      Device::Cpu(_) => cpu::max_pool2d(self, args, &mut output),
      #[cfg(feature="cuda")]
      Device::Cuda(_) => cuda::max_pool2d(self, args, &mut output)
    }
    output
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::{ArrayView2, ArrayViewMut2};
  
  fn test_tensor_from_vec(device: impl Into<Device>) {
    let device = device.into();
    let vec = vec![1., 2., 3., 4.];
    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let vec_out = x.as_slice().into_owned();
    assert_eq!(vec, vec_out);
  }
  #[test]
  fn test_tensor_from_vec_cpu() {
    test_tensor_from_vec(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_tensor_from_vec_cuda() {
    test_tensor_from_vec(CudaGpu::new(0));
  }
  fn test_u8_to_f32(device: impl Into<Device>) {
    let device = device.into();
    let vec: Vec<u8> = vec![1, 2, 3, 4];
    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let y = x.to_f32();
    let vec_out = y.as_slice().into_owned();
    let scale = 255f32.recip(); 
    let vec_true: Vec<f32> = vec.iter()
      .map(|x| scale * x.to_f32().unwrap())
      .collect();
    assert_eq!(vec_out, vec_true);
  }
  #[test]
  fn test_u8_to_f32_cpu() {
    test_u8_to_f32(Cpu::new());
  }
  #[cfg(feature="cuda")]
  fn test_u8_to_f32_cuda() {
    test_u8_to_f32(CudaGpu::new(0));
  }
  fn test_u8_to_one_hot_f32(device: impl Into<Device>) {
    let device = device.into();
    let vec: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let y = x.to_one_hot_f32(8);
    let mut y_true = Array::zeros([6, 8]);
    y_true.outer_iter_mut()
      .into_iter()
      .zip(vec.iter())
      .for_each(|(mut y, &x)| {
        y[x as usize] = 1.;
      });
    let y_out = y.as_array().into_owned();
    assert_eq!(y_out, y_true);
  }
  #[test]
  fn test_u8_to_one_hot_f32_cpu() {
    test_u8_to_one_hot_f32(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_u8_to_one_hot_f32_cuda() {
    test_u8_to_one_hot_f32(CudaGpu::new(0));
  }
  fn test_fill_u8(device: impl Into<Device>) {
    let device = device.into();
    let n = 10;
    let mut x = Tensor::zeros(&device, n);
    assert_eq!(x.as_slice(), vec![0u8; n].as_slice());
    x.fill(1u8);
    assert_eq!(x.as_slice(), vec![1u8; n].as_slice()); 
  }
  #[test]
  fn test_fill_u8_cpu() {
    test_fill_u8(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_fill_u8_cuda() {
    test_fill_u8(CudaGpu::new(0));
  }
  fn test_fill_f32(device: impl Into<Device>) {
    let device = device.into();
    let n = 10;
    let mut x = Tensor::zeros(&device, n);
    assert_eq!(x.as_slice(), vec![0f32; n].as_slice());
    x.fill(1f32);
    assert_eq!(x.as_slice(), vec![1f32; n].as_slice()); 
  }
  #[test]
  fn test_fill_f32_cpu() {
    test_fill_f32(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_fill_f32_cuda() {
    test_fill_f32(CudaGpu::new(0));
  }
  fn test_broadcast(device: impl Into<Device>) {
    let device = device.into();
    let x = Tensor::from_shape_vec(&device, 4, vec![1., 2., 3., 4.]);
    let mut y = Tensor::zeros(&device, [2, 4]);
    broadcast(&x, &mut y);
    let y_out = y.as_slice().into_owned();
    let y_true = vec![1., 2., 3., 4., 1., 2., 3., 4.];
    assert_eq!(y_out, y_true);
  }
  #[test]
  fn test_broadcast_cpu() {
    test_broadcast(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_broadcast_cuda() {
    test_broadcast(CudaGpu::new(0));
  }
  fn test_broadcast_backward(device: impl Into<Device>) {
    let device = device.into();
    let mut dx = Tensor::zeros(&device, 4);
    let dy = Tensor::from_shape_vec(&device, [2, 4], vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    broadcast_backward(&mut dx, &dy);
    let dx_out = dx.as_slice().into_owned();
    let dx_true = vec![6., 8., 10., 12.];
    assert_eq!(dx_out, dx_true);
  }
  #[test]
  fn test_broadcast_backward_cpu() {
    test_broadcast_backward(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_broadcast_backward_cuda() {
    test_broadcast_backward(CudaGpu::new(0));
  }
  fn compare_vectors(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    // dnnl and cuda have fast mul / approx ops
    // compared to ndarray / matrixmultiply which performs exact ops
    // assert_eq fails for gemm with large matrices
    // https://oneapi-src.github.io/oneDNN/cpu_sgemm_and_matmul_8cpp-example.html
    let mut v1_l2 = 0f64;
    let mut diff_l2 = 0f64;
    a.iter()
      .zip(b.iter())
      .for_each(|(&a, &b)| {
        v1_l2 += (a * b) as f64;
        diff_l2 += (a - b).powi(2) as f64;
      });
    let threshold = (f32::EPSILON as f64) * f64::ln(f64::max(2., k.to_f64().unwrap()));
    assert!(diff_l2.sqrt() <= threshold * v1_l2.sqrt(), "m: {} k: {} n: {} ({} !<= {})", m, k, n, diff_l2.sqrt(), threshold * v1_l2.sqrt());
  }
  fn test_gemm_mkn(m: usize, k: usize, n: usize, device: impl Into<Device>) {
    let device = device.into(); 
    
    let vec1: Vec<f32> = (1 ..= m*k).into_iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    let vec2: Vec<f32> = (1 ..= k*n).into_iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    
    { // MxK * KxN
      let x1 = Tensor::from_shape_vec(&device, [m, k], &vec1);
      let x2 = Tensor::from_shape_vec(&device, [k, n], &vec2);
      let mut y = Tensor::zeros(&device, [m, n]);
      gemm(1., &x1, Transpose::No, &x2, Transpose::No, 0., &mut y);
      let y_true = x1.as_array()
        .dot(&x2.as_array());
      compare_vectors(&y.as_slice(), y_true.as_slice().unwrap(), m, k, n);
    }
    { // KxM^T * KxN
      let x1 = Tensor::from_shape_vec(&device, [k, m], &vec1);
      let x2 = Tensor::from_shape_vec(&device, [k, n], &vec2);
      let mut y = Tensor::zeros(&device, [m, n]);
      gemm(1., &x1, Transpose::Yes, &x2, Transpose::No, 0., &mut y);
      let y_true = x1.as_array()
        .t()
        .dot(&x2.as_array());
      compare_vectors(&y.as_slice(), y_true.as_slice().unwrap(), m, k, n);
    }
    { // MxK * NxK^T
      let x1 = Tensor::from_shape_vec(&device, [m, k], &vec1);
      let x2 = Tensor::from_shape_vec(&device, [n, k], &vec2);
      let mut y = Tensor::zeros(&device, [m, n]);
      gemm(1., &x1, Transpose::No, &x2, Transpose::Yes, 0., &mut y);
      let y_true = x1.as_array()
        .dot(&x2.as_array().t());
      compare_vectors(&y.as_slice(), y_true.as_slice().unwrap(), m, k, n);
    }
    { // KxM^T * NxK^T
      let x1 = Tensor::from_shape_vec(&device, [k, m], &vec1);
      let x2 = Tensor::from_shape_vec(&device, [n, k], &vec2);
      let mut y = Tensor::zeros(&device, [m, n]);
      gemm(1., &x1, Transpose::Yes, &x2, Transpose::Yes, 0., &mut y);
      let y_true: Vec<f32> = x1.as_array()
        .t()
        .dot(&x2.as_array().t())
        .iter()
        .copied()
        .collect();
      compare_vectors(&y.as_slice(), &y_true, m, k, n);
    }
  }
  fn test_gemm(device: impl Into<Device>) {
    test_gemm_mkn(33, 43, 53, device);
  }
  #[test]
  fn test_gemm_cpu() {
    test_gemm(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_gemm_cuda() {
    test_gemm(CudaGpu::new(0));
  }
  fn test_sum(device: impl Into<Device>) {
    let device = device.into();
    
    let vec: Vec<f32> = (1 ..= 100).into_iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    
    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let y = x.sum().as_slice()[0];
    assert_eq!(y, vec.iter().sum::<f32>());
  }
  #[test]
  fn test_sum_cpu() {
    test_sum(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_sum_cuda() {
    test_sum(CudaGpu::new(0));
  }
  fn test_relu(device: impl Into<Device>) {
    let device = device.into();
    let x = Tensor::from_shape_vec(&device, 6, vec![-0.1, -100., 0.0, 0.1, 1., 100.]);
    let y = x.relu();
    let y_vec = y.as_slice().into_owned();
    debug_assert_eq!(y_vec, vec![0., 0., 0., 0.1, 1., 100.]); 
  }
  #[test]
  fn test_relu_cpu() {
    test_relu(Cpu::new());
  }
  #[cfg(feature="cuda")]
  fn test_relu_cuda() {
    test_relu(CudaGpu::new(0));
  }
  fn test_relu_backward(device: impl Into<Device>) {
    let device = device.into();
    let x = Tensor::from_shape_vec(&device, 6, vec![-0.1, -100., 0.0, 0.1, 1., 100.]);
    let mut dx = Tensor1::<f32>::ones(&device, 6);
    let dy = Tensor::from_shape_vec(&device, 6, vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]);
    let dx_vec = dx.as_slice().into_owned();
    let mut dx_vec_true = vec![0.; dx_vec.len()];
    x.as_slice()
      .iter()
      .zip(dx_vec_true.iter_mut())
      .zip(dy.as_slice().iter())
      .for_each(|((&x, dx), &dy)| {
        if x >= 0. {
          *dx += dy;
        }
      });
    debug_assert_eq!(dx_vec, vec![0., 0., 0., 0.1, 1., 100.]); 
  }
  #[test]
  fn test_relu_backard_cpu() {
    test_relu(Cpu::new());
  }
  #[cfg(feature="cuda")]
  fn test_relu_backward_cuda() {
    test_relu(CudaGpu::new(0));
  }
  fn test_scaled_add(device: impl Into<Device>) {
    let device = device.into();
    
    let mut lhs = Tensor::zeros(&device, 10);
    let rhs = Tensor::from_shape_vec(
      &device,
      10,
      vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    );
    
    let alpha = 2.;
    
    lhs.scaled_add(alpha, &rhs);
    
    let mut lhs_true = Array::zeros(lhs.raw_dim());
    lhs_true.scaled_add(alpha, &rhs.as_array());
    
    let success = lhs.as_slice()
      .iter()
      .zip(lhs_true.as_slice().unwrap())
      .all(|(a, b)| {
        approx::relative_eq!(a, b, max_relative = 0.00001)
      });
    assert!(success, "{:?} {:?}", lhs.as_slice(), lhs_true.as_slice().unwrap());
  }
  fn test_cross_entropy(device: impl Into<Device>) {
    let device = device.into();
    
    let batch_size = 3;
    let nclasses = 4;
    
    let input = Tensor::from_shape_vec(
      &device,
      [batch_size, nclasses],
      vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
    );
    
    let target = Tensor::from_shape_vec(
      &device,
      [batch_size, nclasses],
      vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
    );
    
    let mut output = Tensor::zeros(
      &device,
      [batch_size, nclasses]
    );
    
    match &device {
      Device::Cpu(_) => { cpu::cross_entropy(&input, &target, &mut output); }
      #[cfg(feature="cuda")]
      Device::Cuda(_) => { cuda::cross_entropy(&input, &target, &mut output); }
    }
    
    let mut output_true = vec![0.; batch_size*nclasses];
    input.as_slice()
      .chunks_exact(nclasses)
      .zip(target.as_slice().chunks_exact(nclasses))
      .zip(output_true.chunks_exact_mut(nclasses))
      .for_each(|((input, target), mut output)| {
        let mut m = input[0];
        input.iter()
          .for_each(|&x| m = f32::max(x, m));
        output.iter_mut()
          .zip(input.iter())
          .for_each(|(y, &x)| *y = x-m);
        let s: f32 = output.iter()
          .map(|&y| y.exp())
          .sum();
        let ln_s = s.ln();
        output.iter_mut()
          .zip(target.iter())
          .for_each(|(y, t)| *y = (ln_s - *y) * t);  
      });
    let output = output.as_slice();
    let success = output.iter()
      .zip(output_true.as_slice())
      .all(|(a, b)| {
        approx::relative_eq!(a, b, max_relative = 0.00001)
      });
    assert!(success, "{:?} {:?}", output, output_true);
  }
  #[test]
  fn test_cross_entropy_cpu() {
    test_cross_entropy(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_cross_entropy_cuda() {
    test_cross_entropy(CudaGpu::new(0));
  }
  fn test_cross_entropy_backward(device: impl Into<Device>) {
    let device = device.into();
    
    let batch_size = 3;
    let nclasses = 4;
    
    let input = Tensor::from_shape_vec(
      &device,
      [batch_size, nclasses],
      vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
    );
    
    let mut input_grad = Tensor::zeros(
      &device,
      [batch_size, nclasses]
    );
    
    let target = Tensor::from_shape_vec(
      &device,
      [batch_size, nclasses],
      vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
    );
    
    let output_grad = Tensor::from_shape_vec(
      &device,
      (),
      vec![1.]
    );
    
    cross_entropy_backward(&input, &mut input_grad, &target, &output_grad); 
    
    let mut input_grad_true = vec![0.; batch_size*nclasses];
    input.as_slice()
      .iter()
      .zip(input_grad_true.iter_mut())
      .zip(target.as_slice().iter())
      .for_each(|((x, mut dx), t)| {
        *dx = x - t;
      });     
    let input_grad = input_grad.as_slice();
    let success = input_grad.iter()
      .zip(input_grad_true.as_slice())
      .all(|(a, b)| {
        approx::relative_eq!(a, b, max_relative = 0.00001)
      });
    assert!(success, "{:?} {:?}", input_grad, input_grad_true);
  }
  #[test]
  fn test_cross_entropy_backward_cpu() {
    test_cross_entropy_backward(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_cross_entropy_backward_cuda() {
    test_cross_entropy_backward(CudaGpu::new(0));
  }
  fn test_conv2d_with_args(input_dim: impl IntoDimension<Dim=Ix4>, outputs: usize, kernel: impl Into2d, use_bias: bool, args: &Conv2dArgs, device: impl Into<Device>) {
    let kernel = kernel.into_2d();
    let device = device.into();
    let input_dim = input_dim.into_dimension();
    let (batch_size, inputs, ih, iw) = input_dim.into_pattern();
    let [kh, kw] = kernel;
    let input_vec: Vec<f32> = (1 ..= input_dim.size())
      .into_iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    let weight_dim = [outputs, inputs, kh, kw].into_dimension();
    let weight_vec: Vec<f32> = (1 ..= weight_dim.size())
      .into_iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    let bias_vec: Vec<f32> = (1 ..= outputs).into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let output = {
      let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
      let weight = Tensor::from_shape_vec(&device, weight_dim, weight_vec.as_slice());
      let bias = if use_bias {
        Some(Tensor::from_shape_vec(&device, outputs, bias_vec.as_slice()))
      } else { None };
      input.conv2d(&weight.view(), bias.as_ref().map(|b| b.view()).as_ref(), args)
    };
    let output_vec = output.as_slice().into_owned();
    let output_true = {
      let input = tch::Tensor::of_slice(&input_vec)
        .reshape(&[batch_size as i64, inputs as i64, ih as i64, iw as i64]);
      let weight = tch::Tensor::of_slice(&weight_vec)
        .reshape(&[outputs as i64, inputs as i64, kh as i64, kw as i64]);
      let bias = if use_bias {
        Some(tch::Tensor::of_slice(&bias_vec)
          .reshape(&[outputs as i64])
        )
      } else { None }; 
      input.conv2d(
        &weight, 
        bias.as_ref(),
        &[args.strides[0] as i64, args.strides[1] as i64],
        &[args.padding[0] as i64, args.padding[1] as i64],
        &[1, 1],
        1
      )  
    };
    let mut output_true_vec = vec![0f32; output_vec.len()];
    output_true.copy_data(&mut output_true_vec, output_vec.len());
    compare_vectors(&output_vec, &output_true_vec, batch_size, outputs, inputs);
  }
  fn test_conv2d(device: impl Into<Device>) {
    let device = device.into();
    test_conv2d_with_args([8, 16, 20, 20], 12, [3, 3], false, &Conv2dArgs::default(), device.clone());
    test_conv2d_with_args([8, 16, 20, 20], 12, [3, 3], true, &Conv2dArgs::default().strides(2).padding(1), device.clone());
  }
  #[test]
  fn test_conv2d_cpu() {
    test_conv2d(Cpu::new());
  }
  #[cfg(feature="cuda")]
  #[test]
  fn test_conv2d_cuda() {
    test_conv2d(CudaGpu::new(0));
  }
  fn test_max_pool2d_with_args(input_dim: impl IntoDimension<Dim=Ix4>, args: &Pool2dArgs, device: impl Into<Device>) {
    let device = device.into();
    let input_dim = input_dim.into_dimension();
    let (batch_size, inputs, ih, iw) = input_dim.into_pattern();
    let input_vec: Vec<f32> = (1 ..= input_dim.size())
      .into_iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    let output = {
      let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
      input.max_pool2d(&args)
    };
    let output_vec = output.as_slice().into_owned();
    let output_true = {
      let input = tch::Tensor::of_slice(&input_vec)
        .reshape(&[batch_size as i64, inputs as i64, ih as i64, iw as i64]); 
      input.max_pool2d(
        &[args.kernel[0] as i64, args.kernel[1] as i64],
        &[args.strides[0] as i64, args.strides[1] as i64],
        &[args.padding[0] as i64, args.padding[1] as i64],
        &[1, 1],
        false
      )  
    };
    let (bs, o, oh, ow) = output_true.size4().unwrap();
    let output_dim_true = [bs as usize, o as usize, oh as usize, ow as usize].into_dimension();
    assert_eq!(output.raw_dim(), output_dim_true); 
    let mut output_true_vec = vec![0f32; output_vec.len()];
    output_true.copy_data(&mut output_true_vec, output_vec.len());
    assert_eq!(output_vec, output_true_vec);
  }
  fn test_max_pool2d(device: impl Into<Device>) {
    let device = device.into();
    test_max_pool2d_with_args([8, 16, 20, 20], &Pool2dArgs::default(), device.clone());
  }
  #[test]
  fn test_max_pool2d_cpu() {
    test_max_pool2d(Cpu::new());
  }
}

