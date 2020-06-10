use super::{Num, Unsigned, DataRef, DataMut, TensorBase, Transpose};
use std::{sync::{Arc, Mutex}, borrow::Cow, fmt::{self, Debug}, ffi::CString, any::TypeId};
use ndarray::{Dimension, Ix0, Ix1, Ix2};
use rustacuda::{
  CudaFlags, 
  memory::{DeviceBuffer, DeviceSlice, DevicePointer, CopyDestination}, 
  device::Device as CudaDevice, 
  context::{Context, ContextFlags, CurrentContext},
  stream::{Stream, StreamFlags},
  module::Module,
  launch
};
use cuda_sys::cuda::{cudaError_t, cuMemsetD8_v2, cuMemsetD32_v2};
use cuda_sys::cublas::{
  cublasStatus_t, 
  cublasContext, cublasCreate_v2, cublasDestroy_v2, 
  cublasSgemm_v2, cublasOperation_t_CUBLAS_OP_T, cublasOperation_t_CUBLAS_OP_N,
  cublasSaxpy_v2
};

pub struct CudaBuffer<T: Num> {
  data: DeviceBuffer<T>,
  device: Arc<CudaGpu>
}

impl<T: Num> CudaBuffer<T> {
  pub(super) unsafe fn uninitialized(gpu: &Arc<CudaGpu>, len: usize) -> Self {
    gpu.make_current();
    let data = DeviceBuffer::uninitialized(len).unwrap();
    let device = gpu.clone();
    Self{data, device}
  }
  pub(super) fn fill(&mut self, elem: T) {
    self.device.make_current();
    let p = unsafe { self.data.as_mut_ptr() as u64 };
    let len = self.data.len();
    let status = if TypeId::of::<T>() == TypeId::of::<u8>() {
      unsafe {
        cuMemsetD8_v2(
          p,
          std::mem::transmute(elem.to_u8().unwrap()), // u8
          len
        )
      }
    }
    else if TypeId::of::<T>() == TypeId::of::<f32>() {
      unsafe {
        cuMemsetD32_v2(
          p,
          std::mem::transmute(elem.to_f32().unwrap()), // u32
          len
        )
      }
    }
    else {
      unreachable!()
    };
    debug_assert_eq!(status, cudaError_t::CUDA_SUCCESS);
  }
  pub(super) fn len(&self) -> usize {
    self.data.len()
  }
  pub(super) fn as_device_slice(&self) -> &DeviceSlice<T> {
    &self.data
  }
  pub(super) fn as_mut_device_slice(&mut self) -> &mut DeviceSlice<T> {
    &mut self.data
  } 
  pub(super) fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }
  pub(super) fn as_mut_ptr(&mut self) -> *mut T {
    self.data.as_mut_ptr()
  }
  pub(super) fn to_vec(&self) -> Vec<T> { 
    self.device.make_current();
    let mut vec = Vec::with_capacity(self.data.len());
    unsafe { vec.set_len(self.data.len()) };
    self.data.copy_to(&mut vec);
    vec
  }
  pub(super) fn copy_from_slice<'a>(&mut self, slice: impl Into<Cow<'a, [T]>>) {
    let slice = slice.into();
    self.device.make_current();
    self.data.copy_from(slice.as_ref())
      .unwrap();
  }
}

impl<T: Num> Clone for CudaBuffer<T> {
  fn clone(&self) -> Self {
    self.device.make_current();
    let mut output = unsafe { Self::uninitialized(&self.device, self.data.len()) };
    self.data.copy_to(&mut output.data);
    output
  }
}

pub struct CudaGpu {
  index: usize,
  device: CudaDevice,
  stream: Stream,
  kernels: Module,
  context: Context,
  cublas_context: Mutex<*mut cublasContext>
}

impl CudaGpu {
  pub fn new(index: usize) -> Arc<Self> {
    rustacuda::init(CudaFlags::empty())
      .unwrap();
    let device = CudaDevice::get_device(index as u32)
      .unwrap();
    let context = Context::create_and_push(ContextFlags::SCHED_AUTO, device)
      .unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, Some(0))
      .unwrap();
    let src = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx")))
      .unwrap();
    let  kernels = Module::load_from_string(&src)
      .unwrap();
    let mut cublas_context = unsafe { std::ptr::null_mut() };
    let status = unsafe { cublasCreate_v2(&mut cublas_context as *mut *mut cublasContext) };
    match status {
      cublasStatus_t::SUCCESS => (),
      _ => panic!("{:?}", &status)
    }
    let cublas_context = Mutex::new(cublas_context);
    Arc::new(Self {
      index,
      device,
      stream,
      kernels,
      context,
      cublas_context
    })
  }
  fn make_current(&self) {
    CurrentContext::set_current(&self.context)
      .unwrap();
  }
  fn stream(&self) -> &Stream {
    &self.stream
  }
  fn kernels(&self) -> &Module {
    &self.kernels
  }
  fn cublas_context(&self) -> &Mutex<*mut cublasContext> {
    &self.cublas_context
  }
}

impl Debug for CudaGpu {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "CudaGpu({})", self.index)
  }
}

impl Drop for CudaGpu {
  fn drop(&mut self) {
    let cublas_guard = self.cublas_context()
      .lock()
      .unwrap();
    let cublas_handle = unsafe { *cublas_guard as *mut cublasContext };
    let status = unsafe { cublasDestroy_v2(cublas_handle) };
    debug_assert_eq!(status, cublasStatus_t::SUCCESS);
  }
}

pub(super) fn unsigned_to_f32<T: Unsigned, S1: DataRef<Elem=T>, S2: DataMut<Elem=f32>, D: Dimension>
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D>) {
  let gpu = input.device.cuda()
    .unwrap();
  gpu.make_current();
  let x = input.as_cuda_ptr()
    .unwrap();
  let y = output.as_mut_cuda_ptr()
    .unwrap();
  let len = input.len() as u32;
  let nthreads = 32;
  let mut nblocks = len / nthreads;
  if len % nthreads != 0 {
    nblocks += 1;
  }
  let stream = gpu.stream();
  let module = gpu.kernels();
  if TypeId::of::<T>() == TypeId::of::<u8>() {
    unsafe {
      launch!(module.u8_to_f32<<<nblocks, nthreads, 0, stream>>>(
        DevicePointer::wrap(x as *mut f32),
        DevicePointer::wrap(y),
        len
      )).unwrap()
    }
  }
  else {
    unreachable!()
  }
}

pub(super) fn unsigned_to_one_hot_f32<T: Unsigned, S1: DataRef<Elem=T>, S2: DataMut<Elem=f32>>
  (input: &TensorBase<S1, Ix1>, output: &mut TensorBase<S2, Ix2>) {
  let (batch_size, nclasses) = output.dim();
  debug_assert_eq!(batch_size, input.dim());
  let gpu = input.device.cuda()
    .unwrap();
  gpu.make_current();
  let x = input.as_cuda_ptr()
    .unwrap();
  let y = output.as_mut_cuda_ptr()
    .unwrap();
  let nclasses = nclasses as u32;
  let len = input.len() as u32;
  let nthreads = 32;
  let mut nblocks = len / nthreads;
  if len % nthreads != 0 {
    nblocks += 1;
  }
  let stream = gpu.stream();
  let module = gpu.kernels();
  if TypeId::of::<T>() == TypeId::of::<u8>() {
    unsafe {
      launch!(module.u8_to_one_hot_f32<<<nblocks, nthreads, 0, stream>>>(
        DevicePointer::wrap(x as *mut f32),
        nclasses,
        DevicePointer::wrap(y),
        len
      )).unwrap()
    }
  }
  else {
    unreachable!()
  }
}

pub(super) fn broadcast<T: Num, D: Dimension, S1: DataRef<Elem=T>, S2: DataMut<Elem=T>>
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D>) {
  let input = &input.as_cuda_slice()
    .unwrap();
  output.as_mut_cuda_slice()
    .unwrap()
    .chunks_mut(input.len())
    .for_each(|mut output| {
      input.copy_to(output);
    });
}

pub(super) fn broadcast_backward<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, D: Dimension>
  (input_grad: &mut TensorBase<S1, D>, output_grad: &TensorBase<S2, D>) {
  let gpu = output_grad.device.cuda()
    .unwrap();
  let cublas_guard = gpu.cublas_context()
    .lock()
    .unwrap();
  let cublas_handle = unsafe { *cublas_guard as *mut cublasContext };
  let alpha = unsafe { &1f32 as *const f32 };
  let dx = input_grad.as_mut_cuda_ptr()
    .unwrap();
  let len = input_grad.len();
  output_grad.as_cuda_slice()
    .unwrap()
    .chunks(len)
    .for_each(|output_grad| {
      unsafe {
        cublasSaxpy_v2(
          cublas_handle,
          len as i32,
          alpha,
          output_grad.as_ptr(),
          1,
          dx,
          1
        );
      }
    });
} 

pub(super) fn gemm<S1: DataRef<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>>
  (alpha: f32, a: &TensorBase<S1, Ix2>, trans_a: Transpose, b: &TensorBase<S2, Ix2>, trans_b: Transpose, beta: f32, c: &mut TensorBase<S3, Ix2>) {
  let (m, k1) = match trans_b {
    Transpose::Yes => b.dim(),
    Transpose::No => {
      let (k1, m) = b.dim();
      (m, k1)
    }
  };
  let ldb = match trans_b {
    Transpose::No => m,
    Transpose::Yes => k1
  };
  let (k2, n) = match trans_a {
    Transpose::Yes => a.dim(),
    Transpose::No => {
      let (n, k2) = a.dim();
      (k2, n)
    }
  };
  let lda = match trans_a {
    Transpose::No => k2,
    Transpose::Yes => n
  };
  debug_assert_eq!(k1, k2);
  debug_assert_eq!((n, m), c.dim());
  let gpu = a.device.cuda()
    .unwrap();
  gpu.make_current();
  let cublas_guard = gpu.cublas_context()
    .lock()
    .unwrap();
  let cublas_handle = unsafe { *cublas_guard as *mut cublasContext };
  let m = m as i32;
  let k = k1 as i32;
  let n = n as i32;
  let ldb = ldb as i32;
  let lda = lda as i32;
  let alpha = unsafe { &alpha as *const f32 };
  let beta = unsafe { &beta as *const f32 };
  let b = b.as_cuda_ptr().unwrap();
  let a = a.as_cuda_ptr().unwrap();
  let c = c.as_mut_cuda_ptr().unwrap();
  let trans_a = match trans_a {
    Transpose::Yes => cublasOperation_t_CUBLAS_OP_T,
    Transpose::No => cublasOperation_t_CUBLAS_OP_N
  };
  let trans_b = match trans_b {
    Transpose::Yes => cublasOperation_t_CUBLAS_OP_T,
    Transpose::No => cublasOperation_t_CUBLAS_OP_N
  };
  let status = unsafe { 
    cublasSgemm_v2(
      cublas_handle,
      trans_b,
      trans_a,
      m,
      n,
      k,
      alpha,
      b,
      ldb,
      a,
      lda,
      beta,
      c,
      m
    )
  };
  debug_assert_eq!(status, cublasStatus_t::SUCCESS);         
}

pub(super) fn reduce_sum<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, D: Dimension>
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, Ix0>) {
  let gpu = input.device.cuda()
    .unwrap();
  gpu.make_current();
  let mut len = input.len() / (2*256);
  if (input.len() % (2*256)) > 0 {
    len += 1;
  }
  let mut tmp = unsafe { 
    DeviceBuffer::<f32>::uninitialized(len)
      .unwrap() 
  };
  let stream = gpu.stream();
  let module = gpu.kernels();
  { // partial sum
    let x = input.as_cuda_ptr()
      .unwrap();
    let len = input.len() as u32;
    let nblocks = tmp.len() as u32;
    let nthreads = 256;
    unsafe {
      launch!(module.reduce_sum_partial<<<nblocks, nthreads, 0, stream>>>(
        DevicePointer::wrap(x as *mut f32),
        tmp.as_device_ptr(),
        len
      )).unwrap()
    } 
  }
  { // final sum
    let y = output.as_mut_cuda_ptr()
      .unwrap();
    let len = len as u32;
    let nblocks = 1;
    let nthreads = 1;
    unsafe {
      launch!(module.reduce_sum_final<<<nblocks, nthreads, 0, stream>>>(
        tmp.as_device_ptr(),
        DevicePointer::wrap(y),
        len
      )).unwrap()
    } 
  }
}

pub(super) fn scaled_add<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, D: Dimension>
  (lhs: &mut TensorBase<S1, D>, alpha: f32, rhs: &TensorBase<S2, D>) {
  let gpu = rhs.device.cuda()
    .unwrap();
  let cublas_guard = gpu.cublas_context()
    .lock()
    .unwrap();
  let cublas_handle = unsafe { *cublas_guard as *mut cublasContext };
  let a = lhs.as_mut_cuda_ptr()
    .unwrap();
  let alpha = unsafe { &alpha as *const f32 }; 
  let b = rhs.as_cuda_ptr()
    .unwrap();
  let len = lhs.len() as i32;
  unsafe {
    cublasSaxpy_v2(
      cublas_handle,
      len,
      alpha,
      b,
      1,
      a,
      1
    );
  }
} 

pub(super) fn cross_entropy<S1: DataRef<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>>
  (input: &TensorBase<S1, Ix2>, target: &TensorBase<S2, Ix2>, output: &mut TensorBase<S3, Ix2>) {
  let gpu = input.device.cuda()
    .unwrap();
  let stream = gpu.stream();
  let module = gpu.kernels();
  let len = input.len() as u32;
  let (batch_size, nclasses) = input.dim();
  let nthreads = 32;
  let mut nblocks = len / nthreads;
  if len % nthreads != 0 {
    nblocks += 1;
  }
  let x = input.as_cuda_ptr()
    .unwrap();
  let t = target.as_cuda_ptr()
    .unwrap();
  let y = output.as_mut_cuda_ptr()
    .unwrap();
  unsafe {
    launch!(module.cross_entropy_forward<<<nblocks, nthreads, 0, stream>>>(
      DevicePointer::wrap(x as *mut f32),
      nclasses as u32,
      DevicePointer::wrap(t as *mut f32),
      DevicePointer::wrap(y),
      len
    )).unwrap()
  }
}

pub(super) fn cross_entropy_backward<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, S3: DataRef<Elem=f32>, S4: DataRef<Elem=f32>>
  (input: &TensorBase<S1, Ix2>, input_grad: &mut TensorBase<S2, Ix2>,
   target: &TensorBase<S3, Ix2>, 
   output_grad: &TensorBase<S4, Ix0>) {
  let gpu = input.device.cuda()
    .unwrap();
  let stream = gpu.stream();
  let module = gpu.kernels();
  let len = input.len() as u32;
  let (batch_size, nclasses) = input.dim();
  let nthreads = 32;
  let mut nblocks = len / nthreads;
  if len % nthreads != 0 {
    nblocks += 1;
  }
  let x = input.as_cuda_ptr()
    .unwrap();
  let dx = input_grad.as_mut_cuda_ptr()
    .unwrap();
  let t = target.as_cuda_ptr()
    .unwrap();
  let dy = output_grad.as_cuda_ptr()
    .unwrap();
  unsafe {
    launch!(module.cross_entropy_backward<<<nblocks, nthreads, 0, stream>>>(
      DevicePointer::wrap(x as *mut f32),
      DevicePointer::wrap(dx),
      DevicePointer::wrap(t as *mut f32),
      DevicePointer::wrap(dy as *mut f32),
      len
    )).unwrap()
  }
}


