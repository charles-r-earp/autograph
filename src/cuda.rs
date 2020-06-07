use super::{Num, DataRef, DataMut, TensorBase, Transpose};
use std::{sync::{Arc, Mutex}, borrow::Cow, fmt::{self, Debug}, ffi::CString};
use ndarray::Ix2;
use rustacuda::{
  CudaFlags, 
  memory::{DeviceBuffer, CopyDestination}, 
  device::Device as CudaDevice, 
  context::{Context, ContextFlags, CurrentContext},
  stream::{Stream, StreamFlags},
  module::Module
};
use cuda_sys::cuda::{cudaError_t, cuMemsetD8_v2, cuMemsetD32_v2};
use cuda_sys::cublas::{cublasStatus_t, cublasContext, cublasCreate_v2, cublasDestroy_v2, cublasSgemm_v2, cublasOperation_t_CUBLAS_OP_T, cublasOperation_t_CUBLAS_OP_N};

pub struct CudaBuffer<T: Num> {
  data: DeviceBuffer<T>,
  device: Arc<CudaGpu>
}

impl<T: Num> CudaBuffer<T> {
  pub(super) unsafe fn uninitialized(device: &Arc<CudaGpu>, len: usize) -> Self {
    device.make_current();
    let data = DeviceBuffer::uninitialized(len).unwrap();
    let device = device.clone();
    Self{data, device}
  }
  pub(super) fn fill(&mut self, elem: T) {
    self.device.make_current();
    let p = unsafe { self.data.as_mut_ptr() as u64 };
    let len = self.data.len();
    let status = if let Some(elem) = elem.to_u8() {
      unsafe {
        cuMemsetD8_v2(
          p,
          elem,
          len
        )
      }
    }
    else if let Some(elem) = elem.to_f32() {
      unsafe {
        cuMemsetD32_v2(
          p,
          std::mem::transmute(elem),
          len
        )
      }
    }
    else {
      unreachable!();
    };
    debug_assert_eq!(status, cudaError_t::CUDA_SUCCESS);
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

pub(super) fn gemm<S1: DataRef<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>>
  (alpha: f32, a: &TensorBase<S1, Ix2>, trans_a: Transpose, b: &TensorBase<S2, Ix2>, trans_b: Transpose, beta: f32, c: &mut TensorBase<S3, Ix2>) {
  let (m, k1) = match trans_b {
    Transpose::Yes => b.dim(),
    Transpose::No => {
      let (k1, m) = b.dim();
      (m, k1)
    }
  };
  let (k2, n) = match trans_a {
    Transpose::Yes => a.dim(),
    Transpose::No => {
      let (n, k2) = a.dim();
      (k2, n)
    }
  };
  debug_assert_eq!(k1, k2);
  debug_assert_eq!((n, m), c.dim());
  let device = a.device.cuda()
    .unwrap();
  let cublas_guard = device.cublas_context()
    .lock()
    .unwrap();
  let cublas_handle = unsafe { *cublas_guard as *mut cublasContext };
  let m = m as i32;
  let k = k1 as i32;
  let n = n as i32;
  let alpha = unsafe { &alpha as *const f32 };
  let beta = unsafe { &beta as *const f32 };
  let lda = if trans_a == Transpose::No { m } else { k };
  let ldb = if trans_b == Transpose::No { k } else { n };
  let (a, b) = (
    b.as_cuda_ptr().unwrap(),
    a.as_cuda_ptr().unwrap()
  );
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
      trans_a,
      trans_b,
      m,
      n,
      k,
      alpha,
      a,
      lda,
      b,
      ldb,
      beta,
      c,
      m
    )
  };
  debug_assert_eq!(status, cublasStatus_t::SUCCESS);         
}
