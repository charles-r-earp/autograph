use super::{AutographResult, AutographError, Element, Stack, Vertex};
use std::{rc::Rc, cell::{UnsafeCell, RefCell, RefMut}};
use rustacuda::{error::CudaError, CudaFlags, device::Device as CudaDevice, context::{Context as CudaContext, CurrentContext, ContextFlags}, memory::DeviceBuffer, stream::{Stream, StreamFlags}};
use cuda_sys::cublas::{cublasOperation_t_CUBLAS_OP_C as OP_C, cublasOperation_t_CUBLAS_OP_T as OP_T, cublasSgemm_v2};
type CublasContext = cuda_sys::cublas::cublasContext;
type CublasStatus = cuda_sys::cublas::cublasStatus_t;

impl From<CudaError> for AutographError {
  fn from(e: CudaError) -> Self {
    AutographError::CudaError(e)
  }
}

#[cfg(feature="cuda")]
pub struct Gpu {
  index: usize,
  device: CudaDevice,
  stack: RefCell<Stack>,
  stream: Stream,
  cuda_ctx: CudaContext,
  cublas_ctx: UnsafeCell<Box<CublasContext>>,
}

#[cfg(feature="cuda")]
impl Gpu {
  pub fn new(index: usize) -> AutographResult<Rc<super::Device>> {
    rustacuda::init(CudaFlags::empty()).unwrap();
    let device = CudaDevice::get_device(index as u32).unwrap();
    let cuda_ctx = CudaContext::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    let mut cublas_ctx = unsafe { std::ptr::null_mut::<CublasContext>() };
    let status = unsafe { cuda_sys::cublas::cublasCreate_v2(&mut cublas_ctx as *mut *mut CublasContext) };
    assert_eq!(status, CublasStatus::SUCCESS);
    let cublas_ctx = UnsafeCell::new(unsafe { Box::from_raw(cublas_ctx) });
    let stack = RefCell::new(Stack::default());
    let stream = Stream::new(StreamFlags::DEFAULT, Some(0))?;
    Ok(Rc::new(super::Device::Cuda(Self{index, device, stack, stream, cuda_ctx, cublas_ctx})))
  }
  pub fn set_current(&self) -> AutographResult<()> {
    CurrentContext::set_current(&self.cuda_ctx)?;
    Ok(())
  }
  pub fn index(&self) -> usize { self.index }
  pub fn cuda_device(&self) -> &CudaDevice { &self.device }
  pub fn stream(&self) -> &Stream { &self.stream }
  pub fn stack_mut(&self) -> RefMut<Stack> { self.stack.borrow_mut() }
  pub fn cuda_context(&self) -> &CudaContext { &self.cuda_ctx }
  pub unsafe fn cublas_context(&self) -> *mut *mut CublasContext { self.cublas_ctx.get() as *mut *mut CublasContext }
  pub fn sync(&self) -> AutographResult<()> {
    self.stream.synchronize()?;
    self.stack.borrow_mut().clear();
    Ok(())
  }
  pub fn gemm(&self, alpha: f32, a: &Vertex<f32>, b: &Vertex<f32>, beta: f32, c: &Vertex<f32>) -> AutographResult<()> {
    //use std::mem::transmute;
    let mut stack = self.stack.borrow_mut();
    stack.push(a.clone());
    stack.push(b.clone());
    stack.push(c.clone());
    self.set_current()?;
    //let status = unsafe { cublasSetStream_v2(&mut **self.cublas_context.borrow_mut() as *mut CublasContext, transmute(&stream.0)) }; 
    //assert_eq!(status, CublasStatus::SUCCESS);
    let alpha = &alpha as *const f32;
    let beta = &beta as *const f32; 
    let m = a.dims()[0] as i32;
    let k = a.dims()[1] as i32;
    let n = b.dims()[1] as i32;
    let trans_a = if a.is_t() { OP_C } else { OP_T };
    let trans_b = if b.is_t() { OP_C } else { OP_T };
    let a = unsafe { a.buffer().cuda().unwrap().as_ptr() };
    let b = unsafe { b.buffer().cuda().unwrap().as_ptr() };
    let c = unsafe { c.buffer().cuda().unwrap().as_mut_ptr() };
    let status = unsafe {
      cublasSgemm_v2(
        &mut **self.cublas_context() as *mut CublasContext,
        trans_a,
        trans_b,
        m,
        n,
        k,
        alpha,
        a,
        k,
        b,
        n,
        beta,
        c,
        m
      )
    };
    assert_eq!(status, CublasStatus::SUCCESS);   
    Ok(()) 
  }
}


