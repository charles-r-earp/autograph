use super::{
    Conv2dArgs, DataMut, DataRef, Num, Pool2dArgs, Tensor, Tensor4, TensorBase, TensorView1,
    TensorView4, TensorViewMut1, TensorViewMut4, Transpose, Unsigned,
};
use cuda_cudnn_sys::{
    cudnnActivationBackward, cudnnActivationDescriptor_t, cudnnActivationForward,
    cudnnActivationMode_t, cudnnHandle_t, cudnnConvolutionBackwardBias,
    cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
    cudnnConvolutionBiasActivationForward, cudnnConvolutionBwdDataAlgoPerf_t,
    cudnnConvolutionBwdDataAlgo_t, cudnnConvolutionBwdFilterAlgoPerf_t,
    cudnnConvolutionBwdFilterAlgo_t, cudnnConvolutionDescriptor_t, cudnnConvolutionForward,
    cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnCreate, cudnnSetStream,
    cudnnCreateActivationDescriptor, cudnnCreateConvolutionDescriptor, cudnnCreateFilterDescriptor,
    cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDataType_t, cudnnDestroy,
    cudnnDestroyActivationDescriptor, cudnnDestroyConvolutionDescriptor,
    cudnnDestroyFilterDescriptor, cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor,
    cudnnFilterDescriptor_t, cudnnGetConvolutionBackwardDataAlgorithm_v7,
    cudnnGetConvolutionBackwardDataWorkspaceSize, cudnnGetConvolutionBackwardFilterAlgorithm_v7,
    cudnnGetConvolutionBackwardFilterWorkspaceSize, cudnnGetConvolutionForwardAlgorithm_v7,
    cudnnGetConvolutionForwardWorkspaceSize, cudnnMathType_t, cudnnNanPropagation_t,
    cudnnPoolingBackward, cudnnPoolingDescriptor_t, cudnnPoolingForward, cudnnPoolingMode_t,
    cudnnSetActivationDescriptor, cudnnSetConvolution2dDescriptor, cudnnSetConvolutionMathType,
    cudnnSetFilter4dDescriptor, cudnnSetPooling2dDescriptor, cudnnSetTensor4dDescriptor,
    cudnnSetTensor4dDescriptorEx, cudnnStatus_t, cudnnTensorDescriptor_t, cudnnTensorFormat_t,
};
use cuda_sys::cublas::{
    cublasHandle_t, cublasCreate_v2, cublasDestroy_v2, cublasSetStream_v2, cublasOperation_t_CUBLAS_OP_N,
    cublasOperation_t_CUBLAS_OP_T, cublasSaxpy_v2, cublasSgemm_v2, cublasStatus_t,
};
use cuda_sys::cuda::{cuMemsetD32_v2, cuMemsetD8_v2, cudaError_t};
use cuda_sys::cudart::cudaStream_t;
use ndarray::{Dimension, Ix0, Ix1, Ix2, Ix4};
use rustacuda::{
    context::{Context, ContextFlags, CurrentContext},
    device::Device as CudaDevice,
    launch,
    memory::{CopyDestination, DeviceBuffer, DevicePointer, DeviceSlice},
    module::Module,
    stream::{Stream, StreamFlags},
    CudaFlags,
};
use std::{
    any::TypeId,
    borrow::Cow,
    ffi::{CString, c_void},
    fmt::{self, Debug},
    sync::{Arc, Mutex, MutexGuard, LockResult, PoisonError},
};

mod error;
use error::{CudaResult, IntoResult};

trait StreamExt {
    unsafe fn as_mut_ptr(&self) -> cudaStream_t;
}

impl StreamExt for Stream {
    unsafe fn as_mut_ptr(&self) -> cudaStream_t {
        let stream: &cudaStream_t = std::mem::transmute(self);
        *stream
    }
}

#[doc(hidden)]
pub struct Cublas {
    handle: cublasHandle_t                         
}

impl Cublas {
    fn with_stream(stream: &Stream) -> CudaResult<Self> {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        let status = unsafe {
            cublasCreate_v2(
                &mut handle as *mut cublasHandle_t
            )
        };
        status.into_result()?;
        let status = unsafe {
            cublasSetStream_v2(
                handle,
                stream.as_mut_ptr()
            )
        };
        status.into_result()?;
        Ok(Self { handle })
    }
    unsafe fn as_mut_ptr(&self) -> cublasHandle_t {
        self.handle
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        let status = unsafe {
            cublasDestroy_v2(self.handle)
        };
        status.into_result()
            .unwrap(); 
    }
}

#[doc(hidden)]
pub struct Cudnn {
    handle: cudnnHandle_t
}

impl Cudnn {
    fn with_stream(stream: &Stream) -> CudaResult<Self> {
        let mut handle: cudnnHandle_t = std::ptr::null_mut();
        let status = unsafe {
            cudnnCreate(
                &mut handle as *mut cudnnHandle_t
            )
        };
        status.into_result()?;
        let status = unsafe {
            cudnnSetStream(
                handle,
                std::mem::transmute(stream.as_mut_ptr()) // cudart to cuda-cudnn-sys
            )
        };
        status.into_result()?;
        Ok(Self { handle })     
    }
    unsafe fn as_mut_ptr(&self) -> cudnnHandle_t {
        self.handle
    }
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        let status = unsafe {
            cudnnDestroy(self.handle)
        };
        status.into_result()
            .unwrap();
    }
}

#[doc(hidden)]
pub struct CudaGpuBase {
    stream: Stream,
    kernels: Module,
    cublas: Cublas,
    cudnn: Cudnn,
    context: Context
}

impl CudaGpuBase {
    fn stream(&self) -> &Stream { 
        &self.stream
    }
    fn kernels(&self) -> &Module {
        &self.kernels
    }
    fn blas(&self) -> &Cublas {
        &self.cublas
    }
    fn nn(&self) -> &Cudnn {
        &self.cudnn
    }
    fn context(&self) -> &Context {
        &self.context 
    }
}

/// Safe wrapper for several CUDA implementation handles
pub struct CudaGpu {
    index: usize,
    device: CudaDevice,
    base: Mutex<CudaGpuBase>
}

impl CudaGpu {
    /// Constructs a CudaGpu on the given device index wrapped in an Arc
    pub fn new(index: usize) -> Arc<Self> {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = CudaDevice::get_device(index as u32).unwrap();
        let context = Context::create_and_push(ContextFlags::SCHED_AUTO, device).unwrap();
        let stream = Stream::new(StreamFlags::DEFAULT, Some(0))
            .expect("Unable to create Cuda Stream!");
        let src = CString::new(include_str!("cuda/kernels.ptx")).unwrap();
        let kernels = Module::load_from_string(&src).unwrap();
        let cublas = Cublas::with_stream(&stream)
            .expect("Unable to create Cublas!");
        let cudnn = Cudnn::with_stream(&stream)
            .expect("Unable to create Cudnn!");
        let base = Mutex::new(CudaGpuBase {
            stream,
            kernels,
            cublas,
            cudnn,
            context
        });
        Arc::new(Self {
            index,
            device,
            base
        })
    }
    fn lock(&self) -> LockResult<MutexGuard<CudaGpuBase>> {
        self.base.lock()
            .map(|base| {
                CurrentContext::set_current(base.context())
                    .expect("Unable to set CurrentContext!");
                base
            })
            .map_err(|e| {
                let base = e.into_inner();
                CurrentContext::set_current(base.context())
                    .expect("Unable to set CurrentContext!");
                PoisonError::new(base)
            })
    }
    pub(super) fn synchronize(&self) {
        self.lock()
            .expect("Unable to lock CudaGpu!")
            .stream()
            .synchronize()
            .expect("Unable to synchronize Cuda Stream!");
    }
}

impl Debug for CudaGpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CudaGpu({})", self.index)
    }
}

/// Warp Size
const WARP_SIZE: u32 = 32;

include!("cuda/common_impl.rs");
