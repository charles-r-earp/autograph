use super::{
    Conv2dArgs, DataMut, DataRef, Num, Pool2dArgs, Tensor, Tensor4, TensorBase, TensorView1,
    TensorView4, TensorViewMut1, TensorViewMut4, Transpose, Unsigned,
};
use cuda_cudnn_sys::{
    cudnnActivationBackward, cudnnActivationDescriptor_t, cudnnActivationForward,
    cudnnActivationMode_t, cudnnContext, cudnnConvolutionBackwardBias,
    cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
    cudnnConvolutionBiasActivationForward, cudnnConvolutionBwdDataAlgoPerf_t,
    cudnnConvolutionBwdDataAlgo_t, cudnnConvolutionBwdFilterAlgoPerf_t,
    cudnnConvolutionBwdFilterAlgo_t, cudnnConvolutionDescriptor_t, cudnnConvolutionForward,
    cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnCreate,
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
    cublasContext, cublasCreate_v2, cublasDestroy_v2, cublasOperation_t_CUBLAS_OP_N,
    cublasOperation_t_CUBLAS_OP_T, cublasSaxpy_v2, cublasSgemm_v2, cublasStatus_t,
};
use cuda_sys::cuda::{cuMemsetD32_v2, cuMemsetD8_v2, cudaError_t};
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
    ffi::CString,
    fmt::{self, Debug},
    sync::{Arc, Mutex},
};

pub struct CudaBuffer<T: Num> {
    data: DeviceBuffer<T>,
    device: Arc<CudaGpu>,
}

impl<T: Num> CudaBuffer<T> {
    pub(super) unsafe fn uninitialized(gpu: &Arc<CudaGpu>, len: usize) -> Self {
        gpu.make_current();
        let data = DeviceBuffer::uninitialized(len).unwrap();
        let device = gpu.clone();
        Self { data, device }
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
                    len,
                )
            }
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            unsafe {
                cuMemsetD32_v2(
                    p,
                    std::mem::transmute(elem.to_f32().unwrap()), // u32
                    len,
                )
            }
        } else {
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
        self.data.copy_from(slice.as_ref()).unwrap();
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

/// Safe wrapper for several CUDA implementation handles
pub struct CudaGpu {
    index: usize,
    device: CudaDevice,
    stream: Stream,
    kernels: Module,
    context: Context,
    cublas_context: Mutex<*mut cublasContext>,
    cudnn_context: Mutex<*mut cudnnContext>,
}

impl CudaGpu {
    /// Constructs a CudaGpu one the given device index wrapped in an Arc
    pub fn new(index: usize) -> Arc<Self> {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = CudaDevice::get_device(index as u32).unwrap();
        let context = Context::create_and_push(ContextFlags::SCHED_AUTO, device).unwrap();
        let stream = Stream::new(StreamFlags::DEFAULT, Some(0)).unwrap();
        let src = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"))).unwrap();
        let kernels = Module::load_from_string(&src).unwrap();
        let cublas_context = {
            let mut cublas_context = unsafe { std::ptr::null_mut() };
            let status = unsafe { cublasCreate_v2(&mut cublas_context as *mut *mut cublasContext) };
            assert_eq!(status, cublasStatus_t::SUCCESS);
            Mutex::new(cublas_context)
        };
        let cudnn_context = {
            let mut cudnn_context = unsafe { std::ptr::null_mut() };
            let status = unsafe { cudnnCreate(&mut cudnn_context as *mut *mut cudnnContext) };
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);

            Mutex::new(cudnn_context)
        };
        Arc::new(Self {
            index,
            device,
            stream,
            kernels,
            context,
            cublas_context,
            cudnn_context,
        })
    }
    fn make_current(&self) {
        CurrentContext::set_current(&self.context).unwrap();
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
    fn cudnn_context(&self) -> &Mutex<*mut cudnnContext> {
        &self.cudnn_context
    }
    pub(super) fn synchronize(&self) {
        self.stream.synchronize();
    }
}

impl Debug for CudaGpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CudaGpu({})", self.index)
    }
}

impl Drop for CudaGpu {
    fn drop(&mut self) {
        {
            let mut cublas_context = self.cublas_context.lock().unwrap();
            let cublas_handle = unsafe { *cublas_context as *mut cublasContext };
            let status = unsafe { cublasDestroy_v2(cublas_handle) };
            assert_eq!(status, cublasStatus_t::SUCCESS);
        }
        {
            let mut cudnn_context = self.cudnn_context.lock().unwrap();
            let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
            let status = unsafe { cudnnDestroy(cudnn_handle) };
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        }
    }
}

struct TensorDescriptor {
    tensor_descriptor: cudnnTensorDescriptor_t,
}

impl TensorDescriptor {
    fn new(shape: impl AsRef<[usize]>, data_type: cudnnDataType_t) -> Self {
        let mut tensor_descriptor = unsafe { std::ptr::null_mut() };
        unsafe {
            cudnnCreateTensorDescriptor(&mut tensor_descriptor as *mut cudnnTensorDescriptor_t);
        }
        let shape = shape.as_ref();
        if shape.len() <= 4 {
            let mut dims = [1i32; 4];
            dims.as_mut()
                .iter_mut()
                .zip(shape.iter())
                .for_each(|(d, &s)| *d = s as i32);
            let [n, c, h, w] = dims;
            let status = unsafe {
                cudnnSetTensor4dDescriptor(
                    tensor_descriptor,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    data_type,
                    n,
                    c,
                    h,
                    w,
                )
            };
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        } else {
            unimplemented!()
        }
        Self { tensor_descriptor }
    }
    fn new_with_strides(
        dims: impl AsRef<[usize]>,
        strides: impl AsRef<[usize]>,
        data_type: cudnnDataType_t,
    ) -> Self {
        let mut tensor_descriptor = unsafe { std::ptr::null_mut() };
        unsafe {
            cudnnCreateTensorDescriptor(&mut tensor_descriptor as *mut cudnnTensorDescriptor_t);
        }
        let dims = dims.as_ref();
        let strides = strides.as_ref();
        debug_assert_eq!(dims.len(), strides.len());
        if strides.len() <= 4 {
            let status = unsafe {
                cudnnSetTensor4dDescriptorEx(
                    tensor_descriptor,
                    data_type,
                    dims[0] as i32,
                    dims[1] as i32,
                    dims[2] as i32,
                    dims[3] as i32,
                    strides[0] as i32,
                    strides[1] as i32,
                    strides[2] as i32,
                    strides[3] as i32,
                )
            };
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        } else {
            unimplemented!()
        }
        Self { tensor_descriptor }
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        let status = unsafe { cudnnDestroyTensorDescriptor(self.tensor_descriptor) };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    }
}

struct FilterDescriptor {
    filter_descriptor: cudnnFilterDescriptor_t,
}

impl FilterDescriptor {
    fn new(shape: impl AsRef<[usize]>, data_type: cudnnDataType_t) -> Self {
        let mut filter_descriptor = unsafe { std::ptr::null_mut() };
        unsafe {
            cudnnCreateFilterDescriptor(&mut filter_descriptor as *mut cudnnFilterDescriptor_t);
        }
        let shape = shape.as_ref();
        if shape.len() <= 4 {
            let mut dims = [1i32; 4];
            dims.as_mut()
                .iter_mut()
                .zip(shape.iter())
                .for_each(|(d, &s)| *d = s as i32);
            let [n, c, h, w] = dims;
            let status = unsafe {
                cudnnSetFilter4dDescriptor(
                    filter_descriptor,
                    data_type,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    n,
                    c,
                    h,
                    w,
                )
            };
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        } else {
            unimplemented!()
        }
        Self { filter_descriptor }
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        let status = unsafe { cudnnDestroyFilterDescriptor(self.filter_descriptor) };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    }
}

struct ConvolutionDescriptor {
    convolution_descriptor: cudnnConvolutionDescriptor_t,
}

impl ConvolutionDescriptor {
    fn new_conv2d(args: &Conv2dArgs, data_type: cudnnDataType_t) -> Self {
        let mut convolution_descriptor = unsafe { std::ptr::null_mut() };
        let status = unsafe {
            cudnnCreateConvolutionDescriptor(
                &mut convolution_descriptor as *mut cudnnConvolutionDescriptor_t,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        let status = unsafe {
            cudnnSetConvolution2dDescriptor(
                convolution_descriptor,
                args.padding[0] as i32,
                args.padding[1] as i32,
                args.strides[0] as i32,
                args.strides[1] as i32,
                1, // dilation unused
                1, //
                cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
                data_type,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        Self {
            convolution_descriptor,
        }
    }
    fn set_math_type(&mut self, math_type: cudnnMathType_t) {
        let status = unsafe { cudnnSetConvolutionMathType(self.convolution_descriptor, math_type) };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    }
}

struct ActivationDescriptor {
    activation_descriptor: cudnnActivationDescriptor_t,
}

impl ActivationDescriptor {
    fn new(
        mode: cudnnActivationMode_t,
        nan_propagation: cudnnNanPropagation_t,
        coef: Option<f32>,
    ) -> Self {
        let mut activation_descriptor = unsafe { std::ptr::null_mut() };
        let status = unsafe {
            cudnnCreateActivationDescriptor(
                &mut activation_descriptor as *mut cudnnActivationDescriptor_t,
            )
        };
        let status = unsafe {
            cudnnSetActivationDescriptor(
                activation_descriptor,
                mode,
                nan_propagation,
                coef.map_or(0., |c| c as f64),
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        Self {
            activation_descriptor,
        }
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        let status = unsafe { cudnnDestroyConvolutionDescriptor(self.convolution_descriptor) };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    }
}

struct PoolingDescriptor {
    pooling_descriptor: cudnnPoolingDescriptor_t,
}

impl PoolingDescriptor {
    fn new_pool2d(
        mode: cudnnPoolingMode_t,
        nan_propagation: cudnnNanPropagation_t,
        args: &Pool2dArgs,
    ) -> Self {
        let mut pooling_descriptor = unsafe { std::ptr::null_mut() };
        let status = unsafe {
            cudnnCreatePoolingDescriptor(&mut pooling_descriptor as *mut cudnnPoolingDescriptor_t)
        };
        let status = unsafe {
            cudnnSetPooling2dDescriptor(
                pooling_descriptor,
                mode,
                nan_propagation,
                args.kernel[0] as i32,
                args.kernel[1] as i32,
                args.padding[0] as i32,
                args.padding[1] as i32,
                args.strides[0] as i32,
                args.strides[1] as i32,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        Self { pooling_descriptor }
    }
}

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        let status = unsafe { cudnnDestroyPoolingDescriptor(self.pooling_descriptor) };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    }
}

pub(super) fn unsigned_to_f32<
    T: Unsigned,
    S1: DataRef<Elem = T>,
    S2: DataMut<Elem = f32>,
    D: Dimension,
>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D>,
) {
    let gpu = input.device.cuda().unwrap();
    gpu.make_current();
    let x = input.as_cuda_ptr().unwrap();
    let y = output.as_mut_cuda_ptr().unwrap();
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
            ))
            .unwrap()
        }
    } else {
        unreachable!()
    }
}

pub(super) fn unsigned_to_one_hot_f32<
    T: Unsigned,
    S1: DataRef<Elem = T>,
    S2: DataMut<Elem = f32>,
>(
    input: &TensorBase<S1, Ix1>,
    output: &mut TensorBase<S2, Ix2>,
) {
    let (batch_size, nclasses) = output.dim();
    debug_assert_eq!(batch_size, input.dim());
    let gpu = input.device.cuda().unwrap();
    gpu.make_current();
    let x = input.as_cuda_ptr().unwrap();
    let y = output.as_mut_cuda_ptr().unwrap();
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
            ))
            .unwrap()
        }
    } else {
        unreachable!()
    }
}

pub(super) fn broadcast<T: Num, D: Dimension, S1: DataRef<Elem = T>, S2: DataMut<Elem = T>>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D::Larger>,
) {
    let input = &input.as_cuda_slice().unwrap();
    output
        .as_mut_cuda_slice()
        .unwrap()
        .chunks_mut(input.len())
        .for_each(|mut output| {
            input.copy_to(output);
        });
}

pub(super) fn broadcast_backward<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    input_grad: &mut TensorBase<S1, D>,
    output_grad: &TensorBase<S2, D::Larger>,
) {
    let gpu = output_grad.device.cuda().unwrap();
    let cublas_guard = gpu.cublas_context().lock().unwrap();
    let cublas_handle = unsafe { *cublas_guard as *mut cublasContext };
    let alpha = unsafe { &1f32 as *const f32 };
    let dx = input_grad.as_mut_cuda_ptr().unwrap();
    let len = input_grad.len();
    output_grad
        .as_cuda_slice()
        .unwrap()
        .chunks(len)
        .for_each(|output_grad| unsafe {
            cublasSaxpy_v2(
                cublas_handle,
                len as i32,
                alpha,
                output_grad.as_ptr(),
                1,
                dx,
                1,
            );
        });
}

pub(super) fn gemm<S1: DataRef<Elem = f32>, S2: DataRef<Elem = f32>, S3: DataMut<Elem = f32>>(
    alpha: f32,
    a: &TensorBase<S1, Ix2>,
    trans_a: Transpose,
    b: &TensorBase<S2, Ix2>,
    trans_b: Transpose,
    beta: f32,
    c: &mut TensorBase<S3, Ix2>,
) {
    let (m, k1) = match trans_b {
        Transpose::Yes => b.dim(),
        Transpose::No => {
            let (k1, m) = b.dim();
            (m, k1)
        }
    };
    let ldb = match trans_b {
        Transpose::No => m,
        Transpose::Yes => k1,
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
        Transpose::Yes => n,
    };
    debug_assert_eq!(k1, k2);
    debug_assert_eq!((n, m), c.dim());
    let gpu = a.device.cuda().unwrap();
    gpu.make_current();
    let cublas_guard = gpu.cublas_context().lock().unwrap();
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
        Transpose::No => cublasOperation_t_CUBLAS_OP_N,
    };
    let trans_b = match trans_b {
        Transpose::Yes => cublasOperation_t_CUBLAS_OP_T,
        Transpose::No => cublasOperation_t_CUBLAS_OP_N,
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
            m,
        )
    };
    debug_assert_eq!(status, cublasStatus_t::SUCCESS);
}

pub(super) fn reduce_sum<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, Ix0>,
) {
    let gpu = input.device.cuda().unwrap();
    gpu.make_current();
    let mut len = input.len() / (2 * 256);
    if (input.len() % (2 * 256)) > 0 {
        len += 1;
    }
    let mut tmp = unsafe { DeviceBuffer::<f32>::uninitialized(len).unwrap() };
    let stream = gpu.stream();
    let module = gpu.kernels();
    {
        // partial sum
        let x = input.as_cuda_ptr().unwrap();
        let len = input.len() as u32;
        let nblocks = tmp.len() as u32;
        let nthreads = 256;
        unsafe {
            launch!(module.reduce_sum_partial<<<nblocks, nthreads, 0, stream>>>(
              DevicePointer::wrap(x as *mut f32),
              tmp.as_device_ptr(),
              len
            ))
            .unwrap()
        }
    }
    {
        // final sum
        let y = output.as_mut_cuda_ptr().unwrap();
        let len = len as u32;
        let nblocks = 1;
        let nthreads = 1;
        unsafe {
            launch!(module.reduce_sum_final<<<nblocks, nthreads, 0, stream>>>(
              tmp.as_device_ptr(),
              DevicePointer::wrap(y),
              len
            ))
            .unwrap()
        }
    }
}

pub(super) fn relu<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let mut cudnn_context = gpu.cudnn_context.lock().unwrap();
    let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
    let x_desc = TensorDescriptor::new(input.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let x = input.as_cuda_ptr().unwrap();
    let y_desc = TensorDescriptor::new(output.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let y = output.as_mut_cuda_ptr().unwrap();
    let relu_desc = ActivationDescriptor::new(
        cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
        cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        None,
    );
    let status = unsafe {
        cudnnActivationForward(
            cudnn_handle,
            relu_desc.activation_descriptor,
            &1f32 as *const f32 as *const std::ffi::c_void,
            x_desc.tensor_descriptor,
            x as *const std::ffi::c_void,
            &0f32 as *const f32 as *const std::ffi::c_void,
            y_desc.tensor_descriptor,
            y as *mut std::ffi::c_void,
        )
    };
    assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
}

pub(super) fn relu_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
    D: Dimension,
>(
    input: &TensorBase<S1, D>,
    input_grad: &mut TensorBase<S2, D>,
    output_grad: &TensorBase<S3, D>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let mut cudnn_context = gpu.cudnn_context.lock().unwrap();
    let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
    let x_desc = TensorDescriptor::new(input.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let x = input.as_cuda_ptr().unwrap();
    let dx_desc = TensorDescriptor::new(input_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dx = input_grad.as_mut_cuda_ptr().unwrap();
    let dy_desc = TensorDescriptor::new(output_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dy = output_grad.as_cuda_ptr().unwrap();
    let relu_desc = ActivationDescriptor::new(
        cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
        cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        None,
    );
    let status = unsafe {
        cudnnActivationBackward(
            cudnn_handle,
            relu_desc.activation_descriptor,
            &1f32 as *const f32 as *const std::ffi::c_void,
            dy_desc.tensor_descriptor,
            x as *const std::ffi::c_void,
            dy_desc.tensor_descriptor,
            dy as *const std::ffi::c_void,
            x_desc.tensor_descriptor,
            x as *const std::ffi::c_void,
            &0f32 as *const f32 as *const std::ffi::c_void,
            dx_desc.tensor_descriptor,
            dx as *mut std::ffi::c_void,
        )
    };
    assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
}

pub(super) fn add<S1: DataRef<Elem = f32>, S2: DataRef<Elem = f32>, S3: DataMut<Elem = f32>, D: Dimension>(
    lhs: &TensorBase<S1, D>,
    rhs: &TensorBase<S2, D>,
    output: &mut TensorBase<S3, D>
) {
    let gpu = lhs.device.cuda().unwrap();
    gpu.make_current();
    let x1 = lhs.as_cuda_ptr().unwrap();
    let x2 = rhs.as_cuda_ptr().unwrap();
    let y = output.as_mut_cuda_ptr().unwrap();
    let len = lhs.len() as u32;
    let nthreads = 32;
    let mut nblocks = len / nthreads;
    if len % nthreads != 0 {
        nblocks += 1;
    }
    let stream = gpu.stream();
    let module = gpu.kernels();
    unsafe {
        launch!(module.add<<<nblocks, nthreads, 0, stream>>>(
          DevicePointer::wrap(x1 as *mut f32),
          DevicePointer::wrap(x2 as *mut f32),
          DevicePointer::wrap(y),
          len
        ))
        .unwrap()
    }
}

pub(super) fn scaled_add<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    lhs: &mut TensorBase<S1, D>,
    alpha: f32,
    rhs: &TensorBase<S2, D>,
) {
    let gpu = rhs.device.cuda().unwrap();
    let cublas_context = gpu.cublas_context().lock().unwrap();
    let cublas_handle = unsafe { *cublas_context as *mut cublasContext };
    let a = lhs.as_mut_cuda_ptr().unwrap();
    let alpha = unsafe { &alpha as *const f32 };
    let b = rhs.as_cuda_ptr().unwrap();
    let len = lhs.len() as i32;
    unsafe {
        cublasSaxpy_v2(cublas_handle, len, alpha, b, 1, a, 1);
    }
}

pub(super) fn cross_entropy<
    S1: DataRef<Elem = f32>,
    S2: DataRef<Elem = f32>,
    S3: DataMut<Elem = f32>,
>(
    input: &TensorBase<S1, Ix2>,
    target: &TensorBase<S2, Ix2>,
    output: &mut TensorBase<S3, Ix2>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let module = gpu.kernels();
    let (batch_size, nclasses) = input.dim();
    let nthreads = 32;
    let mut nblocks = (batch_size as u32) / nthreads;
    if (batch_size as u32) % nthreads != 0 {
        nblocks += 1;
    }
    let x = input.as_cuda_ptr().unwrap();
    let t = target.as_cuda_ptr().unwrap();
    let y = output.as_mut_cuda_ptr().unwrap();
    unsafe {
        launch!(module.cross_entropy_forward<<<nblocks, nthreads, 0, stream>>>(
          batch_size as u32,
          nclasses as u32,
          DevicePointer::wrap(x as *mut f32),
          DevicePointer::wrap(t as *mut f32),
          DevicePointer::wrap(y)
        ))
        .unwrap()
    }
}

pub(super) fn cross_entropy_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
    S4: DataRef<Elem = f32>,
>(
    input: &TensorBase<S1, Ix2>,
    input_grad: &mut TensorBase<S2, Ix2>,
    target: &TensorBase<S3, Ix2>,
    output_grad: &TensorBase<S4, Ix0>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let module = gpu.kernels();
    let len = input.len() as u32;
    let (batch_size, nclasses) = input.dim();
    let nthreads = 32;
    let mut nblocks = len / nthreads;
    if len % nthreads != 0 {
        nblocks += 1;
    }
    let x = input.as_cuda_ptr().unwrap();
    let dx = input_grad.as_mut_cuda_ptr().unwrap();
    let t = target.as_cuda_ptr().unwrap();
    let dy = output_grad.as_cuda_ptr().unwrap();
    unsafe {
        launch!(module.cross_entropy_backward<<<nblocks, nthreads, 0, stream>>>(
          DevicePointer::wrap(x as *mut f32),
          DevicePointer::wrap(dx),
          DevicePointer::wrap(t as *mut f32),
          DevicePointer::wrap(dy as *mut f32),
          len
        ))
        .unwrap()
    }
}

fn reverse_conv2d_filter(input: &TensorView4<f32>, beta: f32, output: &mut TensorViewMut4<f32>) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let module = gpu.kernels();
    let (outputs, inputs, kh, kw) = input.dim();
    let len = (outputs * inputs) as u32;
    let filter_len = (kh * kw) as u32;
    let nthreads = 32;
    let mut nblocks = len / nthreads;
    if len % nthreads != 0 {
        nblocks += 1;
    }
    let x = input.as_cuda_ptr().unwrap();
    let y = output.as_mut_cuda_ptr().unwrap();
    unsafe {
        launch!(module.reverse_conv_filter<<<nblocks, nthreads, 0, stream>>>(
          DevicePointer::wrap(x as *mut f32),
          beta,
          DevicePointer::wrap(y),
          filter_len,
          len
        ))
        .unwrap()
    }
}

pub(super) fn conv2d<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    bias: Option<&TensorView1<f32>>,
    args: &Conv2dArgs,
    output: &mut TensorBase<S2, Ix4>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let mut cudnn_context = gpu.cudnn_context.lock().unwrap();
    let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
    let x_desc = TensorDescriptor::new(input.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let x = input.as_cuda_ptr().unwrap();
    let weight = {
        // Patch cudnn behavior by reversing filter order ie per patch
        // for kernel of shape (1, 1, 2, 2) and data = vec![1., 2., 3., 4.]
        // output = vec![4., 3., 2., 1.]
        let mut weight_reversed = unsafe { Tensor::uninitialized(&input.device, weight.raw_dim()) };
        reverse_conv2d_filter(&weight.view(), 0., &mut weight_reversed.view_mut());
        weight_reversed
    };
    let w_desc = FilterDescriptor::new(weight.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let w = weight.as_cuda_ptr().unwrap();
    let y_desc = TensorDescriptor::new(output.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let y = output.as_mut_cuda_ptr().unwrap();
    let mut conv2d_desc =
        ConvolutionDescriptor::new_conv2d(args, cudnnDataType_t::CUDNN_DATA_FLOAT);
    conv2d_desc.set_math_type(cudnnMathType_t::CUDNN_TENSOR_OP_MATH);
    let algo = if bias.is_some() {
        // required for identity activation in ConvolutionBiasActivationForward
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    } else {
        let mut perf: cudnnConvolutionFwdAlgoPerf_t = unsafe { std::mem::uninitialized() };
        let mut ret_algo_count = 0i32;
        let status = unsafe {
            cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn_handle,
                x_desc.tensor_descriptor,
                w_desc.filter_descriptor,
                conv2d_desc.convolution_descriptor,
                y_desc.tensor_descriptor,
                1,
                &mut ret_algo_count as *mut i32,
                &mut perf as *mut cudnnConvolutionFwdAlgoPerf_t,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        assert_eq!(ret_algo_count, 1);
        perf.algo
    };
    let mut workspace_size = 0;
    let status = unsafe {
        cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_handle,
            x_desc.tensor_descriptor,
            w_desc.filter_descriptor,
            conv2d_desc.convolution_descriptor,
            y_desc.tensor_descriptor,
            algo,
            &mut workspace_size as *mut usize,
        )
    };
    assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    let mut workspace = unsafe {
        gpu.make_current();
        DeviceBuffer::<u8>::uninitialized(workspace_size).unwrap()
    };
    let status: cudnnStatus_t = if let Some(bias) = &bias {
        let b_desc = TensorDescriptor::new([1, bias.dim()], cudnnDataType_t::CUDNN_DATA_FLOAT);
        let b = bias.as_cuda_ptr().unwrap();
        let activation_desc = ActivationDescriptor::new(
            cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
            cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            None,
        );
        unsafe {
            cudnnConvolutionBiasActivationForward(
                cudnn_handle,
                &1f32 as *const f32 as *const std::ffi::c_void,
                x_desc.tensor_descriptor,
                x as *const std::ffi::c_void,
                w_desc.filter_descriptor,
                w as *const std::ffi::c_void,
                conv2d_desc.convolution_descriptor,
                algo,
                workspace.as_mut_ptr() as *mut std::ffi::c_void,
                workspace_size,
                &0f32 as *const f32 as *const std::ffi::c_void,
                y_desc.tensor_descriptor,
                y as *mut std::ffi::c_void,
                b_desc.tensor_descriptor,
                b as *const std::ffi::c_void,
                activation_desc.activation_descriptor,
                y_desc.tensor_descriptor,
                y as *mut std::ffi::c_void,
            )
        }
    } else {
        unsafe {
            cudnnConvolutionForward(
                cudnn_handle,
                &1f32 as *const f32 as *const std::ffi::c_void,
                x_desc.tensor_descriptor,
                x as *const std::ffi::c_void,
                w_desc.filter_descriptor,
                w as *const std::ffi::c_void,
                conv2d_desc.convolution_descriptor,
                algo,
                workspace.as_mut_ptr() as *mut std::ffi::c_void,
                workspace_size,
                &0f32 as *const f32 as *const std::ffi::c_void,
                y_desc.tensor_descriptor,
                y as *mut std::ffi::c_void,
            )
        }
    };
    assert_eq!(
        status,
        cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        "cudnnConvolutionForward"
    );
}

pub(super) fn conv2d_backward_input<S1: DataMut<Elem = f32>>(
    input_grad: &mut TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    let gpu = weight.device.cuda().unwrap();
    let stream = gpu.stream();
    let mut cudnn_context = gpu.cudnn_context.lock().unwrap();
    let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
    let dx_desc = TensorDescriptor::new(input_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dx = input_grad.as_mut_cuda_ptr().unwrap();
    let weight = {
        // Patch cudnn behavior by reversing filter order ie per patch
        // for kernel of shape (1, 1, 2, 2) and data = vec![1., 2., 3., 4.]
        // output = vec![4., 3., 2., 1.]
        let mut weight_reversed =
            unsafe { Tensor::uninitialized(&weight.device, weight.raw_dim()) };
        reverse_conv2d_filter(&weight.view(), 0., &mut weight_reversed.view_mut());
        weight_reversed
    };
    let w_desc = FilterDescriptor::new(weight.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let w = weight.as_cuda_ptr().unwrap();
    let dy_desc = TensorDescriptor::new(output_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dy = output_grad.as_cuda_ptr().unwrap();
    let mut conv2d_desc =
        ConvolutionDescriptor::new_conv2d(args, cudnnDataType_t::CUDNN_DATA_FLOAT);
    conv2d_desc.set_math_type(cudnnMathType_t::CUDNN_TENSOR_OP_MATH);

    let algo = {
        let mut perf: cudnnConvolutionBwdDataAlgoPerf_t = unsafe { std::mem::uninitialized() };
        let mut ret_algo_count = 0i32;
        let status = unsafe {
            cudnnGetConvolutionBackwardDataAlgorithm_v7(
                cudnn_handle,
                w_desc.filter_descriptor,
                dy_desc.tensor_descriptor,
                conv2d_desc.convolution_descriptor,
                dx_desc.tensor_descriptor,
                1,
                &mut ret_algo_count as *mut i32,
                &mut perf as *mut cudnnConvolutionBwdDataAlgoPerf_t,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        assert_eq!(ret_algo_count, 1);
        perf.algo
    };
    let workspace_size = {
        let mut workspace_size = 0;
        let status = unsafe {
            cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnn_handle,
                w_desc.filter_descriptor,
                dy_desc.tensor_descriptor,
                conv2d_desc.convolution_descriptor,
                dx_desc.tensor_descriptor,
                algo,
                &mut workspace_size as *mut usize,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        workspace_size
    };
    let mut workspace = unsafe {
        gpu.make_current();
        DeviceBuffer::<u8>::uninitialized(workspace_size).unwrap()
    };
    let status = unsafe {
        cudnnConvolutionBackwardData(
            cudnn_handle,
            &1f32 as *const f32 as *const std::ffi::c_void,
            w_desc.filter_descriptor,
            w as *const std::ffi::c_void,
            dy_desc.tensor_descriptor,
            dy as *const std::ffi::c_void,
            conv2d_desc.convolution_descriptor,
            algo,
            workspace.as_mut_ptr() as *mut std::ffi::c_void,
            workspace_size,
            &1f32 as *const f32 as *const std::ffi::c_void,
            dx_desc.tensor_descriptor,
            dx as *mut std::ffi::c_void,
        )
    };
    assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
}

pub(super) fn conv2d_backward_weight_bias<S1: DataRef<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    weight_grad: &mut TensorViewMut4<f32>,
    bias_grad: Option<&mut TensorViewMut1<f32>>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let mut cudnn_context = gpu.cudnn_context.lock().unwrap();
    let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
    let x_desc = TensorDescriptor::new(input.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let x = input.as_cuda_ptr().unwrap();
    // patch cudnn behavior
    let mut weight_grad_reversed = Tensor::zeros(weight_grad.device(), weight_grad.raw_dim());
    let dw_desc = FilterDescriptor::new(weight_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dw = weight_grad_reversed.as_mut_cuda_ptr().unwrap();
    let dy_desc = TensorDescriptor::new(output_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dy = output_grad.as_cuda_ptr().unwrap();
    let mut conv2d_desc =
        ConvolutionDescriptor::new_conv2d(args, cudnnDataType_t::CUDNN_DATA_FLOAT);
    conv2d_desc.set_math_type(cudnnMathType_t::CUDNN_TENSOR_OP_MATH);

    let algo = {
        /*
        let mut perf: cudnnConvolutionBwdFilterAlgoPerf_t = unsafe { std::mem::uninitialized() };
        let mut ret_algo_count = 0i32;
        let status = unsafe {
          cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            cudnn_handle,
            x_desc.tensor_descriptor,
            dy_desc.tensor_descriptor,
            conv2d_desc.convolution_descriptor,
            dw_desc.filter_descriptor,
            1,
            &mut ret_algo_count as *mut i32,
            &mut perf as *mut cudnnConvolutionBwdFilterAlgoPerf_t
          )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        assert_eq!(ret_algo_count, 1);
        perf.algo*/
        cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
    };
    let workspace_size = {
        let mut workspace_size = 0;
        let status = unsafe {
            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnn_handle,
                x_desc.tensor_descriptor,
                dy_desc.tensor_descriptor,
                conv2d_desc.convolution_descriptor,
                dw_desc.filter_descriptor,
                algo,
                &mut workspace_size as *mut usize,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        workspace_size
    };
    let mut workspace = unsafe {
        gpu.make_current();
        DeviceBuffer::<u8>::uninitialized(workspace_size).unwrap()
    };

    let status = unsafe {
        cudnnConvolutionBackwardFilter(
            cudnn_handle,
            &1f32 as *const f32 as *const std::ffi::c_void,
            x_desc.tensor_descriptor,
            x as *const std::ffi::c_void,
            dy_desc.tensor_descriptor,
            dy as *const std::ffi::c_void,
            conv2d_desc.convolution_descriptor,
            algo,
            workspace.as_mut_ptr() as *mut std::ffi::c_void,
            workspace_size,
            &1f32 as *const f32 as *const std::ffi::c_void,
            dw_desc.filter_descriptor,
            dw as *mut std::ffi::c_void,
        )
    };
    assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);

    // apply reversed filter back to weight_grad
    reverse_conv2d_filter(
        &weight_grad_reversed.view(),
        1.,
        &mut weight_grad.view_mut(),
    );

    if let Some(bias_grad) = bias_grad {
        println!("bias_grad.dim(): {:?}", bias_grad.raw_dim());
        let db_desc = TensorDescriptor::new(
            [1, bias_grad.dim(), 1, 1],
            cudnnDataType_t::CUDNN_DATA_FLOAT,
        );
        let db = bias_grad.as_mut_cuda_ptr().unwrap();

        let status = unsafe {
            cudnnConvolutionBackwardBias(
                cudnn_handle,
                &1f32 as *const f32 as *const std::ffi::c_void,
                dy_desc.tensor_descriptor,
                dy as *const std::ffi::c_void,
                &1f32 as *const f32 as *const std::ffi::c_void,
                db_desc.tensor_descriptor,
                db as *mut std::ffi::c_void,
            )
        };
        assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    }
}

pub(super) fn max_pool2d<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    args: &Pool2dArgs,
    output: &mut TensorBase<S2, Ix4>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let mut cudnn_context = gpu.cudnn_context.lock().unwrap();
    let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
    let x_desc = TensorDescriptor::new(input.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let x = input.as_cuda_ptr().unwrap();
    let y_desc = TensorDescriptor::new(output.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let y = output.as_mut_cuda_ptr().unwrap();
    let pool2d_desc = PoolingDescriptor::new_pool2d(
        cudnnPoolingMode_t::CUDNN_POOLING_MAX,
        cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        args,
    );
    let status = unsafe {
        cudnnPoolingForward(
            cudnn_handle,
            pool2d_desc.pooling_descriptor,
            &1f32 as *const f32 as *const std::ffi::c_void,
            x_desc.tensor_descriptor,
            x as *const std::ffi::c_void,
            &0f32 as *const f32 as *const std::ffi::c_void,
            y_desc.tensor_descriptor,
            y as *mut std::ffi::c_void,
        )
    };
    assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
}

pub(super) fn max_pool2d_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
>(
    input: &TensorBase<S1, Ix4>,
    input_grad: &mut TensorBase<S2, Ix4>,
    args: &Pool2dArgs,
    output_grad: &TensorBase<S3, Ix4>,
) {
    let gpu = input.device.cuda().unwrap();
    let stream = gpu.stream();
    let mut cudnn_context = gpu.cudnn_context.lock().unwrap();
    let cudnn_handle = unsafe { *cudnn_context as *mut cudnnContext };
    let x_desc = TensorDescriptor::new(input.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let x = input.as_cuda_ptr().unwrap();
    let dx_desc = TensorDescriptor::new(input_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dx = input_grad.as_mut_cuda_ptr().unwrap();
    let dy_desc = TensorDescriptor::new(output_grad.dim.slice(), cudnnDataType_t::CUDNN_DATA_FLOAT);
    let dy = output_grad.as_cuda_ptr().unwrap();
    let pool2d_desc = PoolingDescriptor::new_pool2d(
        cudnnPoolingMode_t::CUDNN_POOLING_MAX,
        cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        args,
    );
    let status = unsafe {
        cudnnPoolingBackward(
            cudnn_handle,
            pool2d_desc.pooling_descriptor,
            &1f32 as *const f32 as *const std::ffi::c_void,
            dy_desc.tensor_descriptor,
            x as *const std::ffi::c_void,
            dy_desc.tensor_descriptor,
            dy as *const std::ffi::c_void,
            x_desc.tensor_descriptor,
            x as *const std::ffi::c_void,
            &0f32 as *const f32 as *const std::ffi::c_void,
            dx_desc.tensor_descriptor,
            dx as *mut std::ffi::c_void,
        )
    };
    assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
}
