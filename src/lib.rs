#![allow(warnings)]
#![recursion_limit = "1024"]

use argmm::ArgMinMax;
use ndarray::{
    Array, ArrayView, CowArray, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
    RemoveAxis,
};
use num_traits::{Bounded, One, ToPrimitive, Zero};
use rand::Rng;
use rand_distr::Distribution;
#[cfg(feature = "cuda")]
use rustacuda::memory::{DeviceCopy, DeviceSlice};
use std::borrow::Cow;
use std::sync::{Arc, LockResult, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
#[macro_use]
extern crate serde;

#[cfg(feature = "autograph_derive")]
#[macro_use]
extern crate autograph_derive;
#[cfg(feature = "autograph_derive")]
pub use autograph_derive::*;

#[doc(hidden)]
pub mod cpu;
pub use cpu::Cpu;
use cpu::CpuBuffer;

#[doc(hidden)]
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
use cuda::CudaBuffer;
#[cfg(feature = "cuda")]
pub use cuda::CudaGpu;

pub mod nn;
use nn::{Conv2dArgs, Pool2dArgs};

#[cfg(feature = "datasets")]
pub mod datasets;

pub mod utils;

#[cfg(test)]
mod tests;

mod private_num {
    pub trait PrivateNum {}
}
use private_num::PrivateNum;

impl PrivateNum for u8 {}
impl PrivateNum for f32 {}

#[doc(hidden)]
#[cfg(not(feature = "cuda"))]
pub trait DeviceCopy {}

#[cfg(not(feature = "cuda"))]
impl<T: PrivateNum> DeviceCopy for T {}

/// Num is a trait for all data types that Tensor can store, it cannot be implemented for additional types
pub trait Num:
    'static + Copy + DeviceCopy + Default + Zero + One + ToPrimitive + Bounded + PartialEq
{
}

impl Num for u8 {}
impl Num for f32 {}

/// Unsigned is a trait for types which can be treated as an index, ie converted to usize
pub trait Unsigned: Num {}

impl Unsigned for u8 {}

#[doc(hidden)]
#[derive(Clone)]
pub enum Buffer<T: Num> {
    Cpu(CpuBuffer<T>),
    #[cfg(feature = "cuda")]
    Cuda(CudaBuffer<T>),
}

impl<T: Num> From<CpuBuffer<T>> for Buffer<T> {
    fn from(cpu_buffer: CpuBuffer<T>) -> Self {
        Buffer::Cpu(cpu_buffer)
    }
}

#[cfg(feature = "cuda")]
impl<T: Num> From<CudaBuffer<T>> for Buffer<T> {
    fn from(cuda_buffer: CudaBuffer<T>) -> Self {
        Buffer::Cuda(cuda_buffer)
    }
}

impl<T: Num> Buffer<T> {
    unsafe fn uninitialized(device: &Device, len: usize) -> Self {
        match device {
            Device::Cpu(_) => CpuBuffer::uninitialized(len).into(),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => CudaBuffer::uninitialized(cuda_gpu, len).into(),
        }
    }
    fn from_vec<'a>(device: &Device, vec: impl Into<Cow<'a, [T]>>) -> Self {
        match device {
            Device::Cpu(_) => CpuBuffer::from_vec(vec).into(),
            #[cfg(feature = "cuda")]
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
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => {
                let mut buffer = unsafe { CudaBuffer::uninitialized(cuda_gpu, len) };
                buffer.fill(T::zero());
                buffer.into()
            }
        }
    }
    fn len(&self) -> usize {
        match self {
            Buffer::Cpu(cpu_buffer) => cpu_buffer.len(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(cuda_buffer) => cuda_buffer.len(),
        }
    }
    fn fill(&mut self, elem: T) {
        match self {
            Buffer::Cpu(cpu_buffer) => cpu_buffer.fill(elem),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(cuda_buffer) => cuda_buffer.fill(elem),
        }
    }
    fn as_slice(&self) -> Cow<[T]> {
        match self {
            Buffer::Cpu(cpu_buffer) => cpu_buffer.as_slice().into(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(cuda_buffer) => cuda_buffer.to_vec().into(),
        }
    }
    fn copy_from_slice<'a>(&mut self, slice: impl Into<Cow<'a, [T]>>) {
        match self {
            Buffer::Cpu(cpu_buffer) => cpu_buffer.copy_from_slice(slice),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(cuda_buffer) => cuda_buffer.copy_from_slice(slice)
        }
    }
    fn cpu(&self) -> Option<&CpuBuffer<T>> {
        match self {
            Buffer::Cpu(cpu_buffer) => Some(cpu_buffer),
            _ => None,
        }
    }
    fn cpu_mut(&mut self) -> Option<&mut CpuBuffer<T>> {
        match self {
            Buffer::Cpu(cpu_buffer) => Some(cpu_buffer),
            _ => None,
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda(&self) -> Option<&CudaBuffer<T>> {
        match self {
            Buffer::Cuda(cuda_buffer) => Some(cuda_buffer),
            _ => None,
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda_mut(&mut self) -> Option<&mut CudaBuffer<T>> {
        match self {
            Buffer::Cuda(cuda_buffer) => Some(cuda_buffer),
            _ => None,
        }
    }
}

/// Device is an enum that is used to select whether to store and execute operations on the cpu or a gpu.\
///
/// You can use the From trait to create a device from a Cpu or CudaGpu:
/// ```
/// let cpu = Device::from(Cpu::new());
/// ```
/// Device can be cloned, which copies the pointer. Each Tensor will have a copy of the Device so that it can execute operations.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Device {
    Cpu(Arc<Cpu>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaGpu>),
}

impl Device {
    fn cpu(&self) -> Option<&Arc<Cpu>> {
        match self {
            Device::Cpu(cpu) => Some(cpu),
            _ => None,
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda(&self) -> Option<&Arc<CudaGpu>> {
        match self {
            Device::Cuda(cuda_gpu) => Some(cuda_gpu),
            _ => None,
        }
    }
    /// For cpu does nothing. For cuda, blocks until all operations on the device are finished. Only necessary for timing ie for benchmarks. Any tranfers back to the cpu will implicitly synchronize.
    pub fn synchronize(&self) {
        #[cfg(feature = "cuda")]
        {
            self.cuda().map(|gpu| gpu.synchronize());
        }
    }
}

/// Use Device::default() to get a gpu if available, or a cpu
impl Default for Device {
    fn default() -> Self {
        #[cfg(feature = "cuda")] {
            return CudaGpu::new(0).into();
        }
        Cpu::new().into()
    }
}

impl From<Arc<Cpu>> for Device {
    fn from(cpu: Arc<Cpu>) -> Self {
        Device::Cpu(cpu)
    }
}

#[cfg(feature = "cuda")]
impl From<Arc<CudaGpu>> for Device {
    fn from(cuda_gpu: Arc<CudaGpu>) -> Self {
        Device::Cuda(cuda_gpu)
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Device::Cpu(cpu1) => match other {
                Device::Cpu(cpu2) => Arc::ptr_eq(cpu1, cpu2),
                _ => false,
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu1) => match other {
                Device::Cuda(cuda_gpu2) => Arc::ptr_eq(cuda_gpu1, cuda_gpu2),
                _ => false,
            },
        }
    }
}

mod private_data {
    pub trait PrivateData {}
}
use private_data::PrivateData;

/// Main trait for Tensor S generic parameter, similar to ndarray::Data. Elem indicates that the Tensor stores that datatype.
pub trait Data: PrivateData {
    type Elem: Num;
}

/// Trait for Tensors that can be constructed, and do not have references (ie not a View)
pub trait DataOwned: Data + Sized {
    fn from_buffer(buffer: Buffer<Self::Elem>) -> Self;
}

/// Trait for Tensors which can borrow their data immutably
pub trait DataRef: Data {
    #[doc(hidden)]
    fn buffer(&self) -> &Buffer<Self::Elem>;
}

/// Trait for Tensors which can borrow their data mutably
pub trait DataMut: DataRef {
    #[doc(hidden)]
    fn buffer_mut(&mut self) -> &mut Buffer<Self::Elem>;
}

#[doc(hidden)]
#[derive(Clone)]
pub struct OwnedRepr<T: Num> {
    buffer: Buffer<T>,
}

impl<T: Num> PrivateData for OwnedRepr<T> {}

impl<T: Num> Data for OwnedRepr<T> {
    type Elem = T;
}

impl<T: Num> DataOwned for OwnedRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self { buffer }
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
    buffer: V,
}

impl<V> ViewRepr<V> {
    fn new(buffer: V) -> Self {
        Self { buffer }
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
    buffer: Arc<Buffer<T>>,
}

impl<T: Num> PrivateData for ArcRepr<T> {}

impl<T: Num> Data for ArcRepr<T> {
    type Elem = T;
}

impl<T: Num> DataOwned for ArcRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self {
            buffer: Arc::new(buffer),
        }
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
    buffer: Arc<RwLock<Buffer<T>>>,
}

impl<T: Num> RwRepr<T> {
    fn read(&self) -> LockResult<RwReadRepr<T>> {
        self.buffer
            .read()
            .map(|buffer| RwReadRepr { buffer })
            .map_err(|e| {
                PoisonError::new(RwReadRepr {
                    buffer: e.into_inner(),
                })
            })
    }
    fn write(&self) -> LockResult<RwWriteRepr<T>> {
        self.buffer
            .write()
            .map(|buffer| RwWriteRepr { buffer })
            .map_err(|e| {
                PoisonError::new(RwWriteRepr {
                    buffer: e.into_inner(),
                })
            })
    }
}

impl<T: Num> PrivateData for RwRepr<T> {}

impl<T: Num> Data for RwRepr<T> {
    type Elem = T;
}

impl<T: Num> DataOwned for RwRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(buffer)),
        }
    }
}

#[doc(hidden)]
pub struct RwReadRepr<'a, T: Num> {
    buffer: RwLockReadGuard<'a, Buffer<T>>,
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
    buffer: RwLockWriteGuard<'a, Buffer<T>>,
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

/// The core data type of Autograph, which abstracts over data location and representation. Data must be loaded into tensors before operations can be run.
#[derive(Clone)]
pub struct TensorBase<S: Data, D: Dimension> {
    device: Device,
    dim: D,
    data: S,
}

/// Tensor which has exclusive ownership of its data
pub type Tensor<T, D> = TensorBase<OwnedRepr<T>, D>;
pub type Tensor0<T> = Tensor<T, Ix0>;
pub type Tensor1<T> = Tensor<T, Ix1>;
pub type Tensor2<T> = Tensor<T, Ix2>;
pub type Tensor4<T> = Tensor<T, Ix4>;
pub type TensorD<T> = Tensor<T, IxDyn>;

/// Tensor which has an immutable (shared) borrow of its data
pub type TensorView<'a, T, D> = TensorBase<ViewRepr<&'a Buffer<T>>, D>;
pub type TensorView1<'a, T> = TensorView<'a, T, Ix1>;
pub type TensorView2<'a, T> = TensorView<'a, T, Ix2>;
pub type TensorView4<'a, T> = TensorView<'a, T, Ix4>;
pub type TensorViewD<'a, T> = TensorView<'a, T, IxDyn>;

/// Tensor which has a mutable (exclusive) borrow of its data
pub type TensorViewMut<'a, T, D> = TensorBase<ViewRepr<&'a mut Buffer<T>>, D>;
pub type TensorViewMut0<'a, T> = TensorViewMut<'a, T, Ix0>;
pub type TensorViewMut1<'a, T> = TensorViewMut<'a, T, Ix1>;
pub type TensorViewMut2<'a, T> = TensorViewMut<'a, T, Ix2>;
pub type TensorViewMut4<'a, T> = TensorViewMut<'a, T, Ix4>;

/// Tensor which has threadsafe shared immutable access without a lifetime
pub type ArcTensor<T, D> = TensorBase<ArcRepr<T>, D>;
pub type ArcTensor2<T> = ArcTensor<T, Ix2>;
pub type ArcTensor4<T> = ArcTensor<T, Ix4>;
pub type ArcTensorD<T> = ArcTensor<T, IxDyn>;

/// Tensor which allows for either shared immutable access or exclusive mutable access
pub type RwTensor<T, D> = TensorBase<RwRepr<T>, D>;
pub type RwTensor0<T> = RwTensor<T, Ix0>;
pub type RwTensor1<T> = RwTensor<T, Ix1>;
pub type RwTensor2<T> = RwTensor<T, Ix2>;
pub type RwTensor3<T> = RwTensor<T, Ix3>;
pub type RwTensor4<T> = RwTensor<T, Ix4>;
pub type RwTensorD<T> = RwTensor<T, IxDyn>;

/// Represents an immutable borrow of a RwTensor, acts like a TensorView
pub type RwReadTensor<'a, T, D> = TensorBase<RwReadRepr<'a, T>, D>;
/// Represents a mutable borrow of a RwTensor, acts like a TensorViewMut
pub type RwWriteTensor<'a, T, D> = TensorBase<RwWriteRepr<'a, T>, D>;

impl<T: Num, S: DataOwned<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Constructs a Tensor on the device with the given shape, its data is unninitialized.
    /// Unsafe: Rust generally marks these kinds of functions unsafe because reading uninitialized data is undefined behavior
    /// Num is only implemented for types which are safe to read arbitrary bits, so generally this is safe
    /// Use this prior an operation that will only write to the tensor and not read from it
    unsafe fn uninitialized(device: &Device, shape: impl IntoDimension<Dim = D>) -> Self {
        let device = device.clone();
        let dim = shape.into_dimension();
        let data = S::from_buffer(Buffer::uninitialized(&device, dim.size()));
        Self { device, dim, data }
    }
    /// Constructs a Tensor on the device with the given shape. If a Vec is provided, will move the data (ie no copy) if the device is a cpu. Can also provide a slice, which allows for the data to only be copied once (rather than twice in the case of copying to the gpu).
    /// Panics: Asserts that the provided shape matches the length of the provided vec or slice.
    ///
    pub fn from_shape_vec<'a>(
        device: &Device,
        shape: impl IntoDimension<Dim = D>,
        vec: impl Into<Cow<'a, [T]>>,
    ) -> Self {
        let device = device.clone();
        let dim = shape.into_dimension();
        let vec = vec.into();
        assert_eq!(dim.size(), vec.len());
        let data = S::from_buffer(Buffer::from_vec(&device, vec));
        Self { device, dim, data }
    }
    /// Similar to from_shape_vec, can accept either an Array or an ArrayView
    /// Copies that data to standard (packed) layout if necessary
    pub fn from_array<'a>(device: &Device, array: impl Into<CowArray<'a, T, D>>) -> Self {
        let array = array.into();
        if let Some(slice) = array.as_slice() {
            Self::from_shape_vec(&device, array.raw_dim(), slice)
        } else {
            let vec: Vec<T> = array.iter().copied().collect();
            Self::from_shape_vec(&device, array.raw_dim(), vec)
        }
    }
    /// Constructs a Tensor on the device with the give shape filled with zeros
    pub fn zeros(device: &Device, shape: impl IntoDimension<Dim = D>) -> Self {
        let device = device.clone();
        let dim = shape.into_dimension();
        let data = S::from_buffer(Buffer::zeros(&device, dim.size()));
        Self { device, dim, data }
    }
    /// Constructs a Tensor on the device with the give shape filled with ones
    pub fn ones(device: &Device, shape: impl IntoDimension<Dim = D>) -> Self {
        let device = device.clone();
        let dim = shape.into_dimension();
        let mut buffer = unsafe { Buffer::uninitialized(&device, dim.size()) };
        buffer.fill(T::one());
        let data = S::from_buffer(buffer);
        Self { device, dim, data }
    }
    /// Constructs a Tensor on the device with the give shape filled with data sampled from distr using the given rng
    pub fn random(
        device: &Device,
        shape: impl IntoDimension<Dim = D>,
        distr: &impl Distribution<T>,
        mut rng: &mut impl Rng,
    ) -> Self {
        let dim = shape.into_dimension();
        let vec: Vec<T> = distr.sample_iter(&mut rng).take(dim.size()).collect();
        Self::from_shape_vec(device, dim, vec)
    }
}

impl<T: Num, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Copies the stored Dimension
    pub fn raw_dim(&self) -> D {
        self.dim.clone()
    }
    /// Returns the Dimension in pattern form, ie for Ix2 -> (usize, usize)
    pub fn dim(&self) -> D::Pattern {
        self.dim.clone().into_pattern()
    }
    /// Returns the number of elements in the tensor
    pub fn len(&self) -> usize {
        self.dim.size()
    }
    /// Consumes self (ie moves its data without copying) and converts its dimension to IxDyn
    pub fn into_dyn(self) -> TensorBase<S, IxDyn> {
        TensorBase {
            device: self.device,
            dim: self.dim.into_dyn(),
            data: self.data,
        }
    }
    /// Consumes self and attempts to convert to the provided Dimension type
    /// Generally this is used to downcast a TensorBase<_, IxDyn> to a Tensor<_, D2>, in which case it will succeed if the number of dimensions match
    /// Some: If D and D2 have the same number of dimensions
    pub fn into_dimensionality<D2: Dimension>(self) -> Option<TensorBase<S, D2>> {
        D2::from_dimension(&self.dim).map(|dim| TensorBase {
            device: self.device,
            dim,
            data: self.data,
        })
    }
    /// Consumes self and attempts to convert the current dim to the provided shape.
    /// Some: If the new shape is the same size as the current size, that is the same number of elements
    pub fn into_shape<D2: Dimension>(
        self,
        shape: impl IntoDimension<Dim = D2>,
    ) -> Option<TensorBase<S, D2>> {
        let dim = shape.into_dimension();
        if self.dim.size() == dim.size() {
            Some(TensorBase {
                device: self.device,
                dim,
                data: self.data,
            })
        } else {
            None
        }
    }
    /// Consumes self and returns a 2D Tensor with the first dimension (ie the batch size) the same as the input
    pub fn into_flatten(self) -> TensorBase<S, Ix2>
    where
        D: RemoveAxis,
    {
        let batch_size = self.dim[0];
        let inputs = self.dim.slice()[1..].iter().product();
        self.into_shape([batch_size, inputs]).unwrap()
    }
}

impl<T: Num, S: DataRef<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Borrows self as a TensorView
    pub fn view(&self) -> TensorView<T, D> {
        let device = self.device.clone();
        let dim = self.dim.clone();
        let data = ViewRepr::new(self.data.buffer());
        TensorView { device, dim, data }
    }
    /// Borrows self as a Cow slice. If the tensor device is a gpu, copies the data to a vec.
    /// Use the into_owned method to get a Vec<T> without an additional copy.  
    pub fn as_slice(&self) -> Cow<[T]> {
        self.data.buffer().as_slice()
    }
    /// Borrows self as an CowArray. If the tensor device is a gpu, copies the data into an Array.
    /// Use the into_owned method to get an Array<T, D> without an additional copy
    pub fn as_array(&self) -> CowArray<T, D> {
        let dim = self.dim.clone();
        match self.data.buffer().as_slice() {
            Cow::Owned(vec) => unsafe { Array::from_shape_vec_unchecked(dim, vec) }.into(),
            Cow::Borrowed(slice) => {
                unsafe { ArrayView::from_shape_ptr(dim, slice.as_ptr()) }.into()
            }
        }
    }
    fn as_cpu_slice(&self) -> Option<&[T]> {
        self.data.buffer().cpu().map(|b| b.as_slice())
    }
    #[cfg(feature = "cuda")]
    fn as_cuda_slice(&self) -> Option<&DeviceSlice<T>> {
        self.data.buffer().cuda().map(|b| b.as_device_slice())
    }
    fn as_cpu_ptr(&self) -> Option<*const T> {
        self.data.buffer().cpu().map(|b| b.as_ptr())
    }
    #[cfg(feature = "cuda")]
    fn as_cuda_ptr(&self) -> Option<*const T> {
        self.data.buffer().cuda().map(|b| b.as_ptr())
    }
}

impl<S: DataRef<Elem = f32>, D: Dimension> TensorBase<S, D> {
    /// Sums all the elements of the tensor, returning a Tensor with 1 element
    pub fn sum(&self) -> Tensor0<f32> {
        let mut output = unsafe { Tensor::uninitialized(&self.device, ()) };
        match &self.device {
            Device::Cpu(cpu) => cpu::reduce_sum(self, &mut output),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => cuda::reduce_sum(self, &mut output),
        }
        output
    }
    /// Computes the ReLU function
    pub fn relu(&self) -> Tensor<f32, D> {
        let mut output = unsafe { Tensor::uninitialized(&self.device, self.raw_dim()) };
        match &self.device {
            Device::Cpu(cpu) => cpu::relu(self, &mut output),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => cuda::relu(self, &mut output),
        }
        output
    }
}

impl<S: DataMut<Elem = f32>, D: Dimension> TensorBase<S, D> {
    pub fn add<S2: DataRef<Elem = f32>>(&self, rhs: &TensorBase<S2, D>) -> Tensor<f32, D> {
        debug_assert_eq!(&self.device, &rhs.device);
        debug_assert_eq!(&self.dim, &rhs.dim);
        let mut output = unsafe { Tensor::uninitialized(self.device(), self.raw_dim()) };
        match &self.device {
            Device::Cpu(cpu) => cpu::add(self, rhs, &mut output),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => cuda::add(self, rhs, &mut output),
        }
        output
    }
    /// Performs the operation self[i] += alpha * rhs[i] for all elements in self
    pub fn scaled_add<S2: DataRef<Elem = f32>>(&mut self, alpha: f32, rhs: &TensorBase<S2, D>) {
        debug_assert_eq!(&self.device, &rhs.device);
        debug_assert_eq!(&self.dim, &rhs.dim);
        match &self.device {
            Device::Cpu(cpu) => cpu::scaled_add(self, alpha, rhs),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => cuda::scaled_add(self, alpha, rhs),
        }
    }
}

impl<T: Unsigned, S: DataRef<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Copies the data from self to a new Tensor<f32, D>, scaled by the max value of T
    /// ie for u8: y = x / 255.
    /// This is used to covert image data typically stored as u8 to f32 for computation
    /// Performing this conversion on a gpu, rather than on the host prior to copying to the device, greatly reduces the size of the data that is copied
    pub fn to_f32(&self) -> Tensor<f32, D> {
        let mut output = unsafe { Tensor::uninitialized(&self.device, self.dim.clone()) };
        match &self.device {
            Device::Cpu(cpu) => cpu::unsigned_to_f32(self, &mut output),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => cuda::unsigned_to_f32(self, &mut output),
        }
        output
    }
}

impl<T: Unsigned, S: DataRef<Elem = T>> TensorBase<S, Ix1> {
    /// Similar to to_f32, this operation converts each element of the input vector to a new vector filled with zeros, with a single element at the given index set to 1.
    /// Again, performing the coversion on the gpu means that significantly less data must be transfered.
    pub fn to_one_hot_f32(&self, nclasses: usize) -> Tensor2<f32> {
        let mut output = Tensor2::zeros(&self.device, [self.len(), nclasses]);
        match &self.device {
            Device::Cpu(cpu) => cpu::unsigned_to_one_hot_f32(self, &mut output),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => cuda::unsigned_to_one_hot_f32(self, &mut output),
        }
        output
    }
}

impl<T: Num, S: DataMut<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Borrows the Tensor as a mutable view
    pub fn view_mut(&mut self) -> TensorViewMut<T, D> {
        let device = self.device.clone();
        let dim = self.dim.clone();
        let data = ViewRepr::new(self.data.buffer_mut());
        TensorViewMut { device, dim, data }
    }
    pub fn copy_from_slice<'a>(&mut self, slice: impl Into<Cow<'a, [T]>>) {
        self.data.buffer_mut()
            .copy_from_slice(slice);
    }
    fn as_mut_cpu_slice(&mut self) -> Option<&mut [T]> {
        self.data
            .buffer_mut()
            .cpu_mut()
            .map(|mut b| b.as_mut_slice())
    }
    #[cfg(feature = "cuda")]
    fn as_mut_cuda_slice(&mut self) -> Option<&mut DeviceSlice<T>> {
        self.data
            .buffer_mut()
            .cuda_mut()
            .map(|b| b.as_mut_device_slice())
    }
    fn as_mut_cpu_ptr(&mut self) -> Option<*mut T> {
        self.data.buffer_mut().cpu_mut().map(|mut b| b.as_mut_ptr())
    }
    #[cfg(feature = "cuda")]
    fn as_mut_cuda_ptr(&mut self) -> Option<*mut T> {
        self.data
            .buffer_mut()
            .cuda_mut()
            .map(|mut b| b.as_mut_ptr())
    }
    /// Fills the tensor with the provided elem.  
    pub fn fill(&mut self, elem: T) {
        self.data.buffer_mut().fill(elem);
    }
    /// Fills the tensor with data sampled from distr with the given rng. On cuda, samples into a vec and then copies to the device buffer
    pub fn fill_random(&mut self, distr: &impl Distribution<T>, mut rng: &mut impl Rng) {
        match self.data.buffer_mut() {
            Buffer::Cpu(cpu_buffer) => {
                cpu_buffer
                    .as_mut_slice()
                    .iter_mut()
                    .zip(distr.sample_iter(&mut rng))
                    .for_each(|(y, x)| *y = x);
            }
            #[cfg(feature = "cuda")]
            Buffer::Cuda(cuda_buffer) => {
                let vec: Vec<T> = distr
                    .sample_iter(&mut rng)
                    .take(cuda_buffer.len())
                    .collect();
                cuda_buffer.copy_from_slice(&vec);
            }
        }
    }
}

impl<T: Num, D: Dimension> From<Tensor<T, D>> for ArcTensor<T, D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        let Tensor { device, dim, data } = tensor;
        let data = ArcRepr::from_buffer(data.buffer);
        Self { device, dim, data }
    }
} 

impl<T: Num, D: Dimension> RwTensor<T, D> {
    /// Similar to RwLock::read(), blocks the current thread until any write access is released, ensures that no writes occur as long as the RwReadTensor is held
    /// Ok: If the lock can be acquired
    /// Err: Returns the PoisonError if the RwLock is poisoned
    pub fn read(&self) -> LockResult<RwReadTensor<T, D>> {
        match self.data.read() {
            Ok(data) => {
                let device = self.device.clone();
                let dim = self.dim.clone();
                Ok(RwReadTensor { device, dim, data })
            }
            Err(poison_error) => {
                let data = poison_error.into_inner();
                let device = self.device.clone();
                let dim = self.dim.clone();
                Err(PoisonError::new(RwReadTensor { device, dim, data }))
            }
        }
    }
    /// Similar to RwLock::write(), blocks until exclusive mutable access can be acquired
    /// Ok: If the lock can be acquired
    /// Err: Returns the PoisonError if the RwLock is poisoned
    pub fn write(&self) -> LockResult<RwWriteTensor<T, D>> {
        match self.data.write() {
            Ok(data) => {
                let device = self.device.clone();
                let dim = self.dim.clone();
                Ok(RwWriteTensor { device, dim, data })
            }
            Err(poison_error) => {
                let data = poison_error.into_inner();
                let device = self.device.clone();
                let dim = self.dim.clone();
                Err(PoisonError::new(RwWriteTensor { device, dim, data }))
            }
        }
    }
}

impl<T: Num, D: Dimension> From<Tensor<T, D>> for RwTensor<T, D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        let Tensor { device, dim, data } = tensor;
        let data = RwRepr::from_buffer(data.buffer);
        Self { device, dim, data }
    }
}

fn broadcast<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D::Larger>,
) {
    debug_assert_eq!(input.device(), output.device());
    match input.device() {
        Device::Cpu(cpu) => cpu::broadcast(input, output),
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda_gpu) => cuda::broadcast(input, output),
    }
}

fn broadcast_backward<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    input_grad: &mut TensorBase<S1, D>,
    output_grad: &TensorBase<S2, D::Larger>,
) {
    debug_assert_eq!(input_grad.device(), output_grad.device());
    match input_grad.device() {
        Device::Cpu(cpu) => cpu::broadcast_backward(input_grad, output_grad),
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda_gpu) => cuda::broadcast_backward(input_grad, output_grad),
    }
}

fn relu_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
    D: Dimension,
>(
    input: &TensorBase<S1, D>,
    input_grad: &mut TensorBase<S2, D>,
    output_grad: &TensorBase<S3, D>,
) {
    debug_assert_eq!(input.device(), input_grad.device());
    debug_assert_eq!(input.device(), output_grad.device());
    debug_assert_eq!(input.raw_dim(), input_grad.raw_dim());
    debug_assert_eq!(input.raw_dim(), output_grad.raw_dim());
    match input.device() {
        Device::Cpu(cpu) => cpu::relu_backward(input, input_grad, output_grad),
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda_gpu) => cuda::relu_backward(input, input_grad, output_grad),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Transpose {
    No,
    Yes,
}

fn gemm<S1: DataRef<Elem = f32>, S2: DataRef<Elem = f32>, S3: DataMut<Elem = f32>>(
    alpha: f32,
    a: &TensorBase<S1, Ix2>,
    trans_a: Transpose,
    b: &TensorBase<S2, Ix2>,
    trans_b: Transpose,
    beta: f32,
    c: &mut TensorBase<S3, Ix2>,
) {
    debug_assert_eq!(&a.device, &b.device);
    debug_assert_eq!(&a.device, &c.device);
    match &a.device {
        Device::Cpu(_) => cpu::gemm(alpha, a, trans_a, b, trans_b, beta, c),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda::gemm(alpha, a, trans_a, b, trans_b, beta, c),
    }
}

fn cross_entropy_backward<
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
    let device = &input.device;
    debug_assert_eq!(device, &input_grad.device);
    debug_assert_eq!(device, &target.device);
    debug_assert_eq!(device, &output_grad.device);
    debug_assert_eq!(input.raw_dim(), input_grad.raw_dim());
    debug_assert_eq!(input.raw_dim(), target.raw_dim());
    match device {
        Device::Cpu(cpu) => cpu::cross_entropy_backward(input, input_grad, target, output_grad),
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda_gpu) => {
            cuda::cross_entropy_backward(input, input_grad, target, output_grad)
        }
    }
}

impl<S1: DataRef<Elem = f32>> TensorBase<S1, Ix2> {
    pub fn dense(
        &self,
        weight: &TensorView2<f32>,
        bias: Option<&TensorView1<f32>>,
    ) -> Tensor2<f32> {
        let (batch_size, inputs) = self.dim();
        let (outputs, inputs2) = weight.dim();
        debug_assert_eq!(inputs, inputs2);
        let mut output = unsafe { Tensor::uninitialized(&self.device, [batch_size, outputs]) };
        match self.device() {
            // Benchmark showed this was slower
            //Device::Cpu(_) => cpu::dense(self, weight, bias, &mut output),
            _ => {
                if let Some(bias) = bias {
                    broadcast(bias, &mut output);
                    gemm(
                        1.,
                        &self,
                        Transpose::No,
                        &weight,
                        Transpose::Yes,
                        1.,
                        &mut output,
                    );
                }   
                else {
                    gemm(
                        1.,
                        &self,
                        Transpose::No,
                        &weight,
                        Transpose::Yes,
                        0.,
                        &mut output,
                    );
                }
            }   
        }
        output
    }
    pub fn cross_entropy_loss(&self, target: &TensorView2<f32>) -> Tensor0<f32> {
        debug_assert_eq!(&self.device, &target.device);
        debug_assert_eq!(self.raw_dim(), target.raw_dim());
        let mut output = unsafe { Tensor::uninitialized(&self.device, self.raw_dim()) };
        match &self.device {
            Device::Cpu(cpu) => cpu::cross_entropy(self, target, &mut output),
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_gpu) => cuda::cross_entropy(self, target, &mut output),
        }
        output.sum()
    }
}

/// Convenience trait to allow passing either usize or [usize, usize] to functions\
/// ie 1.into_2d() == [1, 1]
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

impl<S1: DataRef<Elem = f32>> TensorBase<S1, Ix4> {
    /// Performs a 2D convolution with the given weight, bias, and args.\
    /// Prefer to use [Conv2d](layer/struct.Conv2d.html) instead.\
    /// Inputs:
    ///   * self: Tensor of shape [n, i, ih, iw]
    ///   * weight: Tensor of shape [o, i, kh, kw]
    ///   * bias: Optional Tensor of shape [o]
    ///   * args:
    ///     - strides: [sh, sw]
    ///     - padding: [ph, pw]\
    ///
    /// Returns: Tensor of shape [n, o, oh, ow]\
    /// where:
    ///  - oh = (ih - kh + 2 * ph) / sh + 1
    ///  - ow = (iw - kw + 2 * pw) / sw + 1
    pub fn conv2d(
        &self,
        weight: &TensorView4<f32>,
        bias: Option<&TensorView1<f32>>,
        args: &Conv2dArgs,
    ) -> Tensor4<f32> {
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
        } else {
            Tensor::zeros(&device, [batch_size, outputs, oh, ow])
        };
        match device {
            Device::Cpu(_) => cpu::conv2d(self, weight, bias, args, &mut output),
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => cuda::conv2d(self, weight, bias, args, &mut output),
        }
        output
    }
    /// Computes a 2D max pool\
    /// Inputs:\
    ///   * self: Tensor of shape [n, i, ih, iw]
    ///   * args:
    ///     - kernel: [kh, kw]
    ///     - strides: [sh, sw]
    ///     - padding: [ph, pw]
    ///
    /// Returns: Tensor of shape [n, i, oh, ow]\
    /// where:
    ///   - oh = (ih - (kh - 1) + 2 * ph - 1) / sh + 1
    ///   - ow = (iw - (kw - 1) + 2 * pw - 1) / sw + 1
    pub fn max_pool2d(&self, args: &Pool2dArgs) -> Tensor4<f32> {
        let (output, _) = max_pool2d_forward(self, args, false);
        output
    }
}

fn conv2d_backward_input<S1: DataMut<Elem = f32>>(
    input_grad: &mut TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    debug_assert_eq!(input_grad.device(), weight.device());
    debug_assert_eq!(input_grad.device(), output_grad.device());
    match input_grad.device() {
        Device::Cpu(_) => cpu::conv2d_backward_input(input_grad, weight, args, output_grad),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda::conv2d_backward_input(input_grad, weight, args, output_grad),
    }
}

fn conv2d_backward_weight_bias<S1: DataRef<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    weight_grad: &mut TensorViewMut4<f32>,
    bias_grad: Option<&mut TensorViewMut1<f32>>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    debug_assert_eq!(input.device(), weight_grad.device());
    #[cfg(debug_assertions)]
    {
        if let Some(bias_grad) = &bias_grad {
            assert_eq!(input.device(), bias_grad.device());
        }
    }
    debug_assert_eq!(input.device(), output_grad.device());
    match input.device() {
        Device::Cpu(_) => {
            cpu::conv2d_backward_weight_bias(input, weight_grad, bias_grad, args, output_grad)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            cuda::conv2d_backward_weight_bias(input, weight_grad, bias_grad, args, output_grad)
        }
    }
}

fn max_pool2d_forward<S1: DataRef<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    args: &Pool2dArgs,
    train: bool,
) -> (Tensor4<f32>, Option<Buffer<u8>>) {
    let device = &input.device;
    let (batch_size, inputs, ih, iw) = input.dim();
    let [kh, kw] = args.kernel;
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let oh = (ih - (kh - 1) + 2 * ph - 1) / sh + 1;
    let ow = (iw - (kw - 1) + 2 * pw - 1) / sw + 1;
    let mut output = unsafe { Tensor::uninitialized(&device, [batch_size, inputs, oh, ow]) };
    let workspace: Option<Buffer<u8>> = match device {
        Device::Cpu(_) => {
            let workspace = cpu::max_pool2d_forward(input, args, train, &mut output);
            workspace.map(|ws| ws.into())
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            cuda::max_pool2d(input, args, &mut output);
            None
        }
    };
    (output, workspace)
}

fn max_pool2d_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
>(
    input: &TensorBase<S1, Ix4>,
    input_grad: &mut TensorBase<S2, Ix4>,
    args: &Pool2dArgs,
    workspace: Option<&Buffer<u8>>,
    output_grad: &TensorBase<S3, Ix4>,
) {
    debug_assert_eq!(input.device(), input_grad.device());
    debug_assert_eq!(input.device(), output_grad.device());
    debug_assert_eq!(input.raw_dim(), input_grad.raw_dim());
    match input.device() {
        Device::Cpu(_) => cpu::max_pool2d_backward(input, input_grad, args, workspace, output_grad),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda::max_pool2d_backward(input, input_grad, args, output_grad),
    }
}

fn sgd_with_momentum<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>, D: Dimension>
    (weight: &mut TensorBase<S1, D>, weight_grad: &TensorBase<S2, D>,
     learning_rate: f32, momentum: f32,
     velocity: &mut TensorBase<S3, D>) {
    debug_assert_eq!(weight.device(), weight_grad.device());
    debug_assert_eq!(weight.device(), velocity.device());
    debug_assert_eq!(weight.raw_dim(), weight_grad.raw_dim());
    debug_assert_eq!(weight.raw_dim(), velocity.raw_dim());
    match weight.device() {
        Device::Cpu(_) => cpu::sgd_with_momentum(weight, weight_grad, learning_rate, momentum, velocity),
        #[cfg(feature="cuda")]
        Device::Cuda(_) => cuda::sgd_with_momentum(weight, weight_grad, learning_rate, momentum, velocity)
    }                                     
} 
