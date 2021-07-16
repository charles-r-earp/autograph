use super::{
    len_checked, BufferHandle, BufferId, Device, DeviceBase, ReadGuard as RawReadGuard, WriteOnly,
};
use crate::Result;
use std::{
    marker::PhantomData,
    mem::{forget, transmute},
    ops::{Deref, DerefMut},
    sync::Arc,
};

mod sealed {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    pub trait ScalarBase: Default + Copy + Pod + Zeroable + 'static {}

    pub trait DataBase {
        #[doc(hidden)]
        const OWNED: bool;
        type Elem: Scalar;
    }
}
use sealed::{DataBase, ScalarBase};

pub trait Scalar: ScalarBase {}

impl ScalarBase for u32 {}

impl<T: ScalarBase> Scalar for T {}

pub trait Data: DataBase {}

pub trait DataMut: Data {}

impl<S: DataBase> Data for S {}

pub struct OwnedRepr<T> {
    _m: PhantomData<T>,
}

impl<T: Scalar> DataBase for OwnedRepr<T> {
    const OWNED: bool = true;
    type Elem = T;
}

impl<T: Scalar> DataMut for OwnedRepr<T> {}

pub struct SliceRepr<'a, T> {
    _m: PhantomData<&'a T>,
}

impl<T: Scalar> DataBase for SliceRepr<'_, T> {
    const OWNED: bool = false;
    type Elem = T;
}

pub struct SliceMutRepr<'a, T> {
    _m: PhantomData<&'a mut T>,
}

impl<T: Scalar> DataBase for SliceMutRepr<'_, T> {
    const OWNED: bool = false;
    type Elem = T;
}

impl<T: Scalar> DataMut for SliceMutRepr<'_, T> {}

#[derive(Debug)]
struct HostBufferBase<S: Data> {
    ptr: *mut S::Elem,
    len: usize,
    capacity: usize,
}

type HostBuffer<T> = HostBufferBase<OwnedRepr<T>>;
type HostSlice<'a, T> = HostBufferBase<SliceRepr<'a, T>>;
type HostSliceMut<'a, T> = HostBufferBase<SliceMutRepr<'a, T>>;

impl<T: Scalar, S: Data<Elem = T>> HostBufferBase<S> {
    fn len(&self) -> usize {
        self.len
    }
    fn into_vec(self) -> Vec<T> {
        if S::OWNED {
            unsafe { Vec::from_raw_parts(self.ptr, self.len, self.capacity) }
        } else {
            self.to_vec()
        }
    }
    fn into_owned(self) -> HostBuffer<T> {
        if S::OWNED {
            unsafe { transmute(self) }
        } else {
            self.to_vec().into()
        }
    }
    fn as_slice(&self) -> HostSlice<T> {
        HostSlice::from(self.deref())
    }
    fn as_mut_slice(&mut self) -> HostSliceMut<T>
    where
        S: DataMut,
    {
        HostSliceMut::from(self.deref_mut())
    }
}

impl<T: Scalar> HostBuffer<T> {
    unsafe fn alloc(len: usize) -> Self {
        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);
        vec.into()
    }
}

impl<T: Scalar> From<Vec<T>> for HostBuffer<T> {
    fn from(vec: Vec<T>) -> Self {
        let ptr = vec.as_ptr() as *mut T;
        let len = vec.len();
        let capacity = vec.capacity();
        forget(vec);
        Self { ptr, len, capacity }
    }
}

impl<'a, T: Scalar> From<&'a [T]> for HostSlice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self {
            ptr: slice.as_ptr() as *mut T,
            len: slice.len(),
            capacity: slice.len(),
        }
    }
}

impl<'a, T: Scalar> From<&'a mut [T]> for HostSliceMut<'a, T> {
    fn from(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            capacity: slice.len(),
        }
    }
}

impl<T, S: Data<Elem = T>> Deref for HostBufferBase<S> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T, S: DataMut<Elem = T>> DerefMut for HostBufferBase<S> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<S: Data> Drop for HostBufferBase<S> {
    fn drop(&mut self) {
        if S::OWNED {
            unsafe {
                Vec::from_raw_parts(self.ptr, self.len, self.capacity);
            }
        }
    }
}

struct DeviceBufferBase<S: Data> {
    device: Arc<DeviceBase>,
    id: BufferId,
    offset: usize,
    len: usize,
    _m: PhantomData<S>,
}

type DeviceBuffer<T> = DeviceBufferBase<OwnedRepr<T>>;
type DeviceSlice<'a, T> = DeviceBufferBase<SliceRepr<'a, T>>;
type DeviceSliceMut<'a, T> = DeviceBufferBase<SliceMutRepr<'a, T>>;

impl<T: Scalar, S: Data<Elem = T>> DeviceBufferBase<S> {
    unsafe fn from_raw_parts(
        device: Arc<DeviceBase>,
        id: BufferId,
        offset: usize,
        len: usize,
    ) -> Self {
        Self {
            device,
            id,
            offset,
            len,
            _m: PhantomData::default(),
        }
    }
    fn handle(&self) -> BufferHandle {
        BufferHandle::new_unchecked::<T>(self.id, self.offset, self.len)
    }
    fn device(&self) -> &Arc<DeviceBase> {
        &self.device
    }
    fn len(&self) -> usize {
        self.len
    }
    /*fn try_write<'a, T2, E, F>(&'a mut self, f: F) -> Result<Result<T2, E>>
        where S: DataMut, T2: 'a, E: 'a, F: FnOnce(WriteOnly<[T]>) -> Result<T2, E> {
        Ok(self.device.try_write(self.handle(), |slice| {
            f(slice.cast_slice_mut())
        })?)
    }*/
    fn write<'a, T2, F>(&'a mut self, f: F) -> Result<T2>
    where
        S: DataMut,
        T2: 'a,
        F: FnOnce(WriteOnly<[T]>) -> T2,
    {
        Ok(self
            .device
            .write(self.handle(), |slice| f(slice.cast_slice_mut()))?)
    }
    fn into_owned(self) -> Result<DeviceBuffer<T>> {
        todo!()
    }
    async fn into_device(self, device: Arc<DeviceBase>) -> Result<DeviceBuffer<T>> {
        if Arc::ptr_eq(self.device(), &device) {
            self.into_owned()
        } else {
            let buffer = unsafe { DeviceBuffer::alloc(device, self.len)? };
            let guard_fut = self.device.read(self.handle())?;
            buffer.device.transfer(self.handle(), guard_fut).await?;
            Ok(buffer)
        }
    }
    async fn read(self) -> Result<DeviceReadGuard<S>> {
        let guard = self.device.read(self.handle())?.finish().await?;
        Ok(DeviceReadGuard {
            _buffer: self,
            guard,
        })
    }
    fn as_slice(&self) -> DeviceSlice<T> {
        unsafe { DeviceSlice::from_raw_parts(self.device.clone(), self.id, self.offset, self.len) }
    }
    fn as_mut_slice(&mut self) -> DeviceSliceMut<T>
    where
        S: DataMut,
    {
        unsafe {
            DeviceSliceMut::from_raw_parts(self.device.clone(), self.id, self.offset, self.len)
        }
    }
}

impl<T: Scalar> DeviceBuffer<T> {
    unsafe fn alloc(device: Arc<DeviceBase>, len: usize) -> Result<Self> {
        let id = device.alloc(len_checked::<T>(len)?)?;
        Ok(Self::from_raw_parts(device, id, 0, len))
    }
}

impl<S: Data> Drop for DeviceBufferBase<S> {
    fn drop(&mut self) {
        if S::OWNED {
            self.device.dealloc(self.id);
        }
    }
}

struct DeviceReadGuard<S: Data> {
    _buffer: DeviceBufferBase<S>,
    guard: RawReadGuard,
}

impl<T: Scalar, S: Data<Elem = T>> DeviceReadGuard<S> {
    fn as_slice(&self) -> &[T] {
        bytemuck::cast_slice(self.guard.deref())
    }
}

impl<T: Scalar, S: Data<Elem = T>> Deref for DeviceReadGuard<S> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

enum DynBufferBase<S: Data> {
    Host(HostBufferBase<S>),
    Device(DeviceBufferBase<S>),
}

impl<S: Data> From<HostBufferBase<S>> for DynBufferBase<S> {
    fn from(buffer: HostBufferBase<S>) -> Self {
        Self::Host(buffer)
    }
}

impl<S: Data> From<DeviceBufferBase<S>> for DynBufferBase<S> {
    fn from(buffer: DeviceBufferBase<S>) -> Self {
        Self::Device(buffer)
    }
}

/// A storage buffer.
///
/// Like [`Device`], [`BufferBase`] comes in 2 forms, Host and Device.
///
/// # Host
/// Host buffers are trivially constructible (without copying) from Vec / [T]. Note that generally in order to perform computations, the buffer must be transfered to the device via [`.to_device()`].
///
/// # Device
/// Allocated on the device, like a GPU. Operations are asynchronous with the host, only made visible after awaiting ['.read()`].
pub struct BufferBase<S: Data> {
    base: DynBufferBase<S>,
}

pub type Buffer<T> = BufferBase<OwnedRepr<T>>;
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;
pub type SliceMut<'a, T> = BufferBase<SliceMutRepr<'a, T>>;

impl<T: Scalar> Buffer<T> {
    /// Allocate a buffer with length `len`.
    ///
    /// # Safety
    /// The buffer will not be initialized.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub unsafe fn alloc(device: Device, len: usize) -> Result<Self> {
        Ok(match device.into_base() {
            Some(device) => DeviceBuffer::alloc(device, len)?.into(),
            None => HostBuffer::alloc(len).into(),
        })
    }
}

impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    /// Transfers the buffer into the `device`.
    ///
    /// # Host to Host
    /// Moves a Buffer or copies a Slice into a new Buffer.
    ///
    /// # Host to Device
    /// Copies the data into a staging buffer and schedules a copy to a device Buffer. Does not wait for the copy to be submittted to the device queue.
    ///
    /// # Device to Device
    /// Reads the src buffer, scheduling a write into a new Buffer on the dst device. The future waits until the read is submitted to the device queue (but not completed) to avoid a deadlock.
    ///
    /// # Device to Host
    /// Reads the data back into a new Buffer. The future will resolve when the data is ready. Prefer [`.read()`] for direct access to a Vec<T> or [T].
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub async fn into_device(self, device: Device) -> Result<Buffer<T>> {
        match (self.base, device.into_base()) {
            (DynBufferBase::Host(buffer), None) => Ok(buffer.into_owned().into()),
            (DynBufferBase::Host(src), Some(device)) => {
                let mut buffer = unsafe { DeviceBuffer::alloc(device, src.len())? };
                buffer.write(|mut slice| slice.copy_from_slice(&src))?;
                Ok(buffer.into())
            }
            (DynBufferBase::Device(src), Some(device)) => Ok(src.into_device(device).await?.into()),
            (DynBufferBase::Device(src), None) => {
                let buffer = HostBuffer::from(src.read().await?.to_vec());
                Ok(buffer.into())
            }
        }
    }
    /// Reads a buffer asynchronously.
    ///
    /// # Host
    /// NOOP
    ///
    /// # Device
    /// Returns a [`ReadGuard`] that can be converted to a slice or a vec. The future will resolve when all previous operations have been completed and the transfer is complete.
    ///
    /// # Example
    ///```
    /// use autograph::{Result, device::{Device, buffer::{Buffer, Slice}}};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let device = Device::new()?;
    ///     let a = Buffer::from(vec![1, 2, 3, 4])
    ///         .into_device(device.clone())
    ///         .await?;
    ///     // Note that we can also borrow the data
    ///     let b = Slice::from([5, 6, 7, 8].as_ref())
    ///         .into_device(device.clone())
    ///         .await?;
    ///     let (a, b) = tokio::try_join!(a.read(), b.read())?;
    ///     let a_vec: Vec<u32> = a.to_vec();
    ///     let b_slice: &[u32] = b.as_slice();
    ///     Ok(())
    /// }
    ///```
    /// Note: Use the [`BlockingFuture`](crate::future::BlockingFuture) trait to block on futures without an executor.
    ///```
    /// use autograph::{Result, device::{Device, buffer::Buffer}, future::BlockingFuture};
    ///
    /// fn main() -> Result<()> {
    ///     let device = Device::new()?;
    ///     let a = Buffer::from(vec![1, 2, 3, 4])
    ///         .into_device(device.clone())
    ///         .block()?;
    ///     let a_vec = a.read().block()?;
    ///     Ok(())
    /// }
    pub async fn read(self) -> Result<ReadGuard<S>> {
        match self.base {
            DynBufferBase::Host(buffer) => Ok(ReadGuardBase::Host(buffer).into()),
            DynBufferBase::Device(buffer) => Ok(ReadGuardBase::Device(buffer.read().await?).into()),
        }
    }
    /// Borrows the buffer as a Slice.
    pub fn as_slice(&self) -> Slice<T> {
        match &self.base {
            DynBufferBase::Host(buffer) => buffer.as_slice().into(),
            DynBufferBase::Device(buffer) => buffer.as_slice().into(),
        }
    }
    /// Mutably borrows the buffer as a SliceMut.
    pub fn as_mut_slice(&mut self) -> SliceMut<T>
    where
        S: DataMut,
    {
        match &mut self.base {
            DynBufferBase::Host(buffer) => buffer.as_mut_slice().into(),
            DynBufferBase::Device(buffer) => buffer.as_mut_slice().into(),
        }
    }
    /// The length of the buffer.
    pub fn len(&self) -> usize {
        match &self.base {
            DynBufferBase::Host(buffer) => buffer.len(),
            DynBufferBase::Device(buffer) => buffer.len(),
        }
    }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S: Data> From<DynBufferBase<S>> for BufferBase<S> {
    fn from(base: DynBufferBase<S>) -> Self {
        Self { base }
    }
}

impl<S: Data> From<HostBufferBase<S>> for BufferBase<S> {
    fn from(buffer: HostBufferBase<S>) -> Self {
        DynBufferBase::Host(buffer).into()
    }
}

impl<S: Data> From<DeviceBufferBase<S>> for BufferBase<S> {
    fn from(buffer: DeviceBufferBase<S>) -> Self {
        DynBufferBase::Device(buffer).into()
    }
}

impl<T: Scalar> From<Vec<T>> for Buffer<T> {
    fn from(vec: Vec<T>) -> Self {
        HostBuffer::from(vec).into()
    }
}

impl<'a, T: Scalar> From<&'a [T]> for Slice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        HostSlice::from(slice).into()
    }
}

impl<'a, T: Scalar> From<&'a mut [T]> for SliceMut<'a, T> {
    fn from(slice: &'a mut [T]) -> Self {
        HostSliceMut::from(slice).into()
    }
}

enum ReadGuardBase<S: Data> {
    Host(HostBufferBase<S>),
    Device(DeviceReadGuard<S>),
}

impl<T: Scalar, S: Data<Elem = T>> ReadGuardBase<S> {
    fn into_vec(self) -> Vec<T> {
        match self {
            Self::Host(buffer) => buffer.into_vec(),
            Self::Device(guard) => guard.as_slice().to_vec(),
        }
    }
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Host(buffer) => buffer.deref(),
            Self::Device(guard) => guard.deref(),
        }
    }
}

impl<T: Scalar, S: Data<Elem = T>> Deref for ReadGuardBase<S> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

/// RAII guard for mapped memory
///
/// # Host Memory
/// Typically devices like discrete GPU's have dedicated device memory which is not visible from the host. Reads and writes are performed via staging buffers, allocated within the driver (for Vulkan, Metal, or DX12). In this case, a guard will hold host memory managed by the driver.
///
/// # Shared Memory
/// Some devices, like integrated GPU's, have device memory that is visible from the host. In this case, the [`Device`] may use this memory for staging, overflowing into host memory when it runs out.
///
/// # Device Memory
/// The guard holds onto the [`BufferBase`], which ensures that the data is not modified.
pub struct ReadGuard<S: Data> {
    base: ReadGuardBase<S>,
}

impl<T: Scalar, S: Data<Elem = T>> ReadGuard<S> {
    /// Moves into a Vec, copying if necessary.
    pub fn into_vec(self) -> Vec<T> {
        self.base.into_vec()
    }
    /// Returns a slice.
    pub fn as_slice(&self) -> &[T] {
        self.base.as_slice()
    }
}

impl<T: Scalar, S: Data<Elem = T>> Deref for ReadGuard<S> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.base.deref()
    }
}

impl<S: Data> From<ReadGuardBase<S>> for ReadGuard<S> {
    fn from(base: ReadGuardBase<S>) -> Self {
        Self { base }
    }
}
