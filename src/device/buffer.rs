use super::{
    len_checked, BufferHandle, BufferId, Device, DeviceBase, ReadGuard as RawReadGuard, Result,
    WriteOnly,
};
use crate::{scalar::Scalar, util::size_eq};
use anyhow::bail;
use bytemuck::{Pod, Zeroable};
use std::{
    marker::PhantomData,
    mem::{forget, transmute},
    ops::{Deref, DerefMut},
    sync::Arc,
};

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Marker trait for BufferBase representation.
///
/// Typically use [`Buffer'] / [`Slice`] / [`SliceMut`] types directly.
pub trait Data: Sealed {
    #[doc(hidden)]
    const OWNED: bool;
    /// The element type.
    type Elem;
}

/// Mutable data marker.
pub trait DataMut: Data {}

/// Data for Buffer
pub struct OwnedRepr<T> {
    _m: PhantomData<T>,
}

impl<T> Sealed for OwnedRepr<T> {}

impl<T> Data for OwnedRepr<T> {
    const OWNED: bool = true;
    type Elem = T;
}

impl<T> DataMut for OwnedRepr<T> {}

/// Data for Slice
pub struct SliceRepr<'a, T> {
    _m: PhantomData<&'a T>,
}

impl<T> Sealed for SliceRepr<'_, T> {}

impl<T> Data for SliceRepr<'_, T> {
    const OWNED: bool = false;
    type Elem = T;
}

/// Data for SliceMut
pub struct SliceMutRepr<'a, T> {
    _m: PhantomData<&'a mut T>,
}

impl<T> Sealed for SliceMutRepr<'_, T> {}

impl<T> Data for SliceMutRepr<'_, T> {
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

impl<T, S: Data<Elem = T>> HostBufferBase<S> {
    fn len(&self) -> usize {
        self.len
    }
    fn into_vec(self) -> Vec<T>
    where
        T: Copy,
    {
        if S::OWNED {
            unsafe { Vec::from_raw_parts(self.ptr, self.len, self.capacity) }
        } else {
            self.to_vec()
        }
    }
    fn into_owned(self) -> HostBuffer<T>
    where
        T: Copy,
    {
        if S::OWNED {
            unsafe { transmute(self) }
        } else {
            self.to_owned()
        }
    }
    fn to_owned(&self) -> HostBuffer<T>
    where
        T: Copy,
    {
        self.to_vec().into()
    }
    fn as_slice(&self) -> HostSlice<T> {
        HostSlice::from(self.deref())
    }
    fn as_slice_mut(&mut self) -> HostSliceMut<T>
    where
        S: DataMut,
    {
        HostSliceMut::from(self.deref_mut())
    }
    fn copy_from_slice(&mut self, slice: HostSlice<T>)
    where
        T: Copy,
        S: DataMut,
    {
        self.deref_mut().copy_from_slice(slice.deref());
    }
}

impl<T: Zeroable> HostBuffer<T> {
    unsafe fn alloc(len: usize) -> Self {
        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);
        vec.into()
    }
}

impl<T> From<Vec<T>> for HostBuffer<T> {
    fn from(vec: Vec<T>) -> Self {
        let ptr = vec.as_ptr() as *mut T;
        let len = vec.len();
        let capacity = vec.capacity();
        forget(vec);
        Self { ptr, len, capacity }
    }
}

impl<'a, T> From<&'a [T]> for HostSlice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self {
            ptr: slice.as_ptr() as *mut T,
            len: slice.len(),
            capacity: slice.len(),
        }
    }
}

impl<'a, T> From<&'a mut [T]> for HostSliceMut<'a, T> {
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

unsafe impl<S: Data> Send for HostBufferBase<S> {}
unsafe impl<S: Data> Sync for HostBufferBase<S> {}

pub(super) struct DeviceBufferBase<S: Data> {
    device: Arc<DeviceBase>,
    id: BufferId,
    offset: usize,
    len: usize,
    _m: PhantomData<S>,
}

type DeviceBuffer<T> = DeviceBufferBase<OwnedRepr<T>>;
pub(super) type DeviceSlice<'a, T> = DeviceBufferBase<SliceRepr<'a, T>>;
pub(super) type DeviceSliceMut<'a, T> = DeviceBufferBase<SliceMutRepr<'a, T>>;

impl<T, S: Data<Elem = T>> DeviceBufferBase<S> {
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
    pub(super) fn handle(&self) -> BufferHandle {
        BufferHandle::new_unchecked::<T>(self.id, self.offset, self.len)
    }
    pub(super) fn device(&self) -> &Arc<DeviceBase> {
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
        T: Pod,
        S: DataMut,
        T2: 'a,
        F: FnOnce(WriteOnly<[T]>) -> T2,
    {
        Ok(self
            .device
            .write(self.handle(), |slice| f(slice.cast_slice_mut()))?)
    }
    fn into_owned(self) -> Result<DeviceBuffer<T>>
    where
        T: Pod,
    {
        if S::OWNED {
            unsafe { Ok(transmute(self)) }
        } else {
            self.to_owned()
        }
    }
    fn to_owned(&self) -> Result<DeviceBuffer<T>>
    where
        T: Pod,
    {
        let mut buffer = unsafe { DeviceBuffer::alloc(self.device().clone(), self.len())? };
        buffer.copy_from_slice(self.as_slice())?;
        Ok(buffer)
    }
    async fn into_device(self, device: Arc<DeviceBase>) -> Result<DeviceBuffer<T>>
    where
        T: Pod,
    {
        if Arc::ptr_eq(self.device(), &device) {
            self.into_owned()
        } else {
            let buffer = unsafe { DeviceBuffer::alloc(device, self.len)? };
            let guard_fut = self.device.read(self.handle())?;
            buffer.device.transfer(self.handle(), guard_fut).await?;
            Ok(buffer)
        }
    }
    async fn read(self) -> Result<DeviceReadGuard<S>>
    where
        T: Pod,
    {
        let guard = self.device.read(self.handle())?.finish().await?;
        Ok(DeviceReadGuard {
            _buffer: self,
            guard,
        })
    }
    fn as_slice(&self) -> DeviceSlice<T> {
        unsafe { DeviceSlice::from_raw_parts(self.device.clone(), self.id, self.offset, self.len) }
    }
    fn as_slice_mut(&mut self) -> DeviceSliceMut<T>
    where
        S: DataMut,
    {
        unsafe {
            DeviceSliceMut::from_raw_parts(self.device.clone(), self.id, self.offset, self.len)
        }
    }
    fn copy_from_slice(&mut self, slice: DeviceSlice<T>) -> Result<()>
    where
        S: DataMut,
    {
        Ok(self.device.copy(slice.handle(), self.handle())?)
    }
}

impl<T: Zeroable> DeviceBuffer<T> {
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

impl<T: Pod, S: Data<Elem = T>> DeviceReadGuard<S> {
    fn as_slice(&self) -> &[T] {
        bytemuck::cast_slice(self.guard.deref())
    }
}

impl<T: Pod, S: Data<Elem = T>> Deref for DeviceReadGuard<S> {
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

/// Host / Device Buffer
///
/// Like [`Device`], [`BufferBase`] comes in 2 forms, Host and Device.
///
/// # Host
/// Host buffers are trivially constructible (without copying) from Vec / \[T\]. Note that generally in order to perform computations, the buffer must be transfered to the device via [`.to_device()`].
///
/// # Device
/// Allocated on the device, like a GPU. Operations are asynchronous with the host, only made visible after awaiting ['.read()`].
pub struct BufferBase<S: Data> {
    base: DynBufferBase<S>,
}

/// Owned buffer, like a Vec<T>.
pub type Buffer<T> = BufferBase<OwnedRepr<T>>;

/// Borrowed buffer, like a &\[T\].
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;

/// Mutably borrowed buffer, like a &mut \[T\].
pub type SliceMut<'a, T> = BufferBase<SliceMutRepr<'a, T>>;

impl<T> Buffer<T> {
    /// Allocate a buffer with length `len`.
    ///
    /// # Safety
    /// The buffer will not be initialized.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    ///
    /// # Note
    /// - For constructing a buffer on the host, prefer [`Buffer::from`].
    /// - See [`.zeros()`](BufferBase::zeros) for a safe alternative.
    pub unsafe fn alloc(device: Device, len: usize) -> Result<Self>
    where
        T: Zeroable,
    {
        Ok(match device.into_base() {
            Some(device) => DeviceBuffer::alloc(device, len)?.into(),
            None => HostBuffer::alloc(len).into(),
        })
    }
    /// Creates a buffer with length `len` filled with `elem`.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub fn from_elem(device: Device, len: usize, elem: T) -> Result<Self>
    where
        T: Scalar,
    {
        Ok(match device.into_base() {
            Some(device) => {
                let mut buffer = Self::from(unsafe { DeviceBuffer::alloc(device, len)? });
                buffer.fill(elem)?;
                buffer
            }
            None => Self::from(vec![elem; len]),
        })
    }
    /// Creates a buffer with length `len` filled with 0's.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub fn zeros(device: Device, len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        Self::from_elem(device, len, T::default())
    }
}

impl<T, S: Data<Elem = T>> BufferBase<S> {
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
    /// Reads the data back into a new Buffer. The future will resolve when the data is ready. Prefer [`.read()`](BufferBase::read) for direct access to a Vec<T> or \[T\].
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub async fn into_device(self, device: Device) -> Result<Buffer<T>>
    where
        T: Pod,
    {
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
    /// **Errors**
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub async fn read(self) -> Result<ReadGuard<S>>
    where
        T: Pod,
    {
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
    pub fn as_slice_mut(&mut self) -> SliceMut<T>
    where
        S: DataMut,
    {
        match &mut self.base {
            DynBufferBase::Host(buffer) => buffer.as_slice_mut().into(),
            DynBufferBase::Device(buffer) => buffer.as_slice_mut().into(),
        }
    }
    /// Copies from a slice
    ///
    /// **Errors**
    /// - The source length must match the destination length.
    /// - The source device must match the destination device.
    pub fn copy_from_slice(&mut self, slice: Slice<T>) -> Result<()>
    where
        T: Copy,
        S: DataMut,
    {
        if self.len() != slice.len() {
            bail!(
                "Source buffer length ({}) does not match destination buffer length ({})!",
                self.len(),
                slice.len()
            );
        }
        if self.device() != slice.device() {
            bail!(
                "Source device {:?} does not match destination device {:?}!",
                self.device(),
                slice.device()
            )
        }
        match (&mut self.base, slice.base) {
            (DynBufferBase::Host(buffer), DynBufferBase::Host(slice)) => {
                buffer.copy_from_slice(slice);
            }
            (DynBufferBase::Device(buffer), DynBufferBase::Device(slice)) => {
                buffer.copy_from_slice(slice)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }
    /// Converts into an owned buffer.
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_owned(self) -> Result<Buffer<T>>
    where
        T: Pod,
    {
        match self.base {
            DynBufferBase::Host(buffer) => Ok(buffer.into_owned().into()),
            DynBufferBase::Device(buffer) => Ok(buffer.into_owned()?.into()),
        }
    }
    /// Copies into a new buffer.
    ///
    /// **Errors**
    /// - Allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn to_owned(&self) -> Result<Buffer<T>>
    where
        T: Pod,
    {
        match &self.base {
            DynBufferBase::Host(buffer) => Ok(buffer.to_owned().into()),
            DynBufferBase::Device(buffer) => Ok(buffer.to_owned()?.into()),
        }
    }
    /// The device of the buffer.
    pub fn device(&self) -> Device {
        match &self.base {
            DynBufferBase::Host(_) => Device::host(),
            DynBufferBase::Device(buffer) => Device::from(buffer.device().clone()),
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
    pub(super) fn into_device_buffer_base(self) -> Option<DeviceBufferBase<S>> {
        match self.base {
            DynBufferBase::Device(buffer) => Some(buffer),
            DynBufferBase::Host(_) => None,
        }
    }
    /// Fills the buffer with `elem`.
    ///
    /// **Errors**
    /// - Not supported on the host.
    /// - The device panicked or disconnected.
    /// - The operation could not be performed.
    pub fn fill(&mut self, elem: T) -> Result<()>
    where
        T: Scalar,
        S: DataMut,
    {
        let n = self.len() as u32;
        let name = if size_eq::<T, u64>() {
            "fill_u64"
        } else {
            "fill_u32"
        };
        let builder = if option_env!("RUST_SHADERS").is_some() {
            crate::rust_shaders::core()?.compute_pass(name)?
        } else {
            crate::glsl_shaders::module(name)?.compute_pass("main")?
        };
        let builder = builder.slice_mut(self.as_slice_mut())?;
        if size_eq::<T, u8>() {
            let n = if n % 4 == 0 { n / 4 } else { n / 4 + 1 };
            builder.push(n)?.push([elem; 4])?.submit([n, 1, 1])
        } else if size_eq::<T, u16>() {
            let n = if n % 2 == 0 { n / 2 } else { n / 2 + 1 };
            builder.push(n)?.push([elem; 2])?.submit([n, 1, 1])
        } else {
            builder.push(n)?.push(elem)?.submit([n, 1, 1])
        }
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

impl<T> From<Vec<T>> for Buffer<T> {
    fn from(vec: Vec<T>) -> Self {
        HostBuffer::from(vec).into()
    }
}

impl<'a, T> From<&'a [T]> for Slice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        HostSlice::from(slice).into()
    }
}

impl<'a, T> From<&'a mut [T]> for SliceMut<'a, T> {
    fn from(slice: &'a mut [T]) -> Self {
        HostSliceMut::from(slice).into()
    }
}

enum ReadGuardBase<S: Data> {
    Host(HostBufferBase<S>),
    Device(DeviceReadGuard<S>),
}

impl<T: Pod, S: Data<Elem = T>> ReadGuardBase<S> {
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

impl<T: Pod, S: Data<Elem = T>> Deref for ReadGuardBase<S> {
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

impl<T: Pod, S: Data<Elem = T>> ReadGuard<S> {
    /// Moves into a Vec, copying if necessary.
    pub fn into_vec(self) -> Vec<T> {
        self.base.into_vec()
    }
    /// Returns a slice.
    pub fn as_slice(&self) -> &[T] {
        self.base.as_slice()
    }
}

impl<T: Pod, S: Data<Elem = T>> Deref for ReadGuard<S> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn host_buffer_copy_from_slice() -> Result<()> {
        let mut a = Buffer::from(vec![0; 4]);
        let b_vec = vec![1, 2, 3, 4];
        a.copy_from_slice(b_vec.as_slice().into())?;
        let a_guard = a.read().await?;
        assert_eq!(a_guard.as_slice(), b_vec.as_slice());
        Ok(())
    }

    #[tokio::test]
    async fn device_buffer_copy_from_slice() -> Result<()> {
        let device = Device::new()?;
        let mut a = Slice::from([0; 4].as_ref())
            .into_device(device.clone())
            .await?;
        let b_vec = vec![1, 2, 3, 4];
        let b = Slice::from(b_vec.as_slice())
            .into_device(device.clone())
            .await?;
        a.copy_from_slice(b.as_slice())?;
        //device.sync().await?;
        let a_guard = a.read().await?;
        assert_eq!(a_guard.as_slice(), b_vec.as_slice());
        Ok(())
    }

    async fn fill<T: Scalar>(n: usize, elem: T) -> Result<()> {
        let vec: Vec<T> = (0..n)
            .into_iter()
            .map(|n| T::from_usize(n).unwrap())
            .collect();
        let device = Device::new()?;
        let mut y = Slice::from(vec.as_slice())
            .into_device(device.clone())
            .await?;
        y.fill(elem)?;
        let y_guard = y.read().await?;
        assert!(
            y_guard.iter().copied().all(|y| y == elem),
            "{:?} != {:?}",
            y_guard.as_slice(),
            elem
        );
        Ok(())
    }

    #[tokio::test]
    async fn fill_u8() -> Result<()> {
        fill(15, 11u8).await?;
        fill(100, 251).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_i8() -> Result<()> {
        fill(10, 11u8).await?;
        fill(100, -111i8).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_u16() -> Result<()> {
        fill(10, 11u16).await?;
        fill(1000, 211u16).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_i16() -> Result<()> {
        fill(10, 11i16).await?;
        fill(1000, -211i16).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_u32() -> Result<()> {
        fill(10, 11u32).await?;
        fill(1000, 211u32).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_i32() -> Result<()> {
        fill(10, 11i32).await?;
        fill(1000, -211i32).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_f32() -> Result<()> {
        fill(10, 11.11f32).await?;
        fill(1000, -211.11f32).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_u64() -> Result<()> {
        fill(10, 11u64).await?;
        fill(1000, 211u64).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_i64() -> Result<()> {
        fill(10, 11i64).await?;
        fill(1000, -211i64).await?;
        Ok(())
    }

    #[tokio::test]
    async fn fill_f64() -> Result<()> {
        fill(10, 11.11f64).await?;
        fill(1000, -211.11f64).await?;
        Ok(())
    }
}
