use super::{
    len_checked, BufferHandle, BufferId, Device, DeviceBase, ReadGuard as RawReadGuard, Result,
    WriteOnly,
};
use crate::{
    rust_shaders,
    scalar::{Scalar, ScalarType},
    util::{elem_type_name, type_eq},
};
use anyhow::bail;
use bytemuck::Pod;
use serde::{de::Deserializer, ser::Serializer, Deserialize, Serialize};
use std::{
    fmt::{self, Debug},
    marker::PhantomData,
    mem::{forget, transmute, size_of},
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

impl<T> DataMut for SliceMutRepr<'_, T> {}

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
    fn slice_impl(&self, offset: usize, len: usize) -> HostSlice<T> {
        self.deref()[offset..len].into()
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

impl<T> Clone for HostSlice<'_, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            capacity: self.capacity,
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

impl<T, S: Data<Elem = T>> Debug for HostBufferBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("HostBufferBase")
            .field("len", &self.len)
            .field("elem", &elem_type_name::<T>())
            .finish()
    }
}

impl<T: Serialize, S: Data<Elem = T>> Serialize for HostBufferBase<S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: Serializer,
    {
        self.deref().serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for HostBuffer<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Vec::deserialize(deserializer)?.into())
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
    fn into_owned(self) -> Result<DeviceBuffer<T>> {
        if S::OWNED {
            unsafe { Ok(transmute(self)) }
        } else {
            self.to_owned()
        }
    }
    fn to_owned(&self) -> Result<DeviceBuffer<T>> {
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
    fn slice_impl(&self, offset: usize, len: usize) -> DeviceSlice<T> {
        assert!(offset <= len);
        assert!(offset + len <= self.len());
        unsafe {
            DeviceSlice::from_raw_parts(self.device.clone(), self.id, self.offset + offset, len)
        }
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

impl<T> DeviceBuffer<T> {
    unsafe fn alloc(device: Arc<DeviceBase>, len: usize) -> Result<Self> {
        let id = device.alloc(len_checked::<T>(len)?)?;
        Ok(Self::from_raw_parts(device, id, 0, len))
    }
}

impl<T> Clone for DeviceSlice<'_, T> {
    fn clone(&self) -> Self {
        unsafe { Self::from_raw_parts(self.device.clone(), self.id, self.offset, self.len) }
    }
}

impl<T: Pod + Serialize, S: Data<Elem = T>> Serialize for DeviceBufferBase<S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: Serializer,
    {
        use serde::ser::Error;
        let guard = smol::block_on(self.as_slice().read()).map_err(Se::Error::custom)?;
        guard.deref().serialize(serializer)
    }
}

impl<T, S: Data<Elem = T>> Debug for DeviceBufferBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("DeviceBufferBase")
            .field("device", &self.device.id)
            .field("len", &self.len)
            .field("elem", &elem_type_name::<T>())
            .finish()
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

#[derive(Debug, Serialize)]
#[serde(bound = "S::Elem: Pod + Serialize")]
#[serde(untagged)]
enum DynBufferBase<S: Data> {
    Host(HostBufferBase<S>),
    Device(DeviceBufferBase<S>),
}

type DynBuffer<T> = DynBufferBase<OwnedRepr<T>>;
type DynSlice<'a, T> = DynBufferBase<SliceRepr<'a, T>>;
//type DynSliceMut<'a, T> = DynBufferBase<SliceMutRepr<'a, T>>;

impl<T> Clone for DynSlice<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Host(slice) => slice.clone().into(),
            Self::Device(slice) => slice.clone().into(),
        }
    }
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

impl<'de, T: Deserialize<'de>> Deserialize<'de> for DynBuffer<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(HostBuffer::deserialize(deserializer)?.into())
    }
}

/// Host / Device Buffer
///
/// Like [`Device`], [`BufferBase`] comes in 2 forms, Host and Device.
///
/// # Host
/// Host buffers are trivially constructible (without copying) from Vec / \[T\]. Note that generally in order to perform computations, the buffer must be transfered to the device via [`.to_device()`](BufferBase::into_device()).
///
/// # Device
/// Allocated on the device, like a GPU. Operations are asynchronous with the host, only made visible after awaiting ['.read()`].
#[derive(Serialize)]
#[serde(bound = "S::Elem: Pod + Serialize")]
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
        T: Default + Copy,
    {
        Ok(match device.into_base() {
            Some(device) => DeviceBuffer::alloc(device, len)?.into(),
            None => HostBuffer::from(vec![T::default(); len]).into(),
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
    /// Transfers the buffer into the `device` as an [`ArcBuffer`].
    pub async fn into_device_shared(self, device: Device) -> Result<ArcBuffer<T>>
    where
        T: Pod,
    {
        Ok(self.into_device(device).await?.into())
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
    fn slice_impl(&self, offset: usize, len: usize) -> Slice<T> {
        match &self.base {
            DynBufferBase::Host(buffer) => buffer.slice_impl(offset, len).into(),
            DynBufferBase::Device(buffer) => buffer.slice_impl(offset, len).into(),
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
        T: Copy,
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
        T: Copy,
    {
        match &self.base {
            DynBufferBase::Host(buffer) => Ok(buffer.to_owned().into()),
            DynBufferBase::Device(buffer) => Ok(buffer.to_owned()?.into()),
        }
    }
    /// Converts into a [`ArcBuffer`].
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_shared(self) -> Result<ArcBuffer<T>>
    where
        T: Copy,
    {
        Ok(self.into_owned()?.into())
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
    // Gets a vec when on the host
    #[cfg(feature = "tensor")]
    pub(crate) fn into_vec(self) -> Option<Vec<T>>
    where
        T: Copy,
    {
        match self.base {
            DynBufferBase::Host(buffer) => Some(buffer.into_vec()),
            DynBufferBase::Device(_) => None,
        }
    }
    pub(super) fn into_device_buffer_base(self) -> Option<DeviceBufferBase<S>> {
        match self.base {
            DynBufferBase::Host(_) => None,
            DynBufferBase::Device(buffer) => Some(buffer),
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
        let size = size_of::<T>();
        let core = crate::rust_shaders::core()?;
        let (builder, gs) = match size {
            1 => {
                let builder = core.compute_pass("fill::fill_u32")?;
                let n = if n % 4 == 0 { n / 4 } else { n / 4 + 1 };
                let builder = builder.push(n)?.push([elem; 4])?;
                (builder, n)
            }
            2 => {
                let builder = core.compute_pass("fill::fill_u32")?;
                let n = if n % 2 == 0 { n / 2 } else { n / 2 + 1 };
                let builder = builder.push(n)?.push([elem; 2])?;
                (builder, n)
            }
            4 if n % 2 == 0 => {
                let builder = core.compute_pass("fill::fill_u32x2")?;
                let n = n / 2;
                let builder = builder.push(n)?.push([elem; 2])?;
                (builder, n)
            }
            4 => {
                let builder = core.compute_pass("fill::fill_u32")?;
                let builder = builder.push(n)?.push(elem)?;
                (builder, n)
            }
            8 => {
                let builder = core.compute_pass("fill::fill_u32x2")?;
                let builder = builder.push(n)?.push(elem)?;
                (builder, n)
            }
            _ => unreachable!(),
        };
        let builder = builder.slice_mut(self.as_slice_mut())?;
        unsafe { builder.submit([gs, 1, 1]) }
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

impl<T> Clone for Slice<'_, T> {
    fn clone(&self) -> Self {
        self.base.clone().into()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Buffer<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(DynBufferBase::deserialize(deserializer)?.into())
    }
}

impl<T, S: Data<Elem = T>> Debug for BufferBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BufferBase")
            .field("device", &self.device())
            .field("len", &self.len())
            .field("elem", &elem_type_name::<T>())
            .finish()
    }
}

/// Casts
impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    /// Casts the buffer into a new buffer.
    ///
    /// # Note
    /// NOOP for Buffer<T> -> Buffer<T>.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_into<T2: Scalar>(self) -> Result<Buffer<T2>> {
        self.scale_into(T2::one())
    }
    /// Casts the buffer into a new buffer.
    ///
    /// Returns a CowBuffer, where for T -> T, borrows the buffer.
    ///
    /// **Errors**
    /// See [`BufferBase::scale_into()`].
    pub fn cast_to<T2: Scalar>(&self) -> Result<CowBuffer<T2>> {
        if type_eq::<T, T2>() {
            Ok(unsafe { transmute(CowBuffer::from(self.as_slice())) })
        } else {
            Ok(self.as_slice().cast_into::<T2>()?.into())
        }
    }
    /// Scales the buffer into a new buffer.
    ///
    /// # Note
    /// NOOP for Buffer<T> -> Buffer<T> where alpha == 1.
    ///
    /// # Supported Types
    /// T: u8, u16, bf16, u32, i32, f32
    /// T2: bf16, u32, i32, f32
    ///
    /// **Errors**
    /// - Not implemented on the host.
    /// - Types not implemented.
    /// - Potentially allocates, see [`Buffer::alloc()`](BufferBase::alloc()).
    /// - The device paicked or disconnected.
    pub fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Buffer<T2>> {
        use ScalarType::*;
        if alpha == T2::one() && type_eq::<T, T2>() {
            Ok(unsafe { transmute(self.into_owned()?) })
        } else {
            if !matches!(T::scalar_type(), U8 | U16 | BF16 | U32 | I32 | F32)
                || !matches!(T2::scalar_type(), BF16 | U32 | I32 | F32)
            {
                bail!(
                    "scale_into {} -> {} not implemented!",
                    T::scalar_name(),
                    T2::scalar_name(),
                )
            }
            let mut output = unsafe { Buffer::alloc(self.device(), self.len())? };
            let n = self.len() as u32;
            // TODO: DX12 not working
            //let api = self.device().info().map_or(Api::Vulkan, |i| i.api());
            /*let (module, entry, gs) = match (T::scalar_type(), T2::scalar_type()) {
                (U8, BF16 | F32) => {
                    let module = rust_shaders::core()?;
                    let entry = format!("cast::scale_{}_{}", T::scalar_name(), T2::scalar_name());
                    (module, entry, n)
                }
                (BF16, BF16) => {
                    let module = rust_shaders::core()?;
                    let entry = format!("cast::scale_{}_{}", T::scalar_name(), T2::scalar_name());
                    (module, entry, n)
                }
                _ => {
                    // patch for old impl
                    if matches!(T2::scalar_type(), BF16) {
                        output.fill(T2::zero())?;
                    }
                    let name = format!("scaled_cast_{}_{}", T::scalar_name(), T2::scalar_name());
                    let module = glsl_shaders::module(name)?;
                    let entry = String::from("main");
                    (module, entry, n)
                }
            };*/
            let entry = format!("cast::scale_{}_{}", T::scalar_name(), T2::scalar_name());
            let builder = rust_shaders::core()?
                .compute_pass(entry)?
                .slice(self.as_slice())?
                .slice_mut(output.as_slice_mut())?
                .push(n)?;
            let builder = {
                use ScalarType::*;
                match T2::scalar_type() {
                    U8 | U16 => builder.push(alpha.to_u32().unwrap())?,
                    I8 | I16 => builder.push(alpha.to_i32().unwrap())?,
                    F16 | BF16 => builder.push(alpha.to_f32().unwrap())?,
                    _ => builder.push(alpha)?,
                }
            };
            unsafe {
                builder.submit([n, 1, 1])?;
            }
            Ok(output)
        }
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

/// Shared Buffer
#[derive(Clone)]
pub struct ArcBuffer<T> {
    buffer: Arc<Buffer<T>>,
    offset: usize,
    len: usize,
}

impl<T> ArcBuffer<T> {
    unsafe fn from_raw_parts(buffer: Arc<Buffer<T>>, offset: usize, len: usize) -> Self {
        debug_assert!(
            offset + len <= buffer.len(),
            "{} > {}",
            offset + len,
            buffer.len()
        );
        Self {
            buffer,
            offset,
            len,
        }
    }
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
        T: Default + Copy,
    {
        Ok(Buffer::alloc(device, len)?.into())
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
        Ok(Buffer::from_elem(device, len, elem)?.into())
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
        Ok(Buffer::zeros(device, len)?.into())
    }
    /// Borrows the buffer as a Slice.
    pub fn as_slice(&self) -> Slice<T> {
        self.buffer.slice_impl(self.offset, self.len)
    }
    pub(crate) fn try_unwrap(self) -> Result<Buffer<T>, Self> {
        if self.offset == 0 && self.len == self.buffer.len() {
            match Arc::try_unwrap(self.buffer) {
                Ok(buffer) => Ok(buffer),
                Err(buffer) => Err(Self {
                    buffer,
                    offset: self.offset,
                    len: self.len,
                }),
            }
        } else {
            Err(self)
        }
    }
    #[allow(unused)]
    pub(crate) fn get_mut(&mut self) -> Option<SliceMut<T>> {
        if self.offset == 0 && self.len == self.buffer.len() {
            Arc::get_mut(&mut self.buffer).map(Buffer::as_slice_mut)
        } else {
            None
        }
    }
    #[allow(unused)]
    pub(crate) fn make_mut(&mut self) -> Result<SliceMut<T>>
    where
        T: Copy,
    {
        if let Some(slice) = self.get_mut() {
            return Ok(unsafe { transmute(slice) });
        }
        let buffer = self.to_owned()?;
        self.buffer = Arc::new(buffer);
        let buffer = unsafe { &mut *(Arc::as_ptr(&self.buffer) as *mut Buffer<T>) };
        Ok(unsafe { transmute(buffer.as_slice_mut()) })
    }
    /// Converts into a [`Buffer`].
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_owned(self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        match self.try_unwrap() {
            Ok(buffer) => Ok(buffer),
            Err(buffer) => buffer.to_owned(),
        }
    }
    /// Copies into a new [`Buffer`].
    ///
    /// **Errors**
    /// - Allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn to_owned(&self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        self.as_slice().to_owned()
    }
    /// Transfers the buffer into the `device`.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device(self, device: Device) -> Result<Buffer<T>>
    where
        T: Pod,
    {
        if self.device() == device {
            self.into_owned()
        } else {
            self.as_slice().into_device(device).await
        }
    }
    /// Transfers the buffer into the `device` as an [`ArcBuffer`].
    ///
    /// NOOP if the buffer is on `device`.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device_shared(self, device: Device) -> Result<Self>
    where
        T: Pod,
    {
        if self.device() == device {
            Ok(self)
        } else {
            Ok(self.into_device(device).await?.into())
        }
    }
    /// The device of the buffer.
    pub fn device(&self) -> Device {
        self.buffer.device()
    }
    /// The length of the buffer.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> From<Buffer<T>> for ArcBuffer<T> {
    fn from(buffer: Buffer<T>) -> Self {
        let len = buffer.len();
        unsafe { Self::from_raw_parts(buffer.into(), 0, len) }
    }
}

impl<T: Pod + Serialize> Serialize for ArcBuffer<T> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ArcBuffer<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Buffer::deserialize(deserializer)?.into())
    }
}

impl<T> Debug for ArcBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ArcBuffer")
            .field("device", &self.buffer.device())
            .field("len", &self.len)
            .field("elem", &elem_type_name::<T>())
            .finish()
    }
}

/// Casts
impl<T: Scalar> ArcBuffer<T> {
    /// Casts the buffer into a new buffer.
    ///
    /// # Note
    /// NOOP for Buffer<T> -> Buffer<T>.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_into<T2: Scalar>(self) -> Result<Buffer<T2>> {
        match self.try_unwrap() {
            Ok(buffer) => buffer.cast_into(),
            Err(this) => this.as_slice().cast_into(),
        }
    }
    /// Casts the buffer into a new buffer.
    ///
    /// Returns a CowBuffer, where for T -> T, borrows the buffer.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_to<T2: Scalar>(&self) -> Result<CowBuffer<T2>> {
        Ok(self.as_slice().cast_into()?.into())
    }
    /// Scales the buffer into a new buffer.
    ///
    /// # Note
    /// NOOP for Buffer<T> -> Buffer<T> where alpha == 1.
    ///
    /// See [`BufferBase::scale_into()`].
    pub fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Buffer<T2>> {
        self.as_slice().scale_into(alpha)
    }
}

/// A Borrowed or Owned Buffer
#[derive(Debug)]
pub enum CowBuffer<'a, T> {
    /// Borrowed
    Slice(Slice<'a, T>),
    /// Owned
    Buffer(Buffer<T>),
}

impl<T> CowBuffer<'_, T> {
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
        T: Default + Copy,
    {
        Ok(Buffer::alloc(device, len)?.into())
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
        Ok(Buffer::from_elem(device, len, elem)?.into())
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
        Ok(Buffer::zeros(device, len)?.into())
    }
    /// Borrows the buffer as a Slice.
    pub fn as_slice(&self) -> Slice<T> {
        match self {
            Self::Slice(slice) => slice.as_slice(),
            Self::Buffer(buffer) => buffer.as_slice(),
        }
    }
    pub(crate) fn try_unwrap(self) -> Result<Buffer<T>, Self> {
        match self {
            Self::Buffer(buffer) => Ok(buffer),
            slice => Err(slice),
        }
    }
    #[allow(unused)]
    pub(crate) fn get_mut(&mut self) -> Option<SliceMut<T>> {
        match self {
            Self::Slice(_) => None,
            Self::Buffer(buffer) => Some(buffer.as_slice_mut()),
        }
    }
    #[allow(unused)]
    pub(crate) fn make_mut(&mut self) -> Result<SliceMut<T>>
    where
        T: Copy,
    {
        match self {
            Self::Slice(slice) => {
                let buffer = slice.to_owned()?;
                *self = Self::Buffer(buffer);
                match self {
                    Self::Slice(_) => unreachable!(),
                    Self::Buffer(buffer) => Ok(buffer.as_slice_mut()),
                }
            }
            Self::Buffer(buffer) => Ok(buffer.as_slice_mut()),
        }
    }
    /// Transfers the buffer into the `device`.
    ///
    /// **Errors**
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device(self, device: Device) -> Result<Buffer<T>>
    where
        T: Pod,
    {
        if self.device() == device {
            self.into_owned()
        } else {
            self.as_slice().into_device(device).await
        }
    }
    /// Transfers the buffer into the `device` as an [`ArcBuffer`].
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device_shared(self, device: Device) -> Result<ArcBuffer<T>>
    where
        T: Pod,
    {
        if self.device() == device {
            self.into_shared()
        } else {
            Ok(self.into_device(device).await?.into())
        }
    }
    /// Converts into a [`Buffer`].
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_owned(self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        match self {
            Self::Slice(slice) => slice.to_owned(),
            Self::Buffer(buffer) => Ok(buffer),
        }
    }
    /// Copies into a new [`Buffer`].
    ///
    /// **Errors**
    /// - Allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn to_owned(&self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        match self {
            Self::Slice(slice) => slice.to_owned(),
            Self::Buffer(buffer) => buffer.to_owned(),
        }
    }
    /// Converts into a [`ArcBuffer`].
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_shared(self) -> Result<ArcBuffer<T>>
    where
        T: Copy,
    {
        Ok(self.into_owned()?.into())
    }
    /// The device of the buffer.
    pub fn device(&self) -> Device {
        match self {
            Self::Slice(slice) => slice.device(),
            Self::Buffer(buffer) => buffer.device(),
        }
    }
    /// The length of the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Slice(slice) => slice.len(),
            Self::Buffer(buffer) => buffer.len(),
        }
    }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T> From<Slice<'a, T>> for CowBuffer<'a, T> {
    fn from(slice: Slice<'a, T>) -> Self {
        Self::Slice(slice)
    }
}

impl<'a, T> From<&'a [T]> for CowBuffer<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self::Slice(slice.into())
    }
}

impl<T> From<Buffer<T>> for CowBuffer<'_, T> {
    fn from(buffer: Buffer<T>) -> Self {
        Self::Buffer(buffer)
    }
}

impl<T> From<Vec<T>> for CowBuffer<'_, T> {
    fn from(vec: Vec<T>) -> Self {
        Self::Buffer(vec.into())
    }
}

/// Casts
impl<T: Scalar> CowBuffer<'_, T> {
    /// Casts the buffer into a new buffer.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_into<T2: Scalar>(self) -> Result<Buffer<T2>> {
        match self.try_unwrap() {
            Ok(buffer) => buffer.cast_into(),
            Err(this) => this.as_slice().cast_into(),
        }
    }
    /// Casts the buffer into a new buffer.
    ///
    /// Returns a CowBuffer, where for T -> T, borrows the buffer.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_to<T2: Scalar>(&self) -> Result<CowBuffer<T2>> {
        Ok(self.as_slice().cast_into()?.into())
    }
    /// Scales the buffer into a new buffer.
    ///
    /// See [`BufferBase::scale_into()`].
    pub fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Buffer<T2>> {
        self.as_slice().scale_into(alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "device_tests")]
    use half::{bf16, f16};

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
    async fn host_buffer_serde() -> Result<()> {
        let x = [1u32, 2, 3, 4];
        let slice = Slice::from(x.as_ref());
        let buffer: Buffer<u32> = bincode::deserialize(&bincode::serialize(&slice)?)?;
        assert_eq!(x.as_ref(), buffer.read().await?.as_slice());
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn device_buffer_copy_from_slice() -> Result<()> {
        let device = Device::new()?;
        let _s = device.acquire().await;
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

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn device_buffer_serde() -> Result<()> {
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = [1u32, 2, 3, 4];
        let buffer = Slice::from(x.as_ref()).into_device(device.clone()).await?;
        let buffer: Buffer<u32> = bincode::deserialize(&bincode::serialize(&buffer)?)?;
        assert_eq!(x.as_ref(), buffer.read().await?.as_slice());
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    async fn fill<T: Scalar>(n: usize, elem: T) -> Result<()> {
        let vec: Vec<T> = (0..n)
            .into_iter()
            .map(|n| T::from_usize(n).unwrap())
            .collect();
        let device = Device::new()?;
        let _s = device.acquire().await;
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

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u8() -> Result<()> {
        fill(15, 11u8).await?;
        fill(100, 251).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i8() -> Result<()> {
        fill(10, 11u8).await?;
        fill(100, -111i8).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u16() -> Result<()> {
        fill(10, 11u16).await?;
        fill(1000, 211u16).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i16() -> Result<()> {
        fill(10, 11i16).await?;
        fill(1000, -211i16).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_f16() -> Result<()> {
        fill(10, f16::from_f32(11.11)).await?;
        fill(1000, f16::from_f32(-211.11)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_bf16() -> Result<()> {
        fill(10, bf16::from_f32(11.11)).await?;
        fill(1000, bf16::from_f32(-211.11)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u32() -> Result<()> {
        fill(10, 11u32).await?;
        fill(1000, 211u32).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i32() -> Result<()> {
        fill(10, 11i32).await?;
        fill(1000, -211i32).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_f32() -> Result<()> {
        fill(10, 11.11f32).await?;
        fill(1000, -211.11f32).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u64() -> Result<()> {
        fill(10, 11u64).await?;
        fill(1000, 211u64).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i64() -> Result<()> {
        fill(10, 11i64).await?;
        fill(1000, -211i64).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_f64() -> Result<()> {
        fill(10, 11.11f64).await?;
        fill(1000, -211.11f64).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    mod scale {
        use super::{Result, Scalar};

        async fn scale<T1: Scalar, T2: Scalar>(n: usize, alpha: T2) -> Result<()> {
            use super::*;

            let device = Device::new()?;
            let _s = device.acquire().await;
            let x_vec = (0..n)
                .into_iter()
                .map(|x| T1::from(x).unwrap())
                .collect::<Vec<_>>();
            let y_vec = x_vec
                .iter()
                .copied()
                .map(|x| T2::from_f32(x.to_f32().unwrap() * alpha.to_f32().unwrap()).unwrap())
                .collect::<Vec<_>>();
            let x_buffer = Slice::from(x_vec.as_slice())
                .into_device(device.clone())
                .await?;
            let y_buffer = x_buffer.scale_into(alpha)?;
            let y_guard = y_buffer.read().await?;
            assert_eq!(y_guard.as_slice(), y_vec.as_slice());
            Ok(())
        }

        macro_rules! impl_scale_tests {
            (($($t1:ident),+) => $t2s:tt) => {
                $(
                    mod $t1 {
                        use super::scale;
                        use crate::result::Result;

                        impl_scale_tests!{@Inner $t1 => $t2s}
                    }
                )+
            };
            (@Inner $t1:ident => ($($t2:ident),+)) => {
                $(
                    #[tokio::test]
                    async fn $t2() -> Result<()> {
                        #[allow(unused_imports)]
                        use half::{f16, bf16};
                        use num_traits::FromPrimitive;

                        for n in [10, 67] {
                            for alpha in [2i8, -2i8, 3i8] {
                                if let Some(alpha) = $t2::from_i8(alpha) {
                                    scale::<$t1, _>(n, alpha).await?;
                                }
                            }
                        }
                        Ok(())
                    }
                )+
            };
        }

        impl_scale_tests! { (u8, u16, bf16, u32, i32, f32) => (bf16, u32, i32, f32) }
    }
    /*
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_bf16() -> Result<()> {
        scale::<u8, bf16>(10, 3u8.into()).await?;
        scale::<u8, bf16>(100, bf16::from_f32(-2f32)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_u32() -> Result<()> {
        scale::<u8, u32>(10, 3).await?;
        scale::<u8, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_i32() -> Result<()> {
        scale::<u8, i32>(10, -3).await?;
        scale::<u8, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_f32() -> Result<()> {
        scale::<u8, f32>(10, -3.).await?;
        scale::<u8, f32>(100, 17.).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_bf16() -> Result<()> {
        scale::<u16, bf16>(10, 3u8.into()).await?;
        scale::<u16, bf16>(100, bf16::from_f32(-2f32)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_u32() -> Result<()> {
        scale::<u16, u32>(10, 3).await?;
        scale::<u16, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_i32() -> Result<()> {
        scale::<u16, i32>(10, -3).await?;
        scale::<u16, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_f32() -> Result<()> {
        scale::<u16, f32>(10, -3.).await?;
        scale::<u16, f32>(100, 17.).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_bf16() -> Result<()> {
        scale::<u16, bf16>(10, 3u8.into()).await?;
        scale::<u16, bf16>(100, bf16::from_f32(-2f32)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_u32() -> Result<()> {
        scale::<u16, u32>(10, 3).await?;
        scale::<u16, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_i32() -> Result<()> {
        scale::<u16, i32>(10, -3).await?;
        scale::<u16, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_f32() -> Result<()> {
        scale::<u16, f32>(10, -3.).await?;
        scale::<u16, f32>(100, 17.).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_bf16() -> Result<()> {
        scale::<bf16, _>(10, bf16::from_f32(-3.)).await?;
        scale::<bf16, _>(100, bf16::from_f32(17)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_u32() -> Result<()> {
        scale::<bf16, u32>(10, 3).await?;
        scale::<bf16, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_i32() -> Result<()> {
        scale::<bf16, i32>(10, -3).await?;
        scale::<bf16, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_f32() -> Result<()> {
        scale::<u16, f32>(10, -3.).await?;
        scale::<u16, f32>(100, 17.).await?;
        Ok(())
    }*/
}
