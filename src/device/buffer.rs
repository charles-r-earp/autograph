use crate::{
    buffer::{ArcBuffer, CowBuffer},
    device::{BufferBinding, Device, DeviceBase, StorageBuffer, StorageBufferReadGuard},
    result::Result,
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
    mem::{forget, size_of, transmute},
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
    device: DeviceBase,
    storage: Arc<StorageBuffer>,
    offset: usize,
    len: usize,
    _m: PhantomData<S>,
}

type DeviceBuffer<T> = DeviceBufferBase<OwnedRepr<T>>;
pub(super) type DeviceSlice<'a, T> = DeviceBufferBase<SliceRepr<'a, T>>;
pub(super) type DeviceSliceMut<'a, T> = DeviceBufferBase<SliceMutRepr<'a, T>>;

impl<T, S: Data<Elem = T>> DeviceBufferBase<S> {
    pub(super) fn device(&self) -> DeviceBase {
        self.device.clone()
    }
    fn len(&self) -> usize {
        self.len
    }
    fn into_owned(self) -> Result<DeviceBuffer<T>> {
        if S::OWNED {
            unsafe { Ok(transmute(self)) }
        } else {
            self.to_owned()
        }
    }
    fn to_owned(&self) -> Result<DeviceBuffer<T>> {
        let mut buffer = unsafe { DeviceBuffer::alloc(self.device(), self.len())? };
        buffer.copy_from_slice(self.as_slice())?;
        Ok(buffer)
    }
    async fn into_device(self, device: DeviceBase) -> Result<DeviceBuffer<T>>
    where
        T: Pod,
    {
        if self.device == device {
            self.into_owned()
        } else {
            /*let buffer = unsafe { DeviceBuffer::alloc(device, self.len)? };
            let guard_fut = self.device.read(self.handle())?;
            buffer.device.transfer(self.handle(), guard_fut).await?;
            Ok(buffer)*/
            todo!()
        }
    }
    async fn read(self) -> Result<DeviceReadGuard<S>>
    where
        T: Pod,
    {
        let guard = self
            .device
            .download(
                self.storage.clone(),
                self.offset * size_of::<T>(),
                self.len * size_of::<T>(),
            )
            .await?;
        Ok(DeviceReadGuard {
            _buffer: self,
            guard,
        })
    }
    fn as_slice(&self) -> DeviceSlice<T> {
        DeviceSlice {
            device: self.device.clone(),
            storage: self.storage.clone(),
            offset: self.offset,
            len: self.len,
            _m: PhantomData::default(),
        }
    }
    fn slice_impl(&self, offset: usize, len: usize) -> DeviceSlice<T> {
        assert!(offset <= len);
        assert!(offset + len <= self.len());
        DeviceSlice {
            device: self.device.clone(),
            storage: self.storage.clone(),
            offset: self.offset + offset,
            len,
            _m: PhantomData::default(),
        }
    }
    fn as_slice_mut(&mut self) -> DeviceSliceMut<T>
    where
        S: DataMut,
    {
        DeviceSliceMut {
            device: self.device.clone(),
            storage: self.storage.clone(),
            offset: self.offset,
            len: self.len,
            _m: PhantomData::default(),
        }
    }
    fn copy_from_slice(&mut self, slice: DeviceSlice<T>) -> Result<()>
    where
        S: DataMut,
    {
        assert!(self.len == slice.len(), "{} != {}", self.len, slice.len);
        let size = self.len * size_of::<T>();
        let n = if size % 4 != 0 {
            size / 4 + 1
        } else {
            size / 4
        };
        let lhs: DeviceSliceMut<u8> = DeviceSliceMut {
            device: self.device.clone(),
            storage: self.storage.clone(),
            offset: self.offset * size_of::<T>(),
            len: self.len * size_of::<T>(),
            _m: PhantomData::default(),
        };
        let rhs: DeviceSlice<u8> = DeviceSlice {
            device: slice.device.clone(),
            storage: slice.storage.clone(),
            offset: slice.offset * size_of::<T>(),
            len: slice.len * size_of::<T>(),
            _m: PhantomData::default(),
        };
        let builder = rust_shaders::core()?
            .compute_pass("copy::copy_u32")?
            .slice(rhs.into())?
            .slice_mut(lhs.into())?
            .push(n as u32)?;
        unsafe {
            builder.submit([n as u32, 1, 1])?;
        }
        Ok(())
    }
}

impl<T> DeviceBuffer<T> {
    unsafe fn alloc(device: DeviceBase, len: usize) -> Result<Self> {
        let storage = unsafe { device.alloc(len * size_of::<T>())? };
        Ok(Self {
            device,
            storage,
            offset: 0,
            len,
            _m: PhantomData::default(),
        })
    }
    fn upload(device: DeviceBase, slice: &[T]) -> Result<Self>
    where
        T: Pod,
    {
        let storage = device.upload(bytemuck::cast_slice(slice))?;
        Ok(Self {
            device,
            storage,
            offset: 0,
            len: slice.len(),
            _m: PhantomData::default(),
        })
    }
}

impl<T> DeviceSlice<'_, T> {
    pub(super) fn binding(&self) -> BufferBinding {
        BufferBinding {
            storage: self.storage.clone(),
            offset: self.offset * size_of::<T>(),
            len: self.len * size_of::<T>(),
            mutable: false,
        }
    }
}

impl<T> DeviceSliceMut<'_, T> {
    pub(super) fn binding_mut(&self) -> BufferBinding {
        BufferBinding {
            storage: self.storage.clone(),
            offset: self.offset * size_of::<T>(),
            len: self.len * size_of::<T>(),
            mutable: true,
        }
    }
}

impl<T> Clone for DeviceSlice<'_, T> {
    fn clone(&self) -> Self {
        DeviceSlice {
            device: self.device.clone(),
            storage: self.storage.clone(),
            offset: self.offset,
            len: self.len,
            _m: PhantomData::default(),
        }
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
            //.field("device", &self.device.id)
            .field("len", &self.len)
            .field("elem", &elem_type_name::<T>())
            .finish()
    }
}

struct DeviceReadGuard<S: Data> {
    _buffer: DeviceBufferBase<S>,
    guard: StorageBufferReadGuard,
}

impl<T: Pod, S: Data<Elem = T>> DeviceReadGuard<S> {
    fn as_slice(&self) -> &[T] {
        bytemuck::cast_slice(&self.guard)
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
    /// - Out of device memory.
    ///
    /// # Note
    /// - For constructing a buffer on the host, prefer [`Buffer::from`].
    /// - See [`Buffer::zeros()`](BufferBase::zeros) for a safe alternative.
    pub unsafe fn alloc(device: Device, len: usize) -> Result<Self>
    where
        T: Default + Copy,
    {
        Ok(match device.into_base() {
            Some(device) => unsafe { DeviceBuffer::alloc(device, len)?.into() },
            None => HostBuffer::from(vec![T::default(); len]).into(),
        })
    }
    /// Creates a buffer with length `len` filled with `elem`.
    ///
    /// **Errors**
    /// - Out of device memory.
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
    /// - Out of device memory.
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
                Ok(DeviceBuffer::upload(device, &src)?.into())
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
    pub(crate) fn slice_impl(&self, offset: usize, len: usize) -> Slice<T> {
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
            DynBufferBase::Device(buffer) => Device::from(buffer.device.clone()),
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
