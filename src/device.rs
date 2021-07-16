use crate::{future::BlockingFuture, Result};
use anyhow::anyhow;
use derive_more::Display;
pub use gfx_hal::adapter::DeviceType;
use hibitset::BitSet;
use lazy_static::lazy_static;
use smol::lock::Mutex;
use std::{
    fmt::{self, Debug},
    mem::size_of,
    sync::Arc,
};

mod engine;
use engine::{builders::EngineBuilder, Engine, ReadGuard, ReadGuardFuture, MAX_ALLOCATION};
pub mod buffer;

const MAX_DEVICE_ID: u32 =
    (size_of::<usize>() * size_of::<usize>() * size_of::<usize>() * size_of::<usize>()) as u32;

lazy_static! {
    static ref DEVICE_IDS: Mutex<BitSet> = Mutex::default();
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeviceId(u32);

impl DeviceId {
    async fn create() -> Result<Self> {
        let mut ids = DEVICE_IDS.lock().await;
        for id in 0..MAX_DEVICE_ID {
            if !ids.contains(id) {
                ids.add(id);
                return Ok(Self(id));
            }
        }
        Err(anyhow!(""))
    }
    async fn destroy(self) {
        DEVICE_IDS.lock().await.remove(self.0);
    }
    fn as_u32(&self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct BufferId(u64);

#[derive(Clone, Copy, Debug, Display, Eq, PartialEq, thiserror::Error)]
#[repr(u8)]
enum DeviceError {
    /// Max allocation size is 256 MB
    AllocationTooLarge,
    // gfx_hal
    DeviceUnsupported,
    /// Host memory exhausted
    OutOfHostMemory,
    /// Device memory exhausted
    OutOfDeviceMemory,

    OutOfBounds,
    InitializationFailed,
    MissingFeature,
    TooManyObjects,
    /// The device panicked or disconnected
    DeviceLost,
    MappingFailed,
    Access,
}

type DeviceResult<T> = Result<T, DeviceError>;

mod write_only {
    use bytemuck::Pod;
    use std::fmt::{self, Debug};

    pub(super) struct WriteOnly<'a, T: ?Sized>(&'a mut T);

    impl<'a, T: ?Sized> WriteOnly<'a, T> {
        pub(super) fn new(x: &'a mut T) -> Self {
            Self(x)
        }
        /*
        /// Returns a mutable reference to the contents.\
        ///
        /// # Safety
        /// T may be uninitialized. Don't read from it.
        pub(super) unsafe fn write_only(&mut self) -> &mut T {
            &mut self.0
        }*/
    }

    impl<'a, T: Copy + Pod> WriteOnly<'a, [T]> {
        pub(super) fn cast_slice_mut<T2: Pod>(self) -> WriteOnly<'a, [T2]> {
            WriteOnly(bytemuck::cast_slice_mut(self.0))
        }
        pub(super) fn copy_from_slice(&mut self, slice: &[T]) {
            self.0.copy_from_slice(slice);
        }
    }

    impl<T: ?Sized> Debug for WriteOnly<'_, T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("WriteOnly").finish()
        }
    }
}
#[doc(inline)]
use write_only::WriteOnly;

/*

pub(super) struct ComputePass {
    module: ModuleId,
    entry: EntryId,
    local_size: [u32; 3],
    buffers: Vec<BufferRef>,
    push_constants: Vec<u8>,
}*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DeviceScore {
    device_memory: u64,
    device_type_score: u8,
}

pub mod builders {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DeviceBuilder {
        pub(super) engine_builder: EngineBuilder,
    }

    impl From<EngineBuilder> for DeviceBuilder {
        fn from(engine_builder: EngineBuilder) -> Self {
            Self { engine_builder }
        }
    }

    impl DeviceBuilder {
        /// [`API`] of the device.
        pub fn api(&self) -> Api {
            self.engine_builder.api()
        }
        /// [`DeviceType`] of the device.
        pub fn device_type(&self) -> DeviceType {
            self.engine_builder.adapter_info().device_type.clone()
        }
        fn device_type_score(&self) -> u8 {
            use DeviceType::*;
            match self.device_type() {
                DiscreteGpu => 0,
                IntegratedGpu => 1,
                VirtualGpu => 2,
                Cpu => 3,
                Other => 4,
            }
        }
        pub(super) fn score(&self) -> DeviceScore {
            DeviceScore {
                device_memory: self.device_memory(),
                device_type_score: self.device_type_score(),
            }
        }
        /// Memory of the device in bytes.
        ///
        /// This is the maximum memory that can potentially be allocated on the device. The actual\
        /// amount that can be allocated may be less, due to fragementation or use by other\
        /// processes.
        pub fn device_memory(&self) -> u64 {
            self.engine_builder.device_memory()
        }
        pub(super) fn with_device_id(&self, id: DeviceId) -> Self {
            let mut builder = self.clone();
            builder.engine_builder.set_device_id(id);
            builder
        }
        /// Builds the device.
        ///
        /// **Errors**
        /// - Initialization failed (the device could not be created).
        /// - OutOfDeviceMemory: Preallocates ~1 GB of device memory.
        pub fn build(&self) -> Result<Device> {
            Device::build(self)
        }
    }
}
use builders::DeviceBuilder;

/// Supported API's
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Api {
    Vulkan,
    Metal,
    DX12,
}

struct BufferHandle {
    id: BufferId,
    offset: u32,
    len: u32,
}

impl BufferHandle {
    fn new_unchecked<T>(id: BufferId, offset: usize, len: usize) -> Self {
        let offset = len_unchecked::<T>(offset);
        let len = len_unchecked::<T>(len);
        Self { id, offset, len }
    }
}

fn len_checked<T>(len: usize) -> DeviceResult<u32> {
    if let Some(len) = len.checked_mul(size_of::<T>()) {
        if len <= MAX_ALLOCATION {
            return Ok(len as u32);
        }
    }
    Err(DeviceError::AllocationTooLarge)
}

fn len_unchecked<T>(len: usize) -> u32 {
    let len = len * size_of::<T>();
    debug_assert!(len <= MAX_ALLOCATION);
    len as u32
}

#[derive(Debug)]
pub(super) struct DeviceBase {
    id: DeviceId,
    engine: Engine,
}

impl DeviceBase {
    fn alloc(&self, len: u32) -> DeviceResult<BufferId> {
        self.engine.alloc(len)
    }
    fn dealloc(&self, id: BufferId) {
        self.engine.dealloc(id)
    }
    fn try_write<'a, T, E, F>(&'a self, buffer: BufferHandle, f: F) -> DeviceResult<Result<T, E>>
    where
        T: 'a,
        E: 'a,
        F: FnOnce(WriteOnly<[u8]>) -> Result<T, E>,
    {
        self.engine.try_write(buffer, f)
    }
    fn write<'a, T2, F>(&'a self, buffer: BufferHandle, f: F) -> DeviceResult<T2>
    where
        T2: 'a,
        F: FnOnce(WriteOnly<[u8]>) -> T2,
    {
        self.try_write(buffer, |slice| Ok(f(slice)))
            .map(Result::<_, ()>::unwrap)
    }
    async fn transfer(
        &self,
        buffer: BufferHandle,
        read_guard_fut: ReadGuardFuture,
    ) -> DeviceResult<()> {
        self.engine.transfer(buffer, read_guard_fut).await?;
        Ok(())
    }
    fn read(&self, buffer: BufferHandle) -> DeviceResult<ReadGuardFuture> {
        self.engine.read(buffer)
    }
}

impl Drop for DeviceBase {
    fn drop(&mut self) {
        self.id.destroy().block();
    }
}

/// The Host or a CPU or GPU
///
/// # Host
/// A device can refer to the Host [`Device::host()`]. This supports allocation, some initialization functions, and copying / transfers to a device. Notably it does not support shader execution, [`Buffer`](buffer::Buffer)'s must be transfered via [`.into_device()`](buffer::Buffer::into_device) before shaders can be executed.
///
/// # Device
/// A device can refer to a logical device [`Device::new()`]. In addition to allocation, initialization, and transfer, buffers on the device can be used in compute shaders.

/// ## API's
/// - Vulkan (All platforms)
/// - Metal (MacOS / iOS)
/// - DX12 (Windows)
///
/// ## Hardware
/// - DiscreteGpu
/// - IntegratedGpu
/// - VirtualGpu
/// - CPU
///
/// ## Logical Device
/// A device is a logical device, more than one device can potentially be created (depending on the driver) for a phyical GPU or CPU. Each [`Device::new()`] with create a new device, independent of any others created for the same physical device, other than sharing driver resources (like memory and queues).
///
/// # Asynchronous
/// Each device spawns its own worker thread to submit work to, connected via a channel. This allows work to be scheduled even while the device is busy. The channel is bounded, however, so if the device falls behind too far it will block the host from submitting more work. Results can be read back via [`BufferBase::read()`](buffer::BufferBase::read()).
///
/// # Clone
/// Devices can be cloned to create shared references.
///```
/// # use autograph::{Result, device::Device};
/// # fn main() -> Result<()> {
/// // Devices are independent.
/// let a = Device::new().unwrap();
/// let b = Device::new().unwrap();
/// assert_ne!(&a, &b);
///
/// // Cloning creates a copy
/// let b = a.clone();
/// assert_eq!(a, b);
/// # Ok(())
/// # }
///```
#[derive(Clone)]
pub struct Device {
    base: Option<Arc<DeviceBase>>,
}

impl Device {
    /// Returns a host device.
    ///
    /// The host is stateless and trivially constructable. That is, all [`host()`](Device::host())'s are equivalent. The host supports basic operations like initialization and copying. It does not support shader execution, so most computations will fail.
    ///
    /// # Example
    ///```
    /// # use autograph::device::Device;
    /// assert_eq!(Device::host(), Device::host());
    ///```
    pub fn host() -> Self {
        Self { base: None }
    }
    /// Enumerates available [`DeviceBuilder`]'s.
    ///
    /// See also [`new()`](Device::new()) for a quick and easy alternative.
    ///
    /// # Example
    ///```no_run
    /// # use autograph::{Result, device::{Device, Api, DeviceType}};
    /// # fn main() -> Result<()> {
    /// use anyhow::anyhow;
    /// // Filter for a Vulkan DiscreteGpu with 4 GB of device memory
    /// let device = Device::builder_iter()
    ///     .filter(|b| b.api() == Api::Vulkan)
    ///     .filter(|b| b.device_type() == DeviceType::DiscreteGpu)
    ///     .filter(|b| b.device_memory() >= 4_000_000_000)
    ///     .next()
    ///     .ok_or(anyhow!("No valid device!"))?
    ///     .build()?;
    /// # Ok(())
    /// # }
    ///```
    pub fn builder_iter() -> impl Iterator<Item = DeviceBuilder> {
        Engine::builder_iter().map(EngineBuilder::into)
    }
    /// Creates a device.
    ///
    /// The device will not necessarily be the first. Will try to select the best device. Prefer [Device::builder_iter] for more control.
    pub fn new() -> Result<Self> {
        let mut builders: Vec<_> = Self::builder_iter().collect();
        if builders.is_empty() {
            return Err(anyhow!("No device!"));
        }
        builders.sort_by_key(DeviceBuilder::score);
        // Find any device that will build
        builders
            .iter()
            .take(builders.len() - 1)
            .find_map(|b| b.build().ok())
            .map_or(builders.last().unwrap().build(), Ok)
    }
    fn build(builder: &DeviceBuilder) -> Result<Self> {
        let id = DeviceId::create().block()?;
        let engine = builder.with_device_id(id).engine_builder.build()?;
        let base = Some(Arc::new(DeviceBase { id, engine }));
        Ok(Self { base })
    }
    pub(super) fn into_base(self) -> Option<Arc<DeviceBase>> {
        self.base
    }
}

/// Either "Host" or "Device(\<id\>)", where id is a unique integer
impl Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(base) = self.base.as_ref() {
            f.write_str(&format!("Device({})", base.id.as_u32()))
        } else {
            f.write_str("Host")
        }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match (&self.base, &other.base) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => true,
        }
    }
}

impl Eq for Device {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_new() -> Result<()> {
        Device::new()?;
        Ok(())
    }

    #[test]
    fn device_builder_iter_build() -> Result<()> {
        for builder in Device::builder_iter() {
            builder.build()?;
        }
        Ok(())
    }
}
