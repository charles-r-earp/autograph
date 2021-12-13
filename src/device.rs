use crate::result::Result;
use anyhow::anyhow;
use derive_more::Display;
use hibitset::{AtomicBitSet, BitSet};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
#[cfg(all(test, feature = "device_tests"))]
use smol::lock::{Semaphore, SemaphoreGuard};
use std::{
    fmt::{self, Debug},
    mem::size_of,
    sync::Arc,
};

#[cfg(feature = "profile")]
mod profiler;

mod engine;
use engine::{builders::EngineBuilder, Engine, ReadGuard, ReadGuardFuture, MAX_ALLOCATION};

#[doc(hidden)]
pub mod buffer;
#[doc(hidden)]
pub use buffer::{ArcBuffer, Buffer, BufferBase, CowBuffer, Data, DataMut, Slice, SliceMut};

#[doc(hidden)]
pub mod shader;
#[doc(inline)]
pub use shader::Module;
use shader::{EntryId, ModuleId};

const MAX_DEVICE_ID: u32 =
    (size_of::<usize>() * size_of::<usize>() * size_of::<usize>() * size_of::<usize>()) as u32;

static DEVICE_IDS: Lazy<Mutex<BitSet>> = Lazy::new(Mutex::default);

#[derive(Debug, Clone, Copy)]
pub(super) struct DeviceId(u32);

impl DeviceId {
    fn create() -> Result<Self> {
        let mut ids = DEVICE_IDS.lock();
        for id in 0..MAX_DEVICE_ID {
            if !ids.contains(id) {
                ids.add(id);
                return Ok(Self(id));
            }
        }
        Err(anyhow!(""))
    }
    fn destroy(self) {
        DEVICE_IDS.lock().remove(self.0);
    }
    fn as_u32(&self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct BufferId(u64);

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
    ShaderCompilationFailed,
    // Unable to open summary file
    #[cfg(feature = "profile")]
    ProfileSummaryError,
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

/// Info about a [`Device`].
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    name: String,
    #[allow(unused)]
    vendor: usize,
    #[allow(unused)]
    device: usize,
    api: Api,
    device_type: DeviceType,
    memory: u64,
}

impl DeviceInfo {
    /// The name of the device.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// [`Api`] of the device.
    pub fn api(&self) -> Api {
        self.api
    }
    /// [`DeviceType`] of the device.
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }
    /// Memory of the device in bytes.
    ///
    /// This is the maximum amount of memory that can be used for Buffers. However, some or all of this memory may currently be in use by other processes. Additionally, not all memory is guaranteed to be utilized due to fragmentation.
    pub fn memory(&self) -> u64 {
        self.memory
    }
}

/// Type of a device.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum DeviceType {
    /// A discrete gpu.
    DiscreteGpu,
    /// An integrated gpu.
    IntegratedGpu,
    /// A cpu shader engine.
    Cpu,
    /// A virtual or hosted gpu.
    VirtualGpu,
    /// An unknown device.
    Other,
}

/// Device builders.
pub mod builders {
    use super::*;
    use anyhow::bail;
    use buffer::{Slice, SliceMut};
    use bytemuck::Pod;
    use shader::EntryDescriptor;
    use std::marker::PhantomData;

    /// Builds a [`Device`].
    #[derive(Clone)]
    pub struct DeviceBuilder {
        pub(super) engine_builder: EngineBuilder,
        pub(super) info: DeviceInfo,
    }

    impl From<EngineBuilder> for DeviceBuilder {
        fn from(engine_builder: EngineBuilder) -> Self {
            let info = DeviceInfo {
                name: engine_builder.name().to_string(),
                vendor: engine_builder.vendor(),
                device: engine_builder.device(),
                api: engine_builder.api(),
                device_type: engine_builder.device_type(),
                memory: engine_builder.memory(),
            };
            Self {
                engine_builder,
                info,
            }
        }
    }

    impl DeviceBuilder {
        /// [`DeviceInfo`]
        pub fn info(&self) -> &DeviceInfo {
            &self.info
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

    impl Debug for DeviceBuilder {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.info().fmt(f)
        }
    }

    /// A builder for executing compute shaders.
    ///
    /// See [`Module::compute_pass()`].
    pub struct ComputePassBuilder<'m, 'b> {
        device: Option<Arc<DeviceBase>>,
        module: &'m Module,
        entry: String,
        descriptor: &'m EntryDescriptor,
        args: Vec<ComputePassArg>,
        push_constants: Vec<u8>,
        specialization: Vec<u8>,
        spec_index: usize,
        _m: PhantomData<&'b ()>,
    }

    impl<'m, 'b> ComputePassBuilder<'m, 'b> {
        fn new(module: &'m Module, entry: String) -> Result<Self> {
            let descriptor = module
                .descriptor
                .entries
                .get(&entry)
                .ok_or_else(|| anyhow!("Entry not found {:?}!", entry))?;
            Ok(Self {
                device: None,
                module,
                entry,
                descriptor,
                args: Vec::with_capacity(descriptor.buffers.len()),
                push_constants: Vec::with_capacity(descriptor.push_constant_size()),
                specialization: Vec::with_capacity(descriptor.specialization_size()),
                spec_index: 0,
                _m: PhantomData::default(),
            })
        }
        /// Adds a slice as an argument to the shader at the next binding.
        ///
        /// **Errors**
        /// - The slice is on the host.
        /// - The slice is not on the same device as previous arguments.
        /// - The slice is modified in the shader.
        /// - There are no more arguments to be bound.
        ///
        /// # Note
        /// - Does not check the numerical type of the buffer.
        pub fn slice<'b2, T>(self, slice: Slice<'b2, T>) -> Result<ComputePassBuilder<'m, 'b2>>
        where
            'b2: 'b,
            T: Pod,
        {
            if let Some(buffer) = slice.into_device_buffer_base() {
                self.slice_impl(Some((buffer.handle(), buffer.device())), false)
            } else {
                self.slice_impl(None, false)
            }
        }
        /// Adds a mutable slice as an argument to the shader at the next binding.
        ///
        /// **Errors**
        /// - The slice is on the host.
        /// - The slice is not on the same device as previous arguments.
        /// - There are no more arguments to be bound.
        ///
        /// # Note
        /// - Does not check the numerical type of the buffer.
        pub fn slice_mut<'b2, T>(
            self,
            slice: SliceMut<'b2, T>,
        ) -> Result<ComputePassBuilder<'m, 'b2>>
        where
            'b2: 'b,
            T: Pod,
        {
            if let Some(buffer) = slice.into_device_buffer_base() {
                self.slice_impl(Some((buffer.handle(), buffer.device())), true)
            } else {
                self.slice_impl(None, true)
            }
        }
        fn slice_impl<'b2>(
            mut self,
            buffer_device: Option<(BufferHandle, &Arc<DeviceBase>)>,
            mutable: bool,
        ) -> Result<ComputePassBuilder<'m, 'b2>> {
            let (buffer, device) = buffer_device.ok_or_else(|| {
                anyhow!("Compute passes can only be executed on a device, not the host!")
            })?;
            if let Some(current_device) = self.device.as_ref() {
                if !Arc::ptr_eq(current_device, device) {
                    bail!(
                        "Provided slice at binding {} is on {:?} but previous slices are on {:?}!",
                        self.args.len(),
                        Device::from(device.clone()),
                        Device::from(current_device.clone()),
                    );
                }
            } else {
                self.device.replace(device.clone());
            }
            self.check_args_size(self.args.len() + 1, false)?;
            let declared_mutable = *self.descriptor.buffers.get(self.args.len()).unwrap();
            if !mutable && declared_mutable {
                bail!(
                    "Provided slice at binding {}, but it is modified in {:?} entry {:?}!",
                    self.args.len(),
                    self.module,
                    &self.entry,
                )
            }
            self.args.push(ComputePassArg { buffer, mutable });
            Ok(ComputePassBuilder {
                device: self.device,
                module: self.module,
                entry: self.entry,
                descriptor: self.descriptor,
                args: self.args,
                push_constants: self.push_constants,
                specialization: self.specialization,
                spec_index: self.spec_index,
                _m: PhantomData::default(),
            })
        }
        fn check_args_size(&mut self, size: usize, finished: bool) -> Result<()> {
            let buffers_size = self.descriptor.buffers.len();
            if size > buffers_size || (finished && size != buffers_size) {
                Err(anyhow!(
                    "Provided number of slices {} does not match the number of bindings {} declared in {:?} entry {:?}!",
                    size,
                    buffers_size,
                    self.module,
                    &self.entry,
                ))
            } else {
                Ok(())
            }
        }
        /// Adds one or more push constants.
        ///
        /// `push_constant` can be a numerical type, an array, or a struct implementing [`Pod`](bytemuck::Pod).
        ///
        /// **Errors**
        /// - The push constants exceed the size declared in the module.
        pub fn push<T: Pod>(self, push_constant: T) -> Result<Self> {
            self.push_bytes(bytemuck::cast_slice(&[push_constant]))
        }
        /// Adds push constants as bytes.
        ///
        /// Like [`.push()`](ComputePassBuilder::push), but accepts bytes directly.
        ///
        /// **Errors**
        /// - The push constants exceed the size declared in the module.
        pub fn push_bytes(mut self, bytes: &[u8]) -> Result<Self> {
            let push_constant_size = self.push_constants.len() + bytes.len();
            if cfg!(debug_assertions) {
                self.check_push_constant_size(push_constant_size, false)?;
            }
            self.push_constants.extend_from_slice(bytes);
            Ok(self)
        }
        fn check_push_constant_size(&self, size: usize, finished: bool) -> Result<()> {
            let declared_size = self.descriptor.push_constant_size();
            if size > declared_size || finished && size != declared_size {
                Err(anyhow!(
                    "Provided push constant size {} does not match declared size {} in {:?} entry {:?}!",
                    size,
                    declared_size,
                    self.module,
                    self.entry,
                ))
            } else {
                Ok(())
            }
        }
        /* TODO: Not implemented in Module::parse_spirv
        /// Adds one or more specialization constants.
        ///
        /// `special` can be a numerical type, an array, or a struct implementing [`Pod`](bytemuck::Pod).
        ///
        /// **Errors**
        /// - The specialization constants exceed the size declared in the module.
        ///
        /// # Note
        /// "Specializes" the module on first use, then reuses it on subsequent calls.
        pub fn special<T: Pod>(self, special: T) -> Result<Self> {
            self.special_bytes(bytemuck::cast_slice(&[special]))
        }
        /// Adds specialization constants as bytes.
        ///
        /// Like [`.special()`](ComputePassBuilder::push), but accepts bytes directly.
        ///
        /// **Errors**
        /// - The specialization constants exceed the size declared in the module.
        ///
        /// # Note
        /// "Specializes" the module on first use, then reuses it on subsequent calls.
        pub fn special_bytes(mut self, bytes: &[u8]) -> Result<Self> {
            let specialization_size = self.push_constants.len() + bytes.len();
            self.check_specialization_size(specialization_size, false)?;
            self.specialization.extend_from_slice(bytes);
            Ok(self)
        }*/
        /*fn check_specialization_size(&self) -> Result<()> {
            if self.spec_index != self.descriptor.spec_constants.len() {
                bail!(
                    "Provided specialization {:?}, declared {:?} in {:?} entry {:?}!",
                    &self.descriptor.spec_constants[..self.spec_index],
                    &self.descriptor.spec_constants,
                    self.module,
                    self.entry,
                );
            }
            debug_assert_eq!(self.specialization.len(), self.descriptor.specialization_size());
            Ok(())
        }*/
        /// Submits the compute pass to the device with the given `global_size`.
        ///
        /// The compute pass will be executed with enough "work groups" such that for each dimension GLOBAL_SIZE >= WORK_GROUPS * LOCAL_SIZE.
        ///
        /// # Non-Blocking
        /// Does not block, the implementation submits work to the driver asynchronously.
        ///
        /// # Safety
        /// The caller must ensure:
        /// 1. The type `T` of each slice argument is valid. For example `Slice<u8>` can be bound to shader buffer that is a uint or u32, or any arbitrary type.
        /// 2. The type(s) of the push constants are valid. Only the size (in bytes) is checked, the shader can interpret the bytes arbitrarily.
        /// 3. The `global_size` is valid. If too large, the shader may perform out of bounds reads or writes, which is undefined behavior. If too small, the shader may not fully compute the output(s). Note that the shader will be executed with potentially more invocations than `global_size`, in blocks of the `local_size` declared in the shader. Typically the actual work size (ie the length of the buffer(s)) is passed as a push constant.
        /// 4. The shader is valid. Executing a compute shader is essentially a foreign function call, and is inherently unsafe.
        ///
        /// **Errors**
        /// - The number of arguments does not match the number declared in the module.
        /// - The total size in bytes of all push constants does not match that declared in the module.
        /// - The total size in bytes of all specialization constants does not match that declared in the module.
        /// - The device panicked or disconnected.
        /// - The device could not compile the module (compiles all entries on first use).
        /// - The device could not compile the specialization (compiles each entry / specialization on first use).
        pub unsafe fn submit(mut self, global_size: [u32; 3]) -> Result<()> {
            fn work_groups(global_size: [u32; 3], local_size: [u32; 3]) -> [u32; 3] {
                let mut work_groups = [0; 3];
                for (wg, (gs, ls)) in work_groups
                    .iter_mut()
                    .zip(global_size.iter().copied().zip(local_size.iter().copied()))
                {
                    *wg = if gs % ls == 0 { gs / ls } else { gs / ls + 1 };
                }
                work_groups
            }
            if self.args.is_empty() {
                bail!("No slices passed to compute pass builder!");
            }
            self.check_args_size(self.args.len(), true)?;
            self.check_push_constant_size(self.push_constants.len(), true)?;
            //self.check_specialization_size()?;
            let local_size = self.descriptor.local_size;
            let work_groups = work_groups(global_size, local_size);
            let compute_pass = ComputePass {
                module: self.module.id,
                module_name: self.module.name().map_or_else(String::new, str::to_string),
                entry: self.descriptor.id,
                entry_name: self.entry,
                work_groups,
                local_size,
                args: self.args,
                push_constants: self.push_constants,
                specialization: self.specialization,
            };
            self.device
                .unwrap()
                .compute_pass(self.module, compute_pass)?;
            Ok(())
        }
    }

    impl Module {
        /// Returns a [`ComputePassBuilder`] used to setup and submit a compute shader.
        ///
        /// **Errors**
        /// - The `entry` was not found in the module.
        pub fn compute_pass(&self, entry: impl Into<String>) -> Result<ComputePassBuilder> {
            ComputePassBuilder::new(self, entry.into())
        }
    }
}
use builders::DeviceBuilder;

/// Supported API's
///
/// Most recent GPU's should have drivers for at least one API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Api {
    /// <https://www.vulkan.org/>
    Vulkan,
    /// <https://developer.apple.com/metal/>
    Metal,
    /// <https://docs.microsoft.com/windows/win32/directx>
    DX12,
}

#[derive(Debug)]
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
struct ComputePassArg {
    buffer: BufferHandle,
    mutable: bool,
}

#[derive(Debug)]
struct ComputePass {
    module: ModuleId,
    module_name: String,
    entry: EntryId,
    entry_name: String,
    work_groups: [u32; 3],
    local_size: [u32; 3],
    args: Vec<ComputePassArg>,
    push_constants: Vec<u8>,
    specialization: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct DeviceBase {
    id: DeviceId,
    engine: Engine,
    info: DeviceInfo,
    modules: AtomicBitSet,
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
    fn copy(&self, src: BufferHandle, dst: BufferHandle) -> DeviceResult<()> {
        self.engine.copy(src, dst)
    }
    fn compute_pass(&self, module: &Module, compute_pass: ComputePass) -> DeviceResult<()> {
        debug_assert_eq!(module.id, compute_pass.module);
        if !self.modules.contains(module.id.0) {
            self.engine.module(module)?;
            self.modules.add_atomic(module.id.0);
        }
        self.engine.compute(compute_pass)
    }
    async fn sync(&self) -> DeviceResult<()> {
        self.engine.sync()?.finish().await
    }
    fn info(&self) -> &DeviceInfo {
        &self.info
    }
}

impl Drop for DeviceBase {
    fn drop(&mut self) {
        self.id.destroy();
    }
}

#[cfg(all(test, feature = "device_tests"))]
static TEST_DEVICE: Mutex<Option<Device>> = parking_lot::const_mutex(None);

#[cfg(all(test, feature = "device_tests"))]
static TEST_DEVICE_SEMAPHORE: Semaphore = Semaphore::new(4);

/// Device.
#[derive(Clone)]
pub struct Device {
    base: Option<Arc<DeviceBase>>,
}

impl Device {
    /// Creates a device.
    ///
    /// This is a simple way to get a single device. Use [`.builder_iter()`](Device::builder_iter()) for more control.
    pub fn new() -> Result<Self> {
        fn new_impl() -> Result<Device> {
            let mut builders: Vec<_> = Device::builder_iter().collect();
            builders.sort_by_key(|b| b.info().device_type());
            dbg!(&builders);
            builders
                .first()
                .ok_or_else(|| anyhow!("No device!"))?
                .build()
        }
        #[cfg(all(test, feature = "device_tests"))]
        {
            let mut guard = TEST_DEVICE.lock();
            if let Some(device) = guard.as_ref() {
                return Ok(device.clone());
            } else {
                let device = if let Some(name) = option_env!("AUTOGRAPH_TEST_DEVICE") {
                    Device::builder_iter()
                        .find(|b| b.info().name() == name)
                        .ok_or(anyhow!("Device {:?} not found!", name))?
                        .build()?
                } else {
                    new_impl()?
                };
                guard.replace(device.clone());
                return Ok(device);
            }
        }
        #[cfg_attr(test, allow(unreachable_code))]
        new_impl()
    }
    /// Enumerates available [`DeviceBuilder`]'s.
    ///
    /// See also [`new()`](Device::new()).
    ///
    /// # Example
    ///```no_run
    /// # use autograph::{result::Result, device::{Device, Api, DeviceType}};
    /// # fn main() -> Result<()> {
    /// use anyhow::anyhow;
    /// // Filter for a Vulkan DiscreteGpu with 4 GB of device memory
    /// let device = Device::builder_iter()
    ///     .find(|builder| {
    ///         let info = builder.info();
    ///         info.api() == Api::Vulkan
    ///             && info.device_type() == DeviceType::DiscreteGpu
    ///             && info.memory() >= 4_000_000_000
    ///     })
    ///     .ok_or(anyhow!("No valid device!"))?
    ///     .build()?;
    /// # Ok(())
    /// # }
    ///```
    pub fn builder_iter() -> impl Iterator<Item = DeviceBuilder> {
        Engine::builder_iter().map(EngineBuilder::into)
    }
    /// Returns a host device.
    ///
    /// The host is stateless and trivially constructable. That is, all [`host()`](Device::host())'s are equivalent. The host supports basic operations like initialization and copying. It does not support shader execution, so most computations will fail.
    pub fn host() -> Self {
        Self { base: None }
    }
    fn build(builder: &DeviceBuilder) -> Result<Self> {
        let id = DeviceId::create()?;
        let engine = builder.with_device_id(id).engine_builder.build()?;
        let base = Some(Arc::new(DeviceBase {
            id,
            engine,
            info: builder.info().clone(),
            modules: AtomicBitSet::new(),
        }));
        Ok(Self { base })
    }
    pub(super) fn into_base(self) -> Option<Arc<DeviceBase>> {
        self.base
    }
    /// [`DeviceInfo`]
    ///
    /// Returns info about the device. The host returns [`None`].
    pub fn info(&self) -> Option<&DeviceInfo> {
        self.base.as_deref().map(DeviceBase::info)
    }
    /// Synchronizes pending operations.
    ///
    /// The returned future will resolve when all previous transfer and compute operations are complete.
    ///
    /// NOOP on the host.
    ///
    /// **Errors**
    /// - The device panicked or disconnected.
    pub async fn sync(&self) -> Result<()> {
        if let Some(base) = self.base.as_ref() {
            base.sync().await?;
        }
        Ok(())
    }
    #[cfg(all(test, feature = "device_tests"))]
    pub(crate) async fn acquire(&self) -> Option<SemaphoreGuard<'static>> {
        if self.base.is_some() {
            Some(TEST_DEVICE_SEMAPHORE.acquire().await)
        } else {
            None
        }
    }
}

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

impl From<Arc<DeviceBase>> for Device {
    fn from(base: Arc<DeviceBase>) -> Self {
        Self { base: Some(base) }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "device_tests")]
    use super::*;

    #[cfg(feature = "device_tests")]
    #[test]
    fn device_new() -> Result<()> {
        Device::new()?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[test]
    fn device_builder_iter_build() -> Result<()> {
        for builder in Device::builder_iter() {
            builder.build()?;
        }
        Ok(())
    }
}
