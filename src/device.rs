//! # device
//! This is the compute backend of **autograph**. It has the following parts:
//! - [`Device`](crate::device::Device) represents either the host or a compute device like a gpu (or potentially cpu shader engine).
//! - [`Buffer`](crate::device::buffer::Buffer) is like a Vec but potentially on the device.
//! - [`Module`](crate::device::shader::Module) compute shader modules.
//!
//! # Compute Example
//! This example shows the basics of creating buffers, executing compute, and reading back the results.
//!```no_run
//! use autograph::{
//!     result::Result,
//!     device::{Device, buffer::{Buffer, Slice}, shader::Module}
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // The spirv source can be created at runtime and imported via include_bytes! or compiled
//!     // at runtime (JIT).
//!     let spirv: Vec<u8> = todo!();
//!     // The module stores the spirv and does reflection on it to extract all of the entry
//!     // functions and their arguments. Module can be serialized and deserialized with serde so
//!     // it can be created at compile time and loaded at runtime as well.
//!     let module = Module::from_spirv(spirv)?;
//!     // Create a device.
//!     let device = Device::new()?;
//!     // Construct a Buffer from a vec and transfer it to the device.
//!     let a = Buffer::from(vec![1, 2, 3, 4]).into_device(device.clone()).await?;
//!     // Slice can be created from a &[T] and transfered into a device buffer.
//!     let b = Slice::from([1, 2, 3, 4].as_ref()).into_device(device).await?;
//!     // Allocate the result on the device. This is unsafe because it is not initialized.
//!     let mut y = unsafe { Buffer::<u32>::alloc(device, a.len())? };
//!     let n = y.len() as u32;
//!     // Enqueue the compute pass
//!     module
//!         .compute_pass("add")?
//!         .slice(a.as_slice())?
//!         .slice(b.as_slice())?
//!         .slice_mut(y.as_slice_mut())?
//!         .push(n)?
//!         .submit([n, 1, 1])?;
//!     // Read the data back.
//!     let output = y.read().await?;
//!     println!("{:?}", output.as_slice());
//!     Ok(())
//! }
//!```

use crate::result::Result;
use anyhow::anyhow;
use derive_more::Display;
use hibitset::{AtomicBitSet, BitSet};
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::{
    fmt::{self, Debug},
    mem::size_of,
    sync::Arc,
};

mod engine;
use engine::{builders::EngineBuilder, Engine, ReadGuard, ReadGuardFuture, MAX_ALLOCATION};

/// Buffers.
pub mod buffer;

/// Compute shaders.
pub mod shader;
use shader::{EntryId, Module, ModuleId};

const MAX_DEVICE_ID: u32 =
    (size_of::<usize>() * size_of::<usize>() * size_of::<usize>() * size_of::<usize>()) as u32;

lazy_static! {
    static ref DEVICE_IDS: Mutex<BitSet> = Mutex::default();
}

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
    vendor: usize,
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
                _m: PhantomData::default(),
            })
        }
        /// Adds a slice as an argument to the shader at the next binding.
        ///
        /// **Errors**
        /// - The slice is on the host.
        /// - The slice is not on the same device as previous arguments.
        /// - The slice was declared mutable (not readonly).
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
        /// - The slice was declared immutable (readonly).
        /// - There are no more arguments to be bound.
        ///
        /// # Note
        /// - Does not check the numerical type of the buffer.
        /// - No special behavior for `write_only`.
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
            let declared_mutable = *self.descriptor.buffers.get(self.args.len()).unwrap();
            if mutable != declared_mutable {
                bail!(
                    "Provided {} at binding {}, but it is declared {} in {:?} entry {:?}!",
                    if mutable { "mutable slice" } else { "slice" },
                    self.args.len(),
                    if declared_mutable {
                        "mutable (not readonly)"
                    } else {
                        "immutable (readonly)"
                    },
                    self.module,
                    &self.entry,
                )
            }
            self.check_args_size(self.args.len() + 1, false)?;
            self.args.push(ComputePassArg { buffer, mutable });
            Ok(ComputePassBuilder {
                device: self.device,
                module: self.module,
                entry: self.entry,
                descriptor: self.descriptor,
                args: self.args,
                push_constants: self.push_constants,
                specialization: self.specialization,
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
            self.check_push_constant_size(push_constant_size, false)?;
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
        fn check_specialization_size(&self, size: usize, finished: bool) -> Result<()> {
            let declared_size = self.descriptor.specialization_size();
            if size > declared_size || (finished && size != declared_size) {
                Err(anyhow!(
                    "Provided specialization size {} B does not match declared size {} B in {:?} entry {:?}!",
                    size,
                    declared_size,
                    self.module,
                    self.entry,
                ))
            } else {
                Ok(())
            }
        }
        /// Submits the compute pass to the device with the given `global_size`.
        ///
        /// Does not block, the implementation submits work to the driver asynchronously.
        ///
        /// **Errors**
        /// - The number of arguments does not match the number declared in the module.
        /// - The total size in bytes of all push constants does not match that declared in the module.
        /// - The total size in bytes of all specialization constants does not match that declared in the module.
        /// - The device panicked or disconnected.
        /// - The device could not compile the module (compiles on first use).
        ///
        /// # Note
        /// Shaders are lazily compiled and executed asynchronously, any errors encountered that can not be handled internally will shutdown the internal device thread. This ensures that results are not invalidated by a shader that did not run. The implementation does not track dependencies beyond ensuring correct execution order, and cannot single out just that buffer and fail any dependencies of it. Any subsequent operations will return an error. Be careful with JIT or specialization constants as they will trigger compilation on first use, which may fail or produce invalid binary. Generally it's best to "warm-up" all needed functions first before operating on actual data, to eliminate the startup cost and reduce potential for failures.
        pub fn submit(mut self, global_size: [u32; 3]) -> Result<()> {
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
            self.check_specialization_size(self.specialization.len(), true)?;
            let work_groups = work_groups(global_size, self.descriptor.local_size);
            let compute_pass = ComputePass {
                module: self.module.id,
                entry: self.descriptor.id,
                work_groups,
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
        /// Returns a ['ComputePassBuilder`] used to setup and submit a compute shader.
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
    entry: EntryId,
    work_groups: [u32; 3],
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
        if !self.modules.add_atomic(module.id.0) {
            self.engine.module(module)?;
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
        let mut builders: Vec<_> = Self::builder_iter().collect();
        builders.sort_by_key(|b| b.info().device_type());
        builders
            .first()
            .ok_or_else(|| anyhow!("No device!"))?
            .build()
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
    ///     .filter(|b| b.info().api() == Api::Vulkan)
    ///     .filter(|b| b.info().device_type() == DeviceType::DiscreteGpu)
    ///     .filter(|b| b.info().memory() >= 4_000_000_000)
    ///     .next()
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
    pub fn info(&self) -> Option<&DeviceInfo> {
        self.base.as_deref().map(DeviceBase::info)
    }
    /// Synchronizes pending operations.
    ///
    /// The returned future will resolve when all previous transfer and compute operations are complete.
    pub async fn sync(&self) -> Result<()> {
        if let Some(base) = self.base.as_ref() {
            base.sync().await?;
        }
        Ok(())
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
