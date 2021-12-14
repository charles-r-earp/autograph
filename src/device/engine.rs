#[cfg(feature = "profile")]
use super::profiler::{ComputePassMetrics, Profiler};
use super::{
    shader::{EntryDescriptor, EntryId, Module, ModuleId, SPECIALIZATION_SIZE},
    Api, BufferHandle, BufferId, ComputePass, DeviceError, DeviceId, DeviceResult, WriteOnly,
};
use crate::{
    result::Result,
    util::{type_eq, UnwrapUnchecked as _},
};
use anyhow::{anyhow, bail};
use crossbeam_channel::{unbounded as unbounded_channel, Receiver, Sender};
use crossbeam_utils::atomic::AtomicCell;
#[cfg(feature = "profile")]
use gfx_hal::query::{
    CreationError as QueryCreationError, Query, ResultFlags as QueryResultFlags, Type as QueryType,
};
use gfx_hal::{
    adapter::{Adapter, DeviceType, MemoryProperties, PhysicalDevice},
    buffer::{CreationError as BufferCreationError, State, SubRange, Usage},
    command::{BufferCopy, CommandBuffer, CommandBufferFlags, Level as CommandLevel},
    device::{
        AllocationError, BindError, CreationError as DeviceCreationError, Device, DeviceLost,
        MapError, OutOfMemory,
    },
    memory::{Barrier, Dependencies, HeapFlags, Properties, Segment, SparseFlags},
    pool::{CommandPool, CommandPoolCreateFlags},
    prelude::DescriptorPool,
    pso::{
        BufferDescriptorFormat, BufferDescriptorType, ComputePipelineDesc, CreationError,
        Descriptor, DescriptorPoolCreateFlags, DescriptorRangeDesc, DescriptorSetLayoutBinding,
        DescriptorSetWrite, DescriptorType, EntryPoint, PipelineStage, ShaderStageFlags,
        Specialization, /* SpecializationConstant, */
    },
    queue::{Queue as CommandQueue, QueueFamily, QueueFamilyId, QueueType},
    Backend, Features, Instance, MemoryTypeId,
};
use hibitset::BitSet;
use once_cell::sync::Lazy;
use parking_lot::{
    MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard,
    RwLockWriteGuard,
};
#[cfg(windows)]
use smol::lock::Semaphore;
#[cfg(feature = "profile")]
use std::mem::size_of;
use std::{
    collections::{HashMap, VecDeque},
    convert::TryFrom,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    iter::{empty, once, repeat},
    mem::{take, transmute, ManuallyDrop, MaybeUninit},
    ops::Deref,
    panic::RefUnwindSafe,
    panic::{catch_unwind, UnwindSafe},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
        Arc, Weak,
    },
    time::Duration,
};
use tinyvec::ArrayVec;

#[cfg(any(
    all(unix, not(any(target_os = "ios", target_os = "macos"))),
    feature = "gfx_backend_vulkan",
    windows,
))]
use gfx_backend_vulkan::Backend as Vulkan;

#[cfg(any(target_os = "ios", target_os = "macos"))]
use gfx_backend_metal::Backend as Metal;

#[cfg(windows)]
use gfx_backend_dx12::Backend as DX12;

#[cfg(any(
    all(unix, not(any(target_os = "ios", target_os = "macos"))),
    feature = "gfx_backend_vulkan",
    windows
))]
type VulkanInstance = <Vulkan as Backend>::Instance;

#[cfg(any(target_os = "ios", target_os = "macos"))]
type MetalInstance = <Metal as Backend>::Instance;

#[cfg(windows)]
type DX12Instance = <DX12 as Backend>::Instance;

const CHUNK_SIZE: u64 = 256_000_000;
const BLOCK_SIZE: u32 = 256;
const BLOCKS: u32 = 1_000_000;
pub(super) const MAX_ALLOCATION: usize = CHUNK_SIZE as usize;

#[cfg(any(
    all(unix, not(any(target_os = "ios", target_os = "macos"))),
    feature = "gfx_backend_vulkan",
    windows
))]
static VULKAN_INSTANCE: Lazy<Mutex<Weak<VulkanInstance>>> = Lazy::new(Mutex::default);

#[cfg(any(target_os = "ios", target_os = "macos"))]
static METAL_INSTANCE: Lazy<Mutex<Weak<MetalInstance>>> = Lazy::new(Mutex::default);

#[cfg(windows)]
static DX12_INSTANCE: Lazy<Mutex<Weak<DX12Instance>>> = Lazy::new(Mutex::default);

fn create_instance<B: Backend>() -> Option<B::Instance> {
    B::Instance::create("autograph", 1).ok()
}

pub(super) mod builders {
    use super::*;

    #[derive(Debug, Clone)]
    pub(in super::super) struct EngineBuilder {
        dyn_engine_builder: DynEngineBuilder,
    }

    impl From<DynEngineBuilder> for EngineBuilder {
        fn from(dyn_engine_builder: DynEngineBuilder) -> Self {
            Self { dyn_engine_builder }
        }
    }

    impl EngineBuilder {
        pub(super) fn iter() -> impl Iterator<Item = Self> {
            Self::vulkan_iter()
                .chain(Self::metal_iter())
                .chain(Self::dx12_iter())
        }
        #[allow(unused)]
        #[cfg(any(
            all(unix, not(any(target_os = "ios", target_os = "macos"))),
            feature = "gfx_backend_vulkan",
            windows
        ))]
        fn vulkan_iter() -> impl Iterator<Item = Self> {
            EngineBuilderBase::iter(&VULKAN_INSTANCE)
                .map(|x| Self::from(DynEngineBuilder::Vulkan(x)))
        }
        #[allow(unused)]
        #[cfg(not(any(
            all(unix, not(any(target_os = "ios", target_os = "macos"))),
            feature = "gfx_backend_vulkan",
            windows
        )))]
        fn vulkan_iter() -> impl Iterator<Item = Self> {
            empty()
        }
        #[allow(unused)]
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        fn metal_iter() -> impl Iterator<Item = Self> {
            EngineBuilderBase::iter(&METAL_INSTANCE).map(|x| Self::from(DynEngineBuilder::Metal(x)))
        }
        #[allow(unused)]
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        fn metal_iter() -> impl Iterator<Item = Self> {
            empty()
        }
        #[allow(unused)]
        #[cfg(windows)]
        fn dx12_iter() -> impl Iterator<Item = Self> {
            EngineBuilderBase::iter(&DX12_INSTANCE).map(|x| Self::from(DynEngineBuilder::DX12(x)))
        }
        #[allow(unused)]
        #[cfg(not(windows))]
        fn dx12_iter() -> impl Iterator<Item = Self> {
            empty()
        }
        pub(in crate::device) fn api(&self) -> Api {
            self.dyn_engine_builder.api()
        }
        pub(in crate::device) fn build(&self) -> DeviceResult<Engine> {
            Ok(Engine {
                dyn_engine: self.dyn_engine_builder.build()?,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub(super) enum DynEngineBuilder {
        #[cfg(any(
            all(unix, not(any(target_os = "ios", target_os = "macos"))),
            feature = "gfx_backend_vulkan",
            windows
        ))]
        Vulkan(EngineBuilderBase<Vulkan>),
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        Metal(EngineBuilderBase<Metal>),
        #[cfg(windows)]
        DX12(EngineBuilderBase<DX12>),
    }

    impl DynEngineBuilder {
        fn api(&self) -> Api {
            match self {
                #[cfg(any(
                    all(unix, not(any(target_os = "ios", target_os = "macos"))),
                    feature = "gfx_backend_vulkan",
                    windows
                ))]
                Self::Vulkan(_) => Api::Vulkan,
                #[cfg(any(target_os = "ios", target_os = "macos"))]
                Self::Metal(_) => Api::Metal,
                #[cfg(windows)]
                Self::DX12(_) => Api::DX12,
            }
        }
        fn build(&self) -> DeviceResult<DynEngine> {
            match self {
                #[cfg(any(
                    all(unix, not(any(target_os = "ios", target_os = "macos"))),
                    feature = "gfx_backend_vulkan",
                    windows
                ))]
                Self::Vulkan(base) => Ok(DynEngine::Vulkan(EngineBase::build(base)?)),
                #[cfg(any(target_os = "ios", target_os = "macos"))]
                Self::Metal(base) => Ok(DynEngine::Metal(EngineBase::build(base)?)),
                #[cfg(windows)]
                Self::DX12(base) => Ok(DynEngine::DX12(EngineBase::build(base)?)),
            }
        }
    }

    #[derive(Debug)]
    pub(super) struct EngineBuilderBase<B: Backend> {
        pub(super) adapter: Arc<Adapter<B>>,
        pub(super) instance: Arc<B::Instance>,
        pub(super) allocator_config: AllocatorConfig,
        pub(super) id: Option<DeviceId>,
    }

    impl<B: Backend> Clone for EngineBuilderBase<B> {
        fn clone(&self) -> Self {
            Self {
                adapter: self.adapter.clone(),
                instance: self.instance.clone(),
                allocator_config: self.allocator_config,
                id: self.id,
            }
        }
    }

    impl<B: Backend> EngineBuilderBase<B> {
        fn new(instance: Arc<B::Instance>, adapter: Adapter<B>) -> Self {
            let memory_properties = adapter.physical_device.memory_properties();
            let allocator_config = AllocatorConfig::new(&memory_properties);
            Self {
                instance,
                adapter: Arc::new(adapter),
                allocator_config,
                id: None,
            }
        }
        pub(super) fn iter(instance: &Mutex<Weak<B::Instance>>) -> impl Iterator<Item = Self> {
            let instance = {
                let mut guard = instance.lock();
                if let instance @ Some(_) = Weak::upgrade(&guard) {
                    instance
                } else if let Some(instance) = create_instance::<B>() {
                    let instance = Arc::new(instance);
                    *guard = Arc::downgrade(&instance);
                    Some(instance)
                } else {
                    None
                }
            };
            instance.into_iter().flat_map(|instance| {
                instance
                    .enumerate_adapters()
                    .into_iter()
                    .map(move |adapter| Self::new(instance.clone(), adapter))
            })
        }
    }

    macro_rules! engine_builder_methods {
        ($($vis:vis fn $fn:ident($(&mut $mut_self:ident)? $(&ref $ref_self:ident)? $(, $arg:ident : $arg_ty:ty)*) $(-> $ret:ty)? $body:block)*) => (
            impl EngineBuilder {
                $(
                    $vis fn $fn($(&mut $mut_self)? $(&$ref_self)? $(, $arg : $arg_ty)*) $(-> $ret)? {
                        $($mut_self)? $($ref_self)? .dyn_engine_builder.$fn($($arg),*)
                    }
                )*
            }
            impl DynEngineBuilder {
                $(
                    fn $fn($(&mut $mut_self)? $(&$ref_self)? $(, $arg : $arg_ty)*) $(-> $ret)? {
                        match  $($mut_self)?  $($ref_self)? {
                            #[cfg(any(all(unix, not(any(target_os = "ios", target_os = "macos"))), feature="gfx_backend_vulkan", windows))]
                            Self::Vulkan(this) => this.$fn($($arg),*),
                            #[cfg(any(target_os = "ios", target_os = "macos"))]
                            Self::Metal(this) => this.$fn($($arg),*),
                            #[cfg(windows)]
                            Self::DX12(this) => this.$fn($($arg),*),
                        }
                    }
                )*
            }
            impl<B: Backend> EngineBuilderBase<B> {
                $(
                    fn $fn($(&mut $mut_self)? $(&$ref_self)? $(, $arg : $arg_ty)*) $(-> $ret)? $body
                )*
            }
        )
    }

    engine_builder_methods! {
        pub(in crate::device) fn name(&ref self) -> &str {
            self.adapter.info.name.as_str()
        }
        pub(in crate::device) fn device(&ref self) -> usize {
            self.adapter.info.device
        }
        pub(in crate::device) fn vendor(&ref self) -> usize {
            self.adapter.info.vendor
        }
        pub(in crate::device) fn device_type(&ref self) -> crate::device::DeviceType {
            use crate::device::DeviceType::*;
            match &self.adapter.info.device_type {
                DeviceType::DiscreteGpu => DiscreteGpu,
                DeviceType::IntegratedGpu => IntegratedGpu,
                DeviceType::Cpu => Cpu,
                DeviceType::VirtualGpu => VirtualGpu,
                DeviceType::Other => Other,
            }
        }
        pub(in crate::device) fn memory(&ref self) -> u64 {
            self.allocator_config.storage_memory()
        }
        pub(in crate::device) fn set_device_id(&mut self, id: DeviceId) {
            self.id.replace(id);
        }
    }
}
use builders::{EngineBuilder, EngineBuilderBase};

#[derive(Debug)]
pub(super) struct Engine {
    dyn_engine: DynEngine,
}

impl Engine {
    pub(super) fn builder_iter() -> impl Iterator<Item = EngineBuilder> {
        EngineBuilder::iter()
    }
    pub(super) async fn transfer(
        &self,
        buffer: BufferHandle,
        read_guard_fut: ReadGuardFuture,
    ) -> DeviceResult<()> {
        self.transfer_impl(buffer, read_guard_fut.clone())?;
        read_guard_fut.submit().await
    }
}

#[derive(Debug)]
enum DynEngine {
    #[cfg(any(
        all(unix, not(any(target_os = "ios", target_os = "macos"))),
        feature = "gfx_backend_vulkan",
        windows
    ))]
    Vulkan(EngineBase<Vulkan>),
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    Metal(EngineBase<Metal>),
    #[cfg(windows)]
    DX12(EngineBase<DX12>),
}

#[derive(Debug)]
struct EngineBase<B: Backend> {
    context: Arc<Context<B>>,
    sender: Sender<Op<B>>,
    done: Arc<AtomicBool>,
    result: Arc<AtomicCell<DeviceResult<()>>>,
    exited: Arc<AtomicBool>,
}

impl<B: Backend> EngineBase<B> {
    fn build(builder: &EngineBuilderBase<B>) -> DeviceResult<Self> {
        let adapter = builder.adapter.clone();
        let instance = builder.instance.clone();
        let compute_family = dbg!(adapter
            .queue_families
            .iter()
            .filter(|f| f.queue_type() == QueueType::Compute)
            .chain(
                adapter
                    .queue_families
                    .iter()
                    .filter(|f| f.queue_type().supports_compute()),
            )
            .next()
            .ok_or(DeviceError::DeviceUnsupported))?;
        let mut gpu = unsafe {
            dbg!(adapter
                .physical_device
                .open(&[(compute_family, &[1.])], Features::empty()))?
        };
        let device = gpu.device;
        let compute_queue = gpu.queue_groups[0].queues.pop().unwrap();
        let compute_id = compute_family.id();
        let allocator = dbg!(Allocator::new(&device, &builder.allocator_config))?;
        // TODO probably convert to using anyhow::Error instead of DeviceError.
        #[cfg(feature = "profile")]
        let profiler = Profiler::get()
            .transpose()
            .map_err(|_| DeviceError::ProfileSummaryError)?;
        dbg!("before context");
        #[cfg(feature = "profile")]
        let context = Arc::new(Context::new(device, allocator, adapter, instance, profiler));
        #[cfg(not(feature = "profile"))]
        let context = Arc::new(Context::new(device, allocator, adapter, instance));
        dbg!("after context");
        let (sender, receiver) = unbounded_channel();
        dbg!("before queue");
        let queue = Queue::new(receiver, context.clone(), compute_queue, compute_id)?;
        dbg!("after queue");
        let done = Arc::new(AtomicBool::default());
        let result = Arc::new(AtomicCell::new(Ok(())));
        let exited = Arc::new(AtomicBool::default());
        dbg!("before launch");
        queue.launch(builder.id, done.clone(), result.clone(), exited.clone());
        dbg!("after launch");
        Ok(Self {
            context,
            sender,
            done,
            result,
            exited,
        })
    }
    fn context(&self) -> &Arc<Context<B>> {
        &self.context
    }
    fn device(&self) -> &B::Device {
        &self.context.device
    }
    fn allocator(&self) -> &Allocator<B> {
        &self.context.allocator
    }
    fn send_op(&self, op: Op<B>) -> DeviceResult<()> {
        self.sender.send(op).or_else(|_| self.result.load())
    }
}

impl<B: Backend> Drop for EngineBase<B> {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        while !self.exited.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

macro_rules! engine_methods {
    ($($vis:vis fn $fn:ident $(<$($gen:tt),+>)? (& $($a:lifetime)? $self:ident $(, $arg:ident : $arg_ty:ty)*) $(-> $ret:ty)? $(where $($gen2:ident : $bound:tt),+)? $body:block)*) => (
        impl Engine {
            $(
                $vis fn $fn $(<$($gen),+>)? (& $($a)? $self $(, $arg : $arg_ty)*) $(-> $ret)? $(where $($gen2 : $bound),+)? {
                    $self.dyn_engine.$fn($($arg),*)
                }
            )*
        }
        impl DynEngine {
            $(
                fn $fn $(<$($gen),+>)? (& $($a)? $self $(, $arg : $arg_ty)*) $(-> $ret)? $(where $($gen2 : $bound),+)? {
                    match $self {
                        #[cfg(any(all(unix, not(any(target_os = "ios", target_os = "macos"))), feature="gfx_backend_vulkan", windows))]
                        Self::Vulkan(this) => this.$fn($($arg),*),
                        #[cfg(any(target_os = "ios", target_os = "macos"))]
                        Self::Metal(this) => this.$fn($($arg),*),
                        #[cfg(windows)]
                        Self::DX12(this) => this.$fn($($arg),*),
                    }
                }
            )*
        }
        impl<B: Backend> EngineBase<B> {
            $(
                fn $fn $(<$($gen),+>)? (& $($a)? $self $(, $arg : $arg_ty)*) $(-> $ret)? $(where $($gen2 : $bound),+)? $body
            )*
        }
    )
}

trait BufferIdExt: Sized {
    fn pack(chunk: ChunkId, range: BlockRange) -> Self;
    fn unpack(self) -> (ChunkId, BlockRange);
}

impl BufferIdExt for BufferId {
    fn pack(chunk: ChunkId, range: BlockRange) -> Self {
        let block = range.block.0;
        let len = range.len.0;
        let block_bytes = block.to_ne_bytes();
        let len_bytes = len.to_ne_bytes();
        BufferId(u64::from_ne_bytes([
            chunk.0,
            block_bytes[0],
            block_bytes[1],
            block_bytes[2],
            len_bytes[0],
            len_bytes[1],
            len_bytes[2],
            0,
        ]))
    }
    fn unpack(self) -> (ChunkId, BlockRange) {
        let bytes = self.0.to_ne_bytes();
        let chunk = ChunkId(bytes[0]);
        let block = BlockId(u32::from_ne_bytes([bytes[1], bytes[2], bytes[3], 0]));
        let len = BlockLen(u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], 0]));
        let range = BlockRange { block, len };
        (chunk, range)
    }
}

engine_methods! {
    pub(super) fn alloc(&self, len: u32) -> DeviceResult<BufferId> {
        self.result.load()?;
        let id = self.allocator().alloc(self.device(), len)?;
        Ok(id)
    }
    pub(super) fn dealloc(&self, id: BufferId) {
        self.allocator().dealloc(id)
    }
    pub(super) fn try_write<'a, T, E>(&'a self, buffer: BufferHandle, f: impl FnOnce(WriteOnly<[u8]>)
        -> Result<T, E> + 'a) -> DeviceResult<Result<T, E>>
        where T: 'a, E: 'a {
        let (storage_slice, mut write_guard_base) = self.allocator().write(self.context(), buffer.id, buffer.offset, buffer.len)?;
        let t = match f(write_guard_base.write_only()) {
            Ok(t) => t,
            Err(e) => {
                return Ok(Err(e));
            }
        };
        let op = Op::Write {
            src: write_guard_base.into_write_slice(),
            dst: storage_slice,
            read_guard_fut: None,
        };
        self.send_op(op)?;
        Ok(Ok(t))
    }
    fn transfer_impl<'a>(&'a self, buffer: BufferHandle, read_guard_fut: ReadGuardFuture) -> DeviceResult<()> {
        let (storage_slice, write_guard_base) = self.allocator().write(self.context(), buffer.id, buffer.offset, buffer.len)?;
        let op = Op::Write {
            src: write_guard_base.into_write_slice(),
            dst: storage_slice,
            read_guard_fut: Some(read_guard_fut),
        };
        self.send_op(op)?;
        Ok(())
    }
    pub(super) fn read(&self, buffer: BufferHandle) -> DeviceResult<ReadGuardFuture> {
        let (storage_slice, read_fut) = self.allocator().read(self.context(), &self.result, buffer.id, buffer.offset, buffer.len)?;
        let read_slice = read_fut.read_slice();
        let op = Op::Read {
            src: storage_slice,
            dst: read_slice,
        };
        self.send_op(op)?;
        Ok(read_fut.read_guard_future())
    }
    pub(super) fn copy(&self, src: BufferHandle, dst: BufferHandle) -> DeviceResult<()> {
        let src = self.allocator().storage_slice(src.id, src.offset, src.len);
        let dst = self.allocator().storage_slice(dst.id, dst.offset, dst.len);
        let op = Op::Copy {
            src,
            dst,
        };
        self.send_op(op)?;
        Ok(())
    }
    pub(super) fn module(&self, module: &Module) -> DeviceResult<()> {
        let descriptor = &module.descriptor;
        let entries = descriptor.entries.iter()
            .map(|(k, v)| (k.clone(), v.clone()));
        let shader_module = ShaderModule::new(self.context.clone(), module)?
            .with_entries(entries)?;
        let op = Op::Module {
            id: module.id,
            module: shader_module,
        };
        self.send_op(op)
    }
    pub(super) fn compute(&self, compute_pass: ComputePass) -> DeviceResult<()> {
        self.result.load()?;
        let args: Vec<_> = compute_pass.args.into_iter()
            .map(|arg| {
                let buffer = arg.buffer;
                let slice = self.allocator().storage_slice(buffer.id, buffer.offset, buffer.len);
                ComputeArg {
                    slice,
                    mutable: arg.mutable,
                }
            }).collect();
        let op = Op::Compute {
            module: compute_pass.module,
            module_name: compute_pass.module_name,
            entry: compute_pass.entry,
            entry_name: compute_pass.entry_name,
            work_groups: compute_pass.work_groups,
            local_size: compute_pass.local_size,
            args,
            push_constants: compute_pass.push_constants,
            specialization: compute_pass.specialization,
        };
        self.send_op(op)
    }
    pub(super) fn sync(&self) -> DeviceResult<SyncFuture> {
        self.result.load()?;
        let finished = Arc::new(AtomicBool::default());
        let op = Op::Sync {
            finished: finished.clone(),
        };
        self.send_op(op)?;
        Ok(SyncFuture {
            finished,
            result: self.result.clone()
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ChunkId(u8);

#[derive(Clone, Copy, Debug)]
struct BlockId(u32);

impl BlockId {
    fn as_offset_bytes(self) -> u32 {
        self.0 * BLOCK_SIZE
    }
}

#[derive(Clone, Copy, Debug)]
struct BlockLen(u32);

impl BlockLen {
    fn as_bytes(&self) -> u32 {
        self.0 * BLOCK_SIZE
    }
}

impl TryFrom<u32> for BlockLen {
    type Error = DeviceError;
    fn try_from(len: u32) -> DeviceResult<BlockLen> {
        let n = if len % BLOCK_SIZE == 0 {
            len / BLOCK_SIZE
        } else {
            len / BLOCK_SIZE + 1
        };
        if n > BLOCKS {
            Err(DeviceError::AllocationTooLarge)
        } else {
            Ok(BlockLen(n))
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BlockRange {
    block: BlockId,
    len: BlockLen,
}

impl BlockRange {
    fn new(block: BlockId, len: BlockLen) -> Self {
        Self { block, len }
    }
}

#[derive(Debug)]
struct ChunkBase<B: Backend> {
    buffer: B::Buffer,
    memory: B::Memory,
}

impl<B: Backend> ChunkBase<B> {
    unsafe fn new(
        device: &B::Device,
        ids: impl Iterator<Item = MemoryTypeId>,
        usage: Usage,
    ) -> DeviceResult<Self> {
        let mut buffer = device.create_buffer(CHUNK_SIZE as u64, usage, SparseFlags::empty())?;
        let size = device.get_buffer_requirements(&buffer).size;
        let mut result = Err(DeviceError::DeviceUnsupported);
        for id in ids {
            result = device.allocate_memory(id, size).map_err(Into::into);
            if result.is_ok() {
                break;
            }
        }
        let memory = match result {
            Ok(memory) => memory,
            Err(e) => {
                device.destroy_buffer(buffer);
                return Err(e);
            }
        };
        match device.bind_buffer_memory(&memory, 0, &mut buffer) {
            Ok(()) => (),
            Err(e) => {
                device.free_memory(memory);
                device.destroy_buffer(buffer);
                return Err(e.into());
            }
        }
        Ok(Self { buffer, memory })
    }
    fn new_storage(
        device: &B::Device,
        ids: impl Iterator<Item = MemoryTypeId>,
    ) -> DeviceResult<Self> {
        let usage = Usage::STORAGE | Usage::TRANSFER_SRC | Usage::TRANSFER_DST;
        unsafe { Self::new(device, ids, usage) }
    }
    fn new_mapping(
        device: &B::Device,
        ids: impl Iterator<Item = MemoryTypeId>,
    ) -> DeviceResult<Self> {
        let usage = Usage::TRANSFER_SRC | Usage::TRANSFER_DST;
        unsafe { Self::new(device, ids, usage) }
    }
    unsafe fn free(self, device: &B::Device) {
        device.destroy_buffer(self.buffer);
        device.free_memory(self.memory);
    }
}

#[derive(Default)]
struct StorageBlocks {
    blocks: BitSet,
}

impl StorageBlocks {
    fn alloc(&mut self, len: BlockLen) -> Option<BlockId> {
        let len = len.0;
        let mut block = 0;
        let mut end = len;
        'outer: while end <= BLOCKS {
            #[allow(clippy::mut_range_bound)]
            for i in block..end {
                if self.blocks.contains(i) {
                    block = i + 1;
                    end = block + len;
                    continue 'outer;
                }
            }
            self.blocks.extend(block..end);
            return Some(BlockId(block));
        }
        None
    }
    fn dealloc(&mut self, range: BlockRange) {
        for b in range.block.0..range.block.0 + range.len.0 {
            debug_assert!(self.blocks.contains(b), "{:?} {} \n{:?}", range, b, self);
            self.blocks.remove(b);
        }
    }
}

impl Debug for StorageBlocks {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = (0..BLOCKS)
            .into_iter()
            .rev()
            .find_map(|i| {
                if self.blocks.contains(i) {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let s = (0..n)
            .into_iter()
            .map(|i| if self.blocks.contains(i) { '1' } else { '0' })
            .collect::<String>();
        f.debug_struct("StorageBlocks").field("blocks", &s).finish()
    }
}

#[derive(Debug)]
struct StorageChunk<B: Backend> {
    base: RwLock<Option<ChunkBase<B>>>,
    blocks: Mutex<StorageBlocks>,
}

impl<B: Backend> StorageChunk<B> {
    fn try_alloc(&self, len: BlockLen) -> Option<BlockId> {
        if self.base.try_read()?.is_some() {
            self.blocks.try_lock()?.alloc(len)
        } else {
            None
        }
    }
    fn alloc(
        &self,
        device: &B::Device,
        ids: impl Iterator<Item = MemoryTypeId>,
        len: BlockLen,
    ) -> DeviceResult<Option<BlockId>> {
        let base = self.base.upgradable_read();
        if base.is_none() {
            let mut base = RwLockUpgradableReadGuard::upgrade(base);
            if base.is_none() {
                base.replace(ChunkBase::new_storage(device, ids)?);
            }
        }
        Ok(self.blocks.lock().alloc(len))
    }
    fn dealloc(&self, range: BlockRange) {
        self.blocks.lock().dealloc(range);
    }
    fn slice(&self, range: BlockRange, offset: u32, len: u32) -> Result<StorageSlice<B>> {
        let base = self
            .base
            .try_read()
            .ok_or_else(|| anyhow!("StorageChunk is not initialized!"))?;
        let buffer = &base
            .as_ref()
            .ok_or_else(|| anyhow!("StorageChunk is not initialized!"))?
            .buffer;
        let buffer: &'static B::Buffer = unsafe { transmute(buffer) };
        if len > range.len.as_bytes() {
            bail!("Slice len out of range!");
        }
        if offset > len {
            bail!("Slice offset {} out of range {}!", offset, len);
        }
        let offset = offset + range.block.as_offset_bytes();
        Ok(StorageSlice {
            buffer,
            offset,
            len,
        })
    }
    unsafe fn slice_unchecked(&self, range: BlockRange, offset: u32, len: u32) -> StorageSlice<B> {
        if cfg!(debug_assertions) {
            self.slice(range, offset, len).unwrap();
        }
        let buffer: &B::Buffer = &(*self.base.data_ptr()).as_ref()._unwrap_unchecked().buffer;
        let buffer: &'static B::Buffer = transmute(buffer);
        let offset = offset + range.block.as_offset_bytes();
        StorageSlice {
            buffer,
            offset,
            len,
        }
    }
    unsafe fn free(self, device: &B::Device) {
        if let Some(base) = self.base.into_inner() {
            base.free(device)
        }
    }
}

impl<B: Backend> Default for StorageChunk<B> {
    fn default() -> Self {
        Self {
            base: RwLock::default(),
            blocks: Mutex::default(),
        }
    }
}

#[derive(Debug)]
struct MappingBlocks {
    map: NonNull<u8>,
    offset: AtomicU32,
    submitted: AtomicU32,
    completed: AtomicU32,
    refcount: AtomicUsize,
}

impl MappingBlocks {
    fn new(map: NonNull<u8>) -> Self {
        Self {
            map,
            offset: AtomicU32::default(),
            submitted: AtomicU32::default(),
            completed: AtomicU32::default(),
            refcount: AtomicUsize::default(),
        }
    }
    fn alloc(&self, len: BlockLen) -> Option<BlockId> {
        let len = len.0;
        self.refcount.fetch_add(1, Ordering::SeqCst);
        if self.offset.load(Ordering::SeqCst) + len <= BLOCKS {
            let block = self.offset.fetch_add(len, Ordering::SeqCst);
            if block + len <= BLOCKS {
                return Some(BlockId(block));
            }
        }
        self.drop_guard();
        None
    }
    fn submit(&self, offset: u32) {
        self.submitted.fetch_max(offset, Ordering::SeqCst);
    }
    fn finish(&self, offset: u32) {
        self.completed.fetch_max(offset, Ordering::SeqCst);
    }
    fn drop_guard(&self) {
        let prev = self.refcount.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.submitted.store(0, Ordering::SeqCst);
            self.completed.store(0, Ordering::SeqCst);
            self.offset.store(0, Ordering::SeqCst);
        }
    }
}

unsafe impl Send for MappingBlocks {}
unsafe impl Sync for MappingBlocks {}
impl UnwindSafe for MappingBlocks {}
impl RefUnwindSafe for MappingBlocks {}

#[derive(Debug)]
struct MappingChunkBase<B: Backend> {
    base: ChunkBase<B>,
    blocks: MappingBlocks,
}

impl<B: Backend> MappingChunkBase<B> {
    fn new(device: &B::Device, ids: impl Iterator<Item = MemoryTypeId>) -> DeviceResult<Self> {
        let mut base = ChunkBase::new_mapping(device, ids)?;
        let map = unsafe {
            device.map_memory(
                &mut base.memory,
                Segment {
                    offset: 0,
                    size: None,
                },
            )?
        };
        let map = NonNull::new(map).ok_or(DeviceError::MappingFailed)?;
        let blocks = MappingBlocks::new(map);
        Ok(Self { base, blocks })
    }
    fn buffer(&self) -> &B::Buffer {
        &self.base.buffer
    }
    fn alloc(&self, len: BlockLen) -> Option<BlockId> {
        self.blocks.alloc(len)
    }
    unsafe fn free(self, device: &B::Device) {
        self.base.free(device);
    }
}

unsafe impl<B: Backend> Send for MappingChunkBase<B> {}
unsafe impl<B: Backend> Sync for MappingChunkBase<B> {}

#[derive(Debug)]
struct MappingChunk<B: Backend> {
    base: RwLock<Option<MappingChunkBase<B>>>,
}

impl<B: Backend> MappingChunk<B> {
    fn get_or_try_init(
        &self,
        device: &B::Device,
        ids: impl Iterator<Item = MemoryTypeId>,
    ) -> DeviceResult<MappedRwLockReadGuard<MappingChunkBase<B>>> {
        let base = self.base.upgradable_read();
        let base = if base.is_some() {
            RwLockUpgradableReadGuard::downgrade(base)
        } else {
            let mut base = RwLockUpgradableReadGuard::upgrade(base);
            if base.is_none() {
                base.replace(MappingChunkBase::new(device, ids)?);
            }
            RwLockWriteGuard::downgrade(base)
        };
        Ok(RwLockReadGuard::map(base, |base| unsafe {
            base.as_ref()._unwrap_unchecked()
        }))
    }
    fn alloc_write(
        &self,
        context: &Arc<Context<B>>,
        ids: impl Iterator<Item = MemoryTypeId>,
        len: u32,
    ) -> DeviceResult<Option<WriteGuardBase<B>>> {
        let chunk = self.get_or_try_init(context.device(), ids)?;
        let chunk: &MappingChunkBase<B> = chunk.deref();
        let chunk: &'static MappingChunkBase<B> = unsafe { transmute(chunk) };
        let block_len = BlockLen::try_from(len)?;
        if let Some(block) = chunk.alloc(block_len) {
            let write_guard =
                WriteGuardBase::new(context.clone(), chunk, block.as_offset_bytes(), len);
            Ok(Some(write_guard))
        } else {
            Ok(None)
        }
    }
    fn alloc_read(
        &self,
        context: &Arc<Context<B>>,
        worker_result: &Arc<AtomicCell<DeviceResult<()>>>,
        ids: impl Iterator<Item = MemoryTypeId>,
        len: u32,
    ) -> DeviceResult<Option<ReadFuture<B>>> {
        let chunk = self.get_or_try_init(context.device(), ids)?;
        let chunk: &MappingChunkBase<B> = chunk.deref();
        let chunk: &'static MappingChunkBase<B> = unsafe { transmute(chunk) };
        let block_len = BlockLen::try_from(len)?;
        if let Some(block) = chunk.alloc(block_len) {
            let read_fut = ReadFuture {
                context: context.clone(),
                worker_result: worker_result.clone(),
                chunk,
                offset: block.as_offset_bytes(),
                len,
            };
            Ok(Some(read_fut))
        } else {
            Ok(None)
        }
    }
    unsafe fn free(self, device: &B::Device) {
        if let Some(base) = self.base.into_inner() {
            base.free(device)
        }
    }
}

impl<B: Backend> Default for MappingChunk<B> {
    fn default() -> Self {
        Self {
            base: RwLock::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct HeapInfo {
    // index: usize,
    size: u64,
    id: MemoryTypeId,
}

impl HeapInfo {
    fn new(
        memory_properties: &MemoryProperties,
        heap_flags: HeapFlags,
        properties: Properties,
    ) -> Option<Self> {
        let mut heaps: Vec<_> = memory_properties
            .memory_heaps
            .iter()
            .enumerate()
            .filter(|(_, heap)| heap.flags == heap_flags)
            .map(|(i, heap)| (i, heap.size))
            .collect();
        heaps.sort_by_key(|(_, size)| *size);
        for (_index, size) in heaps.into_iter().rev() {
            for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
                if memory_type.properties.contains(properties) {
                    return Some(Self {
                        // index,
                        size,
                        id: MemoryTypeId(i),
                    });
                }
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct AllocatorConfig {
    device_heap: Option<HeapInfo>,
    shared_heap: Option<HeapInfo>,
    host_heap: Option<HeapInfo>,
    initial_storage_chunks: usize,
    initial_mapping_chunks: usize,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            device_heap: None,
            shared_heap: None,
            host_heap: None,
            initial_storage_chunks: 4,
            initial_mapping_chunks: 4,
        }
    }
}

impl AllocatorConfig {
    fn device_heap(memory_properties: &MemoryProperties) -> Option<HeapInfo> {
        HeapInfo::new(
            memory_properties,
            HeapFlags::DEVICE_LOCAL,
            Properties::DEVICE_LOCAL,
        )
    }
    fn mapping_properties() -> Properties {
        Properties::CPU_VISIBLE | Properties::COHERENT // | Properties::CPU_CACHED
    }
    fn shared_heap(memory_properties: &MemoryProperties) -> Option<HeapInfo> {
        if let Some(heap) = HeapInfo::new(
            memory_properties,
            HeapFlags::DEVICE_LOCAL,
            Self::mapping_properties(),
        ) {
            if heap.size < 4 * CHUNK_SIZE {
                None
            } else {
                Some(heap)
            }
        } else {
            None
        }
    }
    fn host_heap(memory_properties: &MemoryProperties) -> Option<HeapInfo> {
        HeapInfo::new(
            memory_properties,
            HeapFlags::empty(),
            Self::mapping_properties(),
        )
    }
    fn new(memory_properties: &MemoryProperties) -> Self {
        Self::default().with_memory_properties(memory_properties)
    }
    fn with_memory_properties(mut self, memory_properties: &MemoryProperties) -> Self {
        dbg!(&memory_properties);
        self.device_heap = Self::device_heap(memory_properties);
        self.shared_heap = Self::shared_heap(memory_properties);
        self.host_heap = Self::host_heap(memory_properties);
        dbg!(self)
    }
    fn device_id(&self) -> Option<MemoryTypeId> {
        self.device_heap.as_ref().map(|x| x.id)
    }
    fn shared_id(&self) -> Option<MemoryTypeId> {
        self.shared_heap.as_ref().map(|x| x.id)
    }
    fn host_id(&self) -> Option<MemoryTypeId> {
        self.host_heap.as_ref().map(|x| x.id)
    }
    fn storage_chunks(&self) -> usize {
        let device_chunks = (self.device_heap.map_or(0, |x| x.size) / CHUNK_SIZE as u64) as usize;
        let shared_chunks = (self.shared_heap.map_or(0, |x| x.size) / CHUNK_SIZE as u64) as usize;
        (dbg!(device_chunks) + dbg!(shared_chunks)).min(256)
    }
    fn storage_memory(&self) -> u64 {
        dbg!(dbg!(self.storage_chunks() as u64) * dbg!(CHUNK_SIZE as u64))
    }
    fn mapping_chunks(&self) -> usize {
        let shared_chunks = (self.shared_heap.map_or(0, |x| x.size) / CHUNK_SIZE as u64) as usize;
        let host_chunks = (self.host_heap.map_or(0, |x| x.size) / CHUNK_SIZE as u64) as usize;
        (dbg!(shared_chunks) + dbg!(host_chunks)).min(256)
    }
}

#[derive(Debug)]
struct Allocator<B: Backend> {
    device_id: Option<MemoryTypeId>,
    shared_id: Option<MemoryTypeId>,
    host_id: Option<MemoryTypeId>,
    storage_chunks: Vec<StorageChunk<B>>,
    mapping_chunks: Vec<MappingChunk<B>>,
}

impl<B: Backend> Allocator<B> {
    fn new(_device: &B::Device, config: &AllocatorConfig) -> DeviceResult<Self> {
        let device_id = config.device_id();
        let shared_id = config.shared_id();
        let host_id = config.host_id();
        let storage_chunks = config.storage_chunks();
        let mapping_chunks = config.mapping_chunks();
        if storage_chunks < config.initial_storage_chunks {
            return Err(DeviceError::OutOfDeviceMemory);
        }
        if mapping_chunks < config.initial_mapping_chunks {
            return Err(DeviceError::OutOfHostMemory);
        }
        let storage_chunks: Vec<_> = repeat(())
            .map(|_| StorageChunk::<B>::default())
            .take(config.storage_chunks())
            .collect();
        let mapping_chunks: Vec<_> = repeat(())
            .map(|_| MappingChunk::<B>::default())
            .take(config.mapping_chunks())
            .collect();
        Ok(Self {
            device_id,
            shared_id,
            host_id,
            storage_chunks,
            mapping_chunks,
        })
    }
    fn storage_ids(&self) -> impl Iterator<Item = MemoryTypeId> {
        self.device_id.into_iter().chain(self.shared_id)
    }
    fn mapping_ids(&self) -> impl Iterator<Item = MemoryTypeId> {
        self.shared_id.into_iter().chain(self.host_id)
    }
    /*fn alloc_storage_chunks(&self, device: &B::Device, count: usize) -> DeviceResult<()> {
        let mut allocated = 0;
        for chunk in self.storage_chunks.iter() {
            if allocated >= count {
                break;
            }
            if !chunk.initialized() {
                chunk.init(device, self.storage_ids())?;
                allocated += 1;
            }
        }
        Ok(())
    }
    fn alloc_mapping_chunks(&self, device: &B::Device, count: usize) -> DeviceResult<()> {
        let mut allocated = 0;
        for chunk in self.mapping_chunks.iter() {
            if allocated >= count {
                break;
            }
            if !chunk.initialized() {
                chunk.init(device, self.mapping_ids())?;
                allocated += 1;
            }
        }
        Ok(())
    }*/
    fn alloc(&self, device: &B::Device, len: u32) -> DeviceResult<BufferId> {
        let len = BlockLen::try_from(len)?;
        let (chunk, block) = self.alloc_storage(device, len)?;
        Ok(BufferId::pack(chunk, BlockRange::new(block, len)))
    }
    fn alloc_storage(&self, device: &B::Device, len: BlockLen) -> DeviceResult<(ChunkId, BlockId)> {
        if let Some((c, block)) = self
            .storage_chunks
            .iter()
            .enumerate()
            .find_map(|(c, chunk)| Some((c, chunk.try_alloc(len)?)))
        {
            return Ok((ChunkId(c as u8), block));
        }
        for (c, chunk) in self.storage_chunks.iter().enumerate() {
            if let Some(block) = chunk.alloc(device, self.storage_ids(), len)? {
                return Ok((ChunkId(c as u8), block));
            }
        }
        Err(DeviceError::OutOfDeviceMemory)
    }
    fn write(
        &self,
        context: &Arc<Context<B>>,
        id: BufferId,
        offset: u32,
        len: u32,
    ) -> DeviceResult<(StorageSlice<B>, WriteGuardBase<B>)> {
        let (storage_chunk_id, storage_range) = id.unpack();
        // TODO: check in bounds
        for chunk in self.mapping_chunks.iter() {
            if let Some(write_guard) = chunk.alloc_write(context, self.mapping_ids(), len)? {
                let storage_slice = unsafe {
                    self.storage_chunks
                        .get_unchecked(storage_chunk_id.0 as usize)
                        .slice_unchecked(storage_range, offset, len)
                };
                return Ok((storage_slice, write_guard));
            }
        }
        // TODO: maybe wait for memory to become available?
        // Not necessarily out of host memory
        Err(DeviceError::OutOfHostMemory)
    }
    fn read(
        &self,
        context: &Arc<Context<B>>,
        worker_result: &Arc<AtomicCell<DeviceResult<()>>>,
        id: BufferId,
        offset: u32,
        len: u32,
    ) -> DeviceResult<(StorageSlice<B>, ReadFuture<B>)> {
        let (storage_chunk_id, storage_range) = id.unpack();
        // TODO: check in bounds
        for chunk in self.mapping_chunks.iter() {
            if let Some(read_fut) =
                chunk.alloc_read(context, worker_result, self.mapping_ids(), len)?
            {
                let storage_slice = unsafe {
                    self.storage_chunks
                        .get_unchecked(storage_chunk_id.0 as usize)
                        .slice_unchecked(storage_range, offset, len)
                };
                return Ok((storage_slice, read_fut));
            }
        }
        // TODO: maybe wait for memory to become available?
        // Not necessarily out of host memory
        Err(DeviceError::OutOfHostMemory)
    }
    fn storage_slice(&self, id: BufferId, offset: u32, len: u32) -> StorageSlice<B> {
        let (chunk, range) = id.unpack();
        if cfg!(debug_assertions) {
            return self
                .storage_chunks
                .get(chunk.0 as usize)
                .unwrap()
                .slice(range, offset, len)
                .unwrap();
        }
        unsafe {
            self.storage_chunks
                .get_unchecked(chunk.0 as usize)
                .slice_unchecked(range, offset, len)
        }
    }
    fn dealloc(&self, id: BufferId) {
        let (chunk, range) = id.unpack();
        self.dealloc_storage(chunk, range);
    }
    fn dealloc_storage(&self, chunk: ChunkId, range: BlockRange) {
        let chunk = self.storage_chunks.get(chunk.0 as usize);
        if cfg!(debug_assertions) {
            assert!(chunk.is_some());
        }
        if let Some(chunk) = chunk {
            chunk.dealloc(range);
        }
    }
    unsafe fn free(self, device: &B::Device) {
        for chunk in self.storage_chunks {
            chunk.free(device);
        }
        for chunk in self.mapping_chunks {
            chunk.free(device);
        }
    }
}

// Slices hold a reference to the chunk. This ref is valid because the chunk is immutable, and
// through holding onto the allocator / context.
// Slices must be manually "dropped", ie decrementing the ref count.

#[derive(Debug)]
struct StorageSlice<B: Backend> {
    buffer: &'static B::Buffer,
    offset: u32,
    len: u32,
}

impl<B: Backend> StorageSlice<B> {
    fn buffer(&self) -> &B::Buffer {
        self.buffer
    }
}

#[derive(Default, Clone, Debug)]
struct WriteBarrier(Arc<AtomicBool>);

impl WriteBarrier {
    fn wait(&self) {
        while !self.0.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_nanos(100));
        }
    }
    fn signal(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
struct WriteSlice<B: Backend> {
    chunk: &'static MappingChunkBase<B>,
    barrier: WriteBarrier,
    offset: u32,
    len: u32,
}

impl<B: Backend> WriteSlice<B> {
    fn buffer(&self) -> &B::Buffer {
        self.chunk.buffer()
    }
    unsafe fn write_only(&mut self) -> WriteOnly<[u8]> {
        let blocks = &self.chunk.blocks;
        let slice = std::slice::from_raw_parts_mut(
            blocks.map.as_ptr().offset(self.offset as isize) as *mut u8,
            self.len as usize,
        );
        WriteOnly::new(slice)
    }
}

#[derive(Debug)]
pub(crate) struct WriteGuardBase<B: Backend> {
    _context: Arc<Context<B>>,
    slice: WriteSlice<B>,
    drop: bool,
}

impl<B: Backend> WriteGuardBase<B> {
    fn new(
        context: Arc<Context<B>>,
        chunk: &'static MappingChunkBase<B>,
        offset: u32,
        len: u32,
    ) -> Self {
        Self {
            _context: context,
            slice: WriteSlice {
                chunk,
                barrier: WriteBarrier::default(),
                offset,
                len,
            },
            drop: true,
        }
    }
    fn write_only(&mut self) -> WriteOnly<[u8]> {
        unsafe { self.slice.write_only() }
    }
    fn into_write_slice(mut self) -> WriteSlice<B> {
        let slice = self.slice.clone();
        slice.barrier.signal();
        self.drop = false;
        slice
    }
}

impl<B: Backend> Drop for WriteGuardBase<B> {
    fn drop(&mut self) {
        if self.drop {
            self.slice.barrier.signal();
            self.slice.chunk.blocks.drop_guard();
        }
    }
}

#[derive(Debug)]
struct ReadSlice<B: Backend> {
    chunk: &'static MappingChunkBase<B>,
    offset: u32,
    len: u32,
}

impl<B: Backend> ReadSlice<B> {
    fn buffer(&self) -> &'static B::Buffer {
        self.chunk.buffer()
    }
}

#[derive(Debug)]
struct ReadFuture<B: Backend> {
    context: Arc<Context<B>>,
    worker_result: Arc<AtomicCell<DeviceResult<()>>>,
    chunk: &'static MappingChunkBase<B>,
    offset: u32,
    len: u32,
}

impl<B: Backend> ReadFuture<B> {
    fn read_slice(&self) -> ReadSlice<B> {
        ReadSlice {
            chunk: self.chunk,
            offset: self.offset,
            len: self.len,
        }
    }
    fn read_guard_future(self) -> ReadGuardFuture {
        let fut = ReadGuardFuture {
            context: self.context.clone().into(),
            worker_result: self.worker_result.clone(),
            blocks: &self.chunk.blocks,
            offset: self.offset,
            len: self.len,
        };
        std::mem::forget(self);
        fut
    }
}

impl<B: Backend> Drop for ReadFuture<B> {
    fn drop(&mut self) {
        self.chunk.blocks.drop_guard();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReadGuardFuture {
    context: DynContext,
    worker_result: Arc<AtomicCell<DeviceResult<()>>>,
    blocks: &'static MappingBlocks,
    offset: u32,
    len: u32,
}

impl ReadGuardFuture {
    async fn submit(&self) -> DeviceResult<()> {
        while self.blocks.submitted.load(Ordering::Relaxed) < self.offset + self.len {
            self.worker_result.load()?;
            smol::future::yield_now().await;
        }
        Ok(())
    }
    pub(super) async fn finish(self) -> DeviceResult<ReadGuard> {
        while self.blocks.completed.load(Ordering::Relaxed) < self.offset + self.len {
            self.worker_result.load()?;
            smol::future::yield_now().await;
        }
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.blocks.map.as_ptr().offset(self.offset as isize),
                self.len as usize,
            )
        };
        let read_guard = ReadGuard {
            _context: self.context.clone(),
            blocks: self.blocks,
            slice,
        };
        std::mem::forget(self);
        Ok(read_guard)
    }
}

impl Drop for ReadGuardFuture {
    fn drop(&mut self) {
        self.blocks.drop_guard();
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ReadGuard {
    _context: DynContext,
    blocks: &'static MappingBlocks,
    slice: &'static [u8],
}

impl Drop for ReadGuard {
    fn drop(&mut self) {
        self.blocks.drop_guard();
    }
}

impl Deref for ReadGuard {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.slice
    }
}

#[derive(Debug)]
pub(super) struct SyncFuture {
    finished: Arc<AtomicBool>,
    result: Arc<AtomicCell<DeviceResult<()>>>,
}

impl SyncFuture {
    pub(super) async fn finish(self) -> DeviceResult<()> {
        while !self.finished.load(Ordering::Relaxed) {
            self.result.load()?;
            smol::future::yield_now().await;
        }
        Ok(())
    }
}

struct Queue<B: Backend> {
    receiver: Receiver<Op<B>>,
    queue: B::Queue,
    context: Arc<Context<B>>,
    command_pool: ManuallyDrop<B::CommandPool>,
    ready_frames: Vec<Frame<B>>,
    pending_frames: VecDeque<Frame<B>>,
    modules: HashMap<ModuleId, ShaderModule<B>>,
}

impl<B: Backend> Queue<B> {
    fn new(
        receiver: Receiver<Op<B>>,
        context: Arc<Context<B>>,
        queue: B::Queue,
        id: QueueFamilyId,
    ) -> DeviceResult<Self> {
        let command_pool = unsafe {
            ManuallyDrop::new(
                context
                    .device
                    .create_command_pool(id, CommandPoolCreateFlags::RESET_INDIVIDUAL)?,
            )
        };
        let n_frames = 3;
        let ready_frames = Vec::with_capacity(n_frames);
        let pending_frames = VecDeque::with_capacity(n_frames);
        let timestamp_period_ns = queue.timestamp_period() as u64;
        let mut queue = Self {
            receiver,
            queue,
            context,
            command_pool,
            ready_frames,
            pending_frames,
            modules: HashMap::new(),
        };
        for _ in 0..n_frames {
            let frame = Frame::new(
                &queue.context,
                &mut *queue.command_pool,
                timestamp_period_ns,
            )?;
            queue.ready_frames.push(frame);
        }
        Ok(queue)
    }
    fn launch(
        mut self,
        id: Option<DeviceId>,
        done: Arc<AtomicBool>,
        result: Arc<AtomicCell<DeviceResult<()>>>,
        exited: Arc<AtomicBool>,
    ) {
        let id_string = id.map_or("".into(), |id| format!("({})", id.0));
        let name = format!("autograph::Device{}", id_string);
        std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                dbg!("thread launched");
                let r =
                    catch_unwind(move || self.run(&done)).unwrap_or(Err(DeviceError::DeviceLost));
                result.store(r);
                exited.store(true, Ordering::Relaxed);
                dbg!("thread exit");
            })
            .unwrap();
    }
    fn poll(&mut self) -> DeviceResult<()> {
        let mut finished = None;
        if let Some(frame) = self.pending_frames.front_mut() {
            if frame.poll(&self.context)? {
                finished = self.pending_frames.pop_front();
            }
        }
        if let Some(frame) = self.ready_frames.last_mut() {
            if self.pending_frames.len() <= 1 && frame.ready_to_submit() {
                frame.submit(
                    self.context.device(),
                    self.pending_frames.front(),
                    &mut self.queue,
                )?;
                let frame = self.ready_frames.pop().unwrap();
                self.pending_frames.push_back(frame);
            }
        }
        if let Some(frame) = finished {
            self.ready_frames.push(frame);
        }
        Ok(())
    }
    fn run(&mut self, done: &AtomicBool) -> DeviceResult<()> {
        let mut current_op: Option<Op<B>> = None;
        while !done.load(Ordering::Relaxed) {
            if let Some(frame) = self.ready_frames.last_mut() {
                for op in current_op
                    .take()
                    .into_iter()
                    .chain(self.receiver.try_iter())
                {
                    let op = match op {
                        // Sync only frames that have work or signal immediately.
                        Op::Sync { finished } => {
                            if !frame.ready_to_submit() {
                                if let Some(frame) = self.pending_frames.back_mut() {
                                    frame.syncs.push(finished);
                                } else {
                                    finished.store(true, Ordering::Relaxed);
                                }
                                continue;
                            }
                            Op::Sync { finished }
                        }
                        op => op,
                    };
                    if let Err(op) =
                        frame.try_encode(op, self.context.device(), &mut self.modules)?
                    {
                        current_op.replace(op);
                        break;
                    }
                }
            }
            self.poll()?;
        }
        Ok(())
    }
}

impl<B: Backend> Drop for Queue<B> {
    fn drop(&mut self) {
        self.queue.wait_idle().unwrap();
        for frame in take(&mut self.ready_frames)
            .into_iter()
            .chain(take(&mut self.pending_frames))
        {
            unsafe {
                frame.free(self.context.device());
            }
        }
    }
}

impl<B: Backend> UnwindSafe for Queue<B> {}

#[derive(Debug)]
struct ShaderModule<B: Backend> {
    context: Arc<Context<B>>,
    module: ManuallyDrop<B::ShaderModule>,
    shaders: Vec<Shader<B>>,
    name: Option<String>,
}

impl<B: Backend> ShaderModule<B> {
    fn new(context: Arc<Context<B>>, module: &Module) -> DeviceResult<Self> {
        let name = module.name().map(Into::into);
        // Hack because DX12 doesn't like parallel shader compiliation.
        #[cfg(windows)]
        let s = if type_eq::<B, DX12>() {
            Some(smol::block_on(context.shader_semaphore.acquire()))
        } else {
            None
        };
        let module = unsafe {
            context
                .device()
                .create_shader_module(bytemuck::cast_slice(&module.spirv))
                .unwrap()
        };
        #[cfg(windows)]
        std::mem::drop(s);
        Ok(Self {
            context,
            module: ManuallyDrop::new(module),
            shaders: Vec::new(),
            name,
        })
    }
    fn device(&self) -> &B::Device {
        self.context.device()
    }
    fn with_entries<I>(mut self, entries: I) -> DeviceResult<Self>
    where
        I: ExactSizeIterator<Item = (String, EntryDescriptor)>,
    {
        let mut shaders: Vec<MaybeUninit<Shader<B>>> =
            std::iter::from_fn(|| Some(MaybeUninit::uninit()))
                .take(entries.len())
                .collect();
        for (entry, descriptor) in entries {
            #[cfg(windows)]
            let _s = if type_eq::<B, DX12>() {
                Some(smol::block_on(self.context.shader_semaphore.acquire()))
            } else {
                None
            };
            let shader = Shader::new(self.device(), entry, &descriptor)?;
            unsafe {
                shaders
                    .get_unchecked_mut(descriptor.id.0 as usize)
                    .as_mut_ptr()
                    .write(shader);
            }
        }
        self.shaders = unsafe { transmute(shaders) };
        Ok(self)
    }
    fn pipeline(&mut self, entry: EntryId, specialization: &[u8]) -> DeviceResult<PipelineRef<B>> {
        let context = &self.context;
        let name = &self.name;
        let shader = unsafe { self.shaders.get_unchecked_mut(entry.0 as usize) };
        shader
            .pipeline(context.device(), &*self.module, specialization)
            .map_err(|e| {
                dbg!(name);
                e
            })
    }
}

impl<B: Backend> Drop for ShaderModule<B> {
    fn drop(&mut self) {
        unsafe {
            let module = ManuallyDrop::take(&mut self.module);
            self.device().destroy_shader_module(module);
        }
    }
}

#[derive(Debug)]
struct PipelineRef<'a, B: Backend> {
    descriptor_set_layout: &'a B::DescriptorSetLayout,
    pipeline_layout: &'a B::PipelineLayout,
    compute_pipeline: &'a B::ComputePipeline,
}

#[derive(Debug)]
struct Shader<B: Backend> {
    entry: String,
    descriptor_set_layout: B::DescriptorSetLayout,
    pipeline_layout: B::PipelineLayout,
    compute_pipelines: HashMap<ArrayVec<[u8; SPECIALIZATION_SIZE]>, B::ComputePipeline>,
}

impl<B: Backend> Shader<B> {
    fn new(device: &B::Device, entry: String, descriptor: &EntryDescriptor) -> DeviceResult<Self> {
        let bindings =
            descriptor
                .buffers
                .iter()
                .enumerate()
                .map(|(i, mutable)| DescriptorSetLayoutBinding {
                    binding: i as u32,
                    ty: DescriptorType::Buffer {
                        ty: BufferDescriptorType::Storage {
                            read_only: !mutable,
                        },
                        format: BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                });
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(bindings, empty())? };
        let push_constant_size = descriptor.push_constant_size as u32;
        let push_constant_range = if push_constant_size > 0 {
            Some((ShaderStageFlags::COMPUTE, 0..push_constant_size))
        } else {
            None
        };
        let pipeline_layout_result = unsafe {
            device.create_pipeline_layout(
                once(&descriptor_set_layout),
                push_constant_range.into_iter(),
            )
        };
        let pipeline_layout = match pipeline_layout_result {
            Ok(pipeline_layout) => pipeline_layout,
            Err(e) => {
                unsafe {
                    device.destroy_descriptor_set_layout(descriptor_set_layout);
                }
                return Err(e.into());
            }
        };
        Ok(Self {
            entry,
            descriptor_set_layout,
            pipeline_layout,
            compute_pipelines: HashMap::new(),
        })
    }
    fn pipeline(
        &mut self,
        device: &B::Device,
        module: &B::ShaderModule,
        specialization: &[u8],
    ) -> DeviceResult<PipelineRef<B>> {
        use std::collections::hash_map::Entry;
        let specialization = ArrayVec::try_from(specialization).unwrap();
        let compute_pipeline = match self.compute_pipelines.entry(specialization) {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => {
                let entry_point = EntryPoint {
                    entry: &self.entry,
                    module,
                    specialization: Specialization::default(), // TODO: impl specialization
                };
                let compute_pipeline_result = unsafe {
                    device.create_compute_pipeline(
                        &ComputePipelineDesc::new(entry_point, &self.pipeline_layout),
                        None,
                    )
                };
                let compute_pipeline = match compute_pipeline_result {
                    Ok(compute_pipeline) => compute_pipeline,
                    Err(e) => match e {
                        CreationError::OutOfMemory(out_of_memory) => {
                            return Err(out_of_memory.into());
                        }
                        _ => {
                            if cfg!(debug_assertions) {
                                dbg!(&self.entry);
                                dbg!(e);
                            }
                            return Err(DeviceError::ShaderCompilationFailed);
                        }
                    },
                };
                vacant.insert(compute_pipeline)
            }
        };
        Ok(PipelineRef {
            descriptor_set_layout: &self.descriptor_set_layout,
            pipeline_layout: &self.pipeline_layout,
            compute_pipeline,
        })
    }
}

#[derive(Debug)]
struct ComputeArg<B: Backend> {
    slice: StorageSlice<B>,
    mutable: bool,
}

#[derive(Debug)]
enum Op<B: Backend> {
    Module {
        id: ModuleId,
        module: ShaderModule<B>,
    },
    Compute {
        module: ModuleId,
        module_name: String,
        entry: EntryId,
        entry_name: String,
        work_groups: [u32; 3],
        local_size: [u32; 3],
        args: Vec<ComputeArg<B>>,
        push_constants: Vec<u8>,
        specialization: Vec<u8>,
    },
    Copy {
        src: StorageSlice<B>,
        dst: StorageSlice<B>,
    },
    Write {
        src: WriteSlice<B>,
        dst: StorageSlice<B>,
        read_guard_fut: Option<ReadGuardFuture>,
    },
    Read {
        src: StorageSlice<B>,
        dst: ReadSlice<B>,
    },
    Sync {
        finished: Arc<AtomicBool>,
    },
}

struct MappingChunkRef<B: Backend>(&'static MappingChunkBase<B>);

impl<B: Backend> MappingChunkRef<B> {
    fn as_ptr(&self) -> *const MappingChunkBase<B> {
        self.0 as _
    }
}

impl<B: Backend> From<&'static MappingChunkBase<B>> for MappingChunkRef<B> {
    fn from(chunk: &'static MappingChunkBase<B>) -> Self {
        Self(chunk)
    }
}

impl<B: Backend> PartialEq for MappingChunkRef<B> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr().eq(&other.as_ptr())
    }
}

impl<B: Backend> Eq for MappingChunkRef<B> {}

impl<B: Backend> Hash for MappingChunkRef<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state)
    }
}

impl<B: Backend> Deref for MappingChunkRef<B> {
    type Target = MappingChunkBase<B>;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<B: Backend> Debug for MappingChunkRef<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_ptr().fmt(f)
    }
}

#[derive(Debug)]
struct DescriptorSetAllocator<B: Backend> {
    pools: Vec<B::DescriptorPool>,
}

impl<B: Backend> Default for DescriptorSetAllocator<B> {
    fn default() -> Self {
        Self {
            pools: Vec::default(),
        }
    }
}

impl<B: Backend> DescriptorSetAllocator<B> {
    fn allocate(
        &mut self,
        device: &B::Device,
        layout: &B::DescriptorSetLayout,
    ) -> DeviceResult<Option<B::DescriptorSet>> {
        if let Some(pool) = self.pools.last_mut() {
            if let Ok(set) = unsafe { pool.allocate_one(layout) } {
                return Ok(Some(set));
            }
        }
        // TODO: May be able to use more pools, trouble is that one frame could grab all the pools, exhausting memory, but this is unlikely for small numbers.
        if self.pools.is_empty() {
            let descriptor_ranges =
                once(false)
                    .chain(once(true))
                    .map(|read_only| DescriptorRangeDesc {
                        ty: DescriptorType::Buffer {
                            ty: BufferDescriptorType::Storage { read_only },
                            format: BufferDescriptorFormat::Structured {
                                dynamic_offset: false,
                            },
                        },
                        count: 500,
                    });
            let pool = unsafe {
                device.create_descriptor_pool(
                    1000,
                    descriptor_ranges,
                    DescriptorPoolCreateFlags::empty(),
                )?
            };
            self.pools.push(pool);
            let pool = self.pools.last_mut().unwrap();
            Ok(Some(unsafe { pool.allocate_one(layout).unwrap() }))
        } else {
            Ok(None)
        }
    }
    unsafe fn reset(&mut self) {
        for pool in self.pools.iter_mut() {
            pool.reset();
        }
    }
    unsafe fn free(self, device: &B::Device) {
        for pool in self.pools {
            device.destroy_descriptor_pool(pool);
        }
    }
}

#[allow(unused_mut)]
fn descriptor_range<B: Backend>(mut offset: u32, mut len: u32) -> SubRange {
    #[cfg(windows)]
    if type_eq::<B, DX12>() {
        offset /= 4;
    }
    if len % 4 > 0 {
        len += 4 - len % 4;
    }
    SubRange {
        offset: offset as u64,
        size: Some(len as u64),
    }
}

fn barrier_range(offset: u32, mut len: u32) -> SubRange {
    if len % 4 > 0 {
        len += 4 - (len % 4);
    }
    SubRange {
        offset: offset as u64,
        size: Some(len as u64),
    }
}

#[cfg(feature = "profile")]
#[derive(Debug)]
struct QueryPool<B: Backend> {
    pool: B::QueryPool,
    buffer: Vec<u32>,
    period_ns: u64,
}

#[cfg(feature = "profile")]
impl<B: Backend> QueryPool<B> {
    fn new(device: &B::Device, count: u32, period_ns: u64) -> DeviceResult<Self> {
        let pool = unsafe {
            device
                .create_query_pool(QueryType::Timestamp, count)
                .map_err(|err| match err {
                    QueryCreationError::OutOfMemory(memory_error) => {
                        DeviceError::from(memory_error)
                    }
                    err => todo!("{:?}", err),
                })?
        };
        Ok(Self {
            pool,
            buffer: Vec::new(),
            period_ns,
        })
    }
    unsafe fn get_compute_pass_timestamps(
        &mut self,
        device: &B::Device,
        metrics: &mut [ComputePassMetrics],
    ) -> DeviceResult<()> {
        use gfx_hal::device::WaitError;
        let timestamps = metrics.len() * 2;
        let reserve = timestamps
            .checked_sub(self.buffer.len())
            .unwrap_or_default();
        self.buffer.extend(std::iter::repeat(0).take(reserve));
        let data = bytemuck::cast_slice_mut(&mut self.buffer[0..timestamps]);
        device
            .get_query_pool_results(
                &self.pool,
                0..timestamps as u32,
                data,
                size_of::<u32>() as u32,
                QueryResultFlags::WAIT,
            )
            .map_err(|err| match err {
                WaitError::OutOfMemory(out_of_memory) => DeviceError::from(out_of_memory),
                WaitError::DeviceLost(device_lost) => DeviceError::from(device_lost),
            })?;
        for (duration, timestamp) in metrics
            .iter_mut()
            .flat_map(|metric| [&mut metric.start, &mut metric.end])
            .zip(self.buffer.iter().copied())
        {
            *duration = Duration::from_nanos(timestamp as u64 * self.period_ns)
        }
        Ok(())
    }
    unsafe fn free(self, device: &B::Device) {
        device.destroy_query_pool(self.pool);
    }
}

#[cfg(feature = "profile")]
impl<B: Backend> Deref for QueryPool<B> {
    type Target = B::QueryPool;
    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}

#[derive(Debug)]
struct Frame<B: Backend> {
    semaphore: B::Semaphore,
    fence: B::Fence,
    descriptor_set_allocator: DescriptorSetAllocator<B>,
    command_buffer: B::CommandBuffer,
    writes: Vec<(WriteSlice<B>, Option<ReadGuardFuture>)>,
    mapping_chunks: HashMap<MappingChunkRef<B>, u32>,
    syncs: Vec<Arc<AtomicBool>>,
    ready_to_submit: bool,
    #[cfg(feature = "profile")]
    query_pool: Option<QueryPool<B>>,
    #[cfg(feature = "profile")]
    compute_pass_metrics: Vec<ComputePassMetrics>,
}

impl<B: Backend> Frame<B> {
    fn new(
        context: &Context<B>,
        command_pool: &mut B::CommandPool,
        #[allow(unused)] timestamp_period_ns: u64,
    ) -> DeviceResult<Self> {
        let device = context.device();
        let semaphore = device.create_semaphore()?;
        let fence = match device.create_fence(true) {
            Ok(fence) => fence,
            Err(e) => {
                unsafe {
                    device.destroy_semaphore(semaphore);
                }
                return Err(e.into());
            }
        };
        let mut command_buffer = unsafe { command_pool.allocate_one(CommandLevel::Primary) };
        unsafe {
            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
        }
        #[cfg(feature = "profile")]
        let query_pool = context
            .profiler
            .as_ref()
            .map(|_| QueryPool::new(device, 10_000, timestamp_period_ns))
            .transpose()?;
        Ok(Self {
            semaphore,
            fence,
            descriptor_set_allocator: DescriptorSetAllocator::default(),
            command_buffer,
            writes: Vec::new(),
            mapping_chunks: HashMap::new(),
            syncs: Vec::new(),
            ready_to_submit: false,
            #[cfg(feature = "profile")]
            query_pool,
            #[cfg(feature = "profile")]
            compute_pass_metrics: Vec::new(),
        })
    }
    fn ready_to_submit(&mut self) -> bool {
        self.ready_to_submit
    }
    fn try_encode(
        &mut self,
        op: Op<B>,
        device: &B::Device,
        modules: &mut HashMap<ModuleId, ShaderModule<B>>,
    ) -> DeviceResult<Result<(), Op<B>>> {
        match op {
            Op::Module { id, module } => {
                modules.entry(id).or_insert(module);
            }
            Op::Compute {
                module,
                module_name,
                entry,
                entry_name,
                work_groups,
                local_size,
                args,
                push_constants,
                specialization,
            } => {
                let pipeline = modules
                    .get_mut(&module)
                    .unwrap()
                    .pipeline(entry, &specialization)?;
                let mut descriptor_set = if let Some(descriptor_set) = self
                    .descriptor_set_allocator
                    .allocate(device, pipeline.descriptor_set_layout)?
                {
                    descriptor_set
                } else {
                    return Ok(Err(Op::Compute {
                        module,
                        module_name,
                        entry,
                        entry_name,
                        work_groups,
                        local_size,
                        args,
                        push_constants,
                        specialization,
                    }));
                };
                #[cfg(feature = "profile")]
                let metric_id = self.compute_pass_metrics.len();
                #[cfg(feature = "profile")]
                if self.query_pool.is_some() {
                    self.compute_pass_metrics.push(ComputePassMetrics {
                        module_id: module.0,
                        module_name,
                        entry_name,
                        invocations: work_groups
                            .iter()
                            .zip(local_size.iter())
                            .map(|(wg, ls)| *wg as usize * *ls as usize)
                            .product(),
                        start: Duration::default(),
                        end: Duration::default(),
                    });
                }
                let descriptors = args.iter().map(|arg| {
                    Descriptor::Buffer(
                        arg.slice.buffer(),
                        descriptor_range::<B>(arg.slice.offset, arg.slice.len),
                    )
                });
                unsafe {
                    device.write_descriptor_set(DescriptorSetWrite {
                        set: &mut descriptor_set,
                        binding: 0,
                        array_offset: 0,
                        descriptors,
                    });
                    self.command_buffer
                        .bind_compute_pipeline(pipeline.compute_pipeline);
                    self.command_buffer.bind_compute_descriptor_sets(
                        pipeline.pipeline_layout,
                        0,
                        once(&descriptor_set),
                        empty(),
                    );
                    if !push_constants.is_empty() {
                        self.command_buffer.push_compute_constants(
                            pipeline.pipeline_layout,
                            0,
                            bytemuck::cast_slice(&push_constants),
                        );
                    }
                    let barriers = args.iter().map(|arg| {
                        let mut states =
                            State::TRANSFER_WRITE | State::SHADER_WRITE..State::SHADER_READ;
                        if arg.mutable {
                            states.start |= State::TRANSFER_READ | State::SHADER_READ;
                            states.end |= State::SHADER_WRITE;
                        }
                        // TODO: This prevents a failure on DX12, but likely at a performance cost
                        // since reads can't be concurrent.
                        #[cfg(windows)]
                        if type_eq::<B, DX12>() {
                            states.end |= State::SHADER_WRITE;
                        }
                        Barrier::Buffer {
                            states,
                            target: arg.slice.buffer(),
                            range: barrier_range(arg.slice.offset, arg.slice.len),
                            families: None,
                        }
                    });
                    self.command_buffer.pipeline_barrier(
                        PipelineStage::TRANSFER | PipelineStage::COMPUTE_SHADER
                            ..PipelineStage::COMPUTE_SHADER,
                        Dependencies::empty(),
                        barriers,
                    );
                    #[cfg(feature = "profile")]
                    if let Some(pool) = self.query_pool.as_deref() {
                        self.command_buffer.write_timestamp(
                            PipelineStage::COMPUTE_SHADER,
                            Query {
                                pool,
                                id: (metric_id * 2) as u32,
                            },
                        )
                    }
                    self.command_buffer.dispatch(work_groups);
                    #[cfg(feature = "profile")]
                    if let Some(pool) = self.query_pool.as_ref() {
                        self.command_buffer.write_timestamp(
                            PipelineStage::COMPUTE_SHADER,
                            Query {
                                pool,
                                id: (metric_id * 2 + 1) as u32,
                            },
                        )
                    }
                }
            }
            Op::Copy { src, dst } => {
                let region = BufferCopy {
                    src: src.offset as u64,
                    dst: dst.offset as u64,
                    size: dst.len as u64,
                };
                unsafe {
                    let src_barrier = Barrier::Buffer {
                        states: State::TRANSFER_WRITE
                            | State::TRANSFER_READ
                            | State::SHADER_WRITE
                            | State::SHADER_READ
                            ..State::TRANSFER_READ,
                        target: src.buffer(),
                        range: barrier_range(src.offset, src.len),
                        families: None,
                    };
                    let dst_barrier = Barrier::Buffer {
                        states: State::TRANSFER_WRITE
                            | State::TRANSFER_READ
                            | State::SHADER_WRITE
                            | State::SHADER_READ
                            ..State::TRANSFER_WRITE,
                        target: dst.buffer(),
                        range: barrier_range(dst.offset, dst.len),
                        families: None,
                    };
                    self.command_buffer.pipeline_barrier(
                        PipelineStage::TRANSFER | PipelineStage::COMPUTE_SHADER
                            ..PipelineStage::TRANSFER,
                        Dependencies::empty(),
                        once(src_barrier).chain(once(dst_barrier)),
                    );
                    self.command_buffer
                        .copy_buffer(src.buffer(), dst.buffer(), once(region));
                    let dst_barrier = Barrier::Buffer {
                        states: State::TRANSFER_WRITE
                            ..State::TRANSFER_WRITE
                                | State::TRANSFER_READ
                                | State::SHADER_WRITE
                                | State::SHADER_READ,
                        target: dst.buffer(),
                        range: barrier_range(dst.offset, dst.len),
                        families: None,
                    };
                    self.command_buffer.pipeline_barrier(
                        PipelineStage::TRANSFER | PipelineStage::COMPUTE_SHADER
                            ..PipelineStage::TRANSFER,
                        Dependencies::empty(),
                        once(dst_barrier),
                    );
                }
            }
            Op::Write {
                src,
                dst,
                read_guard_fut,
            } => {
                let offset = self.mapping_chunks.entry(src.chunk.into()).or_default();
                *offset = u32::max(*offset, src.offset + src.len);
                let region = BufferCopy {
                    src: src.offset as u64,
                    dst: dst.offset as u64,
                    size: dst.len as u64,
                };
                unsafe {
                    self.command_buffer.pipeline_barrier(
                        PipelineStage::COMPUTE_SHADER | PipelineStage::TRANSFER
                            ..PipelineStage::TRANSFER,
                        Dependencies::empty(),
                        once(Barrier::Buffer {
                            states: State::SHADER_WRITE | State::SHADER_READ | State::TRANSFER_READ
                                ..State::TRANSFER_WRITE,
                            target: dst.buffer(),
                            range: barrier_range(dst.offset, dst.len),
                            families: None,
                        }),
                    );
                    self.command_buffer
                        .copy_buffer(src.buffer(), dst.buffer(), once(region));
                    self.command_buffer.pipeline_barrier(
                        PipelineStage::TRANSFER
                            ..PipelineStage::COMPUTE_SHADER | PipelineStage::TRANSFER,
                        Dependencies::empty(),
                        once(Barrier::Buffer {
                            states: State::TRANSFER_WRITE
                                ..State::SHADER_WRITE | State::SHADER_READ | State::TRANSFER_READ,
                            target: dst.buffer(),
                            range: barrier_range(dst.offset, dst.len),
                            families: None,
                        }),
                    );
                }
                self.writes.push((src, read_guard_fut));
            }
            Op::Read { src, dst } => {
                let offset = self.mapping_chunks.entry(dst.chunk.into()).or_default();
                *offset = u32::max(*offset, dst.offset + dst.len);
                let region = BufferCopy {
                    src: src.offset as u64,
                    dst: dst.offset as u64,
                    size: dst.len as u64,
                };
                unsafe {
                    self.command_buffer.pipeline_barrier(
                        PipelineStage::COMPUTE_SHADER | PipelineStage::TRANSFER
                            ..PipelineStage::TRANSFER,
                        Dependencies::empty(),
                        once(Barrier::Buffer {
                            states: State::SHADER_WRITE | State::TRANSFER_WRITE
                                ..State::TRANSFER_READ,
                            target: src.buffer(),
                            range: barrier_range(src.offset, src.len),
                            families: None,
                        }),
                    );
                    self.command_buffer
                        .copy_buffer(src.buffer, dst.buffer(), once(region));
                }
            }
            Op::Sync { finished } => {
                self.syncs.push(finished);
            }
        }
        self.ready_to_submit = true;
        Ok(Ok(()))
    }
    fn reset(&mut self, #[allow(unused)] context: &Context<B>) -> DeviceResult<()> {
        for (chunk, offset) in self.mapping_chunks.drain() {
            chunk.blocks.finish(offset);
        }
        for (slice, _) in self.writes.drain(..) {
            slice.chunk.blocks.drop_guard();
        }
        for sync in self.syncs.drain(..) {
            sync.store(true, Ordering::SeqCst);
        }
        #[cfg(feature = "profile")]
        if let Some((profiler, query_pool)) =
            context.profiler.as_ref().zip(self.query_pool.as_mut())
        {
            let device = context.device();
            unsafe {
                query_pool.get_compute_pass_timestamps(device, &mut self.compute_pass_metrics)?;
            }
            for metric in self.compute_pass_metrics.drain(..) {
                profiler.compute_pass(metric);
            }
        }
        unsafe {
            self.command_buffer.reset(false);
            self.command_buffer
                .begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            self.descriptor_set_allocator.reset();
        }
        Ok(())
    }
    fn poll(&mut self, context: &Context<B>) -> DeviceResult<bool> {
        if unsafe { context.device().get_fence_status(&self.fence)? } {
            self.reset(context)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn submit(
        &mut self,
        device: &B::Device,
        previous: Option<&Self>,
        queue: &mut B::Queue,
    ) -> DeviceResult<()> {
        unsafe {
            device.reset_fence(&mut self.fence)?;
        }
        let wait_iter = previous
            .map(|f| {
                (
                    &f.semaphore,
                    PipelineStage::COMPUTE_SHADER | PipelineStage::TRANSFER,
                )
            })
            .into_iter();
        unsafe {
            self.command_buffer.finish();
        }
        for (slice, read_guard_fut) in self.writes.iter_mut() {
            if let Some(read_guard_fut) = read_guard_fut.take() {
                unsafe { slice.write_only() }
                    .copy_from_slice(&smol::block_on(read_guard_fut.finish())?);
            } else {
                slice.barrier.wait();
            }
        }
        for (chunk, offset) in self.mapping_chunks.iter() {
            chunk.blocks.submit(*offset);
        }
        unsafe {
            queue.submit(
                once(&self.command_buffer),
                wait_iter,
                once(&self.semaphore),
                Some(&mut self.fence),
            );
        }
        self.ready_to_submit = false;
        Ok(())
    }
    unsafe fn free(self, device: &B::Device) {
        device.destroy_semaphore(self.semaphore);
        device.destroy_fence(self.fence);
        self.descriptor_set_allocator.free(device);
        #[cfg(feature = "profile")]
        if let Some(query_pool) = self.query_pool {
            query_pool.free(device);
        }
    }
}

struct Context<B: Backend> {
    device: B::Device,
    allocator: ManuallyDrop<Allocator<B>>,
    #[cfg(windows)]
    shader_semaphore: Semaphore,
    _adapter: Arc<Adapter<B>>,
    _instance: Arc<B::Instance>,
    #[cfg(feature = "profile")]
    profiler: Option<Arc<Profiler>>,
}

impl<B: Backend> Context<B> {
    fn new(
        device: B::Device,
        allocator: Allocator<B>,
        adapter: Arc<Adapter<B>>,
        instance: Arc<B::Instance>,
        #[cfg(feature = "profile")] profiler: Option<Arc<Profiler>>,
    ) -> Self {
        Self {
            device,
            allocator: ManuallyDrop::new(allocator),
            #[cfg(windows)]
            shader_semaphore: Semaphore::new(1),
            _adapter: adapter,
            _instance: instance,
            #[cfg(feature = "profile")]
            profiler,
        }
    }
    fn device(&self) -> &B::Device {
        &self.device
    }
}

impl<B: Backend> Drop for Context<B> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::take(&mut self.allocator).free(&self.device);
        }
    }
}

impl<B: Backend> RefUnwindSafe for Context<B> {}

impl<B: Backend> Debug for Context<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Context").finish()
    }
}

#[derive(Debug, Clone)]
enum DynContext {
    #[cfg(any(
        all(unix, not(any(target_os = "ios", target_os = "macos"))),
        feature = "gfx_backend_vulkan",
        windows
    ))]
    Vulkan(Arc<Context<Vulkan>>),
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    Metal(Arc<Context<Metal>>),
    #[cfg(windows)]
    DX12(Arc<Context<DX12>>),
}

impl<B: Backend> From<Arc<Context<B>>> for DynContext {
    fn from(context: Arc<Context<B>>) -> Self {
        #[cfg(any(
            all(unix, not(any(target_os = "ios", target_os = "macos"))),
            feature = "gfx_backend_vulkan",
            windows
        ))]
        if type_eq::<B, Vulkan>() {
            return Self::Vulkan(unsafe { transmute(context) });
        }
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        if type_eq::<B, Metal>() {
            return Self::Metal(unsafe { transmute(context) });
        }
        #[cfg(windows)]
        if type_eq::<B, DX12>() {
            return Self::DX12(unsafe { transmute(context) });
        }
        unreachable!()
    }
}

impl From<OutOfMemory> for DeviceError {
    fn from(from: OutOfMemory) -> Self {
        match from {
            OutOfMemory::Host => Self::OutOfHostMemory,
            OutOfMemory::Device => Self::OutOfDeviceMemory,
        }
    }
}

impl From<DeviceLost> for DeviceError {
    fn from(_from: DeviceLost) -> Self {
        Self::DeviceLost
    }
}

impl From<DeviceCreationError> for DeviceError {
    fn from(from: DeviceCreationError) -> Self {
        match from {
            DeviceCreationError::OutOfMemory(out_of_memory) => out_of_memory.into(),
            DeviceCreationError::InitializationFailed => Self::InitializationFailed,
            DeviceCreationError::MissingFeature => Self::MissingFeature,
            DeviceCreationError::TooManyObjects => Self::TooManyObjects,
            DeviceCreationError::DeviceLost => Self::DeviceLost,
        }
    }
}

impl From<BufferCreationError> for DeviceError {
    fn from(from: BufferCreationError) -> Self {
        match from {
            BufferCreationError::OutOfMemory(out_of_memory) => out_of_memory.into(),
            BufferCreationError::UnsupportedUsage { .. } => unreachable!("{:?}", from),
        }
    }
}

impl From<AllocationError> for DeviceError {
    fn from(from: AllocationError) -> Self {
        match from {
            AllocationError::OutOfMemory(out_of_memory) => out_of_memory.into(),
            AllocationError::TooManyObjects => Self::TooManyObjects,
        }
    }
}

impl From<BindError> for DeviceError {
    fn from(from: BindError) -> Self {
        match from {
            BindError::OutOfMemory(out_of_memory) => out_of_memory.into(),
            BindError::WrongMemory | BindError::OutOfBounds => unreachable!("{:?}", from),
        }
    }
}

impl From<MapError> for DeviceError {
    fn from(from: MapError) -> Self {
        match from {
            MapError::OutOfMemory(out_of_memory) => out_of_memory.into(),
            MapError::OutOfBounds => Self::OutOfBounds,
            MapError::MappingFailed => Self::MappingFailed,
            MapError::Access => Self::Access,
        }
    }
}
