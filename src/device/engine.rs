#[cfg(feature = "profile")]
use super::profiler::{ComputePassMetrics, TransferKind, TransferMetrics};
use crate::{
    device::{
        shader::{EntryId, Module, ModuleId},
        ComputePass,
    },
    result::Result,
};

use anyhow::{anyhow, bail};
use crossbeam_channel::{unbounded, Receiver, Sender};
use dashmap::DashMap;
use fxhash::FxBuildHasher;
use once_cell::sync::OnceCell;
use parking_lot::{Mutex, RwLock};
use rspirv::spirv::Capability;
#[cfg(feature = "profile")]
use std::time::Duration;
use std::{
    collections::VecDeque,
    future::Future,
    iter::once,
    mem::transmute,
    ops::{Deref, Range},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use vulkano::{
    buffer::{
        cpu_access::ReadLock, cpu_pool::CpuBufferPoolChunk, device_local::DeviceLocalBuffer,
        BufferAccess, BufferSlice, BufferUsage, CpuAccessibleBuffer, CpuBufferPool,
    },
    command_buffer::{
        pool::{UnsafeCommandPool, UnsafeCommandPoolAlloc},
        submit::SubmitCommandBufferBuilder,
        sys::{
            UnsafeCommandBuffer, UnsafeCommandBufferBuilder,
            UnsafeCommandBufferBuilderPipelineBarrier,
        },
        CommandBufferLevel, CommandBufferUsage,
    },
    descriptor_set::{
        builder::DescriptorSetBuilder,
        layout::{DescriptorDesc, DescriptorSetDesc, DescriptorSetLayout, DescriptorType},
        pool::{standard::StdDescriptorPoolAlloc, DescriptorPool, DescriptorPoolAlloc},
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, DeviceOwned, Features, Queue,
    },
    instance::{Instance, InstanceCreationError, InstanceExtensions, Version},
    memory::pool::StdMemoryPool,
    pipeline::{layout::PipelineLayoutPcRange, ComputePipeline, PipelineBindPoint, PipelineLayout},
    shader::{
        spirv::ExecutionModel, DescriptorRequirements, EntryPointInfo, ShaderExecution,
        ShaderInterface, ShaderModule, ShaderStages,
    },
    sync::{AccessFlags, Fence, PipelineStages},
    OomError,
};
#[cfg(feature = "profile")]
use vulkano::{
    query::{QueryPool, QueryResultFlags, QueryType},
    sync::PipelineStage,
};

#[cfg(any(target_os = "ios", target_os = "macos"))]
mod molten {
    use ash::vk::Instance;
    use std::os::raw::{c_char, c_void};
    use vulkano::instance::loader::Loader;

    pub(super) struct AshMoltenLoader;

    unsafe impl Loader for AshMoltenLoader {
        fn get_instance_proc_addr(&self, instance: Instance, name: *const c_char) -> *const c_void {
            let entry = ash_molten::load();
            let ptr = unsafe { entry.get_instance_proc_addr(std::mem::transmute(instance), name) };
            if let Some(ptr) = ptr {
                unsafe { std::mem::transmute(ptr) }
            } else {
                std::ptr::null()
            }
        }
    }
}
#[cfg(any(target_os = "ios", target_os = "macos"))]
use molten::AshMoltenLoader;

fn instance() -> Result<Arc<Instance>, InstanceCreationError> {
    static INSTANCE: OnceCell<Arc<Instance>> = OnceCell::new();
    INSTANCE
        .get_or_try_init(|| {
            let app_info = vulkano::app_info_from_cargo_toml!();
            let extensions = InstanceExtensions::none();
            let version = Version::major_minor(1, 1);
            let layers = [];
            #[allow(unused_mut)]
            let mut instance = Instance::new(Some(&app_info), version, &extensions, layers);
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            {
                use vulkano::instance::loader::FunctionPointers;
                if instance.is_err() {
                    instance = Instance::with_loader(
                        FunctionPointers::new(Box::new(AshMoltenLoader)),
                        Some(&app_info),
                        version,
                        &extensions,
                        layers,
                    )
                }
            }
            instance
        })
        .map(Arc::clone)
}

fn physical_device_type_index(device_type: PhysicalDeviceType) -> u8 {
    use PhysicalDeviceType::*;
    match device_type {
        DiscreteGpu => 0,
        IntegratedGpu => 1,
        Cpu => 2,
        _ => 3,
    }
}

fn optimal_device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_16bit_storage: true,
        ..DeviceExtensions::none()
    }
}

fn required_device_features() -> Features {
    Features { ..Features::none() }
}

fn optimal_device_features() -> Features {
    Features {
        vulkan_memory_model: true,
        vulkan_memory_model_device_scope: true,
        storage_buffer16_bit_access: true,
        shader_int16: true,
        ..required_device_features()
    }
}

fn supports_capability(c: Capability, f: &Features) -> bool {
    use Capability::*;
    match c {
        Int16 => f.shader_int16,
        StorageBuffer16BitAccess => f.storage_buffer16_bit_access,
        _ => false,
    }
}

pub(super) struct Engine {
    device: Arc<Device>,
    upload_buffer_pool: CpuBufferPool<u8>,
    storage_allocator: StorageAllocator,
    shader_modules: DashMap<ModuleId, Arc<ShaderModule>, FxBuildHasher>,
    compute_cache: DashMap<(ModuleId, EntryId), ComputeCache, FxBuildHasher>,
    op_sender: Sender<Op>,
    done: Arc<AtomicBool>,
    index: usize,
}

impl Engine {
    pub(super) fn new() -> Result<Arc<Self>> {
        let instance = instance()?;
        let mut physical_devices = PhysicalDevice::enumerate(&instance)
            .enumerate()
            .collect::<Vec<_>>();
        if physical_devices.is_empty() {
            bail!("No device!");
        }
        physical_devices.sort_by_key(|x| physical_device_type_index(x.1.properties().device_type));
        let queue_families = physical_devices
            .into_iter()
            .map(|(i, physical_device)| {
                if physical_device
                    .supported_features()
                    .is_superset_of(&required_device_features())
                {
                    if let Some(compute_family) = physical_device
                        .queue_families()
                        .find(|x| !x.supports_graphics() && x.supports_compute())
                        .or_else(|| {
                            physical_device
                                .queue_families()
                                .find(|x| x.supports_compute())
                        })
                    {
                        Ok((i, physical_device, compute_family))
                    } else {
                        Err(anyhow!("Device doesn't support compute!"))
                    }
                } else {
                    Err(anyhow!(
                        "Device doesn't support required features! Supported features = {:#?}\nRequired features = {:#?}",
                        physical_device.supported_features(),
                        required_device_features()
                    ))
                }
            })
            .collect::<Vec<_>>();
        for queue_family in queue_families.iter() {
            if let &Ok((index, physical_device, compute_family)) = queue_family {
                let required_device_extensions = physical_device.required_extensions();
                let supported_device_extensions = physical_device.supported_extensions();
                let device_extensions = supported_device_extensions
                    .intersection(&required_device_extensions.union(&optimal_device_extensions()));
                let device_features = physical_device
                    .supported_features()
                    .intersection(&optimal_device_features());
                let (device, mut queues) = Device::new(
                    physical_device,
                    &device_features,
                    &device_extensions,
                    once((compute_family, 1.)),
                )?;
                let queue = queues.next().expect("Compute queue not found!");
                let upload_buffer_pool = CpuBufferPool::upload(device.clone());
                let initial_chunks = 1;
                let storage_allocator =
                    StorageAllocator::with_initial_chunks(device.clone(), initial_chunks)?;
                let shader_modules = DashMap::<_, _, FxBuildHasher>::default();
                let compute_cache = DashMap::<_, _, FxBuildHasher>::default();
                let (op_sender, op_receiver) = unbounded();
                let done = Arc::new(AtomicBool::new(false));
                let mut runner = Runner::new(queue, op_receiver, done.clone())?;
                std::thread::spawn(move || runner.run());
                let engine = Arc::new(Self {
                    index,
                    device,
                    upload_buffer_pool,
                    storage_allocator,
                    shader_modules,
                    compute_cache,
                    op_sender,
                    done,
                });
                return Ok(engine);
            }
        }
        let err = queue_families
            .into_iter()
            .next()
            .expect("No queue families!");
        Err(err.expect_err("Expected unsupported device error!"))
    }
    pub(super) fn index(&self) -> usize {
        self.index
    }
    pub(super) fn capabilities(&self) -> impl Iterator<Item = Capability> {
        let features = self.device.enabled_features().clone();
        use Capability::*;
        [Int16, StorageBuffer16BitAccess]
            .iter()
            .copied()
            .filter(move |c| supports_capability(*c, &features))
    }
    pub(super) fn supports_capability(&self, c: Capability) -> bool {
        supports_capability(c, self.device.enabled_features())
    }
    pub(super) unsafe fn alloc(&self, len: usize) -> Result<Arc<StorageBuffer>> {
        self.storage_allocator.alloc(len)
    }
    pub(super) fn upload(self: &Arc<Self>, data: &[u8]) -> Result<Arc<StorageBuffer>> {
        self.upload_buffer_pool.reserve(data.len() as u64)?;
        let sub_buffer = self.upload_buffer_pool.chunk(data.iter().copied())?;
        let storage = unsafe { self.alloc(data.len())? };
        self.op_sender.send(Op::Upload(Upload {
            src: sub_buffer,
            dst: storage.as_slice(),
        }))?;
        Ok(storage)
    }
    pub(super) fn download(
        self: &Arc<Self>,
        storage: Arc<StorageBuffer>,
        offset: usize,
        len: usize,
    ) -> Result<impl Future<Output = Result<StorageBufferReadGuard>>> {
        let usage = BufferUsage {
            transfer_destination: true,
            ..BufferUsage::none()
        };
        let host_cached = true;
        let cpu_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            usage,
            host_cached,
            (0..len).map(|_| 0u8),
        )?;
        let device_slice = storage
            .slice(offset..offset + len)
            .expect("Device slice not large enough!");
        let sync = SyncFuture::default();
        self.op_sender.send(Op::Download(Download {
            src: device_slice,
            dst: cpu_buffer.clone(),
            sync: sync.clone(),
        }))?;
        Ok(async move {
            sync.wait().await?;
            let guard = cpu_buffer.read()?;
            let guard = unsafe { transmute(guard) }; // convert to 'static
            Ok(StorageBufferReadGuard {
                _cpu: cpu_buffer,
                guard,
            })
        })
    }
    pub(super) fn compute_pass(&self, compute_pass: ComputePass<'_>) -> Result<()> {
        let device = &self.device;
        let shader_module: Arc<ShaderModule> = self
            .shader_modules
            .entry(compute_pass.module.id)
            .or_try_insert_with(|| shader_module(device.clone(), compute_pass.module))?
            .clone();
        let entry_point = shader_module
            .entry_point(&compute_pass.entry_name)
            .expect("Entry not found!");
        let specialization_constants = &();
        let cache = None;
        let compute_cache = self
            .compute_cache
            .entry((compute_pass.module.id, compute_pass.entry))
            .or_try_insert_with(|| {
                let layout_desc =
                    DescriptorSetDesc::new((0..compute_pass.buffers.len()).map(|_| {
                        Some(DescriptorDesc {
                            ty: DescriptorType::StorageBuffer,
                            descriptor_count: 1,
                            variable_count: false,
                            stages: ShaderStages {
                                compute: true,
                                ..ShaderStages::default()
                            },
                            immutable_samplers: Vec::new(),
                        })
                    }));
                let descriptor_set_layout = DescriptorSetLayout::new(device.clone(), layout_desc)?;
                let pipeline_layout = PipelineLayout::new(
                    device.clone(),
                    [descriptor_set_layout.clone()],
                    entry_point.push_constant_requirements().copied(),
                )?;
                let compute_pipeline = ComputePipeline::new(
                    device.clone(),
                    entry_point,
                    specialization_constants,
                    cache,
                    |_| (),
                )?;
                Result::<_, anyhow::Error>::Ok(ComputeCache {
                    compute_pipeline,
                    pipeline_layout,
                    descriptor_set_layout,
                })
            })?
            .clone();
        let mut descriptor_set = Device::standard_descriptor_pool(&self.device)
            .alloc(&compute_cache.descriptor_set_layout, 0)?;
        let mut builder = DescriptorSetBuilder::start(compute_cache.descriptor_set_layout.clone());
        let mut slices = Vec::with_capacity(compute_pass.buffers.len());
        for buffer in compute_pass.buffers.iter() {
            let start = buffer.offset;
            let start = (start / 4) * 4 + if start % 4 != 0 { 4 } else { 0 };
            let end = buffer.offset + buffer.len;
            let end = (end / 4) * 4 + if end % 4 != 0 { 4 } else { 0 };
            let slice = buffer
                .storage
                .slice(start..end)
                .expect("Slice out of bounds!");
            builder.add_buffer(slice.clone())?;
            slices.push(ComputeSlice {
                slice,
                mutable: buffer.mutable,
            });
        }
        let output = builder.build()?;
        unsafe {
            descriptor_set
                .inner_mut()
                .write(output.layout(), output.writes());
        }
        self.op_sender.send(Op::Compute(Compute {
            descriptor_set,
            _descriptor_set_layout: compute_cache.descriptor_set_layout,
            pipeline_layout: compute_cache.pipeline_layout,
            compute_pipeline: compute_cache.compute_pipeline,
            push_constants: compute_pass.push_constants,
            work_groups: compute_pass.work_groups,
            slices,
            #[cfg(feature = "profile")]
            module_id: compute_pass.module.id,
            #[cfg(feature = "profile")]
            module_name: compute_pass.module.name.clone(),
            #[cfg(feature = "profile")]
            entry_name: compute_pass.entry_name,
            #[cfg(feature = "profile")]
            local_size: compute_pass.local_size,
        }))?;
        Ok(())
    }
    pub(super) fn sync(&self) -> Result<impl Future<Output = Result<()>>> {
        let sync = SyncFuture::default();
        self.op_sender.send(Op::Sync(sync.clone()))?;
        Ok(async move { sync.wait().await })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        while Arc::strong_count(&self.done) > 1 {}
    }
}

pub(super) struct StorageBuffer {
    chunk: Arc<StorageChunk>,
    offset: u32,
    size: u32,
}

impl StorageBuffer {
    fn as_slice(&self) -> Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>> {
        self.chunk
            .device_local
            .slice(self.offset as u64..(self.offset + self.size) as u64)
            .expect("Slice out of bounds!")
    }
    #[allow(clippy::type_complexity)]
    fn slice(
        &self,
        range: Range<usize>,
    ) -> Option<Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>> {
        let offset = self.offset as usize;
        let start = offset + range.start;
        let end = offset + range.end;
        if end <= start + self.size as usize {
            self.chunk.device_local.slice(start as u64..end as u64)
        } else {
            None
        }
    }
}

impl Drop for StorageBuffer {
    fn drop(&mut self) {
        let start = self.offset as u32;
        let mut blocks = self.chunk.blocks.lock();
        let index = blocks
            .iter()
            .position(|block| block.start == start)
            .expect("Block not found!");
        blocks.remove(index);
    }
}

pub(super) struct StorageBufferReadGuard {
    _cpu: Arc<CpuAccessibleBuffer<[u8]>>,
    guard: ReadLock<'static, [u8]>,
}

impl Deref for StorageBufferReadGuard {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.guard
    }
}

#[derive(Clone, Copy, Debug)]
struct StorageBlock {
    start: u32,
    end: u32,
}

struct StorageChunk {
    device_local: Arc<DeviceLocalBuffer<[u8]>>,
    blocks: Mutex<Vec<StorageBlock>>,
}

impl StorageChunk {
    fn new(device: Arc<Device>, size: usize) -> Result<Arc<Self>> {
        let usage = BufferUsage {
            transfer_source: true,
            transfer_destination: true,
            storage_buffer: true,
            ..BufferUsage::none()
        };
        let device_local = DeviceLocalBuffer::array(
            device.clone(),
            size as u64,
            usage,
            device.active_queue_families(),
        )?;
        let blocks = Mutex::default();
        Ok(Arc::new(Self {
            device_local,
            blocks,
        }))
    }
    fn alloc(self: &Arc<Self>, size: usize) -> Option<Arc<StorageBuffer>> {
        let mut blocks = self.blocks.lock();
        let mut start = 0;
        let size = (size / 4) * 4 + if size % 4 != 0 { 4 } else { 0 };
        let block_size = (size as u32 / 128) * 128 + if size % 128 != 0 { 128 } else { 0 };
        for (i, block) in blocks.iter().enumerate() {
            if start + block_size <= block.start {
                blocks.insert(
                    i,
                    StorageBlock {
                        start,
                        end: start + block_size,
                    },
                );
                return Some(Arc::new(StorageBuffer {
                    chunk: self.clone(),
                    offset: start,
                    size: size as u32,
                }));
            } else {
                start = block.end;
            }
        }
        if (start + block_size) as u64 <= self.device_local.size() {
            blocks.push(StorageBlock {
                start,
                end: start + block_size,
            });
            Some(Arc::new(StorageBuffer {
                chunk: self.clone(),
                offset: start,
                size: size as u32,
            }))
        } else {
            None
        }
    }
}

struct StorageAllocator {
    device: Arc<Device>,
    chunks: RwLock<Vec<Arc<StorageChunk>>>,
}

impl StorageAllocator {
    const CHUNK_SIZE: usize = 256_000_000;
    fn with_initial_chunks(device: Arc<Device>, initial_chunks: usize) -> Result<Self> {
        let mut chunks = Vec::with_capacity(initial_chunks);
        for _ in 0..initial_chunks {
            chunks.push(StorageChunk::new(device.clone(), Self::CHUNK_SIZE)?);
        }
        let chunks = RwLock::new(chunks);
        Ok(Self { device, chunks })
    }
    fn alloc(&self, size: usize) -> Result<Arc<StorageBuffer>> {
        if size > Self::CHUNK_SIZE {
            bail!("Buffer is too large {} B! Max buffer size is 256 MB!", size);
        }
        let mut chunks = self.chunks.write();
        for chunk in chunks.iter() {
            if let Some(buffer) = chunk.alloc(size) {
                return Ok(buffer);
            }
        }
        let chunk = StorageChunk::new(self.device.clone(), Self::CHUNK_SIZE)?;
        let buffer = chunk.alloc(size).expect("Unable to allocate buffer!");
        chunks.push(chunk);
        Ok(buffer)
    }
}

fn shader_module(device: Arc<Device>, module: &Module) -> Result<Arc<ShaderModule>> {
    let bytes = module.spirv.as_ref();
    let version = Version::major_minor(1, 1);
    let capabilities = [];
    let extensions = [];
    let stages = ShaderStages {
        compute: true,
        ..Default::default()
    };
    let entry_points = module.descriptor.entries.iter().map(|(entry, descriptor)| {
        let descriptor_requirements = descriptor
            .buffers
            .iter()
            .copied()
            .enumerate()
            .map(|(binding, mutable)| {
                let reqs = DescriptorRequirements {
                    descriptor_types: vec![DescriptorType::StorageBuffer],
                    descriptor_count: 1,
                    mutable,
                    stages,
                    ..Default::default()
                };
                ((0, binding as u32), reqs)
            })
            .collect();
        let push_constant_requirements = if descriptor.push_constant_size > 0 {
            Some(PipelineLayoutPcRange {
                offset: 0,
                size: descriptor.push_constant_size as u32,
                stages,
            })
        } else {
            None
        };
        let specialization_constant_requirements = Default::default();
        let info = EntryPointInfo {
            execution: ShaderExecution::Compute,
            descriptor_requirements,
            push_constant_requirements,
            specialization_constant_requirements,
            input_interface: ShaderInterface::empty(),
            output_interface: ShaderInterface::empty(),
        };
        (entry.clone(), ExecutionModel::GLCompute, info)
    });
    Ok(unsafe {
        ShaderModule::from_bytes_with_data(
            device,
            bytes,
            version,
            capabilities,
            extensions,
            entry_points,
        )?
    })
}

#[derive(Clone)]
struct ComputeCache {
    compute_pipeline: Arc<ComputePipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
}

struct ComputeSlice {
    slice: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
    mutable: bool,
}

trait WriteCommandBuffer {
    unsafe fn write(&self, builder: &mut UnsafeCommandBufferBuilder);
}

enum Op {
    Upload(Upload),
    Compute(Compute),
    Download(Download),
    Sync(SyncFuture),
}

struct Upload {
    src: Arc<CpuBufferPoolChunk<u8, Arc<StdMemoryPool>>>,
    dst: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
}

impl WriteCommandBuffer for Upload {
    unsafe fn write(&self, builder: &mut UnsafeCommandBufferBuilder) {
        unsafe {
            builder.copy_buffer(&self.src, &self.dst, [(0, 0, self.dst.size())]);
        }
    }
}

struct Compute {
    descriptor_set: StdDescriptorPoolAlloc,
    _descriptor_set_layout: Arc<DescriptorSetLayout>,
    compute_pipeline: Arc<ComputePipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    work_groups: [u32; 3],
    push_constants: Vec<u8>,
    slices: Vec<ComputeSlice>,
    #[cfg(feature = "profile")]
    module_id: ModuleId,
    #[cfg(feature = "profile")]
    module_name: Option<String>,
    #[cfg(feature = "profile")]
    entry_name: String,
    #[cfg(feature = "profile")]
    local_size: [u32; 3],
}

impl WriteCommandBuffer for Compute {
    unsafe fn write(&self, builder: &mut UnsafeCommandBufferBuilder) {
        let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();
        for slice in self.slices.iter() {
            let source_stage = PipelineStages {
                compute_shader: true,
                ..PipelineStages::none()
            };
            let source_access = AccessFlags {
                shader_read: true,
                shader_write: true,
                ..AccessFlags::none()
            };
            let destination_stage = PipelineStages {
                compute_shader: true,
                ..PipelineStages::none()
            };
            let destination_access = AccessFlags {
                shader_read: true,
                shader_write: slice.mutable,
                ..AccessFlags::none()
            };
            let by_region = false;
            let queue_transfer = None;
            let offset = slice.slice.offset();
            let size = slice.slice.len();
            unsafe {
                barrier.add_buffer_memory_barrier(
                    &slice.slice,
                    source_stage,
                    source_access,
                    destination_stage,
                    destination_access,
                    by_region,
                    queue_transfer,
                    offset,
                    size,
                )
            }
        }
        unsafe {
            builder.pipeline_barrier(&barrier);
            builder.bind_pipeline_compute(&self.compute_pipeline);
            let first_set = 0;
            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                &self.pipeline_layout,
                first_set,
                [self.descriptor_set.inner()],
                [],
            );
            let stages = ShaderStages {
                compute: true,
                ..ShaderStages::none()
            };
            if !self.push_constants.is_empty() {
                let offset = 0;
                let size = self.push_constants.len() as u32;
                builder.push_constants(
                    &self.pipeline_layout,
                    stages,
                    offset,
                    size,
                    self.push_constants.as_slice(),
                );
            }
            builder.dispatch(self.work_groups);
        }
    }
}

#[derive(Default, Clone, Debug)]
struct SyncFuture {
    signal: Arc<AtomicBool>,
}

impl SyncFuture {
    fn signal(&self) {
        self.signal.store(true, Ordering::Relaxed);
    }
    async fn wait(mut self) -> Result<()> {
        loop {
            if let Some(guard) = Arc::get_mut(&mut self.signal) {
                if *guard.get_mut() {
                    return Ok(());
                } else {
                    return Err(anyhow!("Device disconnected!"));
                }
            } else {
                smol::future::yield_now().await;
            }
        }
    }
}

struct Download {
    src: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
    dst: Arc<CpuAccessibleBuffer<[u8]>>,
    sync: SyncFuture,
}

impl WriteCommandBuffer for Download {
    unsafe fn write(&self, builder: &mut UnsafeCommandBufferBuilder) {
        unsafe {
            builder.copy_buffer(&self.src, &self.dst, [(0, 0, self.dst.size())]);
        }
    }
}

struct CommandBuffer {
    _alloc: UnsafeCommandPoolAlloc,
    command_buffer: UnsafeCommandBuffer,
}

impl CommandBuffer {
    fn builder(alloc: UnsafeCommandPoolAlloc) -> Result<CommandBufferBuilder, OomError> {
        let builder = unsafe {
            UnsafeCommandBufferBuilder::new(
                &alloc,
                CommandBufferLevel::Primary,
                CommandBufferUsage::OneTimeSubmit,
            )?
        };
        Ok(CommandBufferBuilder { alloc, builder })
    }
}

struct CommandBufferBuilder {
    alloc: UnsafeCommandPoolAlloc,
    builder: UnsafeCommandBufferBuilder,
}

impl CommandBufferBuilder {
    fn build(self) -> Result<CommandBuffer, OomError> {
        Ok(CommandBuffer {
            _alloc: self.alloc,
            command_buffer: self.builder.build()?,
        })
    }
}
#[cfg(feature = "profile")]
enum ProfileEntry {
    Upload {
        size: usize,
    },
    Compute {
        module_id: u32,
        module_name: Option<String>,
        entry_name: String,
        invocations: usize,
    },
    Download {
        size: usize,
    },
}

#[cfg(feature = "profile")]
struct Profiler {
    profiler: Arc<super::profiler::Profiler>,
    device: Arc<Device>,
    query_pool: Arc<QueryPool>,
    entries: Vec<ProfileEntry>,
}

#[cfg(feature = "profile")]
impl Profiler {
    fn new(
        profiler: Arc<super::profiler::Profiler>,
        device: Arc<Device>,
        capacity: usize,
    ) -> Result<Self> {
        let query_pool =
            QueryPool::new(device.clone(), QueryType::Timestamp, (capacity * 2) as u32)?;
        let entries = Vec::with_capacity(capacity);
        Ok(Self {
            profiler,
            device,
            query_pool,
            entries,
        })
    }
    fn begin(&mut self) -> Result<()> {
        if !self.entries.is_empty() {
            let mut results = vec![0u64; self.entries.len() * 2];
            let flags = QueryResultFlags {
                wait: true,
                ..QueryResultFlags::default()
            };
            self.query_pool
                .queries_range(0..(self.entries.len() * 2) as u32)
                .expect("No queries!")
                .get_results(&mut results, flags)?;
            let results: &[[u64; 2]] = bytemuck::cast_slice(&results);
            let period = self.device.physical_device().properties().timestamp_period;
            for (entry, [before, after]) in self.entries.drain(..).zip(results.iter().copied()) {
                let duration =
                    Duration::from_nanos(((after - before) as f64 * period as f64) as u64);
                match entry {
                    ProfileEntry::Upload { size } => {
                        self.profiler.transfer(TransferMetrics {
                            kind: TransferKind::HostToDevice,
                            size,
                            duration,
                        });
                    }
                    ProfileEntry::Compute {
                        module_id,
                        module_name,
                        entry_name,
                        invocations,
                    } => {
                        self.profiler.compute_pass(ComputePassMetrics {
                            module_id,
                            module_name,
                            entry_name,
                            invocations,
                            duration,
                        });
                    }
                    ProfileEntry::Download { size } => {
                        self.profiler.transfer(TransferMetrics {
                            kind: TransferKind::DeviceToHost,
                            size,
                            duration,
                        });
                    }
                }
            }
        }
        Ok(())
    }
    unsafe fn before(&mut self, entry: &ProfileEntry, builder: &mut UnsafeCommandBufferBuilder) {
        let index = self.entries.len() * 2;
        let query = self
            .query_pool
            .query(index as u32)
            .expect("Query out of range!");
        let stage = match &entry {
            ProfileEntry::Upload { .. } | ProfileEntry::Download { .. } => PipelineStage::Transfer,
            ProfileEntry::Compute { .. } => PipelineStage::ComputeShader,
        };
        unsafe {
            builder.write_timestamp(query, stage);
        }
    }
    unsafe fn after(&mut self, entry: ProfileEntry, builder: &mut UnsafeCommandBufferBuilder) {
        let index = self.entries.len() * 2 + 1;
        let query = self
            .query_pool
            .query(index as u32)
            .expect("Query out of range!");
        let stage = match &entry {
            ProfileEntry::Upload { .. } | ProfileEntry::Download { .. } => PipelineStage::Transfer,
            ProfileEntry::Compute { .. } => PipelineStage::ComputeShader,
        };
        unsafe {
            builder.write_timestamp(query, stage);
        }
        self.entries.push(entry);
    }
}

struct Frame {
    queue: Arc<Queue>,
    command_pool: UnsafeCommandPool,
    command_buffer: Option<CommandBuffer>,
    command_buffer_builder: Option<CommandBufferBuilder>,
    fence: Fence,
    ops: Vec<Op>,
    downloads: Vec<Download>,
    syncs: Vec<SyncFuture>,
    #[cfg(feature = "profile")]
    profiler: Option<Profiler>,
}

impl Frame {
    // TODO: Increasing MAX_OPS improves performace but risks running out of memory.
    const MAX_OPS: usize = 1_000; // 400;
    fn new(queue: Arc<Queue>) -> Result<Self> {
        let device = queue.device();
        let transient = true;
        let reset_cb = false;
        let command_pool =
            UnsafeCommandPool::new(device.clone(), queue.family(), transient, reset_cb)?;
        let fence = Fence::alloc_signaled(device.clone())?;
        let ops = Vec::new();
        let downloads = Vec::new();
        let syncs = Vec::new();
        #[cfg(feature = "profile")]
        let profiler = super::profiler::Profiler::get()
            .map(|profiler| Profiler::new(profiler?, device.clone(), Self::MAX_OPS))
            .transpose()?;
        let mut frame = Self {
            queue,
            command_pool,
            command_buffer: None,
            command_buffer_builder: None,
            fence,
            ops,
            downloads,
            syncs,
            #[cfg(feature = "profile")]
            profiler,
        };
        frame.begin()?;
        Ok(frame)
    }
    fn poll(&mut self) -> Result<bool> {
        if self.is_empty() {
            Ok(true)
        } else if self.fence.ready()? {
            self.ops.clear();
            self.syncs.extend(self.downloads.drain(..).map(|x| x.sync));
            for sync in self.syncs.drain(..) {
                sync.signal();
            }
            self.begin()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn len(&self) -> usize {
        self.ops.len() + self.downloads.len() + self.syncs.len()
    }
    fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.downloads.is_empty() && self.syncs.is_empty()
    }
    fn begin(&mut self) -> Result<()> {
        assert!(self.fence.ready()?);
        #[cfg(feature = "profile")]
        if let Some(profiler) = self.profiler.as_mut() {
            profiler.begin()?;
        }
        self.command_buffer.take();
        unsafe {
            self.command_pool.reset(false)?;
        }
        let secondary = false;
        let count = 1;
        let alloc = self
            .command_pool
            .alloc_command_buffers(secondary, count)?
            .next()
            .expect("No command buffer!");
        self.command_buffer_builder
            .replace(CommandBuffer::builder(alloc)?);
        Ok(())
    }
    fn encode(&mut self, op: Op) -> Result<()> {
        let builder = &mut self
            .command_buffer_builder
            .as_mut()
            .expect("No builder!")
            .builder;
        match op {
            Op::Upload(ref upload) => {
                #[cfg(feature = "profile")]
                let entry = if self.profiler.is_some() {
                    Some(ProfileEntry::Upload {
                        size: upload.dst.len() as usize,
                    })
                } else {
                    None
                };
                #[cfg(feature = "profile")]
                if let Some((profiler, entry)) = self.profiler.as_mut().zip(entry.as_ref()) {
                    unsafe {
                        profiler.before(&entry, builder);
                    }
                }
                unsafe {
                    upload.write(builder);
                }
                #[cfg(feature = "profile")]
                if let Some((profiler, entry)) = self.profiler.as_mut().zip(entry) {
                    unsafe {
                        profiler.after(entry, builder);
                    }
                }
                self.ops.push(op);
            }
            Op::Compute(ref compute) => {
                #[cfg(feature = "profile")]
                let entry = if self.profiler.is_some() {
                    Some(ProfileEntry::Compute {
                        module_id: compute.module_id.0,
                        module_name: compute.module_name.clone(),
                        entry_name: compute.entry_name.clone(),
                        invocations: compute
                            .work_groups
                            .iter()
                            .zip(compute.local_size.iter())
                            .map(|(wg, ls)| (wg * ls) as usize)
                            .product(),
                    })
                } else {
                    None
                };
                #[cfg(feature = "profile")]
                if let Some((profiler, entry)) = self.profiler.as_mut().zip(entry.as_ref()) {
                    unsafe {
                        profiler.before(&entry, builder);
                    }
                }
                unsafe {
                    compute.write(builder);
                }
                #[cfg(feature = "profile")]
                if let Some((profiler, entry)) = self.profiler.as_mut().zip(entry) {
                    unsafe {
                        profiler.after(entry, builder);
                    }
                }
                self.ops.push(op);
            }
            Op::Download(download) => {
                #[cfg(feature = "profile")]
                let entry = if self.profiler.is_some() {
                    Some(ProfileEntry::Download {
                        size: download.src.len() as usize,
                    })
                } else {
                    None
                };
                #[cfg(feature = "profile")]
                if let Some((profiler, entry)) = self.profiler.as_mut().zip(entry.as_ref()) {
                    unsafe {
                        profiler.before(&entry, builder);
                    }
                }
                unsafe {
                    download.write(builder);
                }
                #[cfg(feature = "profile")]
                if let Some((profiler, entry)) = self.profiler.as_mut().zip(entry) {
                    unsafe {
                        profiler.after(entry, builder);
                    }
                }
                self.downloads.push(download);
            }
            Op::Sync(sync) => {
                self.syncs.push(sync);
            }
        }
        Ok(())
    }
    fn submit(&mut self) -> Result<()> {
        debug_assert!(self.fence.ready()?);
        self.fence.reset()?;
        let mut builder = self.command_buffer_builder.take().expect("No builder!");
        if !self.syncs.is_empty() {
            let source = PipelineStages {
                transfer: true,
                compute_shader: true,
                ..PipelineStages::none()
            };
            let destination = PipelineStages {
                transfer: true,
                compute_shader: true,
                ..PipelineStages::none()
            };
            let by_region = false;
            unsafe {
                let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();
                barrier.add_execution_dependency(source, destination, by_region);
                builder.builder.pipeline_barrier(&barrier);
            }
        }
        let command_buffer = builder.build()?;
        let mut builder = SubmitCommandBufferBuilder::new();
        unsafe {
            builder.add_command_buffer(&command_buffer.command_buffer);
            builder.set_fence_signal(&self.fence);
        }
        builder.submit(&self.queue)?;
        self.command_buffer.replace(command_buffer);
        Ok(())
    }
}

struct Runner {
    op_receiver: Receiver<Op>,
    ready: VecDeque<Frame>,
    pending: VecDeque<Frame>,
    done: Arc<AtomicBool>,
}

impl Runner {
    fn new(queue: Arc<Queue>, op_receiver: Receiver<Op>, done: Arc<AtomicBool>) -> Result<Self> {
        // TODO: Freezes after a while with 3 frames, maybe need semaphore?
        let nframes = 2;
        let mut ready = VecDeque::with_capacity(nframes);
        for _ in 0..nframes {
            ready.push_back(Frame::new(queue.clone())?);
        }
        let pending = VecDeque::with_capacity(ready.len());
        Ok(Self {
            op_receiver,
            ready,
            pending,
            done,
        })
    }
    fn run(&mut self) {
        while !self.done.load(Ordering::Acquire) {
            {
                let frame = self.ready.front_mut().expect("No frame!");
                if frame.len() < Frame::MAX_OPS {
                    if let Ok(op) = self.op_receiver.try_recv() {
                        frame.encode(op).expect("Frame::encode failed!");
                    }
                }
                if let Some(frame) = self.ready.front_mut() {
                    if frame.len() < Frame::MAX_OPS {
                        if let Ok(op) = self.op_receiver.try_recv() {
                            frame.encode(op).expect("Frame::encode failed!");
                        }
                    }
                }
            }
            if let Some(pending) = self.pending.front_mut() {
                if pending.poll().expect("Frame::poll failed!") {
                    self.ready
                        .push_back(self.pending.pop_front().expect("No frame!"));
                }
            }
            if self.ready.len() >= 2 && !self.ready.front().expect("No frame!").is_empty() {
                let mut frame = self.ready.pop_front().expect("No frame!");
                frame.submit().expect("Frame::submit failed!");
                self.pending.push_back(frame);
            }
        }
    }
}
