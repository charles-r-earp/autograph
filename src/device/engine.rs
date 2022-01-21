use crate::{
    device::{
        shader::{EntryId, Module, ModuleId},
        ComputePass,
    },
    result::Result,
};
use anyhow::{anyhow, bail};
use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use fxhash::FxBuildHasher;
use once_cell::sync::OnceCell;
use smol::lock::{Mutex, MutexGuardArc};
use std::{
    collections::VecDeque,
    future::Future,
    iter,
    iter::once,
    mem::{drop, take, transmute},
    ops::Deref,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    time::Duration,
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
        AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, PrimaryCommandBuffer,
    },
    descriptor_set::{
        builder::DescriptorSetBuilder,
        layout::{DescriptorDesc, DescriptorSetDesc, DescriptorSetLayout, DescriptorType},
        pool::{
            standard::StdDescriptorPoolAlloc, DescriptorPool, DescriptorPoolAlloc,
            StdDescriptorPool,
        },
        sys::{DescriptorWrite, UnsafeDescriptorSet},
        DescriptorSet, DescriptorSetResources, DescriptorSetWithOffsets,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceOwned, Features, Queue,
    },
    instance::{Instance, InstanceCreationError, InstanceExtensions, Version},
    memory::pool::StdMemoryPool,
    memory::DeviceMemoryAllocError,
    pipeline::{layout::PipelineLayoutPcRange, ComputePipeline, PipelineBindPoint, PipelineLayout},
    shader::{
        spirv::ExecutionModel, DescriptorRequirements, EntryPointInfo, ShaderExecution,
        ShaderInterface, ShaderModule, ShaderStages,
    },
    sync::{now, AccessFlags, Fence, FenceSignalFuture, FenceWaitError, GpuFuture, PipelineStages},
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
            let ptr = unsafe {
                entry
                    .get_instance_proc_addr(std::mem::transmute(instance), name)
                    .expect("Unable to load MoltenVK!")
            };
            unsafe { std::mem::transmute(ptr) }
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

fn required_device_features() -> Features {
    Features {
        vulkan_memory_model: true,
        ..Features::none()
    }
}

fn optimal_device_features() -> Features {
    Features {
        ..required_device_features()
    }
}

pub(super) struct Engine {
    device: Arc<Device>,
    upload_buffer_pool: CpuBufferPool<u8>,
    shader_modules: DashMap<ModuleId, Arc<ShaderModule>, FxBuildHasher>,
    compute_cache: DashMap<(ModuleId, EntryId), ComputeCache, FxBuildHasher>,
    op_sender: Sender<Op>,
    done: Arc<AtomicBool>,
}

impl Engine {
    pub(super) fn new() -> Result<Arc<Self>> {
        let instance = instance()?;
        let mut physical_devices = PhysicalDevice::enumerate(&instance).collect::<Vec<_>>();
        if physical_devices.is_empty() {
            bail!("No device!");
        }
        physical_devices.sort_by_key(|x| physical_device_type_index(x.properties().device_type));
        let queue_families = physical_devices
            .into_iter()
            .map(|physical_device| {
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
                        let transfer_family = physical_device.queue_families().find(|x| {
                            x.id() != compute_family.id() && x.explicitly_supports_transfers()
                        });
                        Ok((physical_device, compute_family, transfer_family))
                    } else {
                        Err(anyhow!("Device doesn't support compute!"))
                    }
                } else {
                    Err(anyhow!(
                        "Device doesn't support required_features! {:#?}",
                        required_device_features()
                    ))
                }
            })
            .collect::<Vec<_>>();
        for queue_family in queue_families.iter() {
            if let &Ok((physical_device, compute_family, transfer_family)) = queue_family {
                let device_extensions = physical_device.required_extensions();
                let device_features = physical_device
                    .supported_features()
                    .intersection(&optimal_device_features());
                let (device, mut queues) = Device::new(
                    physical_device,
                    &device_features,
                    device_extensions,
                    once((compute_family, 1.)).chain(transfer_family.map(|x| (x, 0.))),
                )?;
                let compute_queue = queues.next().expect("Compute queue not found!");
                //let transfer_queue = None; // queues.next();
                let upload_buffer_pool = CpuBufferPool::upload(device.clone());
                let shader_modules = DashMap::<_, _, FxBuildHasher>::default();
                let compute_cache = DashMap::<_, _, FxBuildHasher>::default();
                let (op_sender, op_receiver) = bounded(100);
                let done = Arc::new(AtomicBool::new(false));
                let runner = Runner::new(compute_queue, op_receiver, done.clone())?;
                std::thread::spawn(move || runner.run());
                let engine = Arc::new(Self {
                    device,
                    upload_buffer_pool,
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
    pub(super) unsafe fn alloc(self: &Arc<Self>, len: usize) -> Result<StorageBuffer> {
        let usage = BufferUsage {
            transfer_source: true,
            transfer_destination: true,
            storage_buffer: true,
            ..BufferUsage::none()
        };
        let len = (len / 4) * 4 + if len % 4 != 0 { 4 } else { 0 };
        let device_local = DeviceLocalBuffer::array(
            self.device.clone(),
            len as u64,
            usage,
            self.device.active_queue_families(),
        )?;
        Ok(StorageBuffer { device_local })
    }
    pub(super) fn upload(self: &Arc<Self>, data: &[u8]) -> Result<StorageBuffer> {
        self.upload_buffer_pool.reserve(data.len() as u64)?;
        let sub_buffer = self.upload_buffer_pool.chunk(data.iter().copied())?;
        let storage = unsafe { self.alloc(data.len())? };
        self.op_sender.send(Op::Upload(Upload {
            src: sub_buffer,
            dst: storage.device_local.into_buffer_slice(),
        }))?;
        Ok(storage)
    }
    pub(super) fn download(
        self: &Arc<Self>,
        storage: StorageBuffer,
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
        let start = offset as u64;
        let end = start + len as u64;
        let device_slice = storage
            .device_local
            .slice(start..end)
            .expect("Device slice not large enough!");
        self.op_sender.send(Op::Download(Download {
            src: device_slice,
            dst: cpu_buffer.clone(),
        }))?;
        let fut = self.sync()?;
        Ok(async move {
            fut.await?;
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
            let start = start as u64;
            let end = end as u64;
            let slice = buffer
                .storage
                .device_local
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
            descriptor_set_layout: compute_cache.descriptor_set_layout,
            pipeline_layout: compute_cache.pipeline_layout,
            compute_pipeline: compute_cache.compute_pipeline,
            push_constants: compute_pass.push_constants,
            work_groups: compute_pass.work_groups,
            slices,
        }))?;
        Ok(())
    }
    pub(super) fn sync(&self) -> Result<impl Future<Output = Result<()>>> {
        let mut sync = Arc::new(Mutex::default());
        self.op_sender
            .send(Op::SyncGuard(SyncGuard::new(sync.try_lock_arc().unwrap())))?;
        Ok(async move {
            sync.lock().await;
            while Arc::get_mut(&mut sync).is_none() {}
            let sync = Arc::try_unwrap(sync).ok().map(Mutex::into_inner).flatten();
            sync.ok_or_else(|| anyhow!("Disconnected!"))?
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        while Arc::strong_count(&self.done) > 1 {}
    }
}

#[derive(Clone)]
pub(super) struct StorageBuffer {
    device_local: Arc<DeviceLocalBuffer<[u8]>>,
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
/*
async fn wait_for_fence(fence: &Fence) -> Result<()> {
    loop {
        let result = fence.wait(Some(Duration::default()));
        match result {
            Ok(()) => {
                break;
            }
            Err(FenceWaitError::Timeout) => {
                smol::future::yield_now().await;
            }
            Err(err) => {
                return Err(err.into())
            }
        }
    }
    Ok(())
}
*/
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
    SyncGuard(SyncGuard),
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
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    compute_pipeline: Arc<ComputePipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    work_groups: [u32; 3],
    push_constants: Vec<u8>,
    slices: Vec<ComputeSlice>,
}

impl WriteCommandBuffer for Compute {
    unsafe fn write(&self, builder: &mut UnsafeCommandBufferBuilder) {
        let mut barrier = unsafe { UnsafeCommandBufferBuilderPipelineBarrier::new() };
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
            let offset = 0;
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

struct SyncGuard {
    guard: MutexGuardArc<Option<Result<()>>>,
}

impl SyncGuard {
    fn new(guard: MutexGuardArc<Option<Result<()>>>) -> Self {
        Self { guard }
    }
    fn store(&mut self, result: Result<()>) {
        self.guard.replace(result);
    }
}

struct Download {
    src: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
    dst: Arc<CpuAccessibleBuffer<[u8]>>,
}

impl WriteCommandBuffer for Download {
    unsafe fn write(&self, builder: &mut UnsafeCommandBufferBuilder) {
        unsafe {
            builder.copy_buffer(&self.src, &self.dst, [(0, 0, self.dst.size())]);
        }
    }
}

struct Frame {
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_pool: UnsafeCommandPool,
    command_buffer: Option<(UnsafeCommandPoolAlloc, UnsafeCommandBuffer)>,
    fence: Fence,
    uploads: Vec<Upload>,
    computes: Vec<Compute>,
    downloads: Vec<Download>,
    syncs: Vec<SyncGuard>,
}

impl Frame {
    fn new(queue: Arc<Queue>) -> Result<Self> {
        let device = queue.device().clone();
        let transient = true;
        let reset_cb = false;
        let command_pool =
            UnsafeCommandPool::new(device.clone(), queue.family(), transient, reset_cb)?;
        let fence = Fence::alloc_signaled(device.clone())?;
        let uploads = Vec::new();
        let computes = Vec::new();
        let downloads = Vec::new();
        let syncs = Vec::new();
        Ok(Self {
            device,
            queue,
            command_pool,
            command_buffer: None,
            fence,
            uploads,
            computes,
            downloads,
            syncs,
        })
    }
    fn poll(&mut self) -> Result<bool> {
        let result = self.fence.ready();
        match result {
            Ok(true) | Err(_) => {
                self.uploads.clear();
                self.computes.clear();
                self.downloads.clear();
                for mut sync in self.syncs.drain(..) {
                    let result = result.map(|_| ()).map_err(|err| anyhow::Error::new(err));
                    sync.store(result);
                }
                result.map_err(Into::into)
            }
            Ok(false) => Ok(false),
        }
    }
    fn submit(&mut self, ops: impl Iterator<Item = Op>) -> Result<()> {
        assert!(self.fence.ready()?);
        self.fence.reset()?;
        if let Some((command_pool_alloc, command_buffer)) = self.command_buffer.take() {
            std::mem::drop(command_buffer);
            std::mem::drop(command_pool_alloc);
            unsafe {
                self.command_pool.reset(false)?;
            }
        }
        for op in ops {
            match op {
                Op::Upload(x) => {
                    self.uploads.push(x);
                }
                Op::Compute(x) => {
                    self.computes.push(x);
                }
                Op::Download(x) => {
                    self.downloads.push(x);
                }
                Op::SyncGuard(x) => {
                    self.syncs.push(x);
                }
            }
        }
        let secondary = false;
        let count = 1;
        let command_pool_alloc = self
            .command_pool
            .alloc_command_buffers(secondary, count)?
            .next()
            .expect("No command buffer!");
        let mut builder = unsafe {
            UnsafeCommandBufferBuilder::new(
                &command_pool_alloc,
                CommandBufferLevel::Primary,
                CommandBufferUsage::OneTimeSubmit,
            )?
        };
        {
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
                builder.pipeline_barrier(&barrier);
            }
        }
        for upload in self.uploads.iter() {
            unsafe {
                upload.write(&mut builder);
            }
        }
        {
            let source = PipelineStages {
                transfer: true,
                ..PipelineStages::none()
            };
            let destination = PipelineStages {
                compute_shader: true,
                ..PipelineStages::none()
            };
            let by_region = false;
            unsafe {
                let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();
                barrier.add_execution_dependency(source, destination, by_region);
                builder.pipeline_barrier(&barrier);
            }
            for compute in self.computes.iter() {
                unsafe {
                    compute.write(&mut builder);
                }
            }
        }
        {
            let source = PipelineStages {
                transfer: true,
                compute_shader: true,
                ..PipelineStages::none()
            };
            let destination = PipelineStages {
                transfer: true,
                ..PipelineStages::none()
            };
            let by_region = false;
            unsafe {
                let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();
                barrier.add_execution_dependency(source, destination, by_region);
                builder.pipeline_barrier(&barrier);
            }
            for download in self.downloads.iter() {
                unsafe {
                    download.write(&mut builder);
                }
            }
        }
        {
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
                builder.pipeline_barrier(&barrier);
            }
        }
        let command_buffer = builder.build()?;
        let mut builder = SubmitCommandBufferBuilder::new();
        unsafe {
            builder.add_command_buffer(&command_buffer);
            builder.set_fence_signal(&self.fence);
        }
        let result = builder.submit(&self.queue);
        self.command_buffer
            .replace((command_pool_alloc, command_buffer));
        if let Err(err) = result {
            for mut sync in self.syncs.drain(..) {
                sync.store(Err(anyhow::Error::new(err)));
            }
        }
        result?;
        Ok(())
    }
}

struct Runner {
    op_receiver: Receiver<Op>,
    ready_frames: VecDeque<Frame>,
    pending_frames: VecDeque<Frame>,
    done: Arc<AtomicBool>,
}

impl Runner {
    fn new(queue: Arc<Queue>, op_receiver: Receiver<Op>, done: Arc<AtomicBool>) -> Result<Self> {
        let n_frames = 3;
        let mut ready_frames = VecDeque::with_capacity(n_frames);
        for _ in 0..n_frames {
            ready_frames.push_back(Frame::new(queue.clone())?);
        }
        let pending_frames = VecDeque::with_capacity(n_frames);
        Ok(Self {
            op_receiver,
            ready_frames,
            pending_frames,
            done,
        })
    }
    fn run(mut self) {
        while !self.done.load(Ordering::Acquire) {
            if let Some(frame) = self.ready_frames.front_mut() {
                let ops = self.op_receiver.try_iter().collect::<Vec<_>>();
                if !ops.is_empty() {
                    frame
                        .submit(ops.into_iter())
                        .expect("Frame::submit failed!");
                    self.pending_frames.extend(self.ready_frames.pop_front());
                }
            }
            while let Some(frame) = self.pending_frames.front_mut() {
                if frame.poll().expect("Frame::poll failed!") {
                    self.ready_frames.extend(self.pending_frames.pop_front());
                } else {
                    break;
                }
            }
        }
    }
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
#[cfg(test)]
mod tests {
    use crate::result::Result;

    #[test]
    fn instance() -> Result<()> {
        super::instance()?;
        Ok(())
    }
}
