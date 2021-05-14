use super::{BufferId, ComputePass, DeviceResult as Result, ModuleId, Scalar, ShaderModule};
use lazy_static::lazy_static;
use std::{
    borrow::Cow,
    fmt::{self, Debug},
    future::Future,
};

// TODO: Fix issues on DX12 with odd lengthed u8 slices.

#[cfg(all(unix, not(any(target_os = "ios", target_os = "macos"))))]
use gfx_backend_vulkan::Backend as Vulkan;

#[cfg(any(target_os = "ios", target_os = "macos"))]
use gfx_backend_metal::Backend as Metal;

#[cfg(windows)]
use gfx_backend_dx12::Backend as DX12;

lazy_static! {
    static ref GPUS: Vec<Gpu> = Gpu::list();
}

#[derive(Clone)]
pub struct Gpu {
    hal: DynHalGpu,
}

impl Gpu {
    pub(super) fn new(index: usize) -> Option<Result<Self>> {
        GPUS.get(index).map(|x| Ok(x.clone()))
        //Gpu::new_impl(index)
    }
    #[allow(clippy::single_match)]
    fn new_impl(index: usize) -> Option<Result<Self>> {
        let mut dyn_hal_gpu = None;
        #[allow(unused_mut)]
        let mut num_adapters = 0;

        #[cfg(all(unix, not(any(target_os = "ios", target_os = "macos"))))]
        match HalGpu::<Vulkan>::new(index - num_adapters) {
            Ok(hal) => {
                dyn_hal_gpu.replace(hal.unwrap().into());
            }
            #[cfg(any(target_os = "ios", target_os = "macos", windows))]
            Err(n) => {
                num_adapters += n;
            }
            _ => (),
        }

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        match HalGpu::<Metal>::new(index - num_adapters) {
            Ok(hal) => {
                dyn_hal_gpu.replace(hal.unwrap().into());
            }
            #[cfg(windows)]
            Err(n) => {
                num_adapters += n;
            }
            _ => (),
        }

        #[cfg(windows)]
        match HalGpu::<DX12>::new(index - num_adapters) {
            Ok(hal) => {
                dyn_hal_gpu.replace(hal.unwrap().into());
            }
            _ => (),
            /*
            Err(n) => {
                num_adapters += n;
            }*/
        }

        dyn_hal_gpu.map(|hal| Ok(Self { hal }))
    }
    fn list() -> Vec<Self> {
        let mut gpus = Vec::new();
        for i in 0..8 {
            if let Some(Ok(gpu)) = Self::new_impl(i) {
                gpus.push(gpu);
            } else {
                break;
            }
        }
        gpus
    }
}

// DynDevice Methods
impl Gpu {
    pub(super) fn create_buffer(&self, size: usize) -> Result<BufferId> {
        self.hal.create_buffer(size)
    }
    pub(super) fn create_buffer_init<T: Scalar>(&self, data: Cow<[T]>) -> Result<BufferId> {
        self.hal.create_buffer_init(data)
    }
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn copy_buffer_to_buffer(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.hal
            .copy_buffer_to_buffer(src, src_offset, dst, dst_offset, len)
    }
    pub(super) fn drop_buffer(&self, id: BufferId) {
        self.hal.drop_buffer(id);
    }
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn read_buffer<T: Scalar>(
        &self,
        id: BufferId,
        offset: usize,
        len: usize,
    ) -> Result<impl Future<Output = Result<Vec<T>>>> {
        Ok(self.hal.read_buffer(id, offset, len))
    }
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn compile_shader_module(&self, id: ModuleId, module: &ShaderModule) -> Result<()> {
        self.hal.compile_shader_module(id, module)
    }
    pub(super) fn enqueue_compute_pass(&self, compute_pass: ComputePass) -> Result<()> {
        self.hal.enqueue_compute_pass(compute_pass)
    }
    pub(super) fn synchronize(&self) -> Result<impl Future<Output = Result<()>>> {
        self.hal.synchronize()
    }
}

impl Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.hal.fmt(f)
    }
}

#[cfg(all(unix, not(any(target_os = "ios", target_os = "macos"))))]
type VulkanGpu = HalGpu<Vulkan>;

#[cfg(any(target_os = "ios", target_os = "macos"))]
type MetalGpu = HalGpu<Metal>;

#[cfg(windows)]
type DX12Gpu = HalGpu<DX12>;

#[cfg_attr(
    all(unix, not(any(target_os = "ios", target_os = "macos"))),
    proxy_enum::proxy(LinuxDynHalGpu)
)]
#[cfg_attr(
    any(target_os = "ios", target_os = "macos"),
    proxy_enum::proxy(AppleDynHalGpu)
)]
#[cfg_attr(windows, proxy_enum::proxy(WindowsDynHalGpu))]
pub mod dyn_hal_gpu_proxy {
    use super::hal::Result;
    use super::*;

    #[cfg(all(unix, not(any(target_os = "ios", target_os = "macos"))))]
    #[derive(Clone)]
    pub enum LinuxDynHalGpu {
        Vulkan(VulkanGpu),
    }

    #[cfg(all(unix, not(any(target_os = "ios", target_os = "macos"))))]
    pub type DynHalGpu = LinuxDynHalGpu;

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    #[derive(Clone)]
    pub enum AppleDynHalGpu {
        // TODO: The futures need to be the same type
        //#[cfg(feature = "gfx-backend-vulkan")]
        //Vulkan(VulkanGpu),
        Metal(MetalGpu),
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub type DynHalGpu = AppleDynHalGpu;

    #[cfg(windows)]
    #[derive(Clone)]
    pub enum WindowsDynHalGpu {
        // TODO: The futures need to be the same type
        //Vulkan(VulkanGpu),
        DX12(DX12Gpu),
    }

    #[cfg(windows)]
    pub type DynHalGpu = WindowsDynHalGpu;

    // DynDevice Methods
    impl DynHalGpu {
        #[implement]
        pub(super) fn create_buffer(&self, size: usize) -> Result<BufferId> {}
        #[implement]
        pub(super) fn create_buffer_init<T: Scalar>(&self, data: Cow<[T]>) -> Result<BufferId> {}
        #[implement]
        pub(super) fn copy_buffer_to_buffer(
            &self,
            src: BufferId,
            src_offset: usize,
            dst: BufferId,
            dst_offset: usize,
            len: usize,
        ) -> Result<()> {
        }
        #[implement]
        pub(super) fn drop_buffer(&self, id: BufferId) {}
        #[implement]
        pub(super) fn read_buffer<T: Scalar>(
            &self,
            id: BufferId,
            offset: usize,
            len: usize,
        ) -> impl Future<Output = Result<Vec<T>>> {
        }
        #[implement]
        pub(super) fn compile_shader_module(
            &self,
            id: ModuleId,
            module: &ShaderModule,
        ) -> Result<()> {
        }
        #[implement]
        pub(super) fn enqueue_compute_pass(&self, compute_pass: ComputePass) -> Result<()> {}
        #[implement]
        pub(super) fn synchronize(&self) -> Result<impl Future<Output = Result<()>>> {}
    }

    #[external(Debug)]
    trait Debug {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
    }

    #[implement]
    impl Debug for DynHalGpu {}
}
use dyn_hal_gpu_proxy::DynHalGpu;

pub mod hal {
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    use super::Metal;
    #[cfg(windows)]
    use super::DX12;
    use crate::backend::{
        BufferBinding, BufferId, ComputePass, DeviceError as GpuError, EntryDescriptor, EntryId,
        ModuleId, Scalar, ShaderModule, MAX_BUFFERS_PER_COMPUTE_PASS,
    };
    #[cfg(any(target_os = "ios", target_os = "macos", windows))]
    use crate::util::type_eq;
    use futures_channel::oneshot::{channel, Receiver, Sender};
    use gfx_hal::{
        adapter::{Adapter, MemoryProperties, PhysicalDevice},
        buffer::{Access, CreationError as BufferCreationError, SubRange, Usage as BufferUsage},
        command::{BufferCopy, CommandBuffer, CommandBufferFlags, Level as CommandLevel},
        device::{
            AllocationError, BindError, CreationError as DeviceCreationError, Device, DeviceLost,
            MapError, OutOfMemory, ShaderError,
        },
        memory::{Barrier, Dependencies, Properties, Segment},
        pool::{CommandPool, CommandPoolCreateFlags},
        prelude::DescriptorPool,
        pso::{
            BufferDescriptorFormat, BufferDescriptorType, ComputePipelineDesc, CreationError,
            Descriptor, DescriptorPoolCreateFlags, DescriptorRangeDesc, DescriptorSetLayoutBinding,
            DescriptorSetWrite, DescriptorType, EntryPoint, PipelineStage, ShaderStageFlags,
        },
        queue::{CommandQueue, QueueFamily},
        Backend, Features, Instance, Limits, MemoryTypeId,
    };
    use hibitset::BitSet;
    use smol::lock::Mutex;
    use std::{
        borrow::Cow,
        cmp::min,
        collections::hash_map::HashMap,
        fmt::{self, Debug},
        future::Future,
        iter::once,
        marker::PhantomData,
        mem::{size_of, swap, take, transmute, ManuallyDrop},
        ops::{Deref, Range},
        sync::Arc,
    };

    const MAX_CHUNK_SIZE: usize = 2 << 24;
    const MAX_BLOCKS_PER_CHUNK: usize =
        size_of::<usize>() * size_of::<usize>() * size_of::<usize>() * size_of::<usize>();
    const MIN_READ_SIZE: usize = 2 << 12;
    const MIN_WRITE_SIZE: usize = 2 << 12;

    impl From<OutOfMemory> for GpuError {
        fn from(from: OutOfMemory) -> Self {
            match from {
                OutOfMemory::Host => Self::OutOfHostMemory,
                OutOfMemory::Device => Self::OutOfDeviceMemory,
            }
        }
    }

    impl From<DeviceLost> for GpuError {
        fn from(_from: DeviceLost) -> Self {
            Self::DeviceLost
        }
    }

    impl From<DeviceCreationError> for GpuError {
        fn from(from: DeviceCreationError) -> Self {
            match from {
                DeviceCreationError::OutOfMemory(out_of_memory) => out_of_memory.into(),
                DeviceCreationError::InitializationFailed => Self::InitializationFailed,
                DeviceCreationError::MissingExtension => Self::MissingExtension,
                DeviceCreationError::MissingFeature => Self::MissingFeature,
                DeviceCreationError::TooManyObjects => Self::TooManyObjects,
                DeviceCreationError::DeviceLost => Self::DeviceLost,
            }
        }
    }

    impl From<BufferCreationError> for GpuError {
        fn from(from: BufferCreationError) -> Self {
            match from {
                BufferCreationError::OutOfMemory(out_of_memory) => out_of_memory.into(),
                BufferCreationError::UnsupportedUsage { .. } => unreachable!("{:?}", from),
            }
        }
    }

    impl From<AllocationError> for GpuError {
        fn from(from: AllocationError) -> Self {
            match from {
                AllocationError::OutOfMemory(out_of_memory) => out_of_memory.into(),
                AllocationError::TooManyObjects => Self::TooManyObjects,
            }
        }
    }

    impl From<BindError> for GpuError {
        fn from(from: BindError) -> Self {
            match from {
                BindError::OutOfMemory(out_of_memory) => out_of_memory.into(),
                BindError::WrongMemory | BindError::OutOfBounds => unreachable!("{:?}", from),
            }
        }
    }

    pub(super) type Result<T, E = GpuError> = std::result::Result<T, E>;

    pub struct Gpu<B: Backend> {
        index: usize,
        name: String,
        device: B::Device,
        fence: ManuallyDrop<B::Fence>,
        context: ManuallyDrop<Mutex<Context<B>>>,
    }

    impl<B: Backend> Drop for Gpu<B> {
        fn drop(&mut self) {
            unsafe {
                let fence = ManuallyDrop::take(&mut self.fence);
                self.device.destroy_fence(fence);
                ManuallyDrop::take(&mut self.context)
                    .into_inner()
                    .free(&self.device);
            }
        }
    }

    #[derive(Clone)]
    pub struct ArcGpu<B: Backend>(Arc<Gpu<B>>);

    impl<B: Backend> ArcGpu<B> {
        pub(super) fn new(index: usize) -> Result<Result<Self>, usize> {
            let app = "autograph";
            let version = 0;
            let instance = B::Instance::create(&app, version).map_err(|_| 0usize)?;
            use gfx_hal::adapter::DeviceType::*;
            let adapters: Vec<_> = instance
                .enumerate_adapters()
                .into_iter()
                .filter(|x| matches!(x.info.device_type, IntegratedGpu | DiscreteGpu))
                .collect();
            let adapter = adapters.get(index).ok_or(adapters.len())?;
            Ok(Self::with_adapter(index, adapter))
        }
        fn with_adapter(index: usize, adapter: &Adapter<B>) -> Result<Self> {
            let name = adapter.info.name.clone();
            let queue_family = adapter
                .queue_families
                .iter()
                .find(|f| {
                    let t = f.queue_type();
                    t.supports_compute() && t.supports_transfer()
                })
                .ok_or(GpuError::DeviceUnsupported)?;
            let gpu = unsafe {
                adapter
                    .physical_device
                    .open(&[(queue_family, &[1.])], Features::empty())?
            };
            let command_queue = gpu
                .queue_groups
                .into_iter()
                .next()
                .ok_or(GpuError::DeviceUnsupported)?
                .queues
                .into_iter()
                .next()
                .ok_or(GpuError::DeviceUnsupported)?;
            let device = gpu.device;
            let command_pool = unsafe {
                device.create_command_pool(
                    queue_family.id(),
                    CommandPoolCreateFlags::RESET_INDIVIDUAL,
                )?
            };
            let memory_properties = adapter.physical_device.memory_properties();
            let limits = adapter.physical_device.limits();
            let fence = device.create_fence(true)?;
            let context = Context::new(command_queue, command_pool, &memory_properties, &limits);
            Ok(Self(Arc::new(Gpu {
                index,
                name,
                device,
                fence: ManuallyDrop::new(fence),
                context: ManuallyDrop::new(Mutex::new(context)),
            })))
        }
    }

    // DynDeviceMethods
    impl<B: Backend> ArcGpu<B> {
        pub(super) fn create_buffer(&self, size: usize) -> Result<BufferId> {
            smol::block_on(self.context.lock())
                .storage
                .alloc(&self.device, size)
                .map(|buffer| buffer.to_buffer_id())
        }
        pub(super) fn create_buffer_init<T: Scalar>(&self, data: Cow<[T]>) -> Result<BufferId> {
            let mut context = smol::block_on(self.context.lock());
            let size = data.len() * size_of::<T>();
            let buffer = context.storage.alloc(&self.device, size)?;
            let data = unsafe { WriteBufferData::new(data.into_owned()) };
            context.queued.push(Op::WriteBuffer {
                data,
                dst: buffer,
                dst_offset: 0,
            });
            Ok(buffer.to_buffer_id())
        }
        #[allow(clippy::unnecessary_wraps)]
        pub(super) fn copy_buffer_to_buffer(
            &self,
            src: BufferId,
            src_offset: usize,
            dst: BufferId,
            dst_offset: usize,
            len: usize,
        ) -> Result<()> {
            let src = StorageBuffer::from_buffer_id(src);
            let dst = StorageBuffer::from_buffer_id(dst);
            smol::block_on(self.context.lock())
                .queued
                .push(Op::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size: len,
                });
            Ok(())
        }
        pub(super) fn drop_buffer(&self, id: BufferId) {
            smol::block_on(self.context.lock())
                .storage
                .dealloc(StorageBuffer::from_buffer_id(id));
        }
        // TODO: With multiple backends, the futures are not of the same type
        pub(super) fn read_buffer<T: Scalar>(
            &self,
            id: BufferId,
            offset: usize,
            len: usize,
        ) -> impl Future<Output = Result<Vec<T>>> {
            let size = len * size_of::<T>();
            let (sender, receiver) = channel();
            smol::block_on(self.context.lock())
                .queued
                .push(Op::ReadBuffer {
                    src: StorageBuffer::from_buffer_id(id),
                    src_offset: offset,
                    sender: unsafe { ReadBufferSender::new(sender) },
                    size,
                });
            let gpu = self.clone();
            async move {
                let mut receiver = receiver;
                match receiver.try_recv() {
                    Ok(Some(result)) => result,
                    Ok(None) => {
                        gpu.synchronize()?.await?;
                        match receiver.await {
                            Ok(result) => result,
                            Err(e) => todo!("{:?}", e), // maybe unreachable?
                        }
                    }
                    Err(e) => todo!("{:?}", e),
                }
            }
        }
        #[allow(clippy::unnecessary_wraps)]
        pub(super) fn compile_shader_module(
            &self,
            id: ModuleId,
            module: &ShaderModule,
        ) -> Result<()> {
            let shaders = match Shader::create_shaders(&self.device, module) {
                Ok(shaders) => shaders,
                Err(e) => todo!("{:?}", e),
            };
            smol::block_on(self.context.lock())
                .shaders
                .insert(id, shaders);
            Ok(())
        }
        #[allow(clippy::unnecessary_wraps)]
        pub(super) fn enqueue_compute_pass(&self, compute_pass: ComputePass) -> Result<()> {
            smol::block_on(self.context.lock())
                .queued
                .push(Op::ComputePass {
                    module_id: compute_pass.module_id,
                    entry_id: compute_pass.entry_id,
                    buffer_bindings: compute_pass.buffer_bindings,
                    push_constants: compute_pass.push_constants,
                    work_groups: compute_pass.work_groups,
                });
            Ok(())
        }
        // TODO: With multiple backends, the futures are not of the same type
        pub(super) fn synchronize(&self) -> Result<impl Future<Output = Result<()>>> {
            let completion =
                smol::block_on(self.context.lock()).submit(&self.device, &self.fence)?;
            let gpu = self.clone();
            Ok(async move {
                let mut completion = completion;
                loop {
                    let completed = completion.try_recv().ok().flatten().is_some();
                    if completed {
                        return Ok(());
                    }
                    let ready = unsafe { gpu.device.get_fence_status(&gpu.fence)? };
                    if ready {
                        let mut context = smol::block_on(gpu.context.lock());
                        let completed = completion.try_recv().ok().flatten().is_some();
                        if !completed {
                            context.on_completion(&gpu.device)?;
                        }
                        return Ok(());
                    }
                    smol::future::yield_now().await;
                }
            })
        }
    }

    impl<B: Backend> Deref for ArcGpu<B> {
        type Target = Gpu<B>;
        fn deref(&self) -> &Gpu<B> {
            &self.0
        }
    }

    impl<B: Backend> Debug for ArcGpu<B> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("Gpu")
                .field("index", &self.index)
                .field("name", &self.name)
                .finish()
        }
    }

    pub struct Context<B: Backend> {
        read: ReadAllocator<B>,
        read_size: usize,
        write: WriteAllocator<B>,
        storage: StorageAllocator<B>,
        shaders: HashMap<ModuleId, Vec<Shader<B>>>,
        descriptor_pool: Option<B::DescriptorPool>,
        num_descriptors: usize,
        descriptor_sets: HashMap<
            (
                ModuleId,
                EntryId,
                [BufferBinding; MAX_BUFFERS_PER_COMPUTE_PASS],
            ),
            Option<B::DescriptorSet>,
        >,
        command_pool: B::CommandPool,
        command_buffer: B::CommandBuffer,
        command_queue: B::CommandQueue,
        queued: Vec<Op>,
        pending: Vec<Op>,
        completion: Option<Sender<()>>,
    }

    impl<B: Backend> Context<B> {
        fn new(
            command_queue: B::CommandQueue,
            mut command_pool: B::CommandPool,
            memory_properties: &MemoryProperties,
            limits: &Limits,
        ) -> Self {
            let read = ReadAllocator::new(memory_properties, limits);
            let read_size = 0;
            let write = WriteAllocator::new(memory_properties, limits);
            let storage = StorageAllocator::new(&memory_properties, &limits);
            let shaders = HashMap::new();
            let descriptor_pool = None;
            let num_descriptors = 0;
            let descriptor_sets = HashMap::new();
            let command_buffer = unsafe { command_pool.allocate_one(CommandLevel::Primary) };
            let queued = Vec::new();
            let pending = Vec::new();
            let completion = None;
            Self {
                read,
                read_size,
                write,
                storage,
                shaders,
                descriptor_pool,
                num_descriptors,
                descriptor_sets,
                command_pool,
                command_buffer,
                command_queue,
                queued,
                pending,
                completion,
            }
        }
        fn submit(&mut self, device: &B::Device, fence: &B::Fence) -> Result<Receiver<()>> {
            unsafe fn binding_key(
                binding: &[BufferBinding],
            ) -> [BufferBinding; MAX_BUFFERS_PER_COMPUTE_PASS] {
                let mut key: [BufferBinding; MAX_BUFFERS_PER_COMPUTE_PASS] = std::mem::zeroed();
                for (b, k) in binding.iter().zip(key.iter_mut()) {
                    *k = *b;
                }
                key
            }
            if self.completion.is_some() {
                self.on_completion(device)?;
            }
            self.read_size = 0;
            let mut write_size = 0;
            let mut num_descriptors = 0;
            let num_descriptor_sets = self.descriptor_sets.capacity();
            self.descriptor_sets.clear();
            for op in self.queued.iter() {
                match op {
                    Op::CopyBuffer { .. } => (),
                    Op::ReadBuffer { size, .. } => {
                        self.read_size += size;
                    }
                    Op::WriteBuffer { data, .. } => {
                        write_size += data.len();
                    }
                    Op::ComputePass {
                        module_id,
                        entry_id,
                        buffer_bindings,
                        ..
                    } => {
                        let key = unsafe { binding_key(&buffer_bindings) };
                        self.descriptor_sets
                            .entry((*module_id, *entry_id, key))
                            .or_insert_with(|| {
                                num_descriptors += buffer_bindings.len();
                                None
                            });
                    }
                }
            }
            if self.read_size > 0 {
                self.read
                    .reserve(device, self.read_size.max(MIN_READ_SIZE))?;
            }
            if write_size > 0 {
                self.write.reserve(device, write_size.max(MIN_WRITE_SIZE))?;
                let mut write_guard = self.write.map(device, write_size)?;
                let mut write_slice = unsafe { write_guard.as_write_slice() };
                for op in self.queued.iter() {
                    if let Op::WriteBuffer { data, .. } = op {
                        write_slice[..data.len()].copy_from_slice(data);
                        write_slice = &mut write_slice[data.len()..];
                    }
                }
                debug_assert!(write_slice.is_empty());
                unsafe {
                    write_guard.flush()?;
                }
            }
            if num_descriptors > 0 {
                if num_descriptors > self.num_descriptors
                    || self.descriptor_sets.len() > num_descriptor_sets
                    || self.descriptor_pool.is_none()
                {
                    if let Some(descriptor_pool) = self.descriptor_pool.take() {
                        unsafe {
                            device.destroy_descriptor_pool(descriptor_pool);
                        }
                    }
                    let range_desc = DescriptorRangeDesc {
                        ty: DescriptorType::Buffer {
                            ty: BufferDescriptorType::Storage { read_only: false },
                            format: BufferDescriptorFormat::Structured {
                                dynamic_offset: false,
                            },
                        },
                        count: num_descriptors,
                    };
                    let descriptor_pool = unsafe {
                        device.create_descriptor_pool(
                            self.descriptor_sets.capacity(),
                            once(range_desc),
                            DescriptorPoolCreateFlags::empty(),
                        )?
                    };
                    self.descriptor_pool.replace(descriptor_pool);
                    self.num_descriptors = num_descriptors;
                } else {
                    let descriptor_pool = self.descriptor_pool.as_mut().unwrap();
                    unsafe {
                        descriptor_pool.reset();
                    }
                }
                let descriptor_pool = self.descriptor_pool.as_mut().unwrap();
                for op in self.queued.iter() {
                    if let Op::ComputePass {
                        module_id,
                        entry_id,
                        buffer_bindings,
                        ..
                    } = op
                    {
                        let module = self.shaders.get(module_id).unwrap();
                        let shader = module.get(entry_id.0 as usize).unwrap();
                        let key = unsafe { binding_key(&buffer_bindings) };
                        if let cached_set @ None = self
                            .descriptor_sets
                            .get_mut(&(*module_id, *entry_id, key))
                            .unwrap()
                        {
                            let descriptor_set = unsafe {
                                descriptor_pool
                                    .allocate_set(&shader.descriptor_set_layout)
                                    .unwrap()
                            };
                            for binding in buffer_bindings.iter() {
                                let buffer = StorageBuffer::from_buffer_id(binding.id);
                                let buffer = self.storage.get(&buffer);
                                let range = buffer.slice(
                                    binding.offset as usize
                                        ..(binding.offset + binding.len) as usize,
                                );
                                let sub_range = SubRange::from_range_usize(&range).corrected::<B>();
                                unsafe {
                                    device.write_descriptor_sets(once(DescriptorSetWrite {
                                        set: &descriptor_set,
                                        binding: binding.binding,
                                        array_offset: 0,
                                        descriptors: once(Descriptor::Buffer(
                                            buffer.deref(),
                                            sub_range,
                                        )),
                                    }));
                                }
                            }
                            cached_set.replace(descriptor_set);
                        }
                    }
                }
            }
            let command_buffer = &mut self.command_buffer;
            unsafe {
                command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            }
            let read_buffer = self.read.buffer.as_ref();
            let write_buffer = self.write.buffer.as_ref();
            let mut read_offset = 0;
            let mut write_offset = 0;
            let mut access_flags = HashMap::<BufferId, Access>::with_capacity(self.queued.len());
            // TODO: Implement complete barriers
            for op in self.queued.iter() {
                match op {
                    Op::CopyBuffer {
                        src,
                        src_offset,
                        dst,
                        dst_offset,
                        size,
                    } => {
                        let src_buffer = self.storage.get(src);
                        let src_range = src_buffer.slice(*src_offset..*src_offset + *size);
                        let dst_buffer = self.storage.get(dst);
                        let dst_range = dst_buffer.slice(*dst_offset..*dst_offset + *size);
                        access_flags
                            .entry(src.to_buffer_id())
                            .and_modify(|access| {
                                let sub_range =
                                    SubRange::from_range_usize(&src_range).corrected::<B>();
                                let stage = match *access {
                                    Access::TRANSFER_READ | Access::TRANSFER_WRITE => {
                                        PipelineStage::TRANSFER
                                    }
                                    Access::SHADER_READ | Access::SHADER_WRITE => {
                                        PipelineStage::COMPUTE_SHADER
                                    }
                                    _ => unreachable!(),
                                };
                                unsafe {
                                    command_buffer.pipeline_barrier(
                                        stage..PipelineStage::TRANSFER,
                                        Dependencies::empty(),
                                        once(Barrier::Buffer {
                                            states: *access..Access::TRANSFER_READ,
                                            target: src_buffer.deref(),
                                            range: sub_range,
                                            families: None,
                                        }),
                                    );
                                }
                                *access = Access::TRANSFER_READ;
                            })
                            .or_insert(Access::TRANSFER_READ);
                        access_flags
                            .entry(dst.to_buffer_id())
                            .and_modify(|access| {
                                let sub_range =
                                    SubRange::from_range_usize(&dst_range).corrected::<B>();
                                let stage = match *access {
                                    Access::TRANSFER_READ | Access::TRANSFER_WRITE => {
                                        PipelineStage::TRANSFER
                                    }
                                    Access::SHADER_READ | Access::SHADER_WRITE => {
                                        PipelineStage::COMPUTE_SHADER
                                    }
                                    _ => unreachable!(),
                                };
                                unsafe {
                                    command_buffer.pipeline_barrier(
                                        stage..PipelineStage::TRANSFER,
                                        Dependencies::empty(),
                                        once(Barrier::Buffer {
                                            states: *access..Access::TRANSFER_WRITE,
                                            target: dst_buffer.deref(),
                                            range: sub_range,
                                            families: None,
                                        }),
                                    );
                                }
                                *access = Access::TRANSFER_WRITE;
                            })
                            .or_insert(Access::TRANSFER_WRITE);
                        unsafe {
                            command_buffer.copy_buffer(
                                &src_buffer,
                                &dst_buffer,
                                once(BufferCopy {
                                    src: src_range.start as u64,
                                    dst: dst_range.start as u64,
                                    size: min(src_range.len(), dst_range.len()) as u64,
                                }),
                            );
                        }
                    }
                    Op::WriteBuffer {
                        data,
                        dst,
                        dst_offset,
                    } => {
                        let write_buffer = write_buffer.unwrap();
                        let storage_buffer = self.storage.get(dst);
                        let range = storage_buffer.slice(*dst_offset..*dst_offset + data.len());
                        let sub_range = SubRange::from_range_usize(&range).corrected::<B>();
                        unsafe {
                            command_buffer.pipeline_barrier(
                                PipelineStage::BOTTOM_OF_PIPE..PipelineStage::TRANSFER,
                                Dependencies::empty(),
                                once(Barrier::Buffer {
                                    states: Access::all()..Access::TRANSFER_WRITE,
                                    target: &*storage_buffer,
                                    range: sub_range,
                                    families: None,
                                }),
                            );
                        }
                        access_flags.insert(dst.to_buffer_id(), Access::TRANSFER_WRITE);
                        unsafe {
                            command_buffer.copy_buffer(
                                &write_buffer,
                                &storage_buffer,
                                once(BufferCopy {
                                    src: write_offset as u64,
                                    dst: range.start as u64,
                                    size: range.len() as u64,
                                }),
                            );
                        }
                        write_offset += data.len();
                    }
                    Op::ReadBuffer {
                        src,
                        src_offset,
                        size,
                        ..
                    } => {
                        let storage_buffer = self.storage.get(src);
                        let read_buffer = read_buffer.unwrap();
                        let range = storage_buffer.slice(*src_offset..*src_offset + *size);
                        let sub_range = SubRange::from_range_usize(&range).corrected::<B>();
                        access_flags
                            .entry(src.to_buffer_id())
                            .and_modify(|access| {
                                let stage = match *access {
                                    Access::TRANSFER_READ | Access::TRANSFER_WRITE => {
                                        PipelineStage::TRANSFER
                                    }
                                    Access::SHADER_READ | Access::SHADER_WRITE => {
                                        PipelineStage::COMPUTE_SHADER
                                    }
                                    _ => unreachable!(),
                                };
                                unsafe {
                                    command_buffer.pipeline_barrier(
                                        stage..PipelineStage::TRANSFER,
                                        Dependencies::empty(),
                                        once(Barrier::Buffer {
                                            states: *access..Access::TRANSFER_READ,
                                            target: &*storage_buffer,
                                            range: sub_range,
                                            families: None,
                                        }),
                                    );
                                }
                                *access = Access::TRANSFER_READ;
                            })
                            .or_insert(Access::TRANSFER_READ);
                        unsafe {
                            command_buffer.copy_buffer(
                                &storage_buffer,
                                &read_buffer,
                                once(BufferCopy {
                                    src: range.start as u64,
                                    dst: read_offset as u64,
                                    size: range.len() as u64,
                                }),
                            );
                        }
                        read_offset += range.len();
                    }
                    Op::ComputePass {
                        module_id,
                        entry_id,
                        buffer_bindings,
                        push_constants,
                        work_groups,
                    } => {
                        let module = self.shaders.get(module_id).unwrap();
                        let shader = module.get(entry_id.0 as usize).unwrap();
                        let key = unsafe { binding_key(&buffer_bindings) };
                        let descriptor_set = self
                            .descriptor_sets
                            .get(&(*module_id, *entry_id, key))
                            .unwrap()
                            .as_ref()
                            .unwrap();
                        for buffer_binding in buffer_bindings.iter() {
                            let buffer = StorageBuffer::from_buffer_id(buffer_binding.id);
                            let buffer = self.storage.get(&buffer);
                            let range = buffer.slice(
                                buffer_binding.offset as usize
                                    ..buffer_binding.offset as usize + buffer_binding.len as usize,
                            );
                            let sub_range = SubRange::from_range_usize(&range).corrected::<B>();
                            let buffer_access = if buffer_binding.mutable {
                                Access::SHADER_WRITE
                            } else {
                                Access::SHADER_READ
                            };
                            access_flags
                                .entry(buffer_binding.id)
                                .and_modify(|access| {
                                    let stage = match *access {
                                        Access::TRANSFER_READ | Access::TRANSFER_WRITE => {
                                            PipelineStage::TRANSFER
                                        }
                                        Access::SHADER_READ | Access::SHADER_WRITE => {
                                            PipelineStage::COMPUTE_SHADER
                                        }
                                        _ => unreachable!(),
                                    };
                                    unsafe {
                                        command_buffer.pipeline_barrier(
                                            stage..PipelineStage::COMPUTE_SHADER,
                                            Dependencies::empty(),
                                            once(Barrier::Buffer {
                                                states: *access..buffer_access,
                                                target: buffer.deref(),
                                                range: sub_range,
                                                families: None,
                                            }),
                                        );
                                    }
                                    *access = buffer_access;
                                })
                                .or_insert(buffer_access);
                        }
                        unsafe {
                            command_buffer.bind_compute_pipeline(&shader.compute_pipeline);
                            command_buffer.bind_compute_descriptor_sets(
                                &shader.pipeline_layout,
                                0,
                                once(descriptor_set),
                                &[],
                            );
                            if !push_constants.is_empty() {
                                command_buffer.push_compute_constants(
                                    &shader.pipeline_layout,
                                    0,
                                    bytemuck::cast_slice(push_constants),
                                );
                            }
                            command_buffer.dispatch(*work_groups);
                        }
                    }
                }
            }
            debug_assert_eq!(read_offset, self.read_size);
            debug_assert_eq!(write_offset, write_size);
            unsafe {
                self.command_buffer.finish();
                device.reset_fence(fence)?;
                self.command_queue
                    .submit_without_semaphores(once(&self.command_buffer), Some(fence));
            }
            let (sender, receiver) = channel();
            self.completion.replace(sender);
            swap(&mut self.queued, &mut self.pending);
            Ok(receiver)
        }
        fn on_completion(&mut self, device: &B::Device) -> Result<()> {
            self.command_queue.wait_idle()?;
            unsafe {
                self.command_buffer.reset(false);
            }
            let ops = take(&mut self.pending);
            if self.read_size > 0 {
                match self.read.map(device, self.read_size) {
                    Ok(guard) => {
                        let mut data = guard.as_slice();
                        for op in ops {
                            if let Op::ReadBuffer { sender, size, .. } = op {
                                sender.send(Ok(&data[..size]));
                                data = &data[size..];
                            }
                        }
                    }
                    Err(e) => {
                        for op in ops {
                            if let Op::ReadBuffer { sender, .. } = op {
                                sender.send(Err(e));
                            }
                        }
                    }
                }
                self.read_size = 0;
            }
            let _ = self.completion.take().unwrap().send(());
            Ok(())
        }
        unsafe fn free(self, device: &B::Device) {
            self.read.free(device);
            self.write.free(device);
            self.storage.free(device);
            self.shaders
                .into_iter()
                .flat_map(|(_, v)| v)
                .for_each(|x| x.free(device));
            if let Some(descriptor_pool) = self.descriptor_pool {
                device.destroy_descriptor_pool(descriptor_pool);
            }
            device.destroy_command_pool(self.command_pool);
        }
    }

    // DX12 expects words not bytes
    trait SubRangeExt: Sized {
        fn from_range_usize(range: &Range<usize>) -> Self;
        fn corrected<B: Backend>(self) -> Self;
    }

    impl SubRangeExt for SubRange {
        fn from_range_usize(range: &Range<usize>) -> Self {
            Self {
                offset: range.start as u64,
                size: Some(range.end as u64),
            }
        }
        fn corrected<B: Backend>(self) -> Self {
            #[cfg(windows)]
            if type_eq::<B, DX12>() {
                return Self {
                    offset: self.offset / 4,
                    size: self.size.map(|x| x + (x % 4)),
                };
            }
            self
        }
    }

    #[derive(Debug)]
    enum Op {
        CopyBuffer {
            src: StorageBuffer,
            src_offset: usize,
            dst: StorageBuffer,
            dst_offset: usize,
            size: usize,
        },
        WriteBuffer {
            data: WriteBufferData,
            dst: StorageBuffer,
            dst_offset: usize,
        },
        ReadBuffer {
            src: StorageBuffer,
            src_offset: usize,
            sender: ReadBufferSender,
            size: usize,
        },
        ComputePass {
            module_id: ModuleId,
            entry_id: EntryId,
            buffer_bindings: Vec<BufferBinding>,
            push_constants: Vec<u8>,
            work_groups: [u32; 3],
        },
    }

    #[allow(non_camel_case_types)]
    #[derive(Clone, Copy)]
    pub struct u24([u8; 3]);

    impl u24 {
        fn from_u32(x: u32) -> Self {
            let [a, b, c, _] = x.to_ne_bytes();
            Self([a, b, c])
        }
        fn to_u32(&self) -> u32 {
            let [a, b, c] = self.0;
            u32::from_ne_bytes([a, b, c, 0])
        }
    }

    impl Debug for u24 {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.to_u32().fmt(f)
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct StorageBuffer {
        chunk: u16,
        start: u24,
        end: u24,
    }

    impl StorageBuffer {
        fn new(chunk: u16, range: Range<u24>) -> Self {
            Self {
                chunk,
                start: range.start,
                end: range.end,
            }
        }
        fn range(&self) -> Range<u24> {
            self.start..self.end
        }
        fn to_buffer_id(&self) -> BufferId {
            unsafe { transmute(*self) }
        }
        fn from_buffer_id(buffer_id: BufferId) -> Self {
            unsafe { transmute(buffer_id) }
        }
    }

    #[derive(Debug)]
    pub struct StorageChunk<B: Backend> {
        buffer: B::Buffer,
        memory: B::Memory,
        blocks: BitSet,
        blocks_per_chunk: usize,
        block_size: usize,
    }

    impl<B: Backend> StorageChunk<B> {
        fn new(
            device: &B::Device,
            memory_type_id: MemoryTypeId,
            blocks_per_chunk: usize,
            block_size: usize,
        ) -> Result<Self> {
            let chunk_size = blocks_per_chunk * block_size;
            let (buffer, memory) = unsafe {
                let usage =
                    BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST;
                let mut buffer = device.create_buffer(chunk_size as u64, usage)?;
                let requirements = device.get_buffer_requirements(&buffer);
                let memory = match device.allocate_memory(memory_type_id, requirements.size) {
                    Ok(memory) => memory,
                    Err(e) => {
                        device.destroy_buffer(buffer);
                        return Err(e.into());
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
                (buffer, memory)
            };
            let blocks = BitSet::with_capacity(blocks_per_chunk as u32);
            Ok(Self {
                buffer,
                memory,
                blocks,
                blocks_per_chunk,
                block_size,
            })
        }
        fn alloc(&mut self, size: usize) -> Option<Range<u24>> {
            let num_blocks = if size % self.block_size == 0 {
                size / self.block_size
            } else {
                size / self.block_size + 1
            };
            'outer: for start in 0..=(self.blocks_per_chunk - num_blocks) as u32 {
                let end = start + num_blocks as u32;
                for block in start..end {
                    if self.blocks.contains(block) {
                        continue 'outer;
                    }
                }
                self.blocks.extend(start..end);
                let start = u24::from_u32(start);
                let end = u24::from_u32(end);
                return Some(start..end);
            }
            None
        }
        fn dealloc(&mut self, range: Range<u24>) {
            for i in range.start.to_u32()..range.end.to_u32() {
                self.blocks.remove(i);
            }
        }
        unsafe fn free(self, device: &B::Device) {
            device.destroy_buffer(self.buffer);
            device.free_memory(self.memory);
        }
    }

    #[derive(Debug)]
    pub struct StorageAllocator<B: Backend> {
        chunks: Vec<StorageChunk<B>>,
        memory_type_id: MemoryTypeId,
        blocks_per_chunk: usize,
        block_size: usize,
    }

    impl<B: Backend> StorageAllocator<B> {
        fn new(memory_properties: &MemoryProperties, limits: &Limits) -> Self {
            let memory_type_id = memory_properties
                .memory_types
                .iter()
                .enumerate()
                .find_map(|(i, x)| {
                    if x.properties == Properties::DEVICE_LOCAL {
                        Some(MemoryTypeId(i))
                    } else {
                        None
                    }
                })
                .unwrap();
            let chunk_size = min(MAX_CHUNK_SIZE, limits.max_storage_buffer_range as usize);
            let block_align = limits.min_storage_buffer_offset_alignment as usize;
            let blocks_per_chunk = min(chunk_size / block_align, MAX_BLOCKS_PER_CHUNK);
            let block_size = chunk_size / blocks_per_chunk;
            Self {
                chunks: Vec::new(),
                memory_type_id,
                blocks_per_chunk,
                block_size,
            }
        }
        fn alloc(&mut self, device: &B::Device, size: usize) -> Result<StorageBuffer> {
            debug_assert!(size <= self.blocks_per_chunk * self.block_size); // TODO: error here?
            for (c, chunk) in self.chunks.iter_mut().enumerate() {
                if let Some(range) = chunk.alloc(size) {
                    return Ok(StorageBuffer::new(c as u16, range));
                }
            }
            let mut chunk = StorageChunk::new(
                device,
                self.memory_type_id,
                self.blocks_per_chunk,
                self.block_size,
            )?;
            let range = chunk.alloc(size).unwrap();
            let c = self.chunks.len();
            self.chunks.push(chunk);
            Ok(StorageBuffer::new(c as u16, range))
        }
        fn dealloc(&mut self, buffer: StorageBuffer) {
            if let Some(chunk) = self.chunks.get_mut(buffer.chunk as usize) {
                chunk.dealloc(buffer.range());
            }
        }
        fn get(&self, buffer: &StorageBuffer) -> StorageBufferGuard<'_, B> {
            let chunk = self.chunks.get(buffer.chunk as usize).unwrap();
            let start = chunk.block_size * buffer.start.to_u32() as usize;
            let end = chunk.block_size * buffer.end.to_u32() as usize;
            StorageBufferGuard {
                buffer: &chunk.buffer,
                range: start..end,
            }
        }
        unsafe fn free(self, device: &B::Device) {
            for chunk in self.chunks {
                chunk.free(device);
            }
        }
    }

    #[derive(Debug)]
    pub struct StorageBufferGuard<'a, B: Backend> {
        buffer: &'a B::Buffer,
        range: Range<usize>,
    }

    impl<B: Backend> StorageBufferGuard<'_, B> {
        fn slice(&self, range: Range<usize>) -> Range<usize> {
            debug_assert!(
                self.range.start + range.start < self.range.end,
                "{:?} {:?}",
                &self.range,
                &range
            );
            debug_assert!(
                self.range.start + range.end <= self.range.end,
                "{:?} {:?}",
                &self.range,
                &range
            );
            let start = min(self.range.start + range.start, self.range.end);
            let end = min(self.range.start + range.end, self.range.end);
            start..end
        }
    }

    impl<'a, B: Backend> Deref for StorageBufferGuard<'a, B> {
        type Target = B::Buffer;
        fn deref(&self) -> &B::Buffer {
            &self.buffer
        }
    }

    pub trait MappingMode {
        const USAGE: BufferUsage;
        fn properties() -> Vec<Properties>;
    }

    #[derive(Debug)]
    pub enum MappingRead {}

    impl MappingMode for MappingRead {
        const USAGE: BufferUsage = BufferUsage::TRANSFER_DST;
        fn properties() -> Vec<Properties> {
            use Properties as P;
            vec![
                P::CPU_VISIBLE | P::COHERENT | P::CPU_CACHED,
                P::CPU_VISIBLE | P::COHERENT,
            ]
        }
    }

    #[derive(Debug)]
    pub enum MappingWrite {}

    impl MappingMode for MappingWrite {
        const USAGE: BufferUsage = BufferUsage::TRANSFER_SRC;
        fn properties() -> Vec<Properties> {
            use Properties as P;
            vec![P::DEVICE_LOCAL | P::CPU_VISIBLE, P::CPU_VISIBLE]
        }
    }

    #[derive(Debug)]
    pub struct MappingAllocator<B: Backend, M> {
        memory: Option<B::Memory>,
        buffer: Option<B::Buffer>,
        memory_type_id: MemoryTypeId,
        capacity: usize,
        max_capacity: usize,
        _m: PhantomData<M>,
    }

    type ReadAllocator<B> = MappingAllocator<B, MappingRead>;
    type WriteAllocator<B> = MappingAllocator<B, MappingWrite>;

    impl<B: Backend, M: MappingMode> MappingAllocator<B, M> {
        fn new(memory_properties: &MemoryProperties, limits: &Limits) -> Self {
            let memory_type_id = M::properties()
                .into_iter()
                .filter_map(|properties| {
                    memory_properties
                        .memory_types
                        .iter()
                        .position(|x| x.properties.contains(properties))
                        .map(MemoryTypeId)
                })
                .next()
                .unwrap();
            let max_capacity = limits.max_storage_buffer_range as usize;
            Self {
                memory: None,
                buffer: None,
                memory_type_id,
                capacity: 0,
                max_capacity,
                _m: PhantomData::default(),
            }
        }
        fn reserve(&mut self, device: &B::Device, capacity: usize) -> Result<()> {
            if capacity > self.capacity {
                unsafe {
                    if let Some(memory) = self.memory.take() {
                        device.free_memory(memory);
                    }
                    if let Some(buffer) = self.buffer.take() {
                        device.destroy_buffer(buffer);
                    }
                    let mut buffer = device.create_buffer(capacity as u64, M::USAGE)?;
                    let requirements = device.get_buffer_requirements(&buffer);
                    let memory =
                        match device.allocate_memory(self.memory_type_id, requirements.size) {
                            Ok(memory) => memory,
                            Err(e) => {
                                device.destroy_buffer(buffer);
                                return Err(e.into());
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
                    self.capacity = capacity;
                    self.memory.replace(memory);
                    self.buffer.replace(buffer);
                }
            }
            Ok(())
        }
        fn map<'a>(
            &'a self,
            device: &'a B::Device,
            len: usize,
        ) -> Result<MappingAllocatorMapGuard<'a, B, M>> {
            debug_assert!(len <= self.capacity);
            let len = min(len, self.capacity);
            let segment = Segment {
                offset: 0,
                size: Some(len as u64),
            };
            let memory = self.memory.as_ref().unwrap();
            let data = unsafe {
                match device.map_memory(memory, segment) {
                    Ok(p) => std::slice::from_raw_parts_mut(p, len),
                    Err(MapError::OutOfMemory(out_of_memory)) => {
                        return Err(out_of_memory.into());
                    }
                    Err(MapError::MappingFailed) => {
                        return Err(GpuError::MappingFailed);
                    }
                    err => unreachable!("{:?}", err),
                }
            };
            Ok(MappingAllocatorMapGuard {
                device,
                memory,
                data,
                _m: PhantomData::default(),
            })
        }
        unsafe fn free(self, device: &B::Device) {
            if let Some(memory) = self.memory {
                device.free_memory(memory);
            }
            if let Some(buffer) = self.buffer {
                device.destroy_buffer(buffer);
            }
        }
    }

    #[derive(Debug)]
    struct MappingAllocatorMapGuard<'a, B: Backend, M> {
        device: &'a B::Device,
        memory: &'a B::Memory,
        data: &'a mut [u8],
        _m: PhantomData<M>,
    }

    type ReadAllocatorMapGuard<'a, B> = MappingAllocatorMapGuard<'a, B, MappingRead>;
    type WriteAllocatorMapGuard<'a, B> = MappingAllocatorMapGuard<'a, B, MappingWrite>;

    impl<B: Backend> ReadAllocatorMapGuard<'_, B> {
        fn as_slice(&self) -> &[u8] {
            &self.data
        }
    }

    impl<B: Backend> WriteAllocatorMapGuard<'_, B> {
        unsafe fn as_write_slice(&mut self) -> &mut [u8] {
            &mut self.data
        }
        unsafe fn flush(&self) -> Result<()> {
            let segment = Segment {
                offset: 0,
                size: Some(self.data.len() as u64),
            };
            self.device
                .flush_mapped_memory_ranges(once((self.memory, segment)))?;
            Ok(())
        }
    }

    impl<B: Backend, M> Drop for MappingAllocatorMapGuard<'_, B, M> {
        fn drop(&mut self) {
            unsafe {
                self.device.unmap_memory(self.memory);
            }
        }
    }
    /*
    // This system is a bit wild. Basically Vec requires that it be created and dropped with the same layout, ie size of T. As a safer alternative to transmute, we could use enums, one for each type. Ideally the backend should only interact with u8, but because of Vec's requirement we need to keep track of T. This solution, while ugly, is simpler than creating an enum for every single type, since we only need to support a known set of sizes.
    // The vec doesn't change, but is stored as a Vec<u8> via transmute. It is transmuted back to a T with the same size when sending and dropping. Because the size doesn't change, the length and capacity does not need to be updated to match the cast to u8, since it will be dropped with the appropriate size via casting back to Vec<T>.
    #[derive(Debug)]
    pub struct ReadBufferSender {
        sender: Sender<Result<Vec<u8>>>,
        size_of_t: usize,
    }

    impl ReadBufferSender {
        unsafe fn new<T: Scalar>(sender: Sender<Result<Vec<T>>>) -> Self {
            Self {
                sender: transmute(sender),
                size_of_t: size_of::<T>(),
            }
        }
        fn send(self, data: Result<&[u8]>) {
            #[allow(clippy::unsound_collection_transmute)]
            let vec: Result<Vec<u8>> = data.map(|x| match self.size_of_t {
                1 => x.to_vec(),
                2 => {
                    let y: Vec<u16> = bytemuck::cast_slice(x).to_vec();
                    unsafe { transmute(y) }
                }
                4 => {
                    let y: Vec<u32> = bytemuck::cast_slice(x).to_vec();
                    unsafe { transmute(y) }
                }
                8 => {
                    let y: Vec<u64> = bytemuck::cast_slice(x).to_vec();
                    unsafe { transmute(y) }
                }
                _ => unreachable!(),
            });
            let _result = self.sender.send(vec);
        }
    }
    */

    #[derive(Debug)]
    enum ReadBufferSender {
        U8(Sender<Result<Vec<u8>>>),
        U16(Sender<Result<Vec<u16>>>),
        U32(Sender<Result<Vec<u32>>>),
        U64(Sender<Result<Vec<u64>>>),
    }

    impl ReadBufferSender {
        unsafe fn new<T: Scalar>(sender: Sender<Result<Vec<T>>>) -> Self {
            match size_of::<T>() {
                1 => Self::U8(transmute(sender)),
                2 => Self::U16(transmute(sender)),
                4 => Self::U32(transmute(sender)),
                8 => Self::U64(transmute(sender)),
                _ => unreachable!(),
            }
        }
        fn send(self, data: Result<&[u8]>) {
            fn send_impl<T: bytemuck::Pod>(sender: Sender<Result<Vec<T>>>, data: Result<&[u8]>) {
                let data = data.map(|slice| {
                    let len = slice.len() / size_of::<T>();
                    debug_assert_eq!(len * size_of::<T>(), slice.len());
                    let mut vec = Vec::with_capacity(len);
                    unsafe {
                        vec.set_len(len);
                    }
                    bytemuck::cast_slice_mut(vec.as_mut_slice()).copy_from_slice(slice);
                    vec
                });
                let _ = sender.send(data);
            }
            match self {
                Self::U8(sender) => send_impl(sender, data),
                Self::U16(sender) => send_impl(sender, data),
                Self::U32(sender) => send_impl(sender, data),
                Self::U64(sender) => send_impl(sender, data),
            }
        }
    }

    pub struct WriteBufferData {
        data: ManuallyDrop<Vec<u8>>,
        size_of_t: usize,
    }

    impl WriteBufferData {
        unsafe fn new<T: Scalar>(data: Vec<T>) -> Self {
            Self {
                data: ManuallyDrop::new(transmute(data)),
                size_of_t: size_of::<T>(),
            }
        }
    }

    impl Drop for WriteBufferData {
        fn drop(&mut self) {
            unsafe {
                let data = ManuallyDrop::take(&mut self.data);
                #[allow(clippy::unsound_collection_transmute)]
                match self.size_of_t {
                    1 => (),
                    2 => {
                        let _data: Vec<u16> = transmute(data);
                    }
                    4 => {
                        let _data: Vec<u32> = transmute(data);
                    }
                    8 => {
                        let _data: Vec<u64> = transmute(data);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    impl Deref for WriteBufferData {
        type Target = [u8];
        fn deref(&self) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(self.data.as_ptr(), self.data.len() * self.size_of_t)
            }
        }
    }

    impl Debug for WriteBufferData {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("WriteBufferData")
                .field("len", &self.len())
                .field("size_of_t", &self.size_of_t)
                .finish()
        }
    }

    #[derive(Debug)]
    pub struct Shader<B: Backend> {
        descriptor_set_layout: B::DescriptorSetLayout,
        pipeline_layout: B::PipelineLayout,
        compute_pipeline: B::ComputePipeline,
    }

    impl<B: Backend> Shader<B> {
        fn new(
            device: &B::Device,
            shader_module: &B::ShaderModule,
            entry: &EntryDescriptor,
        ) -> Result<Self> {
            let len = entry
                .buffer_descriptors
                .len()
                .min(MAX_BUFFERS_PER_COMPUTE_PASS);
            let buffer_descriptors = &entry.buffer_descriptors[..len];
            let bindings: Vec<_> = buffer_descriptors
                .iter()
                .map(|buffer| DescriptorSetLayoutBinding {
                    binding: buffer.binding,
                    ty: DescriptorType::Buffer {
                        ty: BufferDescriptorType::Storage {
                            read_only: !buffer.mutable,
                        },
                        format: BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                })
                .collect();
            let descriptor_set_layout =
                unsafe { device.create_descriptor_set_layout(&bindings, &[])? };
            let push_constant_range = entry
                .push_constant_descriptor
                .as_ref()
                .map(|x| (ShaderStageFlags::COMPUTE, x.range.into()));
            let pipeline_layout_result = unsafe {
                device.create_pipeline_layout(
                    once(&descriptor_set_layout),
                    push_constant_range.as_ref(),
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
            let entry_point = EntryPoint {
                entry: &entry.name,
                module: shader_module,
                specialization: Default::default(),
            };
            let compute_pipeline_result = unsafe {
                device.create_compute_pipeline(
                    &ComputePipelineDesc::new(entry_point, &pipeline_layout),
                    None,
                )
            };
            let compute_pipeline = match compute_pipeline_result {
                Ok(compute_pipeline) => compute_pipeline,
                Err(e) => {
                    unsafe {
                        device.destroy_pipeline_layout(pipeline_layout);
                        device.destroy_descriptor_set_layout(descriptor_set_layout);
                    }
                    match e {
                        // TODO add Other ie unknown error?
                        // CreationError::Other => ...
                        CreationError::OutOfMemory(out_of_memory) => {
                            return Err(out_of_memory.into());
                        }
                        _ => unimplemented!("{:?}", e),
                    }
                }
            };
            Ok(Self {
                descriptor_set_layout,
                pipeline_layout,
                compute_pipeline,
            })
        }
        fn create_shaders(device: &B::Device, module: &ShaderModule) -> Result<Vec<Self>> {
            let shader_module_result =
                unsafe { device.create_shader_module(bytemuck::cast_slice(&module.spirv)) };
            let shader_module = match shader_module_result {
                Ok(module) => module,
                Err(ShaderError::OutOfMemory(out_of_memory)) => {
                    return Err(out_of_memory.into());
                }
                Err(e) => unreachable!("{:?}", e),
            };
            let mut shaders = Vec::with_capacity(module.entry_descriptors.len());
            for entry in module.entry_descriptors.iter() {
                match Self::new(device, &shader_module, entry) {
                    Ok(shader) => {
                        shaders.push(shader);
                    }
                    Err(e) => {
                        for shader in shaders {
                            unsafe {
                                shader.free(device);
                            }
                        }
                        return Err(e);
                    }
                }
            }
            Ok(shaders)
        }
        unsafe fn free(self, device: &B::Device) {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout);
            device.destroy_pipeline_layout(self.pipeline_layout);
            device.destroy_compute_pipeline(self.compute_pipeline);
        }
    }
}
use hal::ArcGpu as HalGpu;
