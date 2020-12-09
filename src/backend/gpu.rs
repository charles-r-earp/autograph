use super::{BufferId, ComputePass, Cow, Future, ModuleId, ShaderModule};
use super::{MAX_BUFFERS_PER_COMPUTE_PASS, MAX_PUSH_CONSTANT_SIZE};
use crate::error::GpuError;
use crate::Result;
use ahash::RandomState;
use bytemuck::Pod;
use smol::lock::Mutex;
use std::fmt::{self, Debug};
use std::mem::{replace, size_of};
use std::sync::Arc;
use wgpu::{
    AdapterInfo, BackendBit, BindGroupLayout, Buffer, BufferAddress, ComputePipeline, Device,
    Instance, Queue,
};
use wgpu::{BufferDescriptor, BufferUsage, DeviceDescriptor, Features, Limits, Maintain, MapMode};
use wgpu::{CommandBuffer, CommandEncoderDescriptor};

type DashMap<K, V, S = RandomState> = dashmap::DashMap<K, V, S>;
type Entry<'a, K, V, S = RandomState> = dashmap::mapref::entry::Entry<'a, K, V, S>;
type VacantEntry<'a, K, V, S = RandomState> = dashmap::mapref::entry::VacantEntry<'a, K, V, S>;

pub struct GpuBase {
    host_name: Option<String>,
    index: usize,
    #[allow(unused)]
    adapter_info: AdapterInfo,
    device: Device,
    queue: Queue,
    buffers: DashMap<BufferId, Buffer>,
    compute_pipelines: DashMap<ModuleId, Vec<(BindGroupLayout, ComputePipeline)>>,
    stream: Mutex<Stream>,
}

impl GpuBase {
    fn new(
        host_name: Option<String>,
        index: usize,
        adapter_info: AdapterInfo,
        device: Device,
        queue: Queue,
    ) -> Self {
        let label = format!(
            "{:?}",
            GpuPrinter {
                host_name: host_name.as_deref(),
                index
            }
        );
        Self {
            host_name,
            index,
            adapter_info,
            device,
            queue,
            buffers: DashMap::with_hasher(RandomState::default()),
            compute_pipelines: DashMap::with_hasher(RandomState::default()),
            stream: Mutex::new(Stream::new(label)),
        }
    }
}

struct GpuPrinter<'a> {
    host_name: Option<&'a str>,
    index: usize,
}

impl Debug for GpuPrinter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("Gpu");
        if let Some(host_name) = self.host_name.as_ref() {
            builder.field("host_name", host_name);
        }
        builder.field("index", &self.index);
        builder.finish()
    }
}

#[derive(Clone)]
pub struct Gpu {
    base: Arc<GpuBase>,
}

impl Gpu {
    fn backend_bit() -> BackendBit {
        BackendBit::PRIMARY
    }
    fn features() -> Features {
        Features::PUSH_CONSTANTS
    }
    fn limits() -> Limits {
        Limits {
            max_bind_groups: 1,
            max_dynamic_uniform_buffers_per_pipeline_layout: 0,
            max_dynamic_storage_buffers_per_pipeline_layout: 0,
            max_sampled_textures_per_shader_stage: 0,
            max_samplers_per_shader_stage: 0,
            max_storage_buffers_per_shader_stage: MAX_BUFFERS_PER_COMPUTE_PASS as u32,
            max_storage_textures_per_shader_stage: 0,
            max_uniform_buffers_per_shader_stage: 0,
            max_uniform_buffer_binding_size: 0,
            max_push_constant_size: MAX_PUSH_CONSTANT_SIZE as u32,
        }
    }
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(index: usize) -> Option<impl Future<Output = Result<Self>>> {
        let backend_bit = Self::backend_bit();
        let instance = Instance::new(backend_bit);
        instance
            .enumerate_adapters(backend_bit)
            .nth(index)
            .map(|adapter| {
                let desc = DeviceDescriptor {
                    features: Self::features(),
                    limits: Self::limits(),
                    shader_validation: true,
                };
                // TODO: Potentially use this
                let trace_path = None;
                async move {
                    let (device, queue) = adapter.request_device(&desc, trace_path).await?;
                    let host_name = None;
                    Ok(Self {
                        base: Arc::new(GpuBase::new(
                            host_name,
                            index,
                            adapter.get_info(),
                            device,
                            queue,
                        )),
                    })
                }
            })
    }
}

// DynDevice Methods
impl Gpu {
    pub(super) fn create_buffer(&self, size: usize) -> Result<BufferId> {
        let vacant = self.vacant_buffer_entry();
        let id = *vacant.key();
        let buffer = self.base.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{:?}", id)),
            size: size as _,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        vacant.insert(buffer);
        Ok(id)
    }
    pub(super) fn create_buffer_init<T: Pod>(&self, data: Cow<[T]>) -> Result<BufferId> {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let vacant = self.vacant_buffer_entry();
        let id = *vacant.key();
        let buffer = self.base.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("{:?}", id)),
            contents: &bytemuck::cast_slice(&data),
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        });
        vacant.insert(buffer);
        Ok(id)
    }
    #[allow(unused)]
    pub(super) fn copy_buffer_to_buffer(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.op(Op::CopyBufferToBuffer {
            src,
            src_offset: src_offset as _,
            dst,
            dst_offset: dst_offset as _,
            size: len as _,
        })
    }
    pub(super) fn drop_buffer(&self, id: BufferId) -> Result<()> {
        self.op(Op::DropBuffer { id })
    }
    pub(super) fn read_buffer<T: Pod>(
        &self,
        id: BufferId,
        offset: usize,
        len: usize,
    ) -> Result<impl Future<Output = Result<Vec<T>>>> {
        if !self.base.buffers.contains_key(&id) {
            todo!();
        }
        let read_entry = self.vacant_buffer_entry();
        let read_id = *read_entry.key();
        let read_buffer = self.base.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{:?}", read_id)),
            size: (len * size_of::<T>()) as _,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });
        read_entry.insert(read_buffer);
        self.op(Op::CopyBufferToBuffer {
            src: id,
            src_offset: (offset * size_of::<T>()) as _,
            dst: read_id,
            dst_offset: 0,
            size: (len * size_of::<T>()) as _,
        })?;
        let sync_future = self.synchronize()?;
        let gpu = self.clone();
        Ok(async move {
            sync_future.await?;
            if let Some((_k, read_buffer)) = gpu.base.buffers.remove(&read_id) {
                let read_slice = read_buffer.slice(0..(len * size_of::<T>()) as _);
                let map_future = read_slice.map_async(MapMode::Read);
                gpu.base.device.poll(Maintain::Wait);
                map_future.await?;
                let vec = bytemuck::cast_slice(&read_slice.get_mapped_range()).to_vec();
                Ok(vec)
            } else {
                todo!()
            }
        })
    }
    #[allow(unused)]
    pub(super) fn compile_shader_module(&self, module: &ShaderModule) -> Result<()> {
        todo!()
    }
    #[allow(unused)]
    pub(super) fn enqueue_compute_pass(&self, compute_pass: ComputePass) -> Result<()> {
        todo!()
    }
    pub(super) fn synchronize(&self) -> Result<impl Future<Output = Result<()>>> {
        let gpu = self.clone();
        Ok(async move {
            let commands = gpu
                .base
                .stream
                .lock()
                .await
                .flush_commands(
                    &gpu.base.device,
                    &gpu.base.buffers,
                    &gpu.base.compute_pipelines,
                )
                .await?;
            smol::unblock(move || gpu.base.queue.submit(commands)).await;
            Ok(())
        })
    }
}

impl Gpu {
    fn vacant_buffer_entry(&self) -> VacantEntry<BufferId, Buffer> {
        let mut id = BufferId(0);
        loop {
            match self.base.buffers.entry(id) {
                Entry::Occupied(_) => {
                    id.0 += 1;
                }
                Entry::Vacant(vacant) => {
                    return vacant;
                }
            }
        }
    }
    fn op(&self, op: Op) -> Result<()> {
        let gpu = self.clone();
        smol::block_on(async move {
            gpu.base
                .stream
                .lock()
                .await
                .push(
                    op,
                    &gpu.base.device,
                    &gpu.base.buffers,
                    &gpu.base.compute_pipelines,
                )
                .await
        })?;
        Ok(())
    }
}

impl Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        GpuPrinter {
            host_name: self.base.host_name.as_deref(),
            index: self.base.index,
        }
        .fmt(f)
    }
}

#[derive(Debug)]
pub enum Op {
    CopyBufferToBuffer {
        src: BufferId,
        src_offset: BufferAddress,
        dst: BufferId,
        dst_offset: BufferAddress,
        size: BufferAddress,
    },
    ComputePass(ComputePass),
    DropBuffer {
        id: BufferId,
    },
}

#[derive(Copy, Clone, Debug)]
pub struct CommandBufferId(usize);

pub struct Stream {
    label: String,
    command_buffer_id: CommandBufferId,
    ops: Vec<Op>,
    commands: Vec<CommandBuffer>,
}

impl Stream {
    fn new(label: String) -> Self {
        Self {
            label,
            command_buffer_id: CommandBufferId(0),
            ops: Vec::new(),
            commands: Vec::new(),
        }
    }
    // Pushes op, may flush if necessary
    async fn push(
        &mut self,
        op: Op,
        device: &Device,
        buffers: &DashMap<BufferId, Buffer>,
        compute_pipelines: &DashMap<ModuleId, Vec<(BindGroupLayout, ComputePipeline)>>,
    ) -> Result<()> {
        use Op::*;
        // Ensure compute pass occurs prior to subsequent copy
        if let CopyBufferToBuffer { .. } = &op {
            if self
                .ops
                .iter()
                .rev()
                .rfind(|x| matches!(x, ComputePass(_)))
                .is_some()
            {
                self.flush_ops(device, buffers, compute_pipelines).await?;
            }
        }
        self.ops.push(op);
        Ok(())
    }
    // Processes ops into commands
    async fn flush_ops(
        &mut self,
        device: &Device,
        buffers: &DashMap<BufferId, Buffer>,
        #[allow(unused)] compute_pipelines: &DashMap<
            ModuleId,
            Vec<(BindGroupLayout, ComputePipeline)>,
        >,
    ) -> Result<()> {
        use Op::*;
        if self.ops.is_empty() {
            return Ok(());
        }
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{}: {:?}", &self.label, &self.command_buffer_id)),
        });
        self.command_buffer_id.0 += 1;
        let mut buffers_to_drop = Vec::new();
        for op in replace(&mut self.ops, Vec::new()) {
            match op {
                CopyBufferToBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let src_buffer = buffers.get(&src).ok_or(GpuError::BufferNotFound(src))?;
                    let dst_buffer = buffers.get(&dst).ok_or(GpuError::BufferNotFound(dst))?;
                    encoder.copy_buffer_to_buffer(
                        &src_buffer,
                        src_offset,
                        &dst_buffer,
                        dst_offset,
                        size,
                    );
                }
                ComputePass { .. } => {
                    todo!()
                }
                DropBuffer { id } => {
                    buffers_to_drop.push(id);
                }
            }
        }
        self.commands.push(encoder.finish());
        for id in buffers_to_drop {
            buffers.remove(&id).ok_or(GpuError::BufferNotFound(id))?;
        }
        Ok(())
    }
    async fn flush_commands(
        &mut self,
        device: &Device,
        buffers: &DashMap<BufferId, Buffer>,
        compute_pipelines: &DashMap<ModuleId, Vec<(BindGroupLayout, ComputePipeline)>>,
    ) -> Result<Vec<CommandBuffer>> {
        self.flush_ops(device, buffers, compute_pipelines).await?;
        Ok(replace(&mut self.commands, Vec::new()))
    }
}
