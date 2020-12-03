use crate::backend::Mem;
use crate::Result;
use async_std::future::Future;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Clone, Debug)]
pub enum GpuSpecifier {
    Index(u32),
}

pub mod builder {
    use super::*;

    #[derive(Default)]
    pub struct GpuBuilder {
        pub(super) specifier: Option<GpuSpecifier>,
    }

    impl GpuBuilder {
        pub(in super::super) async fn build(self) -> super::super::Result<super::super::Device> {
            Ok(Gpu::new(self).await?.into())
        }
    }
}
pub(super) use builder::GpuBuilder;

pub struct Gpu {
    #[allow(unused)]
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: CommandQueue,
    buffers: BufferMap,
}

impl Gpu {
    pub fn builder() -> GpuBuilder {
        GpuBuilder::default()
    }
    pub(super) async fn new(builder: GpuBuilder) -> Result<Self> {
        let backend = wgpu::BackendBit::all();
        let instance = wgpu::Instance::new(backend);
        let specifier = builder.specifier;
        let adapter = match specifier {
            Some(GpuSpecifier::Index(index)) => instance
                .enumerate_adapters(backend)
                .skip(index as usize)
                .next()
                .ok_or_else(|| format!("Invalid gpu index {:?}!", specifier))?,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: None,
                })
                .await
                .ok_or("No gpus!")?,
        };
        let mut limits = wgpu::Limits::default();
        limits.max_push_constant_size = 128;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::PUSH_CONSTANTS,
                    limits,
                    shader_validation: true,
                },
                None,
            )
            .await?;
        let device = Arc::new(device);
        let queue = CommandQueue::new(queue);
        let buffers = BufferMap::new();
        Ok(Self {
            adapter,
            device,
            queue,
            buffers,
        })
    }
    pub(super) fn alloc<'a>(
        &self,
        mem: Mem,
        size: usize,
        data: Option<Cow<'a, [u8]>>,
    ) -> Result<()> {
        let label = None;
        let usage =
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::COPY_DST;
        let buffer = match data {
            Some(data) => self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data.as_ref(),
                    usage,
                }),
            None => self.device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size: size as _,
                usage,
                mapped_at_creation: false,
            }),
        };
        self.buffers.insert(mem, buffer)
    }
    pub(super) fn dealloc(&self, mem: Mem) -> Result<()> {
        self.buffers.remove(&mem)
    }
    pub(super) fn read<'a>(
        &self,
        mem: Mem,
        offset: usize,
        data: &'a mut [u8],
    ) -> Result<impl Future<Output = Result<()>> + 'a> {
        let offset = offset as wgpu::BufferAddress;
        let size = data.len() as wgpu::BufferAddress;
        let buffer = self
            .buffers
            .get(&mem)
            .ok_or_else(|| format!("Gpu invalid {:?}!", mem))?;
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&buffer, offset, &read_buffer, 0, size);
        self.queue.push(encoder.finish());

        let submit_future = self.queue.submit();

        let device = self.device.clone();
        Ok(async move {
            submit_future.await?;
            let slice = read_buffer.slice(offset..size);
            let slice_future = slice.map_async(wgpu::MapMode::Read);
            device.poll(wgpu::Maintain::Wait);
            slice_future.await?;
            data.copy_from_slice(&slice.get_mapped_range());
            read_buffer.unmap();
            Ok(())
        })
    }
}

impl Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Gpu").finish()
    }
}

#[doc(hidden)]
pub mod buffer_map {
    use super::{Mem, Result};
    use evmap_derive::ShallowCopy;
    use std::hash::{Hash, Hasher};
    use std::ops::Deref;
    use std::sync::{Arc, Mutex};

    #[doc(hidden)]
    #[derive(ShallowCopy)]
    pub struct ArcBuffer(Arc<wgpu::Buffer>);

    impl ArcBuffer {
        fn new(buffer: wgpu::Buffer) -> Self {
            Self(Arc::new(buffer))
        }
        fn id(this: &Self) -> usize {
            Arc::as_ptr(&this.0) as usize
        }
    }

    impl PartialEq for ArcBuffer {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.0, &other.0)
        }
    }

    impl Eq for ArcBuffer {}

    impl Hash for ArcBuffer {
        fn hash<H: Hasher>(&self, state: &mut H) {
            Self::id(self).hash(state);
        }
    }

    impl Deref for ArcBuffer {
        type Target = wgpu::Buffer;
        fn deref(&self) -> &Self::Target {
            &*self.0
        }
    }

    #[doc(hidden)]
    pub struct BufferMap {
        reader: evmap::ReadHandle<Mem, ArcBuffer>,
        writer: Mutex<evmap::WriteHandle<Mem, ArcBuffer>>,
    }

    impl BufferMap {
        pub(super) fn new() -> Self {
            let (reader, writer) = evmap::new();
            let writer = Mutex::new(writer);
            Self { reader, writer }
        }
        pub(super) fn insert(&self, mem: Mem, buffer: wgpu::Buffer) -> Result<()> {
            let buffer = ArcBuffer::new(buffer);
            self.writer
                .lock()
                .or_else(|_| Err("BufferMap is poisoned!"))?
                .insert(mem, buffer)
                .refresh();
            Ok(())
        }
        pub(super) fn remove(&self, mem: &Mem) -> Result<()> {
            self.writer
                .lock()
                .or_else(|_| Err("BufferMap is poisoned!"))?
                .clear(*mem)
                .refresh();
            Ok(())
        }
        pub(super) fn get(&self, mem: &Mem) -> Option<impl Deref<Target = wgpu::Buffer> + '_> {
            self.reader
                .get_one(mem)
                .map(|guard| BufferMapReadGuard(guard))
        }
    }

    pub(super) struct BufferMapReadGuard<'a>(evmap::ReadGuard<'a, ArcBuffer>);

    impl Deref for BufferMapReadGuard<'_> {
        type Target = wgpu::Buffer;
        fn deref(&self) -> &Self::Target {
            &*self.0
        }
    }
}
use buffer_map::BufferMap;

#[doc(hidden)]
pub mod command_queue {
    use super::{Future, Result};
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::{Arc, Mutex, TryLockError};

    pub struct CommandQueue {
        queue: Arc<wgpu::Queue>,
        sender: Sender<wgpu::CommandBuffer>,
        receiver: Arc<Mutex<Receiver<wgpu::CommandBuffer>>>,
    }

    impl CommandQueue {
        pub(super) fn new(queue: wgpu::Queue) -> Self {
            let queue = Arc::new(queue);
            let (sender, receiver) = channel();
            let receiver = Arc::new(Mutex::new(receiver));
            Self {
                queue,
                sender,
                receiver,
            }
        }
        pub(super) fn push(&self, command: wgpu::CommandBuffer) {
            self.sender.send(command).unwrap();
        }
        pub(super) fn submit(&self) -> impl Future<Output = Result<()>> {
            let queue = self.queue.clone();
            let receiver = self.receiver.clone();
            async move {
                loop {
                    match receiver.try_lock() {
                        Ok(receiver) => {
                            let commands = std::iter::from_fn(|| receiver.try_recv().ok());
                            queue.submit(commands);
                            return Ok(());
                        }
                        Err(TryLockError::Poisoned(_)) => {
                            Err("CommandQueue poisoned!")?;
                        }
                        Err(TryLockError::WouldBlock) => {
                            async_std::task::yield_now().await;
                        }
                    }
                }
            }
        }
    }
}
use command_queue::CommandQueue;
