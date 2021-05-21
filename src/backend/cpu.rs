use super::{
    BufferId, ComputePass, DeviceResult as Result, Future, ModuleId, Scalar, ShaderModule,
};

use hibitset::BitSet;
use smol::{future::ready, lock::Mutex};
use std::{
    borrow::Cow,
    cmp::Ordering,
    fmt::{self, Debug},
    mem::{size_of, transmute},
    ops::Range,
    pin::Pin,
    sync::Arc,
};

const CHUNK_SIZE: usize = 2 << 24;
const BLOCKS_PER_CHUNK: usize =
    size_of::<usize>() * size_of::<usize>() * size_of::<usize>() * size_of::<usize>();
const BLOCK_SIZE: usize = CHUNK_SIZE / BLOCKS_PER_CHUNK;

#[derive(Clone)]
pub struct Cpu {
    base: Arc<CpuBase>,
}

impl Cpu {
    pub(super) fn new() -> Self {
        Self {
            base: Arc::new(CpuBase::new()),
        }
    }
    pub(super) fn create_buffer(&self, size: usize) -> Result<BufferId> {
        self.base.create_buffer(size)
    }
    pub(super) fn create_buffer_init<T: Scalar>(&self, data: Cow<[T]>) -> Result<BufferId> {
        self.base.create_buffer_init(data)
    }
    pub(super) fn drop_buffer(&self, id: BufferId) {
        self.base.drop_buffer(id)
    }
    #[allow(clippy::type_complexity)]
    pub(super) fn read_buffer<T: Scalar>(
        &self,
        id: BufferId,
        offset: usize,
        len: usize,
    ) -> Result<Pin<Box<dyn Future<Output = Result<Vec<T>>>>>> {
        self.base.read_buffer(id, offset, len)
    }
    pub(super) fn copy_buffer_to_buffer(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.base
            .copy_buffer_to_buffer(src, src_offset, dst, dst_offset, len)
    }
    #[allow(unused_variables)]
    pub(super) fn compile_shader_module(&self, id: ModuleId, module: &ShaderModule) -> Result<()> {
        unimplemented!("Replace with DeviceError")
    }
    #[allow(unused_variables)]
    pub(super) fn enqueue_compute_pass(&self, compute_pass: ComputePass) -> Result<()> {
        unimplemented!("Replace with DeviceError")
    }
    #[allow(clippy::type_complexity)]
    pub(super) fn synchronize(&self) -> Result<Pin<Box<dyn Future<Output = Result<()>>>>> {
        // This resolves immediately because Cpu doesn't execute shaders, yet.
        Ok(Box::pin(async { Ok(()) }))
    }
}

impl Debug for Cpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Cpu").finish()
    }
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
    fn to_usize(&self) -> usize {
        self.to_u32() as usize
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

struct StorageChunk {
    buffer: Vec<u8>,
    blocks: BitSet,
}

impl StorageChunk {
    fn new() -> Self {
        Self {
            buffer: vec![0u8; CHUNK_SIZE],
            blocks: BitSet::with_capacity(BLOCKS_PER_CHUNK as u32),
        }
    }
    fn alloc(&mut self, size: usize) -> Option<Range<u24>> {
        let num_blocks = if size % BLOCK_SIZE == 0 {
            size / BLOCK_SIZE
        } else {
            size / BLOCK_SIZE + 1
        };
        'outer: for start in 0..=(BLOCKS_PER_CHUNK - num_blocks) as u32 {
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
    #[allow(unused)]
    fn get(&self, range: Range<u24>) -> &[u8] {
        let start = BLOCK_SIZE * range.start.to_usize();
        let end = BLOCK_SIZE * range.end.to_usize();
        &self.buffer[start..end]
    }
    fn get_mut(&mut self, range: Range<u24>) -> &mut [u8] {
        let start = BLOCK_SIZE * range.start.to_usize();
        let end = BLOCK_SIZE * range.end.to_usize();
        &mut self.buffer[start..end]
    }
    fn copy_buffer_to_buffer(
        &mut self,
        src_range: Range<u24>,
        src_offset: usize,
        dst_range: Range<u24>,
        dst_offset: usize,
        len: usize,
    ) {
        let src_start = BLOCK_SIZE * src_range.start.to_usize() + src_offset;
        let src_end = src_start + len;
        let dst_start = BLOCK_SIZE * dst_range.start.to_usize() + dst_offset;
        let dst_end = dst_start + len;
        if src_end < dst_start {
            let (left, right) = self.buffer.split_at_mut(dst_start);
            right[0..len].copy_from_slice(&left[src_start..src_end]);
        } else {
            let (left, right) = self.buffer.split_at_mut(src_start);
            left[dst_start..dst_end].copy_from_slice(&right[0..len]);
        }
    }
    fn dealloc(&mut self, range: Range<u24>) {
        for i in range.start.to_u32()..range.end.to_u32() {
            self.blocks.remove(i);
        }
    }
}

struct StorageAllocator {
    chunks: Vec<StorageChunk>,
}

impl StorageAllocator {
    fn new() -> Self {
        Self { chunks: Vec::new() }
    }
    fn alloc(&mut self, size: usize) -> Result<StorageBuffer> {
        debug_assert!(size <= CHUNK_SIZE); // TODO: error here?
        for (c, chunk) in self.chunks.iter_mut().enumerate() {
            dbg!(c);
            if let Some(range) = chunk.alloc(size) {
                return Ok(StorageBuffer::new(c as u16, range));
            }
        }
        let mut chunk = StorageChunk::new();
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
    fn get(&mut self, buffer: &StorageBuffer) -> &[u8] {
        let chunk = self.chunks.get_mut(buffer.chunk as usize).unwrap();
        chunk.get(buffer.start..buffer.end)
    }
    fn get_mut(&mut self, buffer: &StorageBuffer) -> &mut [u8] {
        let chunk = self.chunks.get_mut(buffer.chunk as usize).unwrap();
        chunk.get_mut(buffer.start..buffer.end)
    }
    fn copy_buffer_to_buffer(
        &mut self,
        src: &StorageBuffer,
        src_offset: usize,
        dst: &mut StorageBuffer,
        dst_offset: usize,
        len: usize,
    ) {
        match src.chunk.cmp(&dst.chunk) {
            Ordering::Less => {
                let (left, right) = self.chunks.split_at_mut(dst.chunk as usize);
                let src_chunk = &left[src.chunk as usize];
                let dst_chunk = &mut right[0];
                dst_chunk
                    .get_mut(dst.range())
                    .copy_from_slice(src_chunk.get(src.range()));
            }
            Ordering::Greater => {
                let (left, right) = self.chunks.split_at_mut(src.chunk as usize);
                let src_chunk = &right[0];
                let dst_chunk = &mut left[dst.chunk as usize - src.chunk as usize];
                dst_chunk
                    .get_mut(dst.range())
                    .copy_from_slice(src_chunk.get(src.range()));
            }
            Ordering::Equal => {
                let chunk = self.chunks.get_mut(src.chunk as usize).unwrap();
                chunk.copy_buffer_to_buffer(src.range(), src_offset, dst.range(), dst_offset, len);
            }
        }
    }
}

struct Context {
    storage_allocator: StorageAllocator,
}

impl Context {
    fn new() -> Self {
        Self {
            storage_allocator: StorageAllocator::new(),
        }
    }
}

#[doc(hidden)]
pub struct CpuBase {
    context: Mutex<Context>,
}

impl CpuBase {
    fn new() -> Self {
        Self {
            context: Mutex::new(Context::new()),
        }
    }
    fn create_buffer(&self, size: usize) -> Result<BufferId> {
        let mut context = smol::block_on(self.context.lock());
        let buffer = context.storage_allocator.alloc(size)?;
        Ok(buffer.to_buffer_id())
    }
    fn create_buffer_init<T: Scalar>(&self, data: Cow<[T]>) -> Result<BufferId> {
        let mut context = smol::block_on(self.context.lock());
        let buffer = context
            .storage_allocator
            .alloc(data.len() * size_of::<T>())?;
        let slice = context.storage_allocator.get_mut(&buffer);
        let data_slice = bytemuck::cast_slice(&data);
        slice[..data_slice.len()].copy_from_slice(data_slice);
        Ok(buffer.to_buffer_id())
    }
    fn drop_buffer(&self, id: BufferId) {
        let mut context = smol::block_on(self.context.lock());
        context
            .storage_allocator
            .dealloc(StorageBuffer::from_buffer_id(id));
    }
    #[allow(clippy::type_complexity)]
    fn read_buffer<T: Scalar>(
        &self,
        id: BufferId,
        offset: usize,
        len: usize,
    ) -> Result<Pin<Box<dyn Future<Output = Result<Vec<T>>>>>> {
        let mut context = smol::block_on(self.context.lock());
        let slice = context
            .storage_allocator
            .get(&StorageBuffer::from_buffer_id(id));
        let mut vec = vec![T::zero(); len];
        bytemuck::cast_slice_mut(&mut vec).copy_from_slice(&slice[offset..len * size_of::<T>()]);
        Ok(Box::pin(ready(Ok(vec))))
    }
    fn copy_buffer_to_buffer(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let mut context = smol::block_on(self.context.lock());
        context.storage_allocator.copy_buffer_to_buffer(
            &StorageBuffer::from_buffer_id(src),
            src_offset,
            &mut StorageBuffer::from_buffer_id(dst),
            dst_offset,
            len,
        );
        Ok(())
    }
}
