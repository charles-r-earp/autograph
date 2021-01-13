use crate::Result;
use anyhow::{anyhow, bail, ensure};
use derive_more::Display;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use smol::future::Future;
use smol::lock::Mutex;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

mod fill;
mod shader_util;

#[doc(hidden)]
pub mod gpu;
use gpu::Gpu;

/// TODO: Document This?
#[derive(Clone, Copy, Debug, Display, Eq, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[repr(u8)]
enum DeviceError {
    DeviceUnsupported,
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
    MissingExtension,
    MissingFeature,
    TooManyObjects,
    DeviceLost,
    MappingFailed,
}

type DeviceResult<T> = std::result::Result<T, DeviceError>;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

macro_rules! impl_sealed {
    ($($t:ty,)+) => (
        $(
            impl Sealed for $t {}
        )+
    )
}

impl_sealed!(u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64,);

/// Base trait for all shader types
pub trait Scalar: Sealed + Copy + Send + Sync + Debug + Default + PartialEq + 'static {
    fn zero() -> Self {
        Self::default()
    }
    fn one() -> Self;
    fn to_bits_u8(&self) -> Option<u8> {
        None
    }
    fn to_bits_u16(&self) -> Option<u16> {
        None
    }
    fn to_bits_u32(&self) -> Option<u32> {
        None
    }
    fn to_bits_u64(&self) -> Option<u64> {
        None
    }
    fn to_f32(&self) -> Option<f32> {
        None
    }
}

impl Scalar for u8 {
    fn one() -> Self {
        1
    }
    fn to_bits_u8(&self) -> Option<u8> {
        Some(*self)
    }
}

impl Scalar for i8 {
    fn one() -> Self {
        1
    }
    fn to_bits_u8(&self) -> Option<u8> {
        Some(self.to_ne_bytes()[0])
    }
}

impl Scalar for u16 {
    fn one() -> Self {
        1
    }
    fn to_bits_u16(&self) -> Option<u16> {
        Some(*self)
    }
}

impl Scalar for i16 {
    fn one() -> Self {
        1
    }
    fn to_bits_u16(&self) -> Option<u16> {
        Some(u16::from_ne_bytes(self.to_ne_bytes()))
    }
}

impl Scalar for f16 {
    fn one() -> Self {
        Self::ONE
    }
    fn to_bits_u16(&self) -> Option<u16> {
        Some(self.to_bits())
    }
    fn to_f32(&self) -> Option<f32> {
        Some(Self::to_f32(*self))
    }
}

impl Scalar for bf16 {
    fn one() -> Self {
        Self::ONE
    }
    fn to_bits_u16(&self) -> Option<u16> {
        Some(self.to_bits())
    }
    fn to_f32(&self) -> Option<f32> {
        Some(Self::to_f32(*self))
    }
}

impl Scalar for u32 {
    fn one() -> Self {
        1
    }
    fn to_bits_u32(&self) -> Option<u32> {
        Some(*self)
    }
}

impl Scalar for i32 {
    fn one() -> Self {
        1
    }
    fn to_bits_u32(&self) -> Option<u32> {
        Some(u32::from_ne_bytes(self.to_ne_bytes()))
    }
}

impl Scalar for f32 {
    fn one() -> Self {
        1.
    }
    fn to_bits_u32(&self) -> Option<u32> {
        Some(u32::from_ne_bytes(self.to_ne_bytes()))
    }
    fn to_f32(&self) -> Option<f32> {
        Some(*self)
    }
}

impl Scalar for u64 {
    fn one() -> Self {
        1
    }
    fn to_bits_u64(&self) -> Option<u64> {
        Some(*self)
    }
}

impl Scalar for i64 {
    fn one() -> Self {
        1
    }
    fn to_bits_u64(&self) -> Option<u64> {
        Some(u64::from_ne_bytes(self.to_ne_bytes()))
    }
}

impl Scalar for f64 {
    fn one() -> Self {
        1.
    }
    fn to_bits_u64(&self) -> Option<u64> {
        Some(u64::from_ne_bytes(self.to_ne_bytes()))
    }
}

/// Marker trait for arithmetic types
pub trait Num: Scalar {}

impl Num for bf16 {}

impl Num for u32 {}

impl Num for i32 {}

impl Num for f32 {}

impl Num for f64 {}

#[doc(hidden)]
#[proxy_enum::proxy(DynDevice)]
pub mod dyn_device_proxy {
    use super::*;
    use DeviceResult as Result;

    #[derive(Clone)]
    pub enum DynDevice {
        Gpu(Gpu),
    }

    impl DynDevice {
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
        pub(super) fn drop_buffer(&self, id: BufferId) -> Result<()> {}
        #[implement]
        pub(super) fn read_buffer<T: Scalar>(
            &self,
            id: BufferId,
            offset: usize,
            len: usize,
        ) -> Result<impl Future<Output = Result<Vec<T>>>> {
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

    #[external(std::fmt::Debug)]
    trait Debug {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result;
    }

    #[implement]
    impl std::fmt::Debug for DynDevice {}
}
#[doc(hidden)]
pub use dyn_device_proxy::DynDevice;

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(u64);

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModuleId(u64);

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntryId(u64);

const MAX_BUFFERS_PER_COMPUTE_PASS: usize = 4;
#[allow(unused)]
const MAX_PUSH_CONSTANT_SIZE: usize = 64; // TODO: This was provided to wgpu, potentially this can be used with const generics

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BufferDescriptor {
    binding: u32,
    mutable: bool,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PushConstantRange {
    start: u32,
    end: u32,
}

impl From<PushConstantRange> for std::ops::Range<u32> {
    fn from(from: PushConstantRange) -> Self {
        from.start..from.end
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PushConstantDescriptor {
    range: PushConstantRange,
}

#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EntryDescriptor {
    name: String,
    local_size: [u32; 3],
    buffer_descriptors: Vec<BufferDescriptor>,
    push_constant_descriptor: Option<PushConstantDescriptor>,
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderModule<'a> {
    spirv: Cow<'a, [u8]>,
    entry_descriptors: Vec<EntryDescriptor>,
}

impl<'a> ShaderModule<'a> {
    fn from_spirv(spirv: impl Into<Cow<'a, [u8]>>) -> Result<Self> {
        let spirv = spirv.into();
        let entry_descriptors = shader_util::entry_descriptors_from_spirv(&spirv)?;
        Ok(Self {
            spirv,
            entry_descriptors,
        })
    }
}

#[derive(Clone)]
pub struct Device {
    dyn_device: DynDevice,
    modules: Arc<Mutex<HashMap<ModuleId, ShaderModule<'static>>>>,
}

impl Device {
    /// Creates a new Gpu at index\
    ///
    /// Supported gfx_hal Backends:
    ///   - Vulkan
    ///   - Metal
    ///   - DX12
    pub fn new_gpu(index: usize) -> Option<Result<Self>> {
        Gpu::new(index).map(|gpu| {
            Ok(Self {
                dyn_device: gpu?.into(),
                modules: Arc::default(),
            })
        })
    }
    /// Returns all available Devices
    pub fn list() -> Vec<Self> {
        Self::list_gpus()
    }
    /// Returns all available Gpus
    pub fn list_gpus() -> Vec<Self> {
        let mut devices = Vec::new();
        for i in 0.. {
            if let Some(Ok(gpu)) = Self::new_gpu(i) {
                devices.push(gpu);
            } else {
                break;
            }
        }
        devices
    }
    /// Create a compute pass\
    ///
    /// Compiles the provided source and performs reflection.\
    /// The module is cached based on the reference, so subsequent\
    /// calls with the same module will reuse it.\
    ///
    /// Err: Errors if the provided entry_point is not found in the
    /// spirv module.
    pub fn compute_pass(
        &self,
        spirv: &'static [u8],
        entry_point: impl AsRef<str>,
    ) -> Result<ComputePassBuilder<()>> {
        use std::collections::hash_map::Entry::*;
        let module_id = ModuleId(spirv.as_ptr() as usize as u64);
        let entry_point = entry_point.as_ref();
        let mut modules = smol::block_on(self.modules.lock());
        let (entry_id, entry_descriptor) = match modules.entry(module_id) {
            Occupied(occupied) => {
                let entry_descriptors = &occupied.get().entry_descriptors;
                let (i, entry_descriptor) = entry_descriptors
                    .iter()
                    .enumerate()
                    .find(|(_, e)| e.name == entry_point)
                    .ok_or_else(|| {
                        let entry_points: Vec<_> =
                            entry_descriptors.iter().map(|e| e.name.clone()).collect();
                        anyhow!(
                            "Entry {} not found! Declared entry points are: {:#?}",
                            &entry_point,
                            entry_points
                        )
                    })?;
                (EntryId(i as u64), entry_descriptor.clone())
            }
            Vacant(vacant) => {
                let module = ShaderModule::from_spirv(spirv)?;
                self.dyn_device.compile_shader_module(module_id, &module)?;
                let module = vacant.insert(module);
                let entry_descriptors = &module.entry_descriptors;
                let (i, entry_descriptor) = entry_descriptors
                    .iter()
                    .enumerate()
                    .find(|(_, e)| e.name == entry_point)
                    .ok_or_else(|| {
                        let entry_points: Vec<_> =
                            entry_descriptors.iter().map(|e| e.name.clone()).collect();
                        anyhow!(
                            "Entry {} not found! Declared entry points are: {:#?}",
                            &entry_point,
                            entry_points
                        )
                    })?;
                (EntryId(i as u64), entry_descriptor.clone())
            }
        };
        let buffer_bindings = Vec::with_capacity(entry_descriptor.buffer_descriptors.len());
        Ok(ComputePassBuilder {
            device: self,
            entry_descriptor,
            compute_pass: ComputePass {
                module_id,
                entry_id,
                buffer_bindings,
                push_constants: Vec::new(),
                work_groups: [1, 1, 1],
            },
            borrows: (),
        })
    }
    fn create_buffer(&self, size: usize) -> Result<BufferId> {
        Ok(self.dyn_device.create_buffer(size)?)
    }
    fn create_buffer_init<T: Scalar>(&self, data: Cow<[T]>) -> Result<BufferId> {
        Ok(self.dyn_device.create_buffer_init(data)?)
    }
    #[allow(unused)]
    fn copy_buffer_to_buffer(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        Ok(self
            .dyn_device
            .copy_buffer_to_buffer(src, src_offset, dst, dst_offset, len)?)
    }
    pub(super) fn drop_buffer(&self, id: BufferId) -> Result<()> {
        Ok(self.dyn_device.drop_buffer(id)?)
    }
    pub(super) fn read_buffer<T: Scalar>(
        &self,
        id: BufferId,
        offset: usize,
        len: usize,
    ) -> Result<impl Future<Output = Result<Vec<T>>>> {
        let f = self.dyn_device.read_buffer(id, offset, len)?;
        Ok(async move { Ok(f.await?) })
    }
    /// Ensures all previous operations are submitted\
    ///
    /// Note: the exact semantics of this function may change and may depend on the backend.\
    ///
    /// Gpu:
    ///
    /// Operations (writes / reads / compute passes) are queued in a stream. On synchronize, blocks on any previously submitted work to finish and dispatches work to the gpu. The returned future will resolve when that work is complete. Dropping or cancelling the future is safe and will have no effect.\
    /// Generally aim to call synchronize as infrequently as possible.\
    /// Currently only one set of operations can execute at once, calling synchronize again will block the calling thread until it finishes. This may change in the future. At the very least a possible improvement would be to submit when limits, such as the memory for reads / writes, are reached.\
    /// There is no worker thread, all cpu operations run on the calling thread. It is safe to enqueue work from multiple threads, however, the current implementation is not concurrent, and should probably be used within a single computation flow, though it may be useful to enqueue operations in parallel and then synchronize.\
    ///
    /// Note: When a buffer is dropped, its memory is made available to the custom allocator immediately, to be reused again and again, even within the same stream / submission. Reads / writes / compute passes will only actually access that memory in order. This means that temporary memory will not grow in a loop, an can be reused for each iteration. For example, a Neural Network could enqueue an entire epoch, if there is room for reads / writes, particularly writes. Larger datasets will require separate submissions.
    pub fn synchronize(&self) -> Result<impl Future<Output = Result<()>>> {
        let f = self.dyn_device.synchronize()?;
        Ok(async move {
            f.await?;
            Ok(())
        })
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.dyn_device.fmt(f)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct BufferBinding {
    binding: u32,
    id: BufferId,
    offset: u64,
    len: u64,
    mutable: bool,
}

#[allow(unused)]
pub struct ComputePassBuilder<'a, B> {
    device: &'a Device,
    entry_descriptor: EntryDescriptor,
    compute_pass: ComputePass,
    borrows: B,
}

impl<'a, B> ComputePassBuilder<'a, B> {
    /// Adds an immutable buffer argument\
    ///
    /// Must be readonly in spirv\
    /// Buffers are bound in binding order, low to high\
    ///
    /// Err: Errors if more buffers are provided than in spirv
    pub fn buffer_slice<T: Scalar>(
        self,
        slice: BufferSlice<T>,
    ) -> Result<ComputePassBuilder<'a, (B, Option<BufferSlice<T>>)>> {
        self.option_buffer_slice(Some(slice))
    }
    /// Adds an optional immutable buffer argument\
    ///
    /// See buffer_slice.
    pub fn option_buffer_slice<T: Scalar>(
        mut self,
        slice: Option<BufferSlice<T>>,
    ) -> Result<ComputePassBuilder<'a, (B, Option<BufferSlice<T>>)>> {
        if let Some(slice) = slice {
            if let Some(buffer_descriptor) = self
                .entry_descriptor
                .buffer_descriptors
                .get(self.compute_pass.buffer_bindings.len())
            {
                ensure!(
                    !buffer_descriptor.mutable,
                    "Buffer {} for entry {:?} declared as mutable!",
                    self.compute_pass.buffer_bindings.len(),
                    &self.entry_descriptor.name,
                );
                self.compute_pass.buffer_bindings.push(BufferBinding {
                    binding: buffer_descriptor.binding,
                    id: slice.id,
                    offset: (slice.offset * size_of::<T>()) as u64,
                    len: (slice.len * size_of::<T>()) as u64,
                    mutable: buffer_descriptor.mutable,
                });
                Ok(ComputePassBuilder {
                    device: self.device,
                    entry_descriptor: self.entry_descriptor,
                    compute_pass: self.compute_pass,
                    borrows: (self.borrows, Some(slice)),
                })
            } else {
                bail!(
                    "Buffer index out of range! Entry {} declares {}.\n\nNote: Only variables used in the entry will be declared.",
                    &self.entry_descriptor.name, self.compute_pass.buffer_bindings.len(),
                );
            }
        } else {
            Ok(ComputePassBuilder {
                device: self.device,
                entry_descriptor: self.entry_descriptor,
                compute_pass: self.compute_pass,
                borrows: (self.borrows, None),
            })
        }
    }
    /// Adds a mutable buffer argument\
    ///
    /// Must not be readonly in spirv. Buffers are bound in binding order, low to high\
    ///
    /// Err: Errors if more buffers are provided than in spirv.
    pub fn buffer_slice_mut<T: Scalar>(
        self,
        slice: BufferSliceMut<T>,
    ) -> Result<ComputePassBuilder<'a, (B, Option<BufferSliceMut<T>>)>> {
        self.option_buffer_slice_mut(Some(slice))
    }
    /// Adds an optional mutable buffer argument\
    ///
    /// See buffer_slice_mut.
    pub fn option_buffer_slice_mut<T: Scalar>(
        mut self,
        slice: Option<BufferSliceMut<T>>,
    ) -> Result<ComputePassBuilder<'a, (B, Option<BufferSliceMut<T>>)>> {
        if let Some(slice) = slice {
            if let Some(buffer_descriptor) = self
                .entry_descriptor
                .buffer_descriptors
                .get(self.compute_pass.buffer_bindings.len())
            {
                ensure!(
                    buffer_descriptor.mutable,
                    "Buffer {} for entry {:?} declared as immutable!",
                    self.compute_pass.buffer_bindings.len(),
                    &self.entry_descriptor.name,
                );
                self.compute_pass.buffer_bindings.push(BufferBinding {
                    binding: buffer_descriptor.binding,
                    id: slice.id,
                    offset: (slice.offset * size_of::<T>()) as u64,
                    len: (slice.len * size_of::<T>()) as u64,
                    mutable: buffer_descriptor.mutable,
                });
                Ok(ComputePassBuilder {
                    device: self.device,
                    entry_descriptor: self.entry_descriptor,
                    compute_pass: self.compute_pass,
                    borrows: (self.borrows, Some(slice)),
                })
            } else {
                bail!(
                    "Buffer index out of range! Entry {:?} declares {}.\n\nNote: Only variables used in the entry will be declared.",
                    &self.entry_descriptor.name, self.compute_pass.buffer_bindings.len(),
                );
            }
        } else {
            Ok(ComputePassBuilder {
                device: self.device,
                entry_descriptor: self.entry_descriptor,
                compute_pass: self.compute_pass,
                borrows: (self.borrows, None),
            })
        }
    }
    /// Sets push constants\
    ///
    /// Only one push constant block per pass. The length of the slice must match the spirv.
    pub fn push_constants(mut self, push_constants: &[u8]) -> Result<Self> {
        if let Some(push_constant_descriptor) =
            self.entry_descriptor.push_constant_descriptor.as_ref()
        {
            let PushConstantRange { start, end } = push_constant_descriptor.range;
            ensure!(
                push_constants.len() == (end - start) as usize,
                "Provided push constant size {} does not match declared size {} in entry {}!",
                push_constants.len(),
                (end - start),
                &self.entry_descriptor.name,
            );
            self.compute_pass.push_constants = push_constants.to_vec();
            Ok(self)
        } else {
            bail!(
                "No push constants declared for entry {}!",
                &self.entry_descriptor.name
            );
        }
    }
    /// Sets the number of work groups\
    ///
    /// Use either this method or global_size. The provided function f takes the local size [x, y, z] and returns the work groups [x, y, z].
    pub fn work_groups(mut self, f: impl Fn([u32; 3]) -> [u32; 3]) -> Self {
        self.compute_pass.work_groups = f(self.entry_descriptor.local_size);
        self
    }
    /// Sets the global size\
    ///
    /// Use either this method or work_groups. This will set the work groups such that work_groups * local_size >= global_size.
    pub fn global_size(mut self, global_size: [u32; 3]) -> Self {
        for (wg, (gs, ls)) in self.compute_pass.work_groups.iter_mut().zip(
            global_size
                .iter()
                .copied()
                .zip(self.entry_descriptor.local_size.iter().copied()),
        ) {
            *wg = if gs % ls == 0 { gs / ls } else { gs / ls + 1 };
        }
        self
    }
    /// Enqueues the compute pass\
    ///
    /// The backend may wait for additional work before executing.\
    /// Use Device::synchronize()?.await to force completion.
    ///
    /// Err: Errors if not all buffers or push_constants are provided
    pub fn enqueue(self) -> Result<()> {
        ensure!(
            self.compute_pass.buffer_bindings.len()
                == self.entry_descriptor.buffer_descriptors.len()
        );
        ensure!(
            self.compute_pass.push_constants.is_empty()
                == self.entry_descriptor.push_constant_descriptor.is_none()
        );
        self.device
            .dyn_device
            .enqueue_compute_pass(self.compute_pass)?;
        Ok(())
    }
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputePass {
    module_id: ModuleId,
    entry_id: EntryId,
    buffer_bindings: Vec<BufferBinding>,
    push_constants: Vec<u8>,
    work_groups: [u32; 3],
}

#[doc(hidden)]
pub trait Data: Sealed + Sized {
    type Elem;
    #[doc(hidden)]
    fn needs_drop() -> bool;
}

pub trait DataMut: Data {}

pub struct BufferRepr<T>(PhantomData<T>);

impl<T> Sealed for BufferRepr<T> {}

impl<T> Data for BufferRepr<T> {
    type Elem = T;
    fn needs_drop() -> bool {
        true
    }
}

impl<T> DataMut for BufferRepr<T> {}

pub struct BufferSliceRepr<S>(PhantomData<S>);

impl<T> Sealed for BufferSliceRepr<&'_ T> {}

impl<T> Data for BufferSliceRepr<&'_ T> {
    type Elem = T;
    fn needs_drop() -> bool {
        false
    }
}

impl<T> Sealed for BufferSliceRepr<&'_ mut T> {}

impl<T> Data for BufferSliceRepr<&'_ mut T> {
    type Elem = T;
    fn needs_drop() -> bool {
        false
    }
}

impl<T> DataMut for BufferSliceRepr<&'_ mut T> {}

pub struct BufferBase<S: Data> {
    device: Device,
    id: BufferId,
    offset: usize,
    len: usize,
    _m: PhantomData<S>,
}

pub type Buffer<T> = BufferBase<BufferRepr<T>>;
pub type BufferSlice<'a, T> = BufferBase<BufferSliceRepr<&'a T>>;
pub type BufferSliceMut<'a, T> = BufferBase<BufferSliceRepr<&'a mut T>>;

impl<T> Buffer<T> {
    /// Constructs the Buffer from a Cow\
    ///
    /// Note both Vec<T> and \[T] impl Into<Cow<\[T]>>.\
    ///
    /// Err: Errors if the backend cannot perform the operation, potentially due to a disconnect or running out of memory.
    pub fn from_cow(device: &Device, cow: Cow<[T]>) -> Result<Self>
    where
        T: Scalar,
    {
        let len = cow.len();
        let id = device.create_buffer_init(cow)?;
        Ok(Self {
            device: device.clone(),
            id,
            offset: 0,
            len,
            _m: PhantomData::default(),
        })
    }
    /// Constructs a new buffer\
    ///
    /// Err: Errors if the backend cannot perform the operation, potentially due to a disconnect or running out of memory.
    ///
    /// # Safety
    ///
    /// The buffer will not be initialized.
    pub unsafe fn uninitialized(device: &Device, len: usize) -> Result<Self> {
        let id = device.create_buffer(len * size_of::<T>())?;
        Ok(Self {
            device: device.clone(),
            id,
            offset: 0,
            len,
            _m: PhantomData::default(),
        })
    }
    /// Construct a buffer filled with zeros\
    ///
    /// Err: Errors if the backend cannot perform the operation, potentially due to a disconnect or running out of memory.
    pub fn zeros(device: &Device, len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        Self::from_elem(device, T::zero(), len)
    }
    /// Construct a buffer filled with elem\
    ///
    /// Err: Errors if the backend cannot perform the operation, potentially due to a disconnect or running out of memory.
    pub fn from_elem(device: &Device, elem: T, len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        let mut buffer = unsafe { Self::uninitialized(device, len)? };
        buffer.fill(elem)?;
        Ok(buffer)
    }
}

impl<T, S: Data<Elem = T>> BufferBase<S> {
    /// Borrows self as a BufferSlice
    pub fn as_buffer_slice(&self) -> BufferSlice<T> {
        BufferBase {
            device: self.device.clone(),
            id: self.id,
            offset: self.offset,
            len: self.len,
            _m: PhantomData::default(),
        }
    }
    /// Copies self into a new Buffer
    ///
    /// Err: Errors if the backend cannot perform the operation, potentially due to a disconnect or running out of memory.
    pub fn to_buffer(&self) -> Result<Buffer<T>>
    where
        T: Scalar,
    {
        let buffer = unsafe { Buffer::uninitialized(&self.device, self.len)? };
        self.device.dyn_device.copy_buffer_to_buffer(
            self.id,
            self.offset * size_of::<T>(),
            buffer.id,
            0,
            self.len * size_of::<T>(),
        )?;
        Ok(buffer)
    }
    /// Reads from device memory to a Vec.\
    ///
    /// This call will enqueue the copy operation. It returns a future, which if awaited, will synchronize if necessary, executing all previous operations. When performing multiple to_vec calls acquire all the futures first, then await any of them:\
    ///
    ///```
    /// # use autograph::{Result, backend::{Device, Buffer}};
    /// # fn main() -> Result<()> {
    /// #     if let Some(Ok(device)) = Device::new_gpu(0) {
    ///           let x1 = Buffer::from_elem(&device, 1f32, 1)?;
    ///           let x2 = Buffer::from_elem(&device, 2f32, 1)?;
    ///           let x1_future = x1.to_vec()?;
    ///           let x2_future = x2.to_vec()?;
    ///           // awaiting the future implicitly synchronizes
    ///           // smol::block_on(device.synchronize()?)?;
    ///           let x1_vec = smol::block_on(x1_future)?;
    ///           let x2_vec = smol::block_on(x2_future)?;
    /// #      }
    /// #      Ok(())
    /// #  }
    ///```
    /// Err: Errors if the backend cannot perform the operation, potentially due to a disconnect or running out of memory.
    pub fn to_vec(&self) -> Result<impl Future<Output = Result<Vec<T>>>>
    where
        T: Scalar,
    {
        self.device.read_buffer(self.id, self.offset, self.len)
    }
}

impl<T, S: DataMut<Elem = T>> BufferBase<S> {
    /// Borrows self as a BufferSliceMut
    pub fn as_buffer_slice_mut(&mut self) -> BufferSliceMut<T> {
        BufferBase {
            device: self.device.clone(),
            id: self.id,
            offset: self.offset,
            len: self.len,
            _m: PhantomData::default(),
        }
    }
    /// Fills the buffer with x.
    ///
    /// Err: Errors if the backend cannot perform the operation, potentially due to a disconnect or running out of memory.
    pub fn fill(&mut self, x: T) -> Result<()>
    where
        T: Scalar,
    {
        fill::fill(&self.device.clone(), self.as_buffer_slice_mut(), x)
    }
}

impl<S: Data> Drop for BufferBase<S> {
    fn drop(&mut self) {
        if S::needs_drop() {
            // We might as well not panic here since there's no way to return a result from drop
            // The primary failure mode would be another error or a disconnect
            #[allow(unused)]
            let result = self.device.drop_buffer(self.id);
            #[cfg(debug_assertions)]
            result.unwrap();
        }
    }
}
