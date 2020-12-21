use crate::{
    error::{ComputePassBuilderError, ShaderModuleError},
    Result,
};
use bytemuck::Pod;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smol::future::Future;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;
use half::{f16, bf16};
use num_traits::{ToPrimitive, FromPrimitive};
use crate::util::type_eq;

mod fill;
pub mod shader_util;

#[doc(hidden)]
pub mod gpu;
use gpu::Gpu;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

pub trait Scalar: Sealed + Debug + Default + ToPrimitive + FromPrimitive + Pod + PartialEq {
    fn zero() -> Self { Self::default() }
    fn one() -> Self { Self::from_u32(1).unwrap() }
    fn to_bits_u32(&self) -> u32 {
        if type_eq::<Self, u8>() || type_eq::<Self, u16>() || type_eq::<Self, u32>() {
            self.to_u32().unwrap()
        } else if type_eq::<Self, i8>() || type_eq::<Self, i16>() || type_eq::<Self, i32>() {
            u32::from_ne_bytes(self.to_i32().unwrap().to_ne_bytes())
        } else if type_eq::<Self, f32>() {
            f32::to_bits(self.to_f32().unwrap())
        } else {
            unreachable!()
        }
    }
}

impl Sealed for u8 {}

impl Scalar for u8 {}

impl Sealed for i8 {}

impl Scalar for i8 {}

impl Sealed for u16 {}

impl Scalar for u16 {}

impl Sealed for i16 {}

impl Scalar for i16 {}

impl Sealed for f16 {}

impl Scalar for f16 {
    fn to_bits_u32(&self) -> u32 {
        self.to_bits() as u32
    }
}

impl Sealed for bf16 {}

impl Scalar for bf16 {
    fn to_bits_u32(&self) -> u32 {
        self.to_bits() as u32
    }
}

impl Sealed for f32 {}

impl Scalar for f32 {}

impl Sealed for u32 {}

impl Scalar for u32 {}

impl Sealed for i32 {}

impl Scalar for i32 {}

pub trait Num: Scalar {}

impl Num for f32 {}

impl Num for u32 {}

impl Num for i32 {}

#[doc(hidden)]
#[proxy_enum::proxy(DynDevice)]
pub mod dyn_device_proxy {
    use super::*;

    #[derive(Clone)]
    pub enum DynDevice {
        Gpu(Gpu),
    }

    impl DynDevice {
        #[implement]
        pub(super) fn create_buffer(&self, size: usize) -> Result<BufferId> {}
        #[implement]
        pub(super) fn create_buffer_init<T: Pod>(&self, data: Cow<[T]>) -> Result<BufferId> {}
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
        pub(super) fn read_buffer<T: Pod>(
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
const MAX_PUSH_CONSTANT_SIZE: usize = 64;

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

impl Into<std::ops::Range<u32>> for PushConstantRange {
    fn into(self) -> std::ops::Range<u32> {
        self.start..self.end
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
    modules: Arc<DashMap<ModuleId, ShaderModule<'static>>>,
}

impl Device {
    pub fn new_gpu(index: usize) -> Option<impl Future<Output = Result<Self>>> {
        Gpu::new(index).map(|gpu| async move {
            Ok(Self {
                dyn_device: gpu.await?.into(),
                modules: Arc::new(DashMap::default()),
            })
        })
    }
    pub fn list() -> Vec<Self> {
        Self::list_gpus()
    }
    pub fn list_gpus() -> Vec<Self> {
        let mut tasks = Vec::new();
        let mut i = 0;
        while let Some(gpu) = Self::new_gpu(i) {
            tasks.push(smol::spawn(gpu));
            i += 1
        }
        tasks
            .into_iter()
            .filter_map(|task| smol::block_on(task).ok())
            .collect()
    }
    pub fn compute_pass(
        &self,
        spirv: &'static [u8],
        entry_point: impl AsRef<str>,
    ) -> Result<ComputePassBuilder<()>> {
        use dashmap::mapref::entry::Entry::*;
        let module_id = ModuleId(spirv.as_ptr() as usize as u64);
        let entry_point = entry_point.as_ref();
        let (entry_id, entry_descriptor) = match self.modules.entry(module_id) {
            Occupied(occupied) => {
                let (i, entry_descriptor) = occupied
                    .get()
                    .entry_descriptors
                    .iter()
                    .enumerate()
                    .find(|(_, e)| e.name == entry_point)
                    .ok_or(ShaderModuleError::EntryNotFound)?;
                (EntryId(i as u64), entry_descriptor.clone())
            }
            Vacant(vacant) => {
                let module = ShaderModule::from_spirv(spirv)?;
                self.dyn_device.compile_shader_module(module_id, &module)?;
                let module = vacant.insert(module);
                let (i, entry_descriptor) = module
                    .entry_descriptors
                    .iter()
                    .enumerate()
                    .find(|(_, e)| e.name == entry_point)
                    .ok_or(ShaderModuleError::EntryNotFound)?;
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
    pub fn synchronize(&self) -> Result<impl Future<Output = Result<()>>> {
        self.dyn_device.synchronize()
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.dyn_device.fmt(f)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BufferBinding {
    binding: u32,
    id: BufferId,
    offset: u64,
    len: u64,
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
    /// Errors if more buffers are provided than in spirv
    pub fn buffer_slice<T: Scalar>(
        self,
        slice: BufferSlice<T>,
    ) -> Result<ComputePassBuilder<'a, (B, Option<BufferSlice<T>>)>> {
        self.option_buffer_slice(Some(slice))
    }
    /// Adds an optional immutable buffer argument\
    ///
    /// See buffer_slice()
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
                if !buffer_descriptor.mutable {
                    self.compute_pass.buffer_bindings.push(BufferBinding {
                        binding: buffer_descriptor.binding,
                        id: slice.id,
                        offset: (slice.offset * size_of::<T>()) as u64,
                        len: (slice.len * size_of::<T>()) as u64,
                    });
                    Ok(ComputePassBuilder {
                        device: self.device,
                        entry_descriptor: self.entry_descriptor,
                        compute_pass: self.compute_pass,
                        borrows: (self.borrows, Some(slice)),
                    })
                } else {
                    Err(ComputePassBuilderError::BufferMutability {
                        binding: buffer_descriptor.binding,
                        spirv_mutable: buffer_descriptor.mutable,
                        rust_mutable: false,
                    }
                    .into())
                }
            } else {
                Err(ComputePassBuilderError::NumberOfBuffers.into())
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
    /// Must not be readonly in spirv\
    /// Buffers are bound in binding order, low to high\
    /// Errors if more buffers are provided than in spirv
    pub fn buffer_slice_mut<T: Scalar>(
        self,
        slice: BufferSliceMut<T>,
    ) -> Result<ComputePassBuilder<'a, (B, Option<BufferSliceMut<T>>)>> {
        self.option_buffer_slice_mut(Some(slice))
    }
    /// Adds an optional mutable buffer argument\
    ///
    /// See buffer_slice_mut()
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
                if buffer_descriptor.mutable {
                    self.compute_pass.buffer_bindings.push(BufferBinding {
                        binding: buffer_descriptor.binding,
                        id: slice.id,
                        offset: (slice.offset * size_of::<T>()) as u64,
                        len: (slice.len * size_of::<T>()) as u64,
                    });
                    Ok(ComputePassBuilder {
                        device: self.device,
                        entry_descriptor: self.entry_descriptor,
                        compute_pass: self.compute_pass,
                        borrows: (self.borrows, Some(slice)),
                    })
                } else {
                    Err(ComputePassBuilderError::BufferMutability {
                        binding: buffer_descriptor.binding,
                        spirv_mutable: buffer_descriptor.mutable,
                        rust_mutable: true,
                    }
                    .into())
                }
            } else {
                Err(ComputePassBuilderError::NumberOfBuffers.into())
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
    /// Only one push constant block per pass\
    /// The size of C must match the spirv
    pub fn push_constants<C>(mut self, push_constants: C) -> Result<Self>
    where
        C: Pod,
    {
        if let Some(push_constant_descriptor) =
            self.entry_descriptor.push_constant_descriptor.as_ref()
        {
            let PushConstantRange { start, end } = push_constant_descriptor.range;
            if size_of::<C>() == (end - start) as usize {
                self.compute_pass.push_constants = bytemuck::cast_slice(&[push_constants]).to_vec();
                Ok(self)
            } else {
                Err(ComputePassBuilderError::PushConstantSize {
                    spirv: end - start,
                    rust: size_of::<C>() as u32,
                }
                .into())
            }
        } else {
            Err(ComputePassBuilderError::PushConstantSize {
                spirv: 0,
                rust: size_of::<C>() as u32,
            }
            .into())
        }
    }
    /// Sets the number of work groups\
    ///
    /// Use either this method or global_size()\
    /// The provided function f takes the local size [x, y, z] and returns\
    /// the work groups [x, y, z]
    pub fn work_groups(mut self, f: impl Fn([u32; 3]) -> [u32; 3]) -> Self {
        self.compute_pass.work_groups = f(self.entry_descriptor.local_size);
        self
    }
    /// Sets the global size\
    ///
    /// Use either this method or work_groups()\
    /// This will set the work groups such that work_groups * local_size >= global_size
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
    pub fn enqueue(self) -> Result<()> {
        self.device
            .dyn_device
            .enqueue_compute_pass(self.compute_pass)
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
    pub fn from_cow(device: &Device, cow: Cow<[T]>) -> Result<Self>
    where
        T: Scalar,
    {
        let len = cow.len();
        let id = device.dyn_device.create_buffer_init(cow)?;
        Ok(Self {
            device: device.clone(),
            id,
            offset: 0,
            len,
            _m: PhantomData::default(),
        })
    }
    pub fn zeros(device: &Device, len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        let id = device.dyn_device.create_buffer(len * size_of::<T>())?;
        Ok(Self {
            device: device.clone(),
            id,
            offset: 0,
            len,
            _m: PhantomData::default(),
        })
    }
    pub fn from_elem(device: &Device, elem: T, len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        let mut buffer = Self::zeros(device, len)?;
        buffer.fill(elem)?;
        Ok(buffer)
    }
}

impl<T, S: Data<Elem = T>> BufferBase<S> {
    pub fn as_buffer_slice(&self) -> BufferSlice<T> {
        BufferBase {
            device: self.device.clone(),
            id: self.id,
            offset: self.offset,
            len: self.len,
            _m: PhantomData::default(),
        }
    }
    pub fn to_buffer(&self) -> Result<Buffer<T>>
    where
        T: Scalar,
    {
        let buffer = Buffer::zeros(&self.device, self.len)?;
        self.device.dyn_device.copy_buffer_to_buffer(
            self.id,
            self.offset * size_of::<T>(),
            buffer.id,
            0,
            self.len * size_of::<T>(),
        )?;
        Ok(buffer)
    }
    pub fn to_vec(&self) -> Result<impl Future<Output = Result<Vec<T>>>>
    where
        T: Scalar,
    {
        self.device
            .dyn_device
            .read_buffer(self.id, self.offset, self.len)
    }
}

impl<T, S: DataMut<Elem = T>> BufferBase<S> {
    pub fn as_buffer_slice_mut(&mut self) -> BufferSliceMut<T> {
        BufferBase {
            device: self.device.clone(),
            id: self.id,
            offset: self.offset,
            len: self.len,
            _m: PhantomData::default(),
        }
    }
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
            let result = self.device.dyn_device.drop_buffer(self.id);
            #[cfg(debug_assertions)]
            result.unwrap();
        }
    }
}
