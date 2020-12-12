use crate::{error::ShaderModuleError, Result};
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

pub mod shader_util;

#[doc(hidden)]
pub mod gpu;
use gpu::Gpu;

#[allow(unused)]
macro_rules! include_bytes_align_as {
    ($align_ty:ty, $file:expr) => {{
        #[repr(C)]
        pub struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }

        static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($file),
        };

        &ALIGNED.bytes
    }};
}

#[macro_export]
macro_rules! include_spirv {
    ($file:expr) => {{
        include_bytes_align_as!(u32, $file)
    }};
}

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
        pub(super) fn compile_shader_module(&self, module: &ShaderModule) -> Result<()> {}
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
const MAX_PUSH_CONSTANT_SIZE: usize = 32;

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
    entries: Vec<EntryDescriptor>,
}

impl<'a> ShaderModule<'a> {
    fn from_spirv(spirv: impl Into<Cow<'a, [u8]>>) -> Result<Self> {
        let spirv = spirv.into();
        let entries = shader_util::entry_descriptors_from_spirv(&spirv)?;
        Ok(Self { spirv, entries })
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
        let (entry_id, entry_descriptor) = match self.modules.entry(module_id) {
            Occupied(occupied) => {
                let (i, entry_descriptor) = occupied
                    .get()
                    .entries
                    .iter()
                    .enumerate()
                    .find(|(_, e)| &e.name == entry_point.as_ref())
                    .ok_or(ShaderModuleError::EntryNotFound)?;
                (EntryId(i as u64), entry_descriptor.clone())
            }
            Vacant(vacant) => {
                let module = vacant.insert(ShaderModule::from_spirv(spirv)?);
                let (i, entry_descriptor) = module
                    .entries
                    .iter()
                    .enumerate()
                    .find(|(_, e)| &e.name == entry_point.as_ref())
                    .ok_or(ShaderModuleError::EntryNotFound)?;
                (EntryId(i as u64), entry_descriptor.clone())
            }
        };
        Ok(ComputePassBuilder {
            device: self,
            entry_descriptor,
            compute_pass: ComputePass {
                module_id,
                entry_id,
                buffer_bindings: Vec::new(),
                push_constants: Vec::new(),
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
pub struct ComputePassBuilder<'a, T> {
    device: &'a Device,
    entry_descriptor: EntryDescriptor,
    compute_pass: ComputePass,
    borrows: T,
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputePass {
    module_id: ModuleId,
    entry_id: EntryId,
    buffer_bindings: Vec<BufferBinding>,
    push_constants: Vec<u8>,
}

#[doc(hidden)]
pub trait Data {
    type Elem;
    #[doc(hidden)]
    fn needs_drop() -> bool;
}

pub trait DataMut: Data {}

pub struct BufferRepr<T>(PhantomData<T>);

impl<T> Data for BufferRepr<T> {
    type Elem = T;
    fn needs_drop() -> bool {
        true
    }
}

impl<T> DataMut for BufferRepr<T> {}

pub struct BufferSliceRepr<S>(PhantomData<S>);

impl<T> Data for BufferSliceRepr<&'_ T> {
    type Elem = T;
    fn needs_drop() -> bool {
        false
    }
}

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
        T: Pod,
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
    pub fn zeros(device: &Device, len: usize) -> Result<Self> {
        let id = device.dyn_device.create_buffer(len * size_of::<T>())?;
        Ok(Self {
            device: device.clone(),
            id,
            offset: 0,
            len,
            _m: PhantomData::default(),
        })
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
    pub fn to_vec(&self) -> Result<impl Future<Output = Result<Vec<T>>>>
    where
        T: Pod,
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
