use crate::Result;
use bytemuck::Pod;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smol::future::Future;
use spirv::AccessQualifier;
use spirv_headers as spirv;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;

#[cfg(feature = "staticvec")]
use staticvec::StaticVec;

#[doc(hidden)]
pub mod gpu;
use gpu::Gpu;

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
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PushConstantRange {
    start: u32,
    end: u32,
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryDescriptor {
    name: String,
    #[cfg(feature = "staticvec")]
    buffers: StaticVec<AccessQualifier, MAX_BUFFERS_PER_COMPUTE_PASS>,
    #[cfg(not(feature = "staticvec"))]
    buffers: Vec<AccessQualifier>,
    #[cfg(feature = "staticvec")]
    push_constants: StaticVec<u8, MAX_PUSH_CONSTANT_SIZE>,
    #[cfg(not(feature = "staticvec"))]
    push_constants: Vec<u8>,
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderModule<'a> {
    spirv: Cow<'a, [u8]>,
    entries: Vec<EntryDescriptor>,
}

#[derive(Clone)]
pub struct Device {
    dyn_device: DynDevice,
    modules: DashMap<ModuleId, ShaderModule<'static>>,
}

impl Device {
    pub fn new_gpu(index: usize) -> Option<impl Future<Output = Result<Self>>> {
        Gpu::new(index).map(|gpu| async move {
            Ok(Self {
                dyn_device: gpu.await?.into(),
                modules: DashMap::default(),
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
    #[allow(unused)]
    pub fn compute_pass(
        &self,
        spirv: &'static [u8],
        entry_point: impl AsRef<str>,
    ) -> Result<ComputePassBuilder<()>> {
        todo!()
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
    id: BufferId,
    access: AccessQualifier,
    offset: u64,
    len: u64,
}

#[allow(unused)]
pub struct ComputePassBuilder<'a, T> {
    device: &'a Device,
    compute_pass: ComputePass,
    borrows: T,
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputePass {
    module: ModuleId,
    entry: EntryId,
    #[cfg(feature = "staticvec")]
    buffer_bindings: StaticVec<BufferBinding, MAX_BUFFERS_PER_COMPUTE_PASS>,
    #[cfg(not(feature = "staticvec"))]
    buffer_bindings: Vec<BufferBinding>,
    #[cfg(feature = "staticvec")]
    push_constants: StaticVec<u8, MAX_PUSH_CONSTANT_SIZE>,
    #[cfg(not(feature = "staticvec"))]
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
