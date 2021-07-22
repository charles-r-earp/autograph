use super::{
    Result,
    shader::{Module, ModuleId, EntryId},
};
use std::{
    sync::Arc,
    future::Future,
};
use derive_more::Display;
use futures_channel::oneshot::Receiver as OneShotReceiver;
use hibitset::AtomicBitSet;
use anyhow::anyhow;
pub use gfx_hal::adapter::DeviceType;

mod local;
use local::Engine;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferId(u64);

#[derive(Clone, Copy, Debug, Display, Eq, PartialEq, thiserror::Error)]
#[repr(u8)]
pub(super) enum DeviceError {
    DeviceUnsupported,
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
    MissingFeature,
    TooManyObjects,
    DeviceLost,
    MappingFailed,
}

type DeviceResult<T> = Result<T, DeviceError>;

enum DataVec {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

enum DataSlice<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

enum Data<'a> {
    Vec(DataVec),
    Slice(DataSlice<'a>),
}

pub(super) struct BufferRef {
    id: BufferId,
    width: u8,
    offset: u32,
    len: u32,
}

pub(super) struct ComputePass {
    module: ModuleId,
    entry: EntryId,
    local_size: [u32; 3],
    buffers: Vec<BufferRef>,
    push_constants: Vec<u8>,
}

pub mod builders {
    use super::*;

    pub struct DeviceBuilder {

    }

    impl DeviceBuilder {
        pub fn backend(&self) -> Backend {
            todo!()
        }
        pub fn device_type(&self) -> DeviceType {
            todo!()
        }
    }
}
use builders::DeviceBuilder;

#[non_exhaustive]
pub enum Backend {
    Vulkan,
    Metal,
    DX12,
}

#[derive(Default, Clone)]
pub struct Device {
    engine: Option<Arc<Engine>>,
    compute: bool,
}

impl Device {
    /// Returns a host device.\
    ///
    /// The host supports basic operations like initialization and copying. It does not support
    /// shader execution, so most computations will fail.
    pub fn host() -> Self {
        Self::default()
    }
    pub fn builder_iter() -> impl Iterator<Item=DeviceBuilder> {
        std::iter::empty()
    }
    /// Returns the device with compute shaders enabled or disabled.\
    ///
    /// Compute passes will still be validated, but not executed. Useful for preventing host\
    /// functions from failing, either when testing or for initialization. Calls to uninitialized\
    /// are replaced with zeros.
    #[doc(hidden)]
    pub fn with_compute(mut self, enable: bool) -> Self {
        self.compute = enable;
        self
    }
}

pub(crate::buffer) struct DeviceBase {

}
