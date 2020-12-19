use crate::backend::{BufferId, EntryId, ModuleId};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug, Display};

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Error {
    ShapeError(ShapeError),
    GpuError(GpuError),
    ShaderModuleError(ShaderModuleError),
    ComputePassBuilder(ComputePassBuilderError),
    Unimplemented,
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &'static str {
        "autograph::error::Error"
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ShapeError {
    IncompatibleShape,
    IncompatibleLayout,
    RangeLimited,
    OutOfBounds,
    Unsupported,
    Overflow,
    Unknown,
}

impl From<ndarray::ShapeError> for ShapeError {
    fn from(shape_error: ndarray::ShapeError) -> Self {
        use ndarray::ErrorKind::*;
        match shape_error.kind() {
            IncompatibleShape => Self::IncompatibleShape,
            IncompatibleLayout => Self::IncompatibleLayout,
            RangeLimited => Self::RangeLimited,
            OutOfBounds => Self::OutOfBounds,
            Unsupported => Self::Unsupported,
            Overflow => Self::Overflow,
            _ => Self::Unknown,
        }
    }
}

impl From<ShapeError> for Error {
    fn from(shape_error: ShapeError) -> Self {
        Self::ShapeError(shape_error)
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(shape_error: ndarray::ShapeError) -> Self {
        ShapeError::from(shape_error).into()
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuError {
    RequestDeviceError,
    BufferAsyncError,
    BufferNotFound(BufferId),
    ModuleNotFound(ModuleId),
    EntryNotFound(EntryId),
}

impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(_e: wgpu::RequestDeviceError) -> Self {
        Self::RequestDeviceError
    }
}

impl From<wgpu::BufferAsyncError> for GpuError {
    fn from(_e: wgpu::BufferAsyncError) -> Self {
        Self::RequestDeviceError
    }
}

impl From<GpuError> for Error {
    fn from(gpu_error: GpuError) -> Self {
        Self::GpuError(gpu_error)
    }
}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        GpuError::from(e).into()
    }
}

impl From<wgpu::BufferAsyncError> for Error {
    fn from(e: wgpu::BufferAsyncError) -> Self {
        GpuError::from(e).into()
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ShaderModuleError {
    EntryNotFound,
    InvalidSpirv,
}

impl From<ShaderModuleError> for Error {
    fn from(e: ShaderModuleError) -> Self {
        Self::ShaderModuleError(e)
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ComputePassBuilderError {
    PushConstantSize { spirv: u32, rust: u32 },
    NumberOfBuffers,
    BufferMutability { binding: u32, spirv_mutable: bool, rust_mutable: bool },
    InvalidDevice,
}

impl From<ComputePassBuilderError> for Error {
    fn from(e: ComputePassBuilderError) -> Self {
        Self::ComputePassBuilder(e)
    }
}
