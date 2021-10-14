#![warn(missing_docs)]

//! # autograph
//! A machine learning library for Rust.
//!
//! To use autograph in your crate, add it as a dependency in Cargo.toml:
//!```text
//! [dependencies]
//! autograph = { git = https://github.com/charles-r-earp/autograph }
//!```
//!
//! # Requirements
//! - A device (typically a gpu) with drivers for a supported API:
//!     - Vulkan (All platforms) <https://www.vulkan.org/>
//!     - Metal (MacOS / iOS) <https://developer.apple.com/metal/>
//!     - DX12 (Windows) <https://docs.microsoft.com/windows/win32/directx>

#![cfg_attr(feature = "bench", feature(test))]

#[cfg(feature = "bench")]
extern crate test;

#[cfg(feature = "derive")]
#[allow(unused_imports)]
#[macro_use]
extern crate autograph_derive;

/// Result type.
pub mod result {
    pub use anyhow::Result;
}
/// Error type.
pub mod error {
    pub use anyhow::Error;
}
/// Devices.
pub mod device;

mod util;

mod glsl_shaders;
mod rust_shaders;

/// Scalar types.
pub mod scalar;

// Linear Algebra.
mod linalg;
/// Numerical operations.
pub mod ops;

/// Buffers.
pub mod buffer;

/// Tensors.
#[cfg(feature = "tensor")]
pub mod tensor;

#[doc(hidden)]
#[cfg(feature = "tensor")]
pub use tensor::float as float_tensor;

/// Datasets.
pub mod dataset;

/// Machine Learning.
#[cfg(feature = "tensor")]
pub mod learn;
