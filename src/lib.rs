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
/// Device level backend.
pub mod device;
#[cfg(feature = "tensor")]
mod glsl_shaders;
/// Linear Algebra.
mod linalg;
/// Numerical operations.
mod ops;
mod rust_shaders;
/// Scalar types.
pub mod scalar;
/// Tensors.
#[cfg(feature = "tensor")]
pub mod tensor;
mod util;

/// Datasets.
pub mod dataset;

/// Machine Learning.
pub mod learn;
