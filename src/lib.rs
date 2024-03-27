/*!
A machine learning library for Rust.

GPGPU kernels implemented with [krnl].
- Host and device execution.
- Tensors emulate [ndarray].
   - Host tensors can be borrowed as arrays.
- Tensors, models, and optimizers can be serialized with [serde](https://docs.rs/serde).
   - Portable between platforms.
   - Save and resume training progress.
- Fully extensible, in Rust.
*/
#![cfg_attr(doc_cfg, feature(doc_auto_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// anyhow
pub extern crate anyhow;
/// half
pub extern crate half;
/// krnl
pub extern crate krnl;
/// ndarray
pub extern crate ndarray;
/// num-traits
pub extern crate num_traits;

/// Datasets.
#[cfg(feature = "dataset")]
pub mod dataset;
/// Machine Learning.
#[cfg(feature = "learn")]
pub mod learn;
/// Ops.
pub mod ops;
/// Tensors.
pub mod tensor;
