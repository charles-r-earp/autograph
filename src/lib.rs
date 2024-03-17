/*!
# **autograph**
A machine learning library for Rust.

GPGPU kernels implemented with [**krnl**](https://github.com/charles-r-earp/krnl).
- Host / Device execution.
- Tensors emulate [ndarray](https://github.com/rust-ndarray/ndarray)
   - Host tensors can be borrowed as arrays.
- Tensors / Models / Optimizers can be serialized with [serde](https://github.com/serde-rs/serde).
   - Portable between platforms.
   - Save / resume training progress.
- Fully extensible, in Rust.
*/
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

/// Ops.
pub mod ops;

/// Tensors.
pub mod tensor;

/// Datasets.
#[cfg(feature = "dataset")]
pub mod dataset;

/// Machine Learning.
#[cfg(feature = "learn")]
pub mod learn;
