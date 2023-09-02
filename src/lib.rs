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
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "derive")]
#[allow(unused_imports)]
#[macro_use]
extern crate autograph_derive;

/// krnl
pub extern crate krnl;
pub use krnl::{buffer, device, scalar};
/// anyhow
pub extern crate anyhow;
/// half
pub extern crate half;
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
