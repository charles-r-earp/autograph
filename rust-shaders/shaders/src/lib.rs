#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, generic_const_exprs, atomic_from_mut),
    register_attr(spirv)
)]
#![allow(incomplete_features)]
#![deny(warnings)]


mod util;
mod half;
pub mod fill;
pub mod cast;
pub mod activation;
pub mod kernel;
pub mod reorder;
pub mod pool;
pub mod linalg;
pub mod criterion;
