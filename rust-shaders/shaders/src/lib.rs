#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]
#![deny(warnings)]

mod util;
mod half;
pub mod fill;
pub mod cast;
pub mod activation;
pub mod kernel;
pub mod reorder;
pub mod pool;
