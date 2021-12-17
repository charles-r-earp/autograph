#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, generic_const_exprs, asm, asm_experimental_arch, asm_const, adt_const_params),
    register_attr(spirv)
)]
#![allow(incomplete_features)]
#![deny(warnings)]


pub mod util;
mod half;
mod atomic;
pub mod fill;
pub mod cast;
pub mod activation;
pub mod kernel;
pub mod reorder;
pub mod pool;
pub mod linalg;
pub mod criterion;

use autograph_derive::autobind;
