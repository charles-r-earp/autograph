#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, generic_const_exprs, asm, asm_experimental_arch, asm_const, adt_const_params),
    register_attr(spirv)
)]
#![allow(incomplete_features)]
#![deny(warnings)]


mod util;
mod byte;
mod short;
mod half;
pub mod atomic;
pub mod fill;
pub mod copy;
pub mod cast;
pub mod reduce;
pub mod activation;
pub mod kernel;
pub mod reorder;
pub mod pool;
pub mod linalg;
pub mod criterion;

use autograph_derive::autobind;
