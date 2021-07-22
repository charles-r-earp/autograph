#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, repr_simd),
    register_attr(spirv)
)]
#![deny(warnings)]

use spirv_std::{
    bindless::ArrayBuffer,
};

#[repr(simd)]
#[allow(unused)]
pub struct GlobalId {
    x: u32,
    y: u32,
    z: u32
}

#[repr(C)]
pub struct FillPushConstsU32 {
    n: u32,
    x: u32,
}

#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn fill_u32(
    #[spirv(global_invocation_id)]
    global_id: GlobalId,
    y: &mut ArrayBuffer<u32>,
    #[spirv(push_constant)]
    push_consts: &FillPushConstsU32,
) {
    if global_id.x < push_consts.n {
        unsafe {
            y.store(global_id.x, push_consts.x);
        }
    }
}

#[repr(C)]
pub struct FillPushConstsU64 {
    n: u32,
    x1: u32,
    x2: u32,
}

#[spirv(compute(threads(64)))]
pub fn fill_u64(
    #[spirv(global_invocation_id)]
    global_id: GlobalId,
    y: &mut ArrayBuffer<u32>,
    #[spirv(push_constant)]
    push_consts: &FillPushConstsU64,
) {
    if global_id.x < push_consts.n {
        unsafe {
            y.store(global_id.x * 2, push_consts.x1);
            y.store(global_id.x * 2 + 1, push_consts.x2);
        }
    }
}
