#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, repr_simd),
    register_attr(spirv)
)]
#![deny(warnings)]

use spirv_std::{
    scalar::Scalar,
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
pub struct FillPushConsts<T: Scalar> {
    x: T,
    n: u32,
}

#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn fill_u32(
    #[spirv(global_invocation_id)]
    global_id: GlobalId,
    y: &mut ArrayBuffer<u32>,
    #[spirv(push_constant)]
    push_consts: &FillPushConsts<u32>,
) {
    if global_id.x < push_consts.n {
        unsafe {
            y.store(global_id.x, push_consts.x);
        }
    }
}

#[cfg(target_feature = "Int64")]
#[spirv(compute(threads(64)))]
pub fn fill_u64(
    #[spirv(global_invocation_id)]
    global_id: GlobalId,
    y: &mut ArrayBuffer<u64>,
    #[spirv(push_constant)]
    push_consts: &FillPushConsts<u64>,
) {
    if global_id.x < push_consts.n {
        unsafe {
            y.store(global_id.x, push_consts.x);
        }
    }
}
