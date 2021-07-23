#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]
#![deny(warnings)]

use spirv_std::glam::UVec3;

#[repr(C)]
pub struct FillPushConstsU32 {
    n: u32,
    x: u32,
}

#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn fill_u32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0, non_readable)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &FillPushConstsU32,
) {
    if global_id.x < push_consts.n {
        unsafe {
            y[global_id.x as usize] = push_consts.x;
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0, non_readable)] y: &mut &[u32],
    #[spirv(push_constant)]
    push_consts: &FillPushConstsU64,
) {
    if global_id.x < push_consts.n {
        let index = (global_id.x as usize) * 2;
        unsafe {
            y[index] = push_consts.x1;
            y[index + 1] = push_consts.x2;
        }
    }
}
