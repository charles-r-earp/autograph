#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, repr_simd),
    register_attr(spirv)
)]
#![deny(warnings)]

use spirv_std::bindless::ArrayBuffer;

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

#[cfg(target_feature = "Int8")]
#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn fill_u8(
    #[spirv(global_invocation_id)]
    global_id: GlobalId,
    y: &mut ArrayBuffer<u32>,
    #[spirv(push_constant)]
    push_consts: &FillPushConstsU32,
) {
    let r = push_consts.n % 4;
    let u = if r == 0 { push_consts.n / 4 } else { push_consts.n / 4 + 1 };
    let mut x = y.load(global_id.x);
    let mask = if r == 1 {
        0xFF
    } else if r == 2 {
        0xFFFF
    } else {
        0xFFFFFF
    };
    x |= push_consts.x & mask;
    if global_id.x < u {
        unsafe {
            y.store(global_id.x, x);
        }
    }
}

/*
#[cfg(target_feature = "Int8")]
#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn fill_u8(
    #[spirv(global_invocation_id)]
    global_id: GlobalId,
    y: &mut ArrayBuffer<u32>,
    #[spirv(push_constant)]
    push_consts: &FillPushConstsU32,
) {
    let n_words = push_consts.n / 4;
    if global_id.x < n_words {
        unsafe {
            y.store(global_id.x, push_consts.x);
        }
    } else if push_consts.n % 4 > 0 && global_id.x == n_words {
        unsafe {
            y.store(global_id.x, 0);
        }
        /*match push_consts.n % 4 {
            //1 => x |= push_consts.x & 0xFF,
            //2 => x |= push_consts.x & 0xFFFF,
            //3 => x |= push_consts.x & 0xFFFFFF,
            _ => (),
        }*/
        /*unsafe {
            y.store(global_id.x, x);
        }*/
    }
}*/

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
