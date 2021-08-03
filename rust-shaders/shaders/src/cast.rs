use crate::{
    util::u8x4_to_uvec4,
    half::{bf16x2_to_vec2, vec2_to_bf16x2, vec4_to_bf16x4}
};
use spirv_std::glam::UVec3;

#[repr(C)]
pub struct ScalePushConsts<A> {
    n: u32,
    alpha: A
}

#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn scale_u8_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0, non_writable)] x: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1, non_readable)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &ScalePushConsts<f32>,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    let yid = gid * 4;
    if yid < n {
        let len = if n - yid >= 4 {
            4
        } else {
            n - yid
        };
        let ys = (alpha * u8x4_to_uvec4(x[gid]).as_f32()).to_array();
        for i in 0 .. len {
            y[yid + i] = ys[i];
        }
    }
}

#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn scale_u8_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0, non_writable)] x: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1, non_readable)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &ScalePushConsts<f32>,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    if gid * 4 < n {
        let ys = vec4_to_bf16x4(alpha * u8x4_to_uvec4(x[gid]).as_f32());
        if gid * 4 + 1 < n {
            y[gid * 2] = ys.0;
        } else {
            y[gid * 2] = ys.0 & 0xFFFF;
        }
        if gid * 4 + 3 < n {
            y[gid * 2 + 1] = ys.1;
        } else if gid * 4 + 2 < n {
            y[gid * 2 + 1] = ys.1 & 0xFFFF;
        }
    }
}

#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn scale_bf16_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0, non_writable)] x: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1, non_readable)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &ScalePushConsts<f32>,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    if gid * 2 < n {
        let ys = vec2_to_bf16x2(alpha * bf16x2_to_vec2(x[gid]));
        if gid * 2 + 1 < n {
            y[gid] = ys;
        } else {
            y[gid] = ys & 0xFFFF;
        }
    }
}

/*
macro_rules! impl_scale {
    ($( $($f1:literal)? $t1:ident),+ => $t2s:tt) => {
        $(
            pub mod $t1 {
                use super::*;

                impl_scale!{@Inner $($f1)? $t1 => $t2s}
            }
        )+
    };
    (@Inner $($f1:literal)? $t1:ident => ($($t2:ident | $a:ident),+)) => {
        $(
            $(#[target_feature(enable = $f1)])?
            #[allow(unused)]
            #[spirv(compute(threads(64)))]
            pub fn $t2(
                #[spirv(global_invocation_id)]
                global_id: UVec3,
                #[spirv(storage_buffer, descriptor_set = 0, binding = 0, non_writable)] x: &mut [$t1],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 1, non_readable)] y: &mut [$t2],
                #[spirv(push_constant)]
                push_consts: &ScalePushConsts<$a>,
            ) {
                let alpha = <$t2 as NumCast>::from(push_consts.alpha).unwrap();
                let vs = vector_size::<$t1>().max(vector_size::<$t2>()) as u32;
                let start = global_id.x as usize;
                let end = (global_id.x + vs).min(push_consts.n) as usize;
                for i in start .. end {
                    y[i] = <$t2 as NumCast>::from(x[i]).unwrap() * alpha;
                }
            }
        )+
    }
}

pub mod scale {
    use super::*;

    impl_scale!{"Int8" u8 => (bf16 | f32, f32 | f32)}
}
*/
