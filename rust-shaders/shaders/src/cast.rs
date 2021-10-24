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

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn scale_u8_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &ScalePushConsts<f32>,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    let yid = gid * 4;
    if yid < n {
        let ys = (alpha * u8x4_to_uvec4(x[gid]).as_vec4()).to_array();
        y[yid + 0] = ys[0];
        if yid + 1 < n {
            y[yid + 1] = ys[1];
        }
        if yid + 2 < n {
            y[yid + 2] = ys[2];
        }
        if yid + 3 < n {
            y[yid + 3] = ys[3];
        }
    }
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn scale_u8_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &ScalePushConsts<f32>,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    if gid * 4 < n {
        // Hack that fixes issue on DX12.
        y[gid * 2] = y[gid * 2];
        let (y0, y1) = vec4_to_bf16x4(alpha * u8x4_to_uvec4(x[gid]).as_vec4());
        if gid * 4 + 1 < n {
            y[gid * 2] = y0;
        }else {
            y[gid * 2] = y0 & 0xFFFF;
        }
        if gid * 4 + 3 < n {
            y[gid * 2 + 1] = y1;
        } else if gid * 4 + 2 < n {
            y[gid * 2 + 1] = y1 & 0xFFFF;
        }
    }
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn scale_bf16_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &ScalePushConsts<f32>,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    if gid * 2 < n {
        // Hack that fixes issue on DX12.
        y[gid] = y[gid];
        let ys = vec2_to_bf16x2(alpha * bf16x2_to_vec2(x[gid]));
        if gid * 2 + 1 < n {
            y[gid] = ys;
        } else {
            y[gid] = ys & 0xFFFF;
        }
    }
}
