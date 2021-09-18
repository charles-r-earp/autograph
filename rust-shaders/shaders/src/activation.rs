use crate::half::{bf16x2_to_vec2, vec2_to_bf16x2};
use spirv_std::glam::{UVec3, vec2};

#[repr(C)]
pub struct ReluPushConsts {
    n: u32,
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn relu_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    if gid * 2 < n {
        let x = bf16x2_to_vec2(x[gid]);
        let y0 = if x.x > 0. { x.x } else { 0. };
        let y1 = if x.y > 0. { x.y } else { 0. };
        /*if gid * 2 + 1 >= n {
            //y1 = bf16x2_to_vec2(y[gid]).y;
        }*/
        y[gid] = vec2_to_bf16x2(vec2(y0, y1));
    }
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn relu_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    if gid < n {
        let x = x[gid];
        y[gid] = if x > 0. { x } else { 0. };
    }
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn relu_backward_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dx: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dy: &[u32],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    if gid * 2 < n {
        let x = bf16x2_to_vec2(x[gid]);
        let _dx = bf16x2_to_vec2(dx[gid]);
        let dy = bf16x2_to_vec2(dy[gid]);
        let dx0 = if x.x > 0. { _dx.x + dy.x } else { _dx.x };
        let dx1 = if x.y > 0. { _dx.y + dy.y } else { _dx.y };
        let dx_out = vec2_to_bf16x2(vec2(dx0, dx1));
        if gid * 2 + 1 < n {
            dx[gid] = dx_out;
        } else {
            dx[gid] = dx_out & 0xFFFF;
        }
    }
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn relu_backward_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dx: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dy: &[f32],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    if gid < n {
        if x[gid] > 0. {
            dx[gid] += dy[gid];
        }
    }
}
