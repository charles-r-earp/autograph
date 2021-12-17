use crate::{util::{Load, Store}, half::bf16x2, autobind};
use spirv_std::glam::UVec3;

#[repr(C)]
pub struct ReluPushConsts {
    n: u32,
}

fn relu<T>(
    global_id: UVec3,
    x: &[T],
    y: &mut [T],
    push_consts: &ReluPushConsts,
) where [T]: Store<f32> {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    if gid < n {
        let x = x.load(gid);
        y.store(gid, if x > 0. { x } else { 0. });
    }
}

#[autobind]
#[spirv(compute(threads(64)))]
pub fn relu_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[bf16x2],
    #[spirv(storage_buffer)] y: &mut [bf16x2],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    relu(global_id, x, y, push_consts);
}

#[autobind]
#[spirv(compute(threads(64)))]
pub fn relu_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    relu(global_id, x, y, push_consts);
}

fn relu_backward<T>(
    global_id: UVec3,
    x: &[T],
    dx: &mut [T],
    dy: &[T],
    push_consts: &ReluPushConsts,
) where [T]: Store<f32> {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    if gid < n {
        if x.load(gid) > 0. {
            dx.store(gid, dx.load(gid) + dy.load(gid));
        }
    }
}

#[autobind]
#[spirv(compute(threads(64)))]
pub fn relu_backward_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[bf16x2],
    #[spirv(storage_buffer)] dx: &mut [bf16x2],
    #[spirv(storage_buffer)] dy: &[bf16x2],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    relu_backward(global_id, x, dx, dy, push_consts);
}

#[autobind]
#[spirv(compute(threads(64)))]
pub fn relu_backward_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] dx: &mut [f32],
    #[spirv(storage_buffer)] dy: &[f32],
    #[spirv(push_constant)]
    push_consts: &ReluPushConsts,
) {
    relu_backward(global_id, x, dx, dy, push_consts);
}
