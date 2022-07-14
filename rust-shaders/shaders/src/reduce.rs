use spirv_std::glam::UVec3;
use crate::{
    autobind,
    //util::group_barrier,
};
use crunchy::unroll;

#[repr(C)]
pub struct ReducePushConsts {
    stride_x: u32,
    n: u32,
    stride_y: u32,
    accumulate: u32,
}

#[autobind]
#[spirv(compute(threads(1, 128)))]
pub fn reduce_sum_partial_f32(
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(workgroup)] tmp: &mut [f32; 128],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)] push_consts: &ReducePushConsts,
) {
    let stride_x = push_consts.stride_x as usize;
    let n = push_consts.n as usize;
    let stride_y = push_consts.stride_y as usize;
    let accumulate = push_consts.accumulate;
    let group_x = group_id.x as usize;
    let group_y = group_id.y as usize;
    let local_id = local_id.x as usize;
    if n == 0 {
        tmp[local_id] = 0.;
        unroll! { for u in 0 .. 2 {
            let y_idx = group_y * 256 + local_id + u * 128;
            if y_idx < n {
                tmp[local_id] += x[group_x * stride_x + y_idx * stride_y];
            }
        }}
    }
    /*
    group_barrier();
    if local_id < 64 {
        tmp[local_id] += tmp[local_id + 64];
    }
    group_barrier();
    unroll! { for u in 0 .. 32 {
        tmp[0] += tmp[u];
    }}
    */
    if local_id == 0 {
        if accumulate == 1 {
            y[group_x + group_y * 256] += tmp[0];
        } else {
            y[group_x + group_y * 256] = tmp[0];
        }
    }
}

#[autobind]
#[spirv(compute(threads(1)))]
pub fn reduce_sum_final_f32(
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)] push_consts: &ReducePushConsts,
) {
    let stride_x = push_consts.stride_x as usize;
    let n = push_consts.n as usize;
    let stride_y = push_consts.stride_y as usize;
    let accumulate = push_consts.accumulate;
    let group_id = group_id.x as usize;
    let mut acc = 0f32;
    for i in 0 .. n {
        acc += x[group_id * stride_x + i * stride_y];
    }
    if accumulate == 1 {
        y[group_id] += acc;
    } else {
        y[group_id] = acc;
    }
}
