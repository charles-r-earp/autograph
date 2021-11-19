use spirv_std::{
    memory::{Scope, Semantics},
    arch::control_barrier,
    glam::UVec3,
};
use num_traits::NumAssign;
use crunchy::unroll;

#[repr(C)]
pub struct GemmPushConsts<T> {
    alpha: T,
    beta: T,
    a0: T,
    m: u32,
    k: u32,
    n: u32,
    rsa: u32,
    csa: u32,
    rsb: u32,
    csb: u32,
    rsc: u32,
    csc: u32,
}

const TS: usize = 16;

pub fn gemm<T: Copy + NumAssign + PartialEq>(
    global_id: UVec3,
    local_id: UVec3,
    a: &[T],
    a_tile: &mut [[T; TS]; TS],
    b: &[T],
    b_tile: &mut [[T; TS]; TS],
    use_bias: u32,
    bias: &[T],
    c: &mut [T],
    push_consts: &GemmPushConsts<T>,
) {
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
    let local_x = local_id.x as usize;
    let local_y = local_id.y as usize;
    let alpha = push_consts.alpha;
    let beta = push_consts.beta;
    let m = push_consts.m as usize;
    let k = push_consts.k as usize;
    let n = push_consts.n as usize;
    let rsa = push_consts.rsa as usize;
    let csa = push_consts.csa as usize;
    let rsb = push_consts.rsb as usize;
    let csb = push_consts.csb as usize;
    let rsc = push_consts.rsc as usize;
    let csc = push_consts.csc as usize;

    let mut ntiles = k / TS;
    if ntiles * TS < k {
        ntiles += 1;
    }

    let mut acc = T::zero();
    for t in 0 .. ntiles {
        let z = t * TS;
        a_tile[local_y][local_x] = if global_x < m {
            if z + local_y < k {
                a[global_x * rsa + (z + local_y) * csa]
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };
        b_tile[local_x][local_y] = if z + local_x < k {
            if global_y < n {
                b[(z + local_x) * rsb + global_y * csb]
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };
        unsafe {
            control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
        }
        unroll! {
            for u in 0 .. 16 {
                acc += a_tile[u][local_x] * b_tile[u][local_y];
            }
        }
        unsafe {
            control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
        }
    }
    if global_x < m { if global_y < n {
        let idx = global_x * rsc + global_y * csc;
        let mut y = alpha * acc;
        if beta != T::zero() {
            y += beta * c[idx];
        }
        if use_bias == 1 {
            y += bias[global_y];
        }
        c[idx] = y;
    }}
}

#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn gemm_u32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[u32],
    #[spirv(workgroup)]
    a_tile: &mut [[u32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[u32],
    #[spirv(workgroup)]
    b_tile: &mut [[u32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<u32>,
) {
    gemm(global_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn gemm_i32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[i32],
    #[spirv(workgroup)]
    a_tile: &mut [[i32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[i32],
    #[spirv(workgroup)]
    b_tile: &mut [[i32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [i32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<i32>,
) {
    gemm(global_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn gemm_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm(global_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn gemm_bias_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; TS]; TS],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    bias: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm(global_id, local_id, a, a_tile, b, b_tile, 1, bias, c, push_consts);
}
