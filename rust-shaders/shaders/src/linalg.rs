use spirv_std::{
    memory::{Scope, Semantics},
    arch::control_barrier,
    glam::UVec3,
};
use num_traits::NumAssign;

use crunchy::unroll;

macro_rules! var_unroll {
    (for $i:ident in 0 .. $n:tt $b:block)=> {{
        let mut $i = 0;
        while $i < $n {
            unroll! { for _u in 0 .. 8 {
                if $i < $n {
                    $b
                }
                $i += 1;
            }}
        }
    }};
}


#[repr(C)]
pub struct GemmPushConsts<T> {
    alpha: T,
    beta: T,
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

pub fn gemm<T: Copy + NumAssign + PartialEq, const TS: usize>(
    group_id: UVec3,
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

    let group_id = group_id.x as usize;
    let n_groups_y = n / TS + if n % TS != 0 { 1 } else { 0 };
    let group_x = group_id / n_groups_y;
    let group_y = group_id % n_groups_y;
    let local_id = local_id.x as usize;
    let local_x = local_id / TS;
    let local_y = local_id % TS;
    let global_x = group_x * TS + local_x;
    let global_y = group_y * TS + local_y;

    let mut a_idx = (group_x * TS + local_x) * rsa + local_y * csa;
    let mut b_idx = local_x * rsb + (group_y * TS + local_y) * csb;

    let ntiles = k / TS + if k % TS != 0 { 1 } else { 0 };
    let mut acc = T::zero();

    for t in 0 .. ntiles {
        let z = t * TS;
        a_tile[local_y][local_x] = if global_x < m {
            if z + local_y < k {
                a[a_idx]
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };
        a_idx += TS * csa;
        b_tile[local_x][local_y] = if z + local_x < k {
            if global_y < n {
                b[b_idx]
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };
        b_idx += TS * rsb;
        unsafe {
            control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
        }
        var_unroll! {
            for u in 0 .. TS {
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
#[spirv(compute(threads(256)))]
pub fn gemm_u32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[u32],
    #[spirv(workgroup)]
    a_tile: &mut [[u32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[u32],
    #[spirv(workgroup)]
    b_tile: &mut [[u32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<u32>,
) {
    gemm(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_i32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[i32],
    #[spirv(workgroup)]
    a_tile: &mut [[i32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[i32],
    #[spirv(workgroup)]
    b_tile: &mut [[i32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [i32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<i32>,
) {
    gemm(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_bias_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    bias: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm(group_id, local_id, a, a_tile, b, b_tile, 1, bias, c, push_consts);
}

/*const MACRO: usize = 128;
const MICRO: usize = 8;
const UNROLL: usize = 8;
const MACRO_PADDED_UNROLL: usize = (MACRO + 1) * UNROLL;

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_v3_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_macro: &mut [f32; MACRO_PADDED_UNROLL],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_macro: &mut [f32; MACRO_PADDED_UNROLL],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    let local_id = local_id.x as usize;
    let group_id = group_id.x as usize;
    let n_micro_in_macro = MACRO / MICRO;
    let micro_id_a = local_id % n_micro_in_macro;
    let micro_id_b = local_id / n_micro_in_macro;
    let n_groups_b = 4;
    let group_id_b = group_id % n_groups_b;
    let group_id_a = group_id / n_groups_b;

    let alpha = push_consts.alpha;
    let beta = push_consts.beta;
    let k = push_consts.k as usize;
    let rsa = push_consts.rsa as usize;
    let csa = push_consts.csa as usize;
    let rsb = push_consts.rsb as usize;
    let csb = push_consts.csb as usize;
    let rsc = push_consts.rsc as usize;
    let csc = push_consts.csc as usize;

    let macro_stride_perp_k_a = rsa;
    let stride_perp_k_a = rsa;
    let stride_pll_k_a = csa;

    let macro_stride_perp_k_b = csb;
    let stride_perp_k_b = csb;
    let stride_pll_k_b = rsb;

    // A setup
    let mut a_micro = <[f32; MICRO]>::default();
    let write_macro_start_a = group_id_a * MACRO;
    let write_start_a = write_macro_start_a + micro_id_a;
    let n_micro_a_tiles_pll_unroll = 8;
    let micro_a_tile_pll_unroll = 1;
    let micro_a_tile_perp_unroll = 4;
    let pll_unroll_a_load_id = local_id % n_micro_a_tiles_pll_unroll;
    let perp_unroll_a_load_id = local_id / n_micro_a_tiles_pll_unroll;
    let read_macro_tile_start_a = group_id_a * MICRO;
    let mut a_idx = read_macro_tile_start_a * macro_stride_perp_k_a;
    let a_offset_pll_unroll = micro_a_tile_pll_unroll * pll_unroll_a_load_id;
    let a_offset_perp_unroll = micro_a_tile_perp_unroll * perp_unroll_a_load_id;
    a_idx += stride_pll_k_a * a_offset_pll_unroll;
    a_idx += stride_perp_k_a * a_offset_perp_unroll;

    // B setup
    let mut b_micro = <[f32; MICRO]>::default();
    let write_macro_start_b = group_id_b * MACRO;
    let write_start_b = write_macro_start_b + micro_id_b;
    let n_micro_b_tiles_pll_unroll = 8;
    let micro_b_tile_pll_unroll = 4;
    let micro_b_tile_perp_unroll = 1;
    let pll_unroll_b_load_id = local_id % n_micro_b_tiles_pll_unroll;
    let perp_unroll_b_load_id = local_id / n_micro_b_tiles_pll_unroll;
    let read_macro_tile_start_b = group_id_b * MICRO;
    let mut b_idx = read_macro_tile_start_b * macro_stride_perp_k_b;
    let b_offset_pll_unroll = micro_b_tile_pll_unroll * pll_unroll_b_load_id;
    let b_offset_perp_unroll = micro_b_tile_perp_unroll * perp_unroll_b_load_id;
    b_idx += stride_pll_k_b * b_offset_pll_unroll;
    b_idx += stride_perp_k_b * b_offset_perp_unroll;

    // C setup
    let mut c_micro = <[[f32; MICRO]; MICRO]>::default();

    let mut z = 0;
    while z < k {


        unroll! { for perp_i in 0 .. 4 /* micro_a_tile_perp_unroll */ {
            a_macro[(MACRO + 1) * a_offset_pll_unroll + (a_offset_perp_unroll + perp_i)] = a[a_idx + perp_i * stride_perp_k_a];
        }}
        a_idx += stride_pll_k_a * UNROLL;

        unroll! { for pll_i in 0 .. 4 /* micro_b_tile_pll_unroll */ {
            b_macro[(MACRO + 1) * (b_offset_pll_unroll + pll_i) + b_offset_perp_unroll] = b[b_idx + pll_i * stride_pll_k_b];
        }}
        b_idx += stride_pll_k_a * UNROLL;

        unsafe {
            control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
        }
    if k == 0 {
        unroll! { for u in 0 .. 8 /* UNROLL */ {
            unroll! { for i in 0 .. 8 /* MICRO */ {
                a_micro[i] = a_macro[micro_id_a + u * (MACRO + 1) + i];
            }}
            unroll! { for j in 0 .. 8 /* MICRO */ {
                b_micro[j] = b_macro[micro_id_b + u * (MACRO + 1) + j];
            }}
            unroll! { for i in 0 .. 8 /* MICRO */ {
                unroll! { for j in 0 .. 8 /* MICRO */ {
                    c_micro[i][j] += a_micro[i] * b_micro[j];
                }}
            }}
        }}
    }
        unsafe {
            control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
        }


        z += UNROLL;
    }

    unroll! { for i in 0 .. 8 /* MICRO */ {
        unroll! { for j in 0 .. 8 /* MICRO */ {
            let idx = (write_start_a + i * n_micro_in_macro) * rsc + (write_start_b + j * n_micro_in_macro) * csc;
            c[idx] *= beta;
            c[idx] += 1. + alpha * c_micro[i][j];
        }}
    }}
}
*/
