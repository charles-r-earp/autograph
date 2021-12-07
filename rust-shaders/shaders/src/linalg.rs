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
    m: u32,
    k: u32,
    n: u32,
    rsa: i32,
    csa: i32,
    rsb: i32,
    csb: i32,
    rsc: i32,
    csc: i32,
}

macro_rules! impl_gemm {
    ($($func:ident<$TS:tt, $WPT:tt>),*) => (
        $(
            fn $func<T: Copy + NumAssign + PartialEq>(
                group_id: UVec3,
                local_id: UVec3,
                a: &[T],
                a_tile: &mut [[T; $TS * $WPT]; $TS],
                b: &[T],
                b_tile: &mut [[T; $TS * $WPT]; $TS],
                use_bias: u32,
                bias: &[T],
                c: &mut [T],
                push_consts: &GemmPushConsts<T>,
            ) where [T; $WPT]: Default, [[T; $WPT]; $WPT]: Default {
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
                let n_groups_y = n / ($TS * $WPT) + if n % ($TS * $WPT) != 0 { 1 } else { 0 };
                let group_x = group_id / n_groups_y;
                let group_y = group_id % n_groups_y;
                let local_id = local_id.x as usize;
                let local_x = local_id / $TS;
                let local_y = local_id % $TS;
                let global_x = group_x * ($TS * $WPT) + local_x;
                let global_y = group_y * ($TS * $WPT) + local_y;

                let mut a_micro = <[T; $WPT]>::default();
                let mut b_micro = <[T; $WPT]>::default();
                let mut c_micro = <[[T; $WPT]; $WPT]>::default();

                let mut a_idx = local_y * csa;
                let mut b_idx = local_x * rsb;
                let mut tiled_row = local_x;
                let mut tiled_col = local_y;
                let ntiles = k / $TS + if k % $TS != 0 { 1 } else { 0 };

                for _ in 0 .. ntiles {
                    unroll! { for i in 0 .. $WPT {
                        let global_row = global_x + i * $TS;
                        a_tile[local_y][local_x + i * $TS] = if global_row < m {
                            if tiled_col < k {
                                a[a_idx + global_row * rsa]
                            } else {
                                T::zero()
                            }
                        } else {
                            T::zero()
                        };
                    }}
                    a_idx += $TS * csa;
                    tiled_col += $TS;
                    unroll! { for j in 0 .. $WPT {
                        let global_col = global_y + j * $TS;
                        b_tile[local_x][local_y + j * $TS] = if tiled_row < k {
                            if global_col < n {
                                b[b_idx + global_col * csb]
                            } else {
                                T::zero()
                            }
                        } else {
                            T::zero()
                        };
                    }}
                    b_idx += $TS * rsb;
                    tiled_row += $TS;
                    unsafe {
                        control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
                    }
                    unroll! { for u in 0 .. $TS {
                        unroll! { for i in 0 .. $WPT {
                            a_micro[i] = a_tile[u][local_x + i * $TS];
                        }}
                        unroll! { for j in 0 .. $WPT {
                            b_micro[j] = b_tile[u][local_y + j * $TS];
                        }}
                        unroll! { for i in 0 .. $WPT {
                            unroll! { for j in 0 .. $WPT {
                                c_micro[i][j] += a_micro[i] * b_micro[j];
                            }}
                        }}
                    }}
                    unsafe {
                        control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
                    }
                }

                unroll! { for i in 0 .. $WPT {
                    let global_row = global_x + i * $TS;
                    unroll! { for j in 0 .. $WPT {
                        let global_col = global_y + j * $TS;
                        if global_row < m { if global_col < n {
                            let idx = global_row * rsc + global_col * csc;
                            let mut y = alpha * c_micro[i][j];
                            if beta != T::zero() {
                                y += beta * c[idx];
                            }
                            if use_bias == 1 {
                                y += bias[global_col];
                            }
                            c[idx] = y;
                        }}
                    }}
                }}
            }
        )*
    )
}

impl_gemm!(gemm_ts16_wpt1<16, 1>, gemm_ts16_wpt2<16, 2>, gemm_ts16_wpt4<16, 4> /*, gemm_ts16_wpt8<16, 8>*/);

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_u32_ts16_wpt1(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[u32],
    #[spirv(workgroup)]
    a_tile: &mut [[u32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[u32],
    #[spirv(workgroup)]
    b_tile: &mut [[u32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<u32>,
) {
    gemm_ts16_wpt1(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_i32_ts16_wpt1(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[i32],
    #[spirv(workgroup)]
    a_tile: &mut [[i32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[i32],
    #[spirv(workgroup)]
    b_tile: &mut [[i32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [i32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<i32>,
) {
    gemm_ts16_wpt1(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_f32_ts16_wpt1(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt1(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_bias_f32_ts16_wpt1(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 1]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    bias: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt1(group_id, local_id, a, a_tile, b, b_tile, 1, bias, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_f32_ts16_wpt2(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 2]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 2]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt2(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_bias_f32_ts16_wpt2(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 2]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 2]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    bias: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt2(group_id, local_id, a, a_tile, b, b_tile, 1, bias, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_f32_ts16_wpt4(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 4]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 4]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt4(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_bias_f32_ts16_wpt4(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 4]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 4]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    bias: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt4(group_id, local_id, a, a_tile, b, b_tile, 1, bias, c, push_consts);
}
/*
#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_f32_ts16_wpt8(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 8]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 8]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt8(group_id, local_id, a, a_tile, b, b_tile, 0, a, c, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn gemm_bias_f32_ts16_wpt8(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    a: &[f32],
    #[spirv(workgroup)]
    a_tile: &mut [[f32; 16 * 8]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    b: &[f32],
    #[spirv(workgroup)]
    b_tile: &mut [[f32; 16 * 8]; 16],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    bias: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
    c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_ts16_wpt8(group_id, local_id, a, a_tile, b, b_tile, 1, bias, c, push_consts);
}
*/
