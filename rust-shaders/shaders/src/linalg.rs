use crate::atomic::atomic_compare_exchange;
use spirv_std::{
    memory::{Scope, Semantics},
    arch::control_barrier,
    glam::UVec3,
};
use num_traits::Zero;
use crunchy::unroll;

#[repr(C)]
pub struct CBetaPushConsts<T> {
    n: u32,
    beta: T,
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn c_beta_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &CBetaPushConsts<f32>,
) {
    let n = push_consts.n as usize;
    let beta = push_consts.beta;
    let idx = global_id.x as usize;
    if idx < n {
        y[idx] *= beta;
    }
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

fn group_barrier() {
    unsafe {
        control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>();
    }
}

// Inspired by https://github.com/ROCmSoftwarePlatform/MIOpenGEMM
macro_rules! impl_gemm {
    ($($func:ident<$(@splitk=$splitk:tt,)? $T:ty, $TC:ty, $TS:tt, $TSA:tt, $TSB:tt, $UNR:tt, $MICA:tt, $MICB:tt>($($bias:tt=true)?)),* $(,)?) => (
        $(
            #[allow(unused_attributes)]
            #[spirv(compute(threads($TS)))]
            pub fn $func(
                #[spirv(workgroup_id)]
                group_id: UVec3,
                #[spirv(local_invocation_id)]
                local_id: UVec3,
                #[spirv(storage_buffer, descriptor_set=0, binding=0)]
                a: &[$T],
                #[spirv(workgroup)]
                a_tile: &mut [[$T; $TSA * $MICA + 1]; $UNR],
                #[spirv(storage_buffer, descriptor_set=0, binding=1)]
                b: &[$T],
                #[spirv(workgroup)]
                b_tile: &mut [[$T; $TSB * $MICB + 1]; $UNR],
                $(
                    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
                    $bias: &[$T],
                    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
                    c: &mut [$TC],
                    #[cfg(feature="false")]
                )?
                #[spirv(storage_buffer, descriptor_set=0, binding=2)]
                c: &mut [$TC],
                #[spirv(push_constant)]
                push_consts: &GemmPushConsts<$T>,
            ) {
                type T = $T;

                let alpha = push_consts.alpha;
                #[allow(unused)]
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
                let n_groups_z = {
                    #[allow(unused_mut, unused_assignments)]
                    let mut n_groups_z = 1;
                    $(
                        n_groups_z = k / $splitk + if k % $splitk != 0 { 1 } else { 0 };
                    )?
                    n_groups_z
                };
                let group_id_xy = group_id / n_groups_z;
                let group_z = group_id % n_groups_z;
                let n_groups_y = n / ($TSB * $MICB) + if n % ($TSB * $MICB) != 0 { 1 } else { 0 };
                let group_x = group_id_xy / n_groups_y;
                let group_y = group_id_xy % n_groups_y;
                let local_id = local_id.x as usize;
                let local_x = local_id / $TSB;
                let local_y = local_id % $TSB;
                let global_x = group_x * ($TSA * $MICA) + local_x;
                let global_y = group_y * ($TSB * $MICB) + local_y;

                let mut a_micro = <[T; $MICA]>::default();
                let mut b_micro = <[T; $MICA]>::default();
                let mut c_micro = <[[T; $MICB]; $MICA]>::default();

                let g_unroll = $UNR * n_groups_z;

                let mut tiled_row = local_x + group_z * $UNR;
                let mut tiled_col = local_y + group_z * $UNR;
                let mut a_idx = tiled_col * csa;
                let mut b_idx = tiled_row * rsb;

                let ntiles = if n_groups_z > 1 {
                    let n_groups_with_one_more = (k % g_unroll) / $UNR;
                    k / g_unroll + if group_z < n_groups_with_one_more { 1 } else { 0 }
                } else {
                    k / $UNR + if k % $UNR != 0 { 1 } else { 0 }
                };

                for _ in 0 .. ntiles {
                    unroll! { for i in 0 .. $MICA {
                        let global_row = global_x + i * $TSA;
                        a_tile[local_y][local_x + i * $TSA] = if tiled_col < k {
                            if global_row < m {
                                a[a_idx + global_row * rsa]
                            } else {
                                T::zero()
                            }
                        } else {
                            T::zero()
                        };
                    }}
                    a_idx += g_unroll * csa;
                    tiled_col += g_unroll;
                    unroll! { for j in 0 .. $MICB {
                        let global_col = global_y + j * $TSB;
                        b_tile[local_x][local_y + j * $TSB] = if tiled_row < k {
                            if global_col < n {
                                b[b_idx + global_col * csb]
                            } else {
                                T::zero()
                            }
                        } else {
                            T::zero()
                        };
                    }}
                    b_idx += g_unroll * rsb;
                    tiled_row += g_unroll;
                    group_barrier();
                    unroll! { for u in 0 .. $UNR {
                        unroll! { for i in 0 .. $MICA {
                            a_micro[i] = a_tile[u][local_x + i * $TSA];
                        }}
                        unroll! { for j in 0 .. $MICB {
                            b_micro[j] = b_tile[u][local_y + j * $TSB];
                        }}
                        unroll! { for i in 0 .. $MICA {
                            unroll! { for j in 0 .. $MICB {
                                c_micro[i][j] += a_micro[i] * b_micro[j];
                            }}
                        }}
                    }}
                    group_barrier();
                }

                unroll! { for i in 0 .. $MICA {
                    let global_row = global_x + i * $TSA;
                    unroll! { for j in 0 .. $MICB {
                        let global_col = global_y + j * $TSB;
                        if global_row < m { if global_col < n {
                            let idx = global_row * rsc + global_col * csc;
                            #[allow(unused_mut)]
                            let mut y = alpha * c_micro[i][j];
                            $(
                                if group_z == 0 {
                                    y += $bias[global_col];
                                }
                            )?
                            // Adapted from https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/blob/master/demokernels/tC0_tA0_tB0_colMaj1_m1000_n2000_k3000_lda1100_ldb3200_ldc1300_ws100000000_f32/A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS1__B_MIC6_PAD1_PLU1_LIW0_MIW1_WOS1__C_UNR8_GAL3_PUN1_ICE2_NAW16_UFO0_MAC256_SKW10/cw_alpha.cl
                            $(
                                let _splitk = $splitk; // need macro binding

                                let mut previous: u32;
                                loop {
                                    previous = c[idx];
                                    let value = (T::from_bits(previous) + y).to_bits();
                                    if unsafe {
                                        atomic_compare_exchange::<u32, {Scope::Device as u32}, {Semantics::NONE.bits()}, {Semantics::NONE.bits()}>(&mut c[idx], value, previous)
                                    } == previous {
                                        break;
                                    }
                                }

                                #[cfg(feature = "false")]
                            )?
                            {
                                c[idx] *= beta;
                                c[idx] += y;
                            }
                        }}
                    }}
                }}
            }
        )*
    );
}

impl_gemm!{
    gemm_u32_tsa16_tsb16_unr16_mica1_micb1<u32, u32, 256, 16, 16, 16, 1, 1>(),
    gemm_i32_tsa16_tsb16_unr16_mica1_micb1<i32, i32, 256, 16, 16, 16, 1, 1>(),
    gemm_f32_tsa16_tsb16_unr16_mica1_micb1<f32, f32, 256, 16, 16, 16, 1, 1>(),
    gemm_bias_f32_tsa16_tsb16_unr16_mica1_micb1<f32, f32, 256, 16, 16, 16, 1, 1>(bias=true),
    gemm_f32_tsa16_tsb16_unr16_mica2_micb2<f32, f32, 256, 16, 16, 16, 2, 2>(),
    gemm_bias_f32_tsa16_tsb16_unr16_mica2_micb2<f32, f32, 256, 16, 16, 16, 2, 2>(bias=true),
    gemm_f32_tsa16_tsb16_unr16_mica4_micb4<f32, f32, 256, 16, 16, 16, 4, 4>(),
    gemm_bias_f32_tsa16_tsb16_unr16_mica4_micb4<f32, f32, 256, 16, 16, 16, 4, 4>(bias=true),
    gemm_f32_tsa16_tsb16_splitk128_unr16_mica1_micb1<@splitk=128, f32, u32, 256, 16, 16, 16, 1, 1>(),
    gemm_bias_f32_tsa16_tsb16_splitk128_unr16_mica1_micb1<@splitk=128, f32, u32, 256, 16, 16, 16, 1, 1>(bias=true),
}
