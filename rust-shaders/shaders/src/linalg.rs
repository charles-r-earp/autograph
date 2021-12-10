use spirv_std::{
    memory::{Scope, Semantics},
    arch::control_barrier,
    glam::UVec3,
};
use num_traits::Zero;
use crunchy::unroll;

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
    ($($func:ident<$T:ty, $TS:tt, $TSA:tt, $TSB:tt, $UNR:tt, $MICA:tt, $MICB:tt>($($bias:tt=true)?)),* $(,)?) => (
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
                    c: &mut [$T],
                    #[cfg(feature="false")]
                )?
                #[spirv(storage_buffer, descriptor_set=0, binding=2)]
                c: &mut [$T],
                #[spirv(push_constant)]
                push_consts: &GemmPushConsts<$T>,
            ) {
                type T = $T;

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
                let n_groups_y = n / ($TSB * $MICB) + if n % ($TSB * $MICB) != 0 { 1 } else { 0 };
                let group_x = group_id / n_groups_y;
                let group_y = group_id % n_groups_y;
                let local_id = local_id.x as usize;
                let local_x = local_id / $TSB;
                let local_y = local_id % $TSB;
                let global_x = group_x * ($TSA * $MICA) + local_x;
                let global_y = group_y * ($TSB * $MICB) + local_y;

                let mut a_micro = <[T; $MICA]>::default();
                let mut b_micro = <[T; $MICA]>::default();
                let mut c_micro = <[[T; $MICB]; $MICA]>::default();

                let mut tiled_row = local_x;
                let mut tiled_col = local_y;
                let mut a_idx = tiled_col * csa;
                let mut b_idx = tiled_row * rsb;

                let ntiles = k / $UNR + if k % $UNR != 0 { 1 } else { 0 };

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
                    a_idx += $UNR * csa;
                    tiled_col += $UNR;
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
                    b_idx += $UNR * rsb;
                    tiled_row += $UNR;
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
                            c[idx] *= beta;
                            c[idx] += alpha * c_micro[i][j];
                            $(
                                c[idx] += $bias[global_col];
                            )?
                        }}
                    }}
                }}
            }
        )*
    );
}

impl_gemm!{
    gemm_u32_tsa16_tsb16_unr16_mica1_micb1<u32, 256, 16, 16, 16, 1, 1>(),
    gemm_i32_tsa16_tsb16_unr16_mica1_micb1<i32, 256, 16, 16, 16, 1, 1>(),
    gemm_f32_tsa16_tsb16_unr16_mica1_micb1<f32, 256, 16, 16, 16, 1, 1>(),
    gemm_bias_f32_tsa16_tsb16_unr16_mica1_micb1<f32, 256, 16, 16, 16, 1, 1>(bias=true),
    gemm_f32_tsa16_tsb16_unr16_mica2_micb2<f32, 256, 16, 16, 16, 2, 2>(),
    gemm_bias_f32_tsa16_tsb16_unr16_mica2_micb2<f32, 256, 16, 16, 16, 2, 2>(bias=true),
    gemm_f32_tsa16_tsb16_unr16_mica4_micb4<f32, 256, 16, 16, 16, 4, 4>(),
    gemm_bias_f32_tsa16_tsb16_unr16_mica4_micb4<f32, 256, 16, 16, 16, 4, 4>(bias=true),
}
