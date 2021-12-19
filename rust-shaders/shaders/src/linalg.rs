use crate::{
    atomic::{atomic_compare_exchange, /*atomic_u32_add_f32*/},
    autobind,
};
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

#[autobind]
#[spirv(compute(threads(64)))]
pub fn c_beta_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)]
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

macro_rules! impl_gemm {
    ($($func:ident<$(@splitk=$splitk:tt,)? $T:ty, TC=$TC:ty, UNR=$UNR:tt, MICA=$MICA:tt, LA=$LA:tt, MICB=$MICB:tt, LB=$LB:tt $(, $bias:tt=true)?>),* $(,)?) => (
        $(
            impl_gemm!{@Impl $func<$(@splitk=$splitk,)? $T, $TC, 256, 16, 16, $UNR, $MICA, $LA, $MICB, $LB $(, $bias=true)?>}
        )*
    );
    (@Impl $func:ident<$(@splitk=$splitk:tt,)? $T:ty, $TC:ty, $TS:tt, $TSA:tt, $TSB:tt, $UNR:tt, $MICA:tt, $LA:tt, $MICB:tt, $LB:tt $(, $bias:tt=true)?>) => (
        #[autobind]
        #[spirv(compute(threads($TS)))]
        pub fn $func(
            #[spirv(workgroup_id)]
            group_id: UVec3,
            #[spirv(local_invocation_id)]
            local_id: UVec3,
            #[spirv(storage_buffer)] a: &[$T],
            #[spirv(workgroup)] a_tile: &mut [[$T; $TSA * $MICA + 1]; $UNR],
            #[spirv(storage_buffer)] b: &[$T],
            #[spirv(workgroup)] b_tile: &mut [[$T; $TSB * $MICB + 1]; $UNR],
            $(
                #[cfg(feature = "false")] $bias: u32,
                #[spirv(storage_buffer)] bias: &[$T],
            )?
            #[spirv(storage_buffer)] c: &mut [$TC],
            #[spirv(push_constant)] push_consts: &GemmPushConsts<$T>,
        ) {
            type T = $T;

            let alpha = push_consts.alpha;
            #[allow(unused)] // unused for splitk
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

            let groups_b = n / ($TSB * $MICB) + if n % ($TSA * $MICB) != 0 { 1 } else { 0 };
            let group_a = group_id_xy / groups_b;
            let group_b = group_id_xy % groups_b;

            let local_id = local_id.x as usize;
            let local_a = local_id / $TSB;
            let local_b = local_id % $TSB;

            // $MICA and $UNR have to be chosen such that $TSA * $MICA * $UNR is multiple of $TS
            //let loads_a = ($TSA * $MICA * $UNR) / $TS;
            let load_stride_a = $UNR / $LA;
            //let loads_b = ($TSB * $MICB * $UNR) / $TS;
            let load_stride_b = $UNR / $LB;

            let tile_col_a = local_id / ($TSA * $MICA);
            let tile_row_a = local_id % ($TSA * $MICA);
            let tile_row_b = local_id / ($TSB * $MICB);
            let tile_col_b = local_id % ($TSB * $MICB);

            let mut a_micro = <[T; $MICA]>::default();
            let mut b_micro = <[T; $MICB]>::default();
            let mut c_micro = <[[T; $MICB]; $MICA]>::default();

            let g_unroll = $UNR * n_groups_z;

            let ntiles = if n_groups_z > 1 {
                let n_groups_with_one_more = (k % g_unroll) / $UNR + if k % g_unroll != 0 { 1 } else { 0 };
                k / g_unroll + if group_z < n_groups_with_one_more { 1 } else { 0 }
            } else {
                k / $UNR + if k % $UNR != 0 { 1 } else { 0 }
            };

            for t in 0 .. ntiles {
                unroll! { for j in 0 .. $LA {
                    let row = group_a * ($TSA * $MICA) + tile_row_a;
                    let col = (t * g_unroll + group_z * $UNR) + j * load_stride_a + tile_col_a;
                    a_tile[tile_col_a + j * load_stride_a][tile_row_a] = if row < m {
                        if col < k {
                            a[row * rsa + col * csa]
                        } else {
                            T::zero()
                        }
                    } else {
                        T::zero()
                    };
                }}
                unroll! { for i in 0 .. $LB {
                    let row = (t * g_unroll + group_z * $UNR) + i * load_stride_b + tile_row_b;
                    let col = group_b * ($TSB * $MICB) + tile_col_b;
                    b_tile[tile_row_b + i * load_stride_b][tile_col_b] = if col < n {
                        if row < k {
                            b[row * rsb + col * csb]
                        } else {
                            T::zero()
                        }
                    } else {
                        T::zero()
                    };
                }}
                group_barrier();
                unroll! { for u in 0 .. $UNR {
                    unroll! { for i in 0 .. $MICA {
                        a_micro[i] = a_tile[u][i * $TSA + local_a];
                    }}
                    unroll! { for j in 0 .. $MICB {
                        b_micro[j] = b_tile[u][j * $TSB + local_b];
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
                unroll! { for j in 0 .. $MICB {
                    let row = (group_a * $MICA + i) * $TSA + local_a;
                    let col = (group_b * $MICB + j) * $TSB + local_b;
                    if row < m { if col < n {
                        let idx = row * rsc + col * csc;
                        #[allow(unused_mut)]
                        let mut y = alpha * c_micro[i][j];
                        $(
                            #[cfg(feature = "false")]
                            {
                                // macro binding
                                let $bias = 0;
                            }
                            if group_z == 0 {
                                y += bias[col];
                            }
                        )?
                        $(
                            // macro binding
                            #[cfg(feature = "false")]
                            {
                                let _splitk = $splitk;
                            }

                            /*unsafe { TODO: Use this instead, wasn't producing correct output.
                                atomic_u32_add_f32::<{Scope::Device as u32}, {Semantics::NONE.bits()}>(c, idx, y);
                            }*/

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
    )
}

impl_gemm! {
    gemm_u32_unr16_mica1_micb1<u32, TC=u32, UNR=16, MICA=1, LA=1, MICB=1, LB=1>,
    gemm_i32_unr16_mica1_micb1<i32, TC=i32, UNR=16, MICA=1, LA=1, MICB=1, LB=1>,
    gemm_f32_unr16_mica1_micb1<f32, TC=f32, UNR=16, MICA=1, LA=1, MICB=1, LB=1>,
    gemm_bias_f32_unr16_mica1_micb1<f32, TC=f32, UNR=16, MICA=1, LA=1, MICB=1, LB=1, bias=true>,
    gemm_f32_unr8_mica2_micb2<f32, TC=f32, UNR=8, MICA=2, LA=1, MICB=2, LB=1>,
    gemm_bias_f32_unr8_mica2_micb2<f32, TC=f32, UNR=8, MICA=2, LA=1, MICB=2, LB=1, bias=true>,
    gemm_f32_unr8_mica4_micb4<f32, TC=f32, UNR=8, MICA=4, LA=2, MICB=4, LB=2>,
    gemm_bias_f32_unr8_mica4_micb4<f32, TC=f32, UNR=8, MICA=4, LA=2, MICB=4, LB=2, bias=true>,
    gemm_f32_splitk256_unr16_mica1_micb1<@splitk=256, f32, TC=u32, UNR=16, MICA=1, LA=1, MICB=1, LB=1>,
    gemm_bias_f32_splitk256_unr16_mica1_micb1<@splitk=256, f32, TC=u32, UNR=16, MICA=1, LA=1, MICB=1, LB=1, bias=true>,
}
