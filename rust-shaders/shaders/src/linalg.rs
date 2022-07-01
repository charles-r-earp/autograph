use crate::{
    autobind,
    util::group_barrier,
};
use spirv_std::{
    glam::UVec3,
};
use num_traits::Zero;
use crunchy::unroll;
use paste::paste;

#[repr(C)]
pub struct CBetaPushConsts<T> {
    n: u32,
    beta: T,
}

#[autobind]
#[spirv(compute(threads(64)))]
pub fn c_beta_u32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)]
    y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &CBetaPushConsts<u32>,
) {
    let n = push_consts.n as usize;
    let beta = push_consts.beta;
    let idx = global_id.x as usize;
    if idx < n {
        y[idx] *= beta;
    }
}

#[autobind]
#[spirv(compute(threads(64)))]
pub fn c_beta_i32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)]
    y: &mut [i32],
    #[spirv(push_constant)]
    push_consts: &CBetaPushConsts<i32>,
) {
    let n = push_consts.n as usize;
    let beta = push_consts.beta;
    let idx = global_id.x as usize;
    if idx < n {
        y[idx] *= beta;
    }
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
pub struct BiasPushConsts {
    bs: u32,
    c: u32,
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn bias_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)]
    bias: &[f32],
    #[spirv(storage_buffer)]
    y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &BiasPushConsts,
) {
    let bs = push_consts.bs as usize;
    let c = push_consts.c as usize;
    let idx = global_id.x as usize;
    if idx < bs * c {
        y[idx] += bias[idx % c];
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
    n_groups_k: u32,
}

macro_rules! impl_gemm {
    /*
    [TM, TK, TN]: Size of a workgroup tile.
    [GM, GK, GN]: Workgroup size (ie number of threads).
    [LM, LK, LN]: Micro tile sizes LM and LN, LK items are reduced producing [LM, LN] items per thread.
    [LAM, LAK]: Number of loads of A in M and K directions. These are needed to unroll loops.
    [LBK, LBN]: Number of loads of B in K and N directons.

    Implementation is defined by [TM, TK, TN] and [LM, LK, LN], all other parameters can be derived.
    */
    ($((T=$T:ty, [$TM:tt, $TK:tt, $TN:tt], [$GM:tt, $GK:tt, $GN:tt], [$LM:tt, $LK:tt, $LN:tt], [$LAM:tt, $LAK:tt], [$LBK:tt, $LBN:tt])),* $(,)?) => (
        paste! {
            $(
                #[autobind]
                #[spirv(compute(threads(256)))]
                pub fn [<gemm_ $T _tm $TM _tk $TK _tn $TN _lm $LM _lk $LK _ln $LN>](
                    #[spirv(workgroup_id)]
                    group_id: UVec3,
                    #[spirv(local_invocation_id)]
                    local_id: UVec3,
                    #[spirv(storage_buffer)] a: &[$T],
                    #[spirv(workgroup)] a_tile: &mut [[$T; $TM + 1]; $TK],
                    #[spirv(storage_buffer)] b: &[$T],
                    #[spirv(workgroup)] b_tile: &mut [[$T; $TK + 1]; $TN],
                    #[spirv(storage_buffer)] c: &mut [$T],
                    #[spirv(push_constant)] push_consts: &GemmPushConsts<$T>,
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
                    let n_groups_k = push_consts.n_groups_k as usize;

                    let n_groups_n = n / $TN + if n % $TN != 0 { 1 } else { 0 };

                    let group_id = group_id.x as usize;
                    let group_mn = group_id / n_groups_k;
                    let group_k = group_id % n_groups_k;
                    let group_m = group_mn / n_groups_n;
                    let group_n = group_mn % n_groups_n;

                    let threads = $GM * $GK * $GN;
                    let local_id = local_id.x as usize;
                    let local_k = local_id / ($GM * $GN);
                    let local_mn = local_id % ($GM * $GN);
                    let local_m = local_mn / $GN;
                    let local_n = local_mn % $GN;

                    let g_unroll = n_groups_k * $TK;

                    let mut a_micro = <[T; $LM]>::default();
                    let mut b_micro = <[T; $LN]>::default();
                    let mut c_micro = <[[T; $LN]; $LM]>::default();

                    let mut ki = group_k * $TK;
                    while ki < k {
                        // TODO: Optimize loading, potentially wide loads of 2 or 4 values per thread.
                        // May benefit from aligning the loads properly to 32 via an offset, but this hasn't shown to help.
                        { // load a
                            let ts_m = if $TM >= 64 { 64 } else { $TM };
                            let ts_k = threads / ts_m;
                            let local_k = local_id / ts_m;
                            let local_m = local_id % ts_m;
                            unroll! { for u in 0 .. $LAK {
                                let tile_k = u * ts_k  + local_k;
                                let global_k = ki + tile_k;
                                unroll! { for i in 0 .. $LAM {
                                    let tile_m = i * ts_m + local_m;
                                    let global_m = group_m * $TM + tile_m;
                                    a_tile[tile_k][tile_m] = if global_m < m && global_k < k {
                                        a[global_m * rsa + global_k * csa]
                                    } else {
                                        T::zero()
                                    };
                                }}
                            }}
                        }
                        { // load b
                            let ts_n = if $TN >= 64 { 64 } else { $TN };
                            let ts_k = threads / ts_n;
                            let local_k = local_id / ts_n;
                            let local_n = local_id % ts_n;
                            unroll! { for u in 0 .. $LBK {
                                let tile_k = u * ts_k  + local_k;
                                let global_k = ki + tile_k;
                                unroll! { for j in 0 .. $LBN {
                                    let tile_n = j * ts_n + local_n;
                                    let global_n = group_n * $TN + tile_n;
                                    b_tile[tile_n][tile_k] = if global_k < k && global_n < n {
                                        b[global_k * rsb + global_n * csb]
                                    } else {
                                        T::zero()
                                    };
                                }}
                            }}
                        }
                        group_barrier();
                        unroll! { for u in 0 .. $LK {
                            unroll! { for i in 0 .. $LM {
                                a_micro[i] = a_tile[u * $GK + local_k][i * $GM + local_m];
                            }}
                            unroll! { for j in 0 .. $LN {
                                b_micro[j] = b_tile[j * $GN + local_n][u * $GK + local_k];
                            }}
                            unroll! { for i in 0 .. $LM {
                                unroll! { for j in 0 .. $LN {
                                    c_micro[i][j] += a_micro[i] * b_micro[j];
                                }}
                            }}
                        }}
                        group_barrier();
                        ki += g_unroll;
                    }
                    unroll! { for i in 0 .. $LM {
                        unroll! { for j in 0 .. $LN {
                            let row = group_m * $TM + i * $GM + local_m;
                            let col = group_n * $TN + j * $GN + local_n;
                            let idx = (row * rsc + col * csc) * n_groups_k * $GK + group_k * $GK + local_k;
                            if row < m && col < n {
                                c[idx] = alpha * c_micro[i][j] + beta * c[idx];
                            }
                        }}
                    }}
                }
            )*
        }
    );
}

impl_gemm! {
    (T=u32, [64, 8, 64], [16, 1, 16], [4, 8, 4], [1, 2], [2, 1]),
    (T=u32, [128, 8, 32], [32, 1, 8], [4, 8, 4], [2, 2], [1, 1]),
    (T=u32, [32, 8, 128], [8, 1, 32], [4, 8, 4], [1, 1], [2, 2]),
    (T=u32, [16, 64, 16], [16, 1, 16], [1, 64, 1], [1, 4], [4, 1]),

    (T=i32, [64, 8, 64], [16, 1, 16], [4, 8, 4], [1, 2], [2, 1]),
    (T=i32, [128, 8, 32], [32, 1, 8], [4, 8, 4], [2, 2], [1, 1]),
    (T=i32, [32, 8, 128], [8, 1, 32], [4, 8, 4], [1, 1], [2, 2]),
    (T=i32, [16, 64, 16], [16, 1, 16], [1, 64, 1], [1, 4], [4, 1]),

    (T=f32, [64, 8, 64], [16, 1, 16], [4, 8, 4], [1, 2], [2, 1]),
    (T=f32, [128, 8, 32], [32, 1, 8], [4, 8, 4], [2, 2], [1, 1]),
    (T=f32, [32, 8, 128], [8, 1, 32], [4, 8, 4], [1, 1], [2, 2]),
    (T=f32, [16, 64, 16], [16, 1, 16], [1, 64, 1], [1, 4], [4, 1]),
}
