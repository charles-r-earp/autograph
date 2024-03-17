#![allow(warnings)]
use super::*;
use dry::macro_wrap;
use half::{bf16, f16};
#[cfg(feature = "device")]
use krnl::{macros::module, scalar::ScalarElem};
use ndarray::linalg::Dot;
use paste::paste;
use std::time::{Duration, Instant};

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "device")]
#[module]
mod kernels {
    #[cfg(target_arch = "spirv")]
    use crunchy::unroll;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{
        buffer::UnsafeIndex,
        half::{bf16, f16},
        num_traits::Zero,
        scalar::Scalar,
        spirv_std::arch::workgroup_memory_barrier_with_group_sync as group_barrier,
    };
    use paste::paste;

    macro_rules! impl_gemm {
        ($t:ty => $a:ty) => {
            paste! {
                #[kernel]
                pub unsafe fn [<gemm_$t>]<
                    const M: u32,
                    const K: u32,
                    const N: u32,
                    const GROUPS_K: u32,
                    const RSA: i32,
                    const CSA: i32,
                    const RSB: i32,
                    const CSB: i32,
                    const RSC: i32,
                    const CSC: i32,
                >(
                    alpha: $a,
                    #[global] a: Slice<$t>,
                    offset_a: u32,
                    #[group] a_group: UnsafeSlice<$a, { 16 * (8 + 1) }>,
                    #[global] b: Slice<$t>,
                    offset_b: u32,
                    #[group] b_group: UnsafeSlice<$a, { 8 * (16 + 1) }>,
                    beta: $a,
                    #[global] c: UnsafeSlice<$t>,
                    offset_c: u32,
                ) {
                    type T = $t;
                    type A = $a;

                    let m = M as usize;
                    let k = K as usize;
                    let n = N as usize;
                    let groups_k = GROUPS_K as usize;

                    let m_thread = 2;
                    let n_thread = 2;
                    let threads_m = 8;
                    let threads_n = 8;
                    let threads = threads_m * threads_n;
                    let m_group = threads_m * m_thread;
                    let n_group = threads_n * n_thread;

                    let unroll = 8;
                    let groups_m = m / m_group + (m % m_group != 0) as usize;
                    let unrolls = k / unroll + (k % unroll != 0) as usize;
                    let groups_n = n / n_group + (n % n_group != 0) as usize;
                    let groups_mn = groups_m * groups_n;
                    let global_unroll = groups_k * unroll;

                    let group_id = kernel.group_id();
                    let group_k = group_id / groups_mn;
                    let group_mn = group_id % groups_mn;
                    let group_m = group_mn / groups_n;
                    let group_n = group_mn % groups_n;
                    let global_m = group_m * m_group;
                    let mut global_k = group_k * unroll;
                    let global_n = group_n * n_group;

                    let thread_id = kernel.thread_id();
                    let thread_m = thread_id / threads_n;
                    let thread_n = thread_id % threads_n;

                    let threads_m_a = m_group;
                    let threads_k_a = threads / threads_m_a;
                    let thread_m_a = thread_id % threads_m_a;
                    let thread_k_a = thread_id / threads_m_a;

                    let threads_n_b = n_group;
                    let threads_k_b = threads / threads_n_b;
                    let thread_n_b = thread_id % threads_n_b;
                    let thread_k_b = thread_id / threads_n_b;

                    let mut a_prefetch = <[T; 2]>::default();
                    let mut b_prefetch = <[T; 2]>::default();

                    let mut a_thread = <[A; 2]>::default();
                    let mut b_thread = <[A; 2]>::default();
                    let mut c_thread = <[[A; 2]; 2]>::default();

                    macro_rules! compute {
                        () => {
                            unroll!(for tile_k in 0 .. 8 {
                                unroll!(for i in 0 .. 2 {
                                    let tile_m = i * threads_m + thread_m;
                                    unsafe {
                                        a_thread[i] = *a_group.unsafe_index(tile_m * (unroll + 1) + tile_k);
                                    }
                                });
                                unroll!(for j in 0 .. 2 {
                                    let tile_n = j * threads_n + thread_n;
                                    unsafe {
                                        b_thread[j] = *b_group.unsafe_index(tile_k * (n_group + 1) + tile_n);
                                    }
                                });
                                unroll!(for i in 0 .. 2 {
                                    unroll!(for j in 0 .. 2 {
                                        c_thread[i][j] += a_thread[i] * b_thread[j];
                                    });
                                });
                            });
                        };
                    }

                    macro_rules! prefetch {
                        () => {
                            {
                                let tile_m = thread_m_a;
                                let global_m = global_m + tile_m;
                                unroll! { for u in 0 .. 2 {
                                    let tile_k = u * threads_k_a + thread_k_a;
                                    let global_k = global_k + tile_k;
                                    a_prefetch[u] = if global_m < m && global_k < k {
                                        a[(global_m as i32 * RSA + global_k as i32 * CSA + offset_a as i32) as usize]
                                    } else {
                                        T::zero()
                                    };
                                }}
                            }
                            {
                                let tile_n = thread_n_b;
                                let global_n = global_n + tile_n;
                                unroll! { for u in 0 .. 2 {
                                    let tile_k = u * threads_k_b + thread_k_b;
                                    let global_k = global_k + tile_k;
                                    b_prefetch[u] = if global_k < k && global_n < n {
                                        b[(global_k as i32 * RSB + global_n as i32 * CSB + offset_b as i32) as usize]
                                    } else {
                                        T::zero()
                                    };
                                }}
                            }
                        };
                    }

                    macro_rules! fetch {
                        () => {
                            {
                                let tile_m = thread_m_a;
                                unroll! { for u in 0 .. 2 {
                                    let tile_k = u * threads_k_a + thread_k_a;
                                    unsafe {
                                        *a_group.unsafe_index_mut(tile_m * (unroll + 1) + tile_k) = a_prefetch[u].cast();
                                    }
                                }}
                            }
                            {
                                let tile_n = thread_n_b;
                                unroll! { for u in 0 .. 2 {
                                    let tile_k = u * threads_k_b + thread_k_b;
                                    unsafe {
                                        *b_group.unsafe_index_mut(tile_k * (n_group + 1) + tile_n) = b_prefetch[u].cast();
                                    }
                                }}
                            }
                        }
                    }

                    prefetch!();
                    fetch!();
                    unsafe {
                        group_barrier();
                    }
                    global_k += global_unroll;

                    while global_k < k {
                        prefetch!();
                        compute!();
                        unsafe {
                            group_barrier();
                        }
                        fetch!();
                        unsafe {
                            group_barrier();
                        }
                        global_k += global_unroll;
                    }
                    compute!();

                    unroll! { for i in 0 .. 2 {
                        let global_m = global_m + i * threads_m + thread_m;
                        unroll! { for j in 0 .. 2 {
                            let global_n = global_n + j * threads_n + thread_n;
                            if global_m < m && global_n < n {
                                let index = ((global_m as i32 * RSC + global_n as i32 * CSC + offset_c as i32) as usize) * groups_k + group_k;
                                if beta == A::zero() {
                                    unsafe {
                                        *c.unsafe_index_mut(index) = (alpha * c_thread[i][j]).cast();
                                    }
                                } else {
                                    unsafe {
                                        *c.unsafe_index_mut(index) = (alpha * c_thread[i][j] + beta * c.unsafe_index(index).cast::<A>()).cast();
                                    }
                                }
                            }
                        }}
                    }}
                }
            }
        };
        ($($t:ty),* => $a:ty) => {
            $(
                impl_gemm!($t => $a);
            )*
        };
        ($($t:ty),*) => {
            $(
                impl_gemm!($t => $t);
            )*
        };
    }

    impl_gemm!(u8, u16 => u32);
    impl_gemm!(i8, i16 => i32);
    impl_gemm!(f16, bf16 => f32);
    impl_gemm!(u32, i32, f32, u64, i64, f64);
}

#[cfg(feature = "device")]
fn gemm(
    alpha: ScalarElem,
    a: ScalarTensorView2,
    b: ScalarTensorView2,
    beta: ScalarElem,
    mut c: ScalarTensorViewMut2,
) -> Result<()> {
    let a_scalar_type = a.scalar_type();
    let b_scalar_type = b.scalar_type();
    let c_scalar_type = c.scalar_type();
    if a_scalar_type != b_scalar_type {
        bail!("a_scalar_type != b_scalar_type, {a_scalar_type:?} != {b_scalar_type:?}");
    }
    if a_scalar_type != c_scalar_type {
        bail!("a_scalar_type != c_scalar_type, {a_scalar_type:?} != {c_scalar_type:?}");
    }
    let scalar_type = c_scalar_type;

    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    let (m2, n2) = c.dim();

    if m != m2 {
        bail!("a_rows != c_rows, {} != {}", m, m2);
    }
    if k != k2 {
        bail!("a_cols != b_rows, {} != {}", k, k2);
    }
    if n != n2 {
        bail!("b_cols != c_rows, {} != {}", n, n2);
    }

    let m = m.to_u32().unwrap();
    let k = k.to_u32().unwrap();
    let n = n.to_u32().unwrap();

    let [rsa, csa]: [isize; 2] = a.strides().try_into().unwrap();
    let [rsa, csa] = [rsa.to_i32().unwrap(), csa.to_i32().unwrap()];

    let [rsb, csb]: [isize; 2] = b.strides().try_into().unwrap();
    let [rsb, csb] = [rsb.to_i32().unwrap(), csb.to_i32().unwrap()];

    let [rsc, csc]: [isize; 2] = c.strides().try_into().unwrap();
    let [rsc, csc] = [rsc.to_i32().unwrap(), csc.to_i32().unwrap()];

    let (a, offset_a) = a.as_raw_scalar_slice_offset();
    let offset_a = offset_a.to_u32().unwrap();
    let (b, offset_b) = b.as_raw_scalar_slice_offset();
    let offset_b = offset_b.to_u32().unwrap();
    let (mut c, offset_c) = c.as_raw_scalar_slice_offset_mut();
    let offset_c = offset_c.to_u32().unwrap();

    let device = c.device();
    let scalar_type = c.scalar_type();
    macro_wrap!(match scalar_type {
        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            $T::SCALAR_TYPE => {
                let groups_k = if k >= (2 * m * n).max(3 * 64) {
                    (k / 64).min(64)
                } else {
                    1
                };
                let a = a.try_into().unwrap();
                let b = b.try_into().unwrap();
                let mut c = c.try_into().unwrap();
                let [m_group, n_group] = [16, 16];
                let groups_m = m / m_group + (m % m_group != 0) as u32;
                let groups_n = n / n_group + (n % n_group != 0) as u32;
                let gemm_kernel = paste! {
                    kernels::[<gemm_ $T>]::builder()?
                    .with_threads(64)
                    .specialize(m, k, n, groups_k, rsa, csa, rsb, csb, rsc, csc)
                    .build(device.clone())?
                    .with_groups(groups_k * groups_m * groups_n)
                };
                if groups_k > 1 {
                    let mut c_tmp = unsafe {
                        Tensor::<$T, _>::uninit(device.clone(), [(m * n) as usize, groups_k as usize])?
                    };
                    unsafe {
                        gemm_kernel.dispatch(
                            alpha.cast(),
                            a,
                            offset_a,
                            b,
                            offset_b,
                            Default::default(),
                            c_tmp.as_slice_mut().unwrap(),
                            offset_c,
                        )?;
                    }
                    c_tmp.sum_axis_with(Axis(1), beta.cast(), &mut TensorViewMut::from(c))?;
                } else {
                    unsafe {
                        gemm_kernel.dispatch(
                            alpha.cast(),
                            a,
                            offset_a,
                            b,
                            offset_b,
                            beta.cast(),
                            c,
                            offset_c,
                        )?;
                    }
                }
            }
        })
        _ => bail!("Dot unimplemented for {scalar_type:?}!"),
    });
    Ok(())
}

impl<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>> Dot<TensorBase<S2, Ix2>>
    for TensorBase<S1, Ix2>
{
    type Output = Result<Tensor2<T>>;
    fn dot(&self, rhs: &TensorBase<S2, Ix2>) -> Self::Output {
        if let Some((lhs_array, rhs_array)) = self.as_array().zip(rhs.as_array()) {
            let (m, k) = lhs_array.dim();
            let n = rhs_array.dim().1;
            let output = if 100 * m * n < k {
                use rayon::prelude::*;

                let threads = rayon::current_num_threads();
                let k_split = k / threads + (k % threads != 0) as usize;
                let k_chunks = k / k_split + (k % k_split != 0) as usize;
                let mut tmp = unsafe { Array::<T, _>::uninitialized([m, n, k_chunks]) };
                (
                    lhs_array.axis_chunks_iter(Axis(1), k_split),
                    rhs_array.axis_chunks_iter(Axis(0), k_split),
                    tmp.axis_iter_mut(Axis(2)),
                )
                    .into_par_iter()
                    .for_each(|(a, b, mut c)| {
                        ndarray::linalg::general_mat_mul(T::one(), &a, &b, T::zero(), &mut c);
                    });
                let output: Vec<_> = tmp
                    .as_slice()
                    .unwrap()
                    .par_chunks(k_chunks)
                    .map(|x| x.iter().copied().reduce(|a, b| a + b).unwrap())
                    .collect();
                Array::from(output).into_shape([m, n]).unwrap()
            } else {
                let mut output_array = unsafe { Array::<T, _>::uninitialized([m, n]) };
                ndarray::linalg::general_mat_mul(
                    T::one(),
                    &lhs_array,
                    &rhs_array,
                    T::zero(),
                    &mut output_array.view_mut(),
                );
                output_array
            };
            return Ok(output.into());
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            let mut output = unsafe { Tensor::uninit(self.device(), [self.dim().0, rhs.dim().1])? };
            gemm(
                T::one().into(),
                self.view().into(),
                rhs.view().into(),
                T::zero().into(),
                output.view_mut().into(),
            )?;
            Ok(output)
        }
    }
}

impl<S1: ScalarData, S2: ScalarData> Dot<ScalarTensorBase<S2, Ix2>> for ScalarTensorBase<S1, Ix2> {
    type Output = Result<ScalarTensor2>;
    fn dot(&self, rhs: &ScalarTensorBase<S2, Ix2>) -> Self::Output {
        if self.scalar_type() != rhs.scalar_type() {
            bail!(
                "Can not dot tensors of different types {:?} != {:?}!",
                self.scalar_type(),
                rhs.scalar_type()
            );
        }
        let device = self.device();
        let scalar_type = self.scalar_type();
        if device.is_host() && rhs.device().is_host() {
            macro_wrap!(paste! { match scalar_type {
                macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    ScalarType::[<$T:upper>] => {
                        let lhs = TensorView2::<$T>::try_from(self.view()).unwrap();
                        let rhs = TensorView2::<$T>::try_from(rhs.view()).unwrap();
                        return lhs.dot(&rhs).map(Into::into);
                    }
                })
                _ => bail!("Dot unimplemented for {scalar_type:?}!"),
            }});
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            let mut output = unsafe {
                ScalarTensor::uninit(self.device(), [self.dim().0, rhs.dim().1], scalar_type)?
            };
            gemm(
                ScalarElem::one(scalar_type),
                self.view(),
                rhs.view(),
                ScalarElem::zero(scalar_type),
                output.view_mut(),
            )?;
            Ok(output)
        }
    }
}

/*
#[cfg(feature = "device")]
#[test]
fn gemm_bench() {
    use std::{
        env::var,
        fmt::Display,
        str::FromStr,
        time::{Duration, Instant},
    };

    let device_index = var("KRNL_DEVICE")
        .map(|s| usize::from_str(&s).unwrap())
        .unwrap_or_default();

    let device = Device::builder().index(device_index).build().unwrap();

    #[derive(Clone, Copy, derive_more::IsVariant)]
    enum Transpose {
        N,
        T,
    }
    use Transpose::*;

    impl Display for Transpose {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if self.is_t() {
                write!(f, "T")
            } else {
                write!(f, "")
            }
        }
    }
    let mut total_duration = Duration::default();
    for (i, ([m, k, n], [at, bt])) in [
        //([128, 128, 128], [N, N]),
        //([8, 8, 8], [N, N]),
        ([57600, 25, 6], [N, N]),
        ([6400, 150, 16], [N, T]),
        ([100, 256, 128], [N, N]),
        ([100, 128, 84], [N, N]),
        ([100, 84, 10], [N, N]),
        ([100, 10, 84], [N, T]),
        ([84, 100, 10], [T, N]),
        ([100, 84, 128], [N, T]),
        ([100, 128, 256], [N, T]),
        ([100, 128, 256], [T, N]),
        ([6400, 16, 150], [N, N]),
        ([16, 6400, 150], [T, N]),
        ([6, 57600, 25], [T, N]),
    ]
    .into_iter()
    .enumerate()
    {
        let a_shape = if at.is_n() { [m, k] } else { [k, m] };
        let b_shape = if bt.is_n() { [k, n] } else { [n, k] };
        let a = Tensor::<f32, _>::zeros(device.clone(), a_shape).unwrap();
        let a: CowTensor<f32, _> = if at.is_t() {
            a.t().into()
        } else {
            a.view().into()
        };
        let b = Tensor::<f32, _>::zeros(device.clone(), b_shape).unwrap();
        let b: CowTensor<f32, _> = if bt.is_t() {
            b.t().into()
        } else {
            b.view().into()
        };
        let mut c = Tensor::<f32, _>::zeros(device.clone(), [m, n]).unwrap();
        let iters = 1000;
        for _ in 0..iters {
            gemm(
                1f32.into(),
                a.view().into(),
                b.view().into(),
                0f32.into(),
                c.view_mut().into(),
            )
            .unwrap();
            device.wait().unwrap();
        }
        let mut duration = Duration::default();
        for _ in 0..iters {
            let start = Instant::now();
            gemm(
                1f32.into(),
                a.view().into(),
                b.view().into(),
                0f32.into(),
                c.view_mut().into(),
            )
            .unwrap();
            device.wait().unwrap();
            duration += start.elapsed() / iters;
        }
        let gflops = (2 * m * k * n) as f64 / (duration.as_secs_f64() * 1_000_000_000f64);
        println!("{i}: {a_shape:?}{at} x {b_shape:?}{bt} {duration:?} @ {gflops:.2} GFLOPS");
        total_duration += duration;
    }
    println!("{total_duration:?}");
}
*/
