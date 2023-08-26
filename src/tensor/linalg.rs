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

                    let group_id = kernel.group_id as usize;
                    let group_k = group_id / groups_mn;
                    let group_mn = group_id % groups_mn;
                    let group_m = group_mn / groups_n;
                    let group_n = group_mn % groups_n;
                    let global_m = group_m * m_group;
                    let mut global_k = group_k * unroll;
                    let global_n = group_n * n_group;

                    let thread_id = kernel.thread_id as usize;
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

                    let compute =
                        |a_thread: &mut [A; 2], b_thread: &mut [A; 2], c_thread: &mut [[A; 2]; 2]| {
                            unroll! { for tile_k in 0 .. 8 {
                                unroll! { for i in 0 .. 2 {
                                    let tile_m = i * threads_m + thread_m;
                                    unsafe {
                                        a_thread[i] = *a_group.unsafe_index(tile_m * (unroll + 1) + tile_k);
                                    }
                                }}
                                unroll! { for j in 0 .. 2 {
                                    let tile_n = j * threads_n + thread_n;
                                    unsafe {
                                        b_thread[j] = *b_group.unsafe_index(tile_k * (n_group + 1) + tile_n);
                                    }
                                }}
                                unroll! { for i in 0 .. 2 {
                                    unroll! { for j in 0 .. 2 {
                                        c_thread[i][j] += a_thread[i] * b_thread[j];
                                    }}
                                }}
                            }}
                        };

                    {
                        let tile_m = thread_m_a;
                        let global_m = global_m + tile_m;
                        unroll! { for u in 0 .. 2 {
                            let tile_k = u * threads_k_a + thread_k_a;
                            let global_k = global_k + tile_k;
                            let a = if global_m < m && global_k < k {
                                a[(global_m as i32 * RSA + global_k as i32 * CSA + offset_b as i32) as usize].cast()
                            } else {
                                A::zero()
                            };
                            unsafe {
                                *a_group.unsafe_index_mut(tile_m * (unroll + 1) + tile_k) = a;
                            }
                        }}
                    }
                    {
                        let tile_n = thread_n_b;
                        let global_n = global_n + tile_n;
                        unroll! { for u in 0 .. 2 {
                            let tile_k = u * threads_k_b + thread_k_b;
                            let global_k = global_k + tile_k;
                            let b = if global_k < k && global_n < n {
                                b[(global_k as i32 * RSB + global_n as i32 * CSB + offset_b as i32) as usize].cast()
                            } else {
                                A::zero()
                            };
                            unsafe {
                                *b_group.unsafe_index_mut(tile_k * (n_group + 1) + tile_n) = b;
                            }
                        }}
                    }
                    unsafe {
                        group_barrier();
                    }
                    global_k += global_unroll;

                    while global_k < k {
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
                        compute(&mut a_thread, &mut b_thread, &mut c_thread);
                        unsafe {
                            group_barrier();
                        }
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
                        unsafe {
                            group_barrier();
                        }
                        global_k += global_unroll;
                    }
                    compute(&mut a_thread, &mut b_thread, &mut c_thread);

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
        }
    }

    impl_gemm!(u8, u16 => u32);
    impl_gemm!(i8, i16 => i32);
    impl_gemm!(f16, bf16 => f32);
    impl_gemm!(u32, i32, f32, u64, i64, f64);

    #[cfg(target_arch = "spirv")]
    unsafe fn subgroup_add_f32(x: f32) -> f32 {
        use core::arch::asm;

        let mut y = 0f32;
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%subgroup = OpConstant %u32 3",
            "%y = OpGroupNonUniformFAdd _ %subgroup Reduce {x}",
            "OpStore {y} %y",
            x = in(reg) x,
            y = in(reg) &mut y,
        }
        y
    }

    #[kernel]
    pub unsafe fn reduce_k_f32<const TS: u32>(
        #[global] x: Slice<f32>,
        #[group] y_group: UnsafeSlice<f32, 1>,
        #[global] y: UnsafeSlice<f32>,
    ) {
        let n = x.len() / y.len();
        let threads = TS as usize;
        let thread_id = kernel.thread_id as usize;
        let groups = kernel.groups as usize;
        let group_id = kernel.group_id as usize;
        let subgroups = kernel.subgroups as usize;
        let subgroup_id = kernel.subgroup_id as usize;
        let subgroup_thread_id = kernel.subgroup_thread_id as usize;
        let mut y_thread = 0f32;
        let mut idx = 0;
        while idx < n {
            y_thread += x[group_id * n + idx + thread_id];
            idx += threads;
        }
        unsafe {
            y_thread = subgroup_add_f32(y_thread);
        };
        if subgroups == 1 {
            if subgroup_thread_id == 0 {
                unsafe {
                    *y.unsafe_index_mut(group_id) = y_thread;
                }
            }
        } else {
            if subgroup_thread_id == 0 {
                unsafe {
                    *y_group.unsafe_index_mut(0) = y_thread;
                }
            }
            for i in 1..subgroups - 1 {
                unsafe {
                    group_barrier();
                }
                if subgroup_id == i && subgroup_thread_id == 0 {
                    unsafe {
                        *y_group.unsafe_index_mut(0) += y_thread;
                    }
                }
            }
            unsafe {
                group_barrier();
            }
            if subgroup_id == subgroups - 1 && subgroup_thread_id == 0 {
                unsafe {
                    *y.unsafe_index_mut(group_id) = y_group.unsafe_index(0) + y_thread;
                }
            }
        }
    }
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
    if scalar_type == ScalarType::F32 {
        let a = Slice::<f32>::try_from(a.clone()).unwrap();
        let b = Slice::<f32>::try_from(b.clone()).unwrap();
        let mut c = SliceMut::<f32>::try_from(c.as_scalar_slice_mut()).unwrap();

        let groups_k = if k >= (2 * m * n).max(3 * 64) {
            (k / 64).min(64)
        } else {
            1
        };

        let alpha = alpha.cast::<f32>();
        let beta = beta.cast::<f32>();
        let [m_group, n_group] = [16, 16];
        let groups_m = m / m_group + (m % m_group != 0) as u32;
        let groups_n = n / n_group + (n % n_group != 0) as u32;
        let gemm_kernel = kernels::gemm_f32::builder()?
            .with_threads(64)
            .specialize(m, k, n, groups_k, rsa, csa, rsb, csb, rsc, csc)
            .build(device.clone())?
            .with_groups(groups_k * groups_m * groups_n);
        if groups_k > 1 {
            let mut c_tmp = unsafe {
                Tensor::<f32, _>::uninit(device.clone(), [(m * n) as usize, groups_k as usize])?
            };
            unsafe {
                gemm_kernel.dispatch(
                    alpha,
                    a,
                    offset_a,
                    b,
                    offset_b,
                    0f32,
                    c_tmp.as_slice_mut().unwrap(),
                    offset_c,
                )?;
                /*if false {
                    unsafe {
                        kernels::reduce_k_f32::builder()?
                            .specialize(64)?
                            .build(device.clone())?
                            .with_groups(m * n)
                            .dispatch(c_tmp.as_slice().unwrap(), c)?;
                    }
                }*/
                c_tmp.sum_axis_with(Axis(1), beta, &mut TensorViewMut::from(c))?;
            }
        } else {
            unsafe {
                gemm_kernel.dispatch(
                    alpha,
                    a,
                    offset_a,
                    b,
                    offset_b,
                    beta,
                    c.as_slice_mut(),
                    offset_c,
                )?;
            }
        }
        return Ok(());
    }
    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, /* f32, */ u64, i64, f64] {
        if scalar_type == $T::scalar_type() {
            let a = Slice::<$T>::try_from(a.clone()).unwrap();
            let b = Slice::<$T>::try_from(b.clone()).unwrap();
            let mut c = SliceMut::<$T>::try_from(c.as_scalar_slice_mut()).unwrap();

            let groups_k = 1;

            let alpha = alpha.cast();
            let beta = beta.cast();
            let [m_group, n_group] = [16, 16];
            let groups_m = m / m_group + (m % m_group != 0) as u32;
            let groups_n = n / n_group + (n % n_group != 0) as u32;
            let gemm_kernel = paste! { kernels::[<gemm_ $T>]::builder()? }
                .with_threads(64)
                .specialize(m, k, n, groups_k, rsa, csa, rsb, csb, rsc, csc)
                .build(device.clone())?
                .with_groups(groups_m * groups_n);
            unsafe {
                gemm_kernel
                    .dispatch(alpha, a, offset_a, b, offset_b, beta, c.as_slice_mut(), offset_c)?;
            }
            return Ok(());
        }
    });
    bail!("Dot unimplemented for {scalar_type:?}!")
}

impl<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>> Dot<TensorBase<S2, Ix2>>
    for TensorBase<S1, Ix2>
{
    type Output = Result<Tensor2<T>>;
    fn dot(&self, rhs: &TensorBase<S2, Ix2>) -> Self::Output {
        if let Some((lhs_array, rhs_array)) = self.as_array().zip(rhs.as_array()) {
            /*
             // TODO: bf16 is very slow because it falls back to naive alg, min is handle more shapes here
            if matches!(T::scalar_type(), ScalarType::BF16)
                && lhs_array.is_standard_layout()
                && rhs_array.is_standard_layout()
            {
                use half::{slice::HalfFloatSliceExt, vec::HalfFloatVecExt};
                let lhs_vec = if let Some(slice) = self.as_slice_memory_order() {
                    let slice = slice.as_host_slice().unwrap();
                    let slice: &[bf16] = bytemuck::cast_slice(slice);
                    slice.to_f32_vec()
                } else {
                    todo!()
                };
                let lhs = Tensor {
                    dim: self.dim.clone(),
                    strides: self.strides.clone(),
                    buffer: Buffer::from(lhs_vec),
                    offset: 0,
                };
                let rhs_vec = if let Some(slice) = rhs.as_slice_memory_order() {
                    let slice = slice.as_host_slice().unwrap();
                    let slice: &[bf16] = bytemuck::cast_slice(slice);
                    slice.to_f32_vec()
                } else {
                    todo!()
                };
                let rhs = Tensor {
                    dim: rhs.dim.clone(),
                    strides: rhs.strides.clone(),
                    buffer: Buffer::from(rhs_vec),
                    offset: 0,
                };
                let output = lhs.as_array().unwrap().dot(&rhs.as_array().unwrap());
                let output_vec = Vec::<bf16>::from_f32_slice(output.as_slice().unwrap());
                return Ok(Tensor::from(
                    Array::from(output_vec)
                        .into_shape(output.raw_dim())
                        .unwrap(),
                )
                .cast_into()
                .unwrap());
            }*/
            return Ok(lhs_array.dot(&rhs_array).into());
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
        /*  ([128, 128, 128], [N, N]),
        ([8, 8, 8], [N, N]),
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
        ([6400, 16, 150], [N, N]), */
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
        let iters = 100;
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
