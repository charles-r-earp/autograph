use super::*;
use dry::{macro_for, macro_wrap};
use half::{bf16, f16};
use krnl::macros::module;
#[cfg(feature = "device")]
use krnl::{
    buffer::{ScalarSlice, ScalarSliceMut},
    krnl_core::num_traits::ToPrimitive,
};
use ndarray::{linalg::Dot, Axis};
use paste::paste;

/*
fn c_beta<T: Scalar, D: Dimension>(beta: T, c: &mut TensorViewMut<T, D>) -> Result<()> {
    let n = c.len() as u32;
    let builder = crate::rust_shaders::compute_pass("linalg::c_beta_f32")?
        .slice_mut(c.as_raw_slice_mut())?
        .push(n)?
        .push(beta)?;
    unsafe { builder.submit([n, 1, 1]) }
}

fn bias_impl<T: Scalar>(bias: &TensorView1<T>, c: &mut TensorViewMut2<T>) -> Result<()> {
    if T::scalar_type() != ScalarType::F32 {
        bail!("{} bias not implemented!", elem_type_name::<T>());
    }
    let _oc = bias.dim();
    let (bs, oc) = c.dim();
    if _oc != oc {
        bail!("bias != c_cols, {} != {}", _oc, oc);
    }
    let builder = crate::rust_shaders::compute_pass("linalg::bias_f32")?
        .slice(bias.as_raw_slice())?
        .slice_mut(c.as_raw_slice_mut())?
        .push([bs as u32, oc as u32])?;
    unsafe { builder.submit([(bs * oc) as u32, 1, 1]) }
}
*/

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
        spirv_std::arch::workgroup_memory_barrier_with_group_sync as group_barrier,
    };
    use krnl_core::{
        half::{bf16, f16},
        scalar::Scalar,
    };
    use paste::paste;
    #[cfg(target_arch = "spirv")]
    use static_assertions::const_assert_eq;

    pub const GEMM_THREAD_TILE_MAX_M: u32 = 4;
    pub const GEMM_THREAD_TILE_MAX_K: u32 = 64;
    pub const GEMM_THREAD_TILE_MAX_N: u32 = 4;

    macro_rules! impl_gemm {
        ($T:ty => $A:ty) => {
            paste! {
                #[kernel(threads(256))]
                pub unsafe fn [<gemm_ $T>]<
                    const M: u32,
                    const K: u32,
                    const N: u32,
                    // Strides
                    const RSA: i32,
                    const CSA: i32,
                    const RSB: i32,
                    const CSB: i32,
                    const RSC: i32,
                    const CSC: i32,
                    // Group Tile
                    const GM: u32,
                    const GK: u32,
                    const GN: u32,
                    // Thread tile
                    const TM: u32,
                    const TK: u32,
                    const TN: u32,
                >(
                    alpha: $A,
                    #[global] a: Slice<$T>,
                    offset_a: u32,
                    #[group] a_group: UnsafeSlice<$A, { ((GM + 1) * GK) as usize }>,
                    #[global] b: Slice<$T>,
                    offset_b: u32,
                    #[group] b_group: UnsafeSlice<$A, { (GK * (GN + 1)) as usize }>,
                    beta: $A,
                    #[global] c: UnsafeSlice<$T>,
                    offset_c: u32,
                    n_groups_k: u32,
                ) {
                    // Threads
                    let tsm = GM / TM;
                    let tsk = GK / TK;
                    let tsn = GN / TN;

                    let n_groups_n = N / GN + (N % GN != 0) as u32;

                    let group_id = kernel.group_id();
                    let group_mn = group_id / n_groups_k;
                    let group_k = group_id % n_groups_k;
                    let group_m = group_mn / n_groups_n;
                    let group_n = group_mn % n_groups_n;

                    let threads = kernel.threads();
                    let thread_id = kernel.thread_id();
                    let thread_k = thread_id / (tsm * tsn);
                    let thread_mn = thread_id % (tsm * tsn);
                    let thread_m = thread_mn / tsn;
                    let thread_n = thread_mn % tsn;

                    let global_unroll = n_groups_k * GK;

                    let mut a_thread = <[$A; GEMM_THREAD_TILE_MAX_M as usize]>::default();
                    let mut b_thread = <[$A; GEMM_THREAD_TILE_MAX_N as usize]>::default();
                    let mut c_thread =
                        <[$A; (GEMM_THREAD_TILE_MAX_M * GEMM_THREAD_TILE_MAX_N) as usize]>::default();

                    let mut global_k = group_k * GK;

                    while global_k < K {
                        {
                            // load a
                            let thread_k = thread_id / 16;
                            let thread_m = thread_id % 16;
                            let tsk = 16;
                            let tsm = 16;
                            for u in 0..4 {
                                let tile_k = u as u32 * tsk + thread_k;
                                if tile_k < GK {
                                    let global_k = global_k + tile_k;
                                    for i in 0..4 {
                                        let tile_m = i as u32 * tsm + thread_m;
                                        if tile_m < GM {
                                            let global_m = group_m * GM + tile_m;
                                            unsafe {
                                                *a_group
                                                    .unsafe_index_mut((tile_m + tile_k * (GM + 1)) as usize) =
                                                    if global_m < M && global_k < K {
                                                        a[(global_m as i32 * RSA
                                                            + global_k as i32 * CSA
                                                            + offset_a as i32)
                                                            as usize].cast()
                                                    } else {
                                                        Default::default()
                                                    };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        {
                            // load b
                            let thread_k = thread_id / 16;
                            let thread_n = thread_id % 16;
                            let tsk = 16;
                            let tsn = 16;
                            for u in 0..4 {
                                let tile_k = u as u32 * tsk + thread_k;
                                if tile_k < GK {
                                    let global_k = global_k + tile_k;
                                    for j in 0..4 {
                                        let tile_n = j as u32 * tsn + thread_n;
                                        if tile_n < GN {
                                            let global_n = group_n * GN + tile_n;
                                            unsafe {
                                                *b_group
                                                    .unsafe_index_mut((tile_k * (GN + 1) + tile_n) as usize) =
                                                    if global_k < K && global_n < N {
                                                        b[(global_k as i32 * RSB
                                                            + global_n as i32 * CSB
                                                            + offset_b as i32)
                                                            as usize].cast()
                                                    } else {
                                                        Default::default()
                                                    };
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        const_assert_eq!(GEMM_THREAD_TILE_MAX_M, 4);
                        const_assert_eq!(GEMM_THREAD_TILE_MAX_K, 64);
                        const_assert_eq!(GEMM_THREAD_TILE_MAX_N, 4);

                        unsafe {
                            group_barrier();
                        }

                        for u in 0..TK {
                            unroll! {
                                for i_ in 0..4 {
                                    let i = i_ as u32;
                                    if i < TM {
                                        unsafe {
                                            a_thread[i as usize] = *a_group.unsafe_index(
                                                ((i * tsm + thread_m) + (u * tsk + thread_k) * (GM + 1))
                                                    as usize,
                                            );
                                        }
                                    }
                                }
                            }
                            unroll! {
                                for j_ in 0..4 {
                                    let j = j_ as u32;
                                    if j < TN {
                                        unsafe {
                                            b_thread[j as usize] = *b_group.unsafe_index(
                                                ((u * tsk + thread_k) * (GN + 1) + (j * tsn + thread_n))
                                                    as usize,
                                            );
                                        }
                                    }
                                }
                            }
                            unroll! {
                                for i_ in 0..4 {
                                    let i = i_ as u32;
                                    if i < TM {
                                        unroll! {
                                            for j_ in 0..4 {
                                                let j = j_ as u32;
                                                if j < TN {
                                                    c_thread[(i * TN + j) as usize] +=
                                                        a_thread[i as usize] * b_thread[j as usize];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        unsafe {
                            group_barrier();
                        }

                        global_k += global_unroll;
                    }

                    const_assert_eq!(GEMM_THREAD_TILE_MAX_M, 4);
                    const_assert_eq!(GEMM_THREAD_TILE_MAX_N, 4);
                    unroll! {
                        for i_ in 0..4 {
                            let i = i_ as u32;
                            if i < TM {
                                unroll! {
                                    for j_ in 0..4 {
                                        let j = j_ as u32;
                                        if j < TN {
                                            let row = group_m * GM + i * tsm + thread_m;
                                            let col = group_n * GN + j * tsn + thread_n;
                                            let idx = ((row as i32 * RSC + col as i32 * CSC + offset_c as i32) as u32
                                                * n_groups_k
                                                * tsk
                                                + group_k * tsk
                                                + thread_k) as usize;
                                            if row < M && col < N {
                                                unsafe {
                                                    *c.unsafe_index_mut(idx) = (alpha * c_thread[(i * TN + j) as usize]
                                                        + beta * c.unsafe_index(idx).cast::<$A>()).cast();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        ($($T:ty),*) => {
            $(
                impl_gemm!($T => $T);
            )*
        };
        ($($T:ty),* => $A:ty) => {
            $(
                impl_gemm!($T => $A);
            )*
        };
    }

    impl_gemm!(u8, u16 => u32);
    impl_gemm!(i8, i16 => i32);
    impl_gemm!(f16, bf16 => f32);
    impl_gemm!(u32, i32, f32, u64, i64, f64);
}

#[cfg(feature = "device")]
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn gemm<T: Scalar>(
    alpha: T,
    a: &TensorView2<T>,
    b: &TensorView2<T>,
    beta: T,
    c: &mut TensorViewMut2<T>,
) -> Result<()> {
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

    let (a, offset_a) = a.as_raw_slice_offset();
    let offset_a = offset_a.to_u32().unwrap();
    let (b, offset_b) = b.as_raw_slice_offset();
    let offset_b = offset_b.to_u32().unwrap();
    let (c, offset_c) = c.as_raw_slice_offset_mut();
    let offset_c = offset_c.to_u32().unwrap();

    let splitk = 512;
    let (n_groups_k, [tsm, tsk, tsn], [tm, tk, tn]) = if k > splitk {
        let n_groups_k = k / splitk + if k % splitk != 0 { 1 } else { 0 };
        (n_groups_k, [16u32, 1, 16], [1u32, 64, 1])
    } else if m >= 64 && n <= 32 {
        (1, [32, 1, 8], [4, 8, 4])
    } else if n >= 64 && m <= 32 {
        (1, [8, 1, 32], [4, 8, 4])
    } else {
        (1, [16, 1, 16], [4, 8, 4])
    };
    assert!(tm <= kernels::GEMM_THREAD_TILE_MAX_M);
    assert!(tk <= kernels::GEMM_THREAD_TILE_MAX_K);
    assert!(tn <= kernels::GEMM_THREAD_TILE_MAX_N);
    let gm = tsm * tm;
    let gk = tsk * tk;
    let gn = tsn * tn;
    let groups_m = m / gm + (m % gm != 0) as u32;
    let groups_n = n / gn + (n % gn != 0) as u32;
    let groups = groups_m * groups_n * n_groups_k;
    //let gemm_beta = if n_groups_k > 1 { T::zero() } else { beta };
    if n_groups_k * tsk > 1 {
        todo!();
        /*
        let mut c_tmp =
            Tensor::zeros(
                a.device(),
                [m as usize, n as usize, (n_groups_k * gk) as usize],
            )?
        };
        let builder = builder.slice_mut(c_tmp.as_raw_slice_mut())?;
        unsafe {
            builder.submit(work_size)?;
        }
        c_beta(beta, c)?;
        c_tmp.sum_axis_with(Axis(2), c)?; */
    } else {
        let device = c.device();
        macro_wrap!(paste! { match T::scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarType::[<$T:upper>] => {
                    let a: Slice<$T> = ScalarSlice::from(a)
                        .try_into()
                        .ok()
                        .unwrap();
                    let b: Slice<$T> = ScalarSlice::from(b)
                        .try_into()
                        .ok()
                        .unwrap();
                    let c: SliceMut<$T> = ScalarSliceMut::from(c)
                        .try_into()
                        .ok()
                        .unwrap();
                    unsafe {
                        kernels::[<gemm_ $T>]::builder()?
                            .specialize(
                                m, k, n, rsa, csa, rsb, csb, rsc, csc, gm, gk, gn, tm, tk, tn,
                            )?
                            .build(device)?
                            .with_groups(groups)
                            .dispatch(
                                alpha.cast(), a, offset_a, b, offset_b, beta.cast(), c, offset_c, n_groups_k,
                            )?;
                    }
                }
            })
            scalar_type => bail!("Dot unimplemented for {scalar_type:?}!"),
        }});
    }
    Ok(())
}

impl<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>> Dot<TensorBase<S2, Ix2>>
    for TensorBase<S1, Ix2>
{
    type Output = Result<Tensor2<T>>;
    fn dot(&self, rhs: &TensorBase<S2, Ix2>) -> Self::Output {
        if let Some((lhs, rhs)) = self.as_array().zip(rhs.as_array()) {
            return Ok(lhs.dot(&rhs).into());
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            let mut output = Tensor::zeros(self.device(), [self.dim().0, rhs.dim().1])?;
            gemm(
                T::one(),
                &self.view(),
                &rhs.view(),
                T::zero(),
                &mut output.view_mut(),
            )?;
            Ok(output)
        }
    }
}
/*
impl<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>, S3: Data<Elem = T>>
    DotBias<TensorBase<S2, Ix2>, TensorBase<S3, Ix1>> for TensorBase<S1, Ix2>
{
    fn dot_bias(
        &self,
        rhs: &TensorBase<S2, Ix2>,
        bias: Option<&TensorBase<S3, Ix1>>,
    ) -> Result<Self::Output> {
        let mut output = unsafe { Tensor::alloc(self.device(), [self.dim().0, rhs.dim().1])? };
        gemm(
            T::one(),
            &self.view(),
            &rhs.view(),
            T::zero(),
            &mut output.view_mut(),
        )?;
        if let Some(bias) = bias {
            bias_impl(&bias.view(), &mut output.view_mut())?;
        }
        Ok(output)
    }
}

impl<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>, S3: DataMut<Elem = T>>
    DotAcc<T, TensorBase<S2, Ix2>, TensorBase<S3, Ix2>> for TensorBase<S1, Ix2>
{
    fn dot_acc(
        &self,
        alpha: T,
        rhs: &TensorBase<S2, Ix2>,
        output: &mut TensorBase<S3, Ix2>,
    ) -> Result<()> {
        gemm(
            alpha,
            &self.view(),
            &rhs.view(),
            T::one(),
            &mut output.view_mut(),
        )
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::util::type_eq;
    use approx::assert_relative_eq;
    use ndarray::Array2;
    use paste::paste;

    #[allow(unused)]
    #[derive(Clone, Copy, Debug)]
    enum Transpose {
        N,
        T,
    }
    use Transpose::*;

    fn gen_array<T: From<u8>>(dim: [usize; 2]) -> Array2<T> {
        let n = dim[0] * dim[1];
        let vec: Vec<T> = (0..n)
            .into_iter()
            .map(|x| T::from((((x + 100) % 100) + 1) as u8))
            .collect();
        Array2::from_shape_vec(dim, vec).unwrap()
    }

    async fn tensor_dot<T: Scalar + From<u8>>(
        [m, k, n]: [usize; 3],
        [a_t, b_t]: [Transpose; 2],
    ) -> Result<()> {
        let device = Device::new()?;
        let dim1 = match a_t {
            Transpose::N => [m, k],
            Transpose::T => [k, m],
        };
        let dim2 = match b_t {
            Transpose::N => [k, n],
            Transpose::T => [n, k],
        };
        let a1 = gen_array::<T>(dim1);
        let t1 = CowTensor::from(a1.view())
            .into_device(device.clone())
            .await?;
        let (a1, t1) = match a_t {
            Transpose::N => (a1.view(), t1.view()),
            Transpose::T => (a1.t(), t1.t()),
        };
        let a2 = gen_array::<T>(dim2);
        let t2 = CowTensor::from(a2.view())
            .into_device(device.clone())
            .await?;
        let (a2, t2) = match b_t {
            Transpose::N => (a2.view(), t2.view()),
            Transpose::T => (a2.t(), t2.t()),
        };
        let a_true = a1.dot(&a2);
        let a_out = t1.dot(&t2)?.read().await?.into_array();
        if type_eq::<T, f32>() {
            let a_true = a_true.map(|x| x.to_f32().unwrap());
            let a_out = a_out.map(|x| x.to_f32().unwrap());
            assert_relative_eq!(a_true, a_out);
        } else {
            assert_eq!(a_true, a_out);
        }
        Ok(())
    }

    macro_rules! test_dot {
        (@Outer $($args:tt)*) => (
            test_dot! {
                @Trans u32;
                $(
                    $args
                )*
            }
            test_dot! {
                @Trans i32;
                $(
                    $args
                )*
            }
            test_dot! {
                @Trans f32;
                $(
                    $args
                )*
            }
        );
        (@Trans $T:ty; $([$M:tt, $K:tt, $N:tt],)*) => (
            test_dot! {
                @Impl $T;
                $(
                    ([$M, $K, $N], [N, N]),
                    ([$M, $K, $N], [T, N]),
                    ([$M, $K, $N], [N, T]),
                    ([$M, $K, $N], [T, T]),
                )*
            }
        );
        (@Impl $T:ty; $(([$M:tt, $K:tt, $N:tt], [$TA:tt, $TB:tt]),)*) => (
            paste! {
                $(
                    #[allow(non_snake_case)]
                    #[tokio::test]
                    async fn [<tensor_dot_ $T _m $M _k $K _n $N _ $TA _ $TB>]() -> Result<()> {
                        tensor_dot::<$T>([$M, $K, $N], [$TA, $TB]).await
                    }
                )*
            }
        );
    }

    test_dot! {
        @Outer
        [21, 31, 41], [31, 41, 21],
        [121, 131, 141], [131, 121, 141],
        [7, 603, 19], [67, 543, 83],
    }

    #[cfg(feature = "bench")]
    mod bench {
        use super::*;
        use num_traits::FromPrimitive;
        use test::Bencher;

        macro_rules! bench_dot {
            ($t:tt; $($name:ident => $args:expr,)+) => (
                $(
                    #[allow(non_snake_case)]
                    #[bench]
                    fn $name (bencher: &mut Bencher) {
                        let device = Device::new().unwrap();
                        let _s = smol::block_on(device.acquire());
                        let (m, k, n, a_t, b_t) = $args;
                        let dim1 = match a_t {
                            Transpose::N => [m, k],
                            Transpose::T => [k, m],
                        };
                        let a = Tensor::from_elem(device.clone(), dim1, $t::from_u32(1).unwrap()).unwrap();
                        let a = match a_t {
                            N => a.view(),
                            T => a.t()
                        };
                        let dim2 = match b_t {
                            Transpose::N => [k, n],
                            Transpose::T => [n, k],
                        };
                        let b = Tensor::from_elem(device.clone(), dim2, $t::from_u32(1).unwrap()).unwrap();
                        let b = match b_t {
                            N => b.view(),
                            T => b.t()
                        };
                        for _ in 0 .. 100 {
                            a.view().dot(&b.view()).unwrap();
                            smol::block_on(device.sync()).unwrap();
                        }
                        bencher.iter(|| {
                            a.view().dot(&b.view()).unwrap();
                            smol::block_on(device.sync()).unwrap();
                        });
                    }
                )+
            );
        }

        bench_dot!(
            f32;
            tensor_dot_f32_m512_k512_n512_N_N => (512, 512, 512, N, N),
            tensor_dot_f32_m1024_k1024_n1024_N_N => (1024, 1024, 1024, N, N),
            tensor_dot_f32_m57600_k25_n6_N_N => (57600, 25, 6, N, N),
            tensor_dot_f32_m57600_k25_n6_N_T => (57600, 25, 6, N, T),
            tensor_dot_f32_m57600_k25_n6_T_N => (57600, 25, 6, T, N),
            tensor_dot_f32_m57600_k25_n6_T_T => (57600, 25, 6, T, T),
            tensor_dot_f32_m25_k57600_n6_T_N => (25, 57600, 6, T, N),
            tensor_dot_f32_m57600_k6_n25_N_T => (57600, 6, 25, N, T),
            tensor_dot_f32_m6400_k150_n16_N_N => (6400, 150, 16, N, T),
            tensor_dot_f32_m150_k6400_n16_T_N => (150, 6400, 16, T, N),
        );
    }
}
*/
