use super::*;
use dry::macro_wrap;
use half::{bf16, f16};
#[cfg(feature = "device")]
use krnl::{macros::module, scalar::ScalarElem};
use ndarray::linalg::Dot;
use paste::paste;

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
    #[allow(unused_imports)]
    use krnl_core::{
        buffer::UnsafeIndex,
        half::{bf16, f16},
        num_traits::One,
        scalar::Scalar,
        spirv_std::arch::workgroup_memory_barrier_with_group_sync as group_barrier,
    };
    use paste::paste;

    pub const THREADS_M: u32 = 16;
    pub const THREADS_N: u32 = 16;
    pub const UNROLL: u32 = 16;
    pub const THREAD_M_MAX: u32 = 8;
    pub const THREAD_N_MAX: u32 = 8;

    #[cfg(target_arch = "spirv")]
    #[rustfmt::skip]
    fn init_c_thread<T: Default + Copy>() -> [T; (THREAD_M_MAX * THREAD_N_MAX) as usize] {
        let z = T::default();
        [
            z, z, z, z, z, z, z, z,
            z, z, z, z, z, z, z, z,
            z, z, z, z, z, z, z, z,
            z, z, z, z, z, z, z, z,
            z, z, z, z, z, z, z, z,
            z, z, z, z, z, z, z, z,
            z, z, z, z, z, z, z, z,
            z, z, z, z, z, z, z, z,
        ]
    }

    macro_rules! impl_gemm {
        ($T:ty => $A:ty) => {
            paste! {
                #[kernel(threads(16, 16, 1))]
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
                    // Thread tile
                    const TM: u32,
                    const TN: u32,
                >(
                    alpha: $A,
                    #[global] a: Slice<$T>,
                    offset_a: u32,
                    #[group] a_group: UnsafeSlice<$A, { ((16 * TM + 1) * 16) as usize }>,
                    #[global] b: Slice<$T>,
                    offset_b: u32,
                    #[group] b_group: UnsafeSlice<$A, { (16 * (16 * TN + 1)) as usize }>,
                    beta: $A,
                    #[global] c: UnsafeSlice<$T>,
                    offset_c: u32,
                ) {
                    let [global_n, global_m, global_k] = kernel.global_id().to_array();
                    let [groups_n, groups_m, groups_k] = kernel.groups().to_array();
                    let [group_n, group_m, group_k] = kernel.group_id().to_array();
                    let [thread_n, thread_m, _] = kernel.thread_id().to_array();

                    if a.len() != (M * K) as usize {
                        panic!();
                    }
                    if b.len() != (K * N) as usize {
                        panic!();
                    }
                    if c.len() != (M * N * groups_k + offset_c) as usize {
                        panic!();
                    }
                    if kernel.threads().y != THREADS_M {
                        panic!();
                    }
                    if kernel.threads().x != THREADS_N {
                        panic!();
                    }
                    if kernel.threads().z != 1 {
                        panic!();
                    }
                    if TM > 8 || TN > 8 || UNROLL > 16 {
                        panic!();
                    }

                    let mut a_thread = <[$A; THREAD_M_MAX as usize]>::default();
                    let mut b_thread = <[$A; THREAD_N_MAX as usize]>::default();
                    let mut c_thread = init_c_thread::<$A>();

                    let mut global_k = group_k * UNROLL;

                    while global_k < K {
                        {
                            let (thread_m, thread_k) = if RSA >= CSA {
                                (thread_m, thread_n)
                            } else {
                                (thread_n, thread_m)
                            };
                            const THREADS_M: u32 = 16;
                            const THREADS_K: u32 = 16;
                            for i in 0 .. TM {
                                let tile_m = i * THREADS_M + thread_m;
                                let tile_k = thread_k;
                                if tile_m < THREADS_M * TM && tile_k < THREADS_K * UNROLL {
                                    let global_m = group_m * THREADS_M * TM + tile_m;
                                    let global_k = global_k + thread_k;
                                    unsafe {
                                        *a_group.unsafe_index_mut(
                                            (tile_m + tile_k * (THREADS_M * TM + 1)) as usize,
                                        ) = if global_m < M && global_k < K {
                                            a[(global_m as i32 * RSA + global_k as i32 * CSA + offset_a as i32)
                                                as usize]
                                                .cast()
                                        } else {
                                            Default::default()
                                        };
                                    }
                                }
                            }
                        }
                        {
                            let (thread_k, thread_n) = if RSB > CSB {
                                (thread_m, thread_n)
                            } else {
                                (thread_n, thread_m)
                            };
                            const THREADS_K: u32 = 16;
                            const THREADS_N: u32 = 16;
                            for j in 0..TN {
                                let tile_k = thread_k;
                                let tile_n = j * THREADS_N + thread_n;
                                if tile_k < THREADS_K * UNROLL && tile_n < THREADS_N * TN {
                                    let global_k = global_k + thread_k;
                                    let global_n = group_n * THREADS_N * TM + tile_n;
                                    unsafe {
                                        *b_group.unsafe_index_mut(
                                            (tile_k * (THREADS_N * TN + 1) + tile_n) as usize,
                                        ) = if global_k < K && global_n < N {
                                            b[(global_k as i32 * RSB + global_n as i32 * CSB + offset_b as i32)
                                                as usize]
                                                .cast()
                                        } else {
                                            Default::default()
                                        };
                                    }
                                }
                            }
                        }

                        unsafe {
                            group_barrier();
                        }

                        for u in 0..UNROLL {
                            unroll! {
                                for _i in 0..8 {
                                    let i = _i as u32;
                                    if i < TM {
                                        unsafe {
                                            a_thread[i as usize] = *a_group.unsafe_index(
                                                (i * THREADS_M + thread_m + u * (THREADS_M * TM + 1)) as usize,
                                            );
                                        }
                                    }
                                }
                            }
                            unroll! {
                                for _j in 0..8 {
                                    let j = _j as u32;
                                    if j < TN {
                                        unsafe {
                                            b_thread[j as usize] = *b_group.unsafe_index(
                                                (u * (THREADS_N * TN + 1) + j * THREADS_N + thread_n) as usize,
                                            );
                                        }
                                    }
                                }
                            }
                            unroll! {
                                for _i in 0..8 {
                                    let i = _i as u32;
                                    unroll! {
                                        for _j in 0..8 {
                                            let j = _j as u32;
                                            if i < TM && j < TN {
                                                c_thread[(i * TN + j) as usize] += a_thread[i as usize] * b_thread[j as usize];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        unsafe {
                            group_barrier();
                        }

                        global_k += groups_k * UNROLL;
                    }

                    let mut row = group_m * THREADS_M * TM + thread_m;
                    for i in 0 .. TM {
                        if row < M {
                            let mut col = group_n * THREADS_N * TN + thread_n;
                            for j in 0 .. TN {
                                if col < N {
                                    let idx = (group_k * M * N) as usize
                                        + (row as i32 * RSC + col as i32 * CSC + offset_c as i32) as usize;
                                    let mut c_out = alpha * c_thread[(i * TN + j) as usize];
                                    if beta > $A::default() {
                                        unsafe {
                                            c_out += beta * c.unsafe_index(idx).cast::<$A>();
                                        }
                                    }
                                    unsafe {
                                        *c.unsafe_index_mut(idx) = c_out.cast();
                                    }
                                } else {
                                    break;
                                }
                                col += THREADS_N;
                            }
                        } else {
                            break;
                        }
                        row += THREADS_M;
                    }
                }

                #[kernel(threads(256))]
                pub fn [<reduce_k_ $T>](#[global] x: Slice<$T>, beta: $A, #[item] y: &mut $T) {
                    let mut idx = kernel.item_id() as usize;
                    let mut acc = if beta > $A::default() {
                        beta * y.cast::<$A>()
                    } else {
                        $A::default()
                    };
                    while idx < x.len() {
                        acc += x[idx].cast::<$A>();
                        idx += kernel.items() as usize;
                    }
                    *y = acc.cast();
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

    impl_gemm!(f16, bf16 => f32);
    impl_gemm!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
}

#[cfg(feature = "device")]
fn gemm(
    alpha: ScalarElem,
    a: ScalarTensorView2,
    b: ScalarTensorView2,
    beta: ScalarElem,
    mut c: ScalarTensorViewMut2,
) -> Result<()> {
    use kernels::{THREADS_M, THREADS_N, THREAD_M_MAX, THREAD_N_MAX, UNROLL};

    assert_eq!(alpha.scalar_type(), c.scalar_type());
    assert_eq!(a.scalar_type(), c.scalar_type());
    assert_eq!(b.scalar_type(), c.scalar_type());
    assert_eq!(beta.scalar_type(), c.scalar_type());

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

    let tm = (m / THREADS_M + (m % THREADS_M != 0) as u32).min(THREAD_M_MAX);
    let tn = (n / THREADS_N + (n % THREADS_N != 0) as u32).min(THREAD_N_MAX);
    let splitk = 4 * UNROLL;
    let groups_k = if c.scalar_type() == ScalarType::F32 {
        k / splitk + ((k % splitk) != 0) as u32
    } else {
        1
    };

    let gm = THREADS_M * tm;
    let gn = THREADS_N * tn;
    let groups_m = m / gm + (m % gm != 0) as u32;
    let groups_n = n / gn + (n % gn != 0) as u32;
    let groups = [groups_n, groups_m, groups_k];

    let device = c.device();
    macro_wrap!(paste! { match c.scalar_type() {
        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            ScalarType::[<$T:upper>] => {
                if groups_k > 1 {
                    assert!(rsc > 0);
                    assert!(csc > 0);
                    assert_eq!(offset_c, 0);
                    let alpha = alpha.cast();
                    let a = Slice::<$T>::try_from(a.clone()).unwrap();
                    let b = Slice::<$T>::try_from(b.clone()).unwrap();
                    let beta = beta.cast();
                    let c = SliceMut::<$T>::try_from(c.as_scalar_slice_mut()).unwrap();
                    let gemm_kernel = kernels::[<gemm_ $T>]::builder()?
                        .specialize(
                            m, k, n, rsa, csa, rsb, csb, rsc, csc, tm, tn,
                        )?
                        .build(device.clone())?
                        .with_groups(groups.into());
                    let reduce_kernel = kernels::[<reduce_k_ $T>]::builder()?.build(device.clone())?;
                    let mut c_tmp = unsafe {
                        Buffer::<$T>::uninit(device, (groups_k * m * n) as usize)?
                    };
                    unsafe {
                        gemm_kernel
                            .dispatch(
                                alpha, a, offset_a, b, offset_b, Default::default(), c_tmp.as_slice_mut(), 0,
                            )?;
                    }
                    reduce_kernel.dispatch(c_tmp.as_slice(), beta, c)?;
                } else {
                    let a = Slice::<$T>::try_from(a.clone()).unwrap();
                    let b = Slice::<$T>::try_from(b.clone()).unwrap();
                    let c = SliceMut::<$T>::try_from(c.as_scalar_slice_mut()).unwrap();
                    let gemm_kernel = kernels::[<gemm_ $T>]::builder()?
                        .specialize(
                            m, k, n, rsa, csa, rsb, csb, rsc, csc, tm, tn,
                        )?
                        .build(device.clone())?
                        .with_groups(groups.into());
                    unsafe {
                        gemm_kernel
                            .dispatch(
                                alpha.cast(), a, offset_a, b, offset_b, beta.cast(), c, offset_c,
                            )?;
                    }
                }
            }
        })
        scalar_type => bail!("Dot unimplemented for {scalar_type:?}!"),
    }});
    Ok(())
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
#[test]
fn gemm_bench() {
    let device = Device::builder().index(1).build().unwrap();
    let x = Tensor::<f32, _>::zeros(device.clone(), [100, 256]).unwrap();
    let w = Tensor::<f32, _>::zeros(device.clone(), [256, 128]).unwrap();
    let mut y = Tensor::<f32, _>::zeros(device.clone(), [100, 128]).unwrap();
    for _ in 0..10 {
        gemm(
            1f32.into(),
            x.view().into(),
            w.view().into(),
            0f32.into(),
            y.view_mut().into(),
        )
        .unwrap();
        device.wait().unwrap();
    }
    use std::time::{Duration, Instant};
    let mut duration = Duration::default();
    for _ in 0..10 {
        let start = Instant::now();
        gemm(
            1f32.into(),
            x.view().into(),
            w.view().into(),
            0f32.into(),
            y.view_mut().into(),
        )
        .unwrap();
        device.wait().unwrap();
        duration += start.elapsed() / 10;
    }
    panic!("{duration:?}");
}

#[test]
fn dot_bench() {
    let device = Device::builder().index(1).build().unwrap();
    let x = Tensor::<f32, _>::zeros(device.clone(), [1000, 28 * 28]).unwrap();
    let w = Tensor::<f32, _>::zeros(device.clone(), [10, 28 * 28]).unwrap();
    for _ in 0..10 {
        let y = x.view().dot(&w.t());
        device.wait().unwrap();
    }
    use std::time::{Duration, Instant};
    let mut duration = Duration::default();
    for _ in 0..10 {
        let start = Instant::now();
        let y = x.view().dot(&w.t());
        device.wait().unwrap();
        duration += start.elapsed() / 10;
    }
    panic!("{duration:?}");
}

#[test]
fn uninit_bench() {
    let device = Device::builder().index(1).build().unwrap();
    for _ in 0..10 {
        let y = unsafe { Tensor::<f32, _>::uninit(device.clone(), [1000, 10]).unwrap() };
        device.wait().unwrap();
    }
    use std::time::{Duration, Instant};
    let mut duration = Duration::default();
    for _ in 0..10 {
        let start = Instant::now();
        let y = unsafe { Tensor::<f32, _>::uninit(device.clone(), [1000, 10]).unwrap() };
        device.wait().unwrap();
        duration += start.elapsed() / 10;
    }
    panic!("{duration:?}");
}

#[test]
fn zeros_bench() {
    let device = Device::builder().index(1).build().unwrap();
    for _ in 0..10 {
        let y = Tensor::<f32, _>::zeros(device.clone(), [1000, 10]).unwrap();
        device.wait().unwrap();
    }
    use std::time::{Duration, Instant};
    let mut duration = Duration::default();
    for _ in 0..10 {
        let start = Instant::now();
        let y = Tensor::<f32, _>::zeros(device.clone(), [1000, 10]).unwrap();
        device.wait().unwrap();
        duration += start.elapsed() / 10;
    }
    panic!("{duration:?}");
}*/

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
