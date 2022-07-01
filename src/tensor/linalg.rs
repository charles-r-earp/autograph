use super::*;
use crate::linalg::{Dot, DotAcc, DotBias};
use ndarray::Axis;

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

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn gemm<T: Scalar>(
    alpha: T,
    a: &TensorView2<T>,
    b: &TensorView2<T>,
    beta: T,
    c: &mut TensorViewMut2<T>,
) -> Result<()> {
    use ScalarType::*;

    if !matches!(T::scalar_type(), U32 | I32 | F32) {
        bail!("{} dot not implemented!", elem_type_name::<T>());
    }

    /*
    // Patch for custom 16 bit ops
    // If 16 bit is supported then this is unnecessary
    if size_eq::<T, u16>() && beta == T::zero() {
        c.as_raw_slice_mut().fill(T::default())?;
    }
    */

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

    let m = m as u32;
    let k = k as u32;
    let n = n as u32;

    // TODO: Potentially support negative strides with an offset?
    // Strides are negative to match ndarray but we can't offset pointers yet.
    // Offsetting pointers of 16 and 8 bit types is a problem because bindings
    // are required to be aligned to 32 bits on most backends.
    let [rsa, csa]: [isize; 2] = a.strides().try_into().unwrap();
    let [rsa, csa] = [rsa as u32, csa as u32];

    let [rsb, csb]: [isize; 2] = b.strides().try_into().unwrap();
    let [rsb, csb] = [rsb as u32, csb as u32];

    let [rsc, csc]: [isize; 2] = c.strides().try_into().unwrap();
    let [rsc, csc] = [rsc as u32, csc as u32];

    let splitk = 512;
    let (n_groups_k, [gm, gk, gn], [lm, lk, ln]) = if k > splitk {
        let n_groups_k = k / splitk + if k % splitk != 0 { 1 } else { 0 };
        (n_groups_k, [16, 1, 16], [1, 64, 1])
    } else if m >= 64 && n <= 32 {
        (1, [32, 1, 8], [4, 8, 4])
    } else if n >= 64 && m <= 32 {
        (1, [8, 1, 32], [4, 8, 4])
    } else {
        (1, [16, 1, 16], [4, 8, 4])
    };
    let tm = gm * lm;
    let tk = gk * lk;
    let tn = gn * ln;
    let entry = format!(
        "linalg::gemm_{}_tm{}_tk{}_tn{}_lm{}_lk{}_ln{}",
        elem_type_name::<T>(),
        tm,
        tk,
        tn,
        lm,
        lk,
        ln,
    );
    let global_m = (m / tm + if m % tm != 0 { 1 } else { 0 }) * gm;
    let global_n = (n / tn + if n % tn != 0 { 1 } else { 0 }) * gn;
    let work_size = [(global_m * global_n * n_groups_k * gk) as u32, 1, 1];
    let gemm_beta = if n_groups_k > 1 { T::zero() } else { beta };
    let builder = crate::rust_shaders::compute_pass(entry)?
        .slice(a.as_raw_slice())?
        .slice(b.as_raw_slice())?
        .push([alpha, gemm_beta])?
        .push([m, k, n])?
        .push([rsa, csa])?
        .push([rsb, csb])?
        .push([rsc, csc])?
        .push(n_groups_k)?;
    if n_groups_k * gk > 1 {
        let mut c_tmp = unsafe {
            Tensor::alloc(
                a.device(),
                [m as usize, n as usize, (n_groups_k * gk) as usize],
            )?
        };
        let builder = builder.slice_mut(c_tmp.as_raw_slice_mut())?;
        unsafe {
            builder.submit(work_size)?;
        }
        c_beta(beta, c)?;
        c_tmp.sum_axis_with(Axis(2), c)?;
    } else {
        let builder = builder.slice_mut(c.as_raw_slice_mut())?;
        unsafe {
            builder.submit(work_size)?;
        }
    }
    Ok(())
}

impl<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>> Dot<TensorBase<S2, Ix2>>
    for TensorBase<S1, Ix2>
{
    type Output = Tensor2<T>;
    fn dot(&self, rhs: &TensorBase<S2, Ix2>) -> Result<Self::Output> {
        let mut output = unsafe { Tensor::alloc(self.device(), [self.dim().0, rhs.dim().1])? };
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
