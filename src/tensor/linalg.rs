use super::*;
use crate::linalg::{Dot, DotAcc, DotBias};

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn gemm_impl<T: Scalar>(
    alpha: T,
    a: &TensorView2<T>,
    b: &TensorView2<T>,
    bias: Option<&TensorView1<T>>,
    beta: T,
    c: &mut TensorViewMut2<T>,
) -> Result<()> {
    use ScalarType::*;
    if !matches!(T::scalar_type(), U32 | I32 | F32 | BF16) {
        bail!("{} dot not implemented!", elem_type_name::<T>());
    }

    // Patch for custom 16 bit ops
    // If 16 bit is supported then this is unnecessary
    if size_eq::<T, u16>() && beta == T::zero() {
        c.as_raw_slice_mut().fill(T::default())?;
    }

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

    let a0 = 0f32;

    let (builder, ts, wpt, splitk) = match T::scalar_type() {
        BF16 => {
            // TODO: Porting to Rust requires atomics
            let ts = 16;
            let wpt = 1;
            let name = format!("gemm{}_bf16", if bias.is_some() { "_bias" } else { "" },);
            let builder = crate::glsl_shaders::module(name)?.compute_pass("main")?;
            (builder, ts, wpt, None)
        }
        _ => {
            let ts = 16;
            let splitk = 256;
            let splitk = if k >= (m * n).max(splitk * 2) {
                Some(splitk)
            } else {
                None
            };
            let wpt = if splitk.is_none() && T::scalar_type() == F32 {
                u32::min(m / ts, n / ts).min(4).max(1)
            } else {
                1
            };
            let unr = if wpt == 1 { 16 } else { 8 };
            let entry = format!(
                "linalg::gemm{}_{}{}_unr{}_mica{}_micb{}",
                if bias.is_some() { "_bias" } else { "" },
                elem_type_name::<T>(),
                splitk.map_or(String::new(), |splitk| format!("_splitk{}", splitk)),
                unr,
                wpt,
                wpt,
            );
            let builder = crate::rust_shaders::core()?.compute_pass(entry)?;
            (builder, ts, wpt, splitk)
        }
    };

    if splitk.is_some() {
        let builder = crate::rust_shaders::core()?
            .compute_pass("linalg::c_beta_f32")?
            .slice_mut(c.as_raw_slice_mut())?
            .push(m * n)?
            .push(beta)?;
        unsafe {
            builder.submit([m * n, 1, 1])?;
        }
    };

    let builder = builder.slice(a.as_raw_slice())?.slice(b.as_raw_slice())?;
    let builder = if let Some(bias) = bias.as_ref() {
        builder.slice(bias.as_raw_slice())?
    } else {
        builder
    };
    let builder = builder.slice_mut(c.as_raw_slice_mut())?;

    let builder = match T::scalar_type() {
        BF16 => builder.push([alpha.to_f32().unwrap(), beta.to_f32().unwrap(), a0])?,
        _ => builder.push([alpha, beta])?,
    };
    let builder = builder
        .push([m, k, n])?
        .push([rsa, csa])?
        .push([rsb, csb])?
        .push([rsc, csc])?;

    let work_size = if T::scalar_type() == BF16 {
        [m, n, 1]
    } else {
        let global_x = m / wpt + if m % (ts * wpt) != 0 { ts } else { 0 };
        let global_y = n / wpt + if n % (ts * wpt) != 0 { ts } else { 0 };
        let global_z = splitk.map_or(1, |splitk| k / splitk + if k % splitk != 0 { 1 } else { 0 });
        [global_x * global_y * global_z, 1, 1]
    };
    unsafe { builder.submit(work_size) }
}

impl<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>> Dot<TensorBase<S2, Ix2>>
    for TensorBase<S1, Ix2>
{
    type Output = Tensor2<T>;
    fn dot(&self, rhs: &TensorBase<S2, Ix2>) -> Result<Self::Output> {
        let mut output = unsafe { Tensor::alloc(self.device(), [self.dim().0, rhs.dim().1])? };
        gemm_impl(
            T::one(),
            &self.view(),
            &rhs.view(),
            None,
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
        gemm_impl(
            T::one(),
            &self.view(),
            &rhs.view(),
            bias.map(TensorBase::view).as_ref(),
            T::zero(),
            &mut output.view_mut(),
        )?;
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
        gemm_impl(
            alpha,
            &self.view(),
            &rhs.view(),
            None,
            T::one(),
            &mut output.view_mut(),
        )
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use half::bf16;
    use ndarray::Array2;

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

    macro_rules! tensor_dot {
        ($t1:ty, $t2:tt, $args:expr) => {{
            let device = Device::new()?;
            let _s = device.acquire().await;
            let (m, k, n, a_t, b_t) = $args;
            let dim1 = match a_t {
                Transpose::N => [m, k],
                Transpose::T => [k, m],
            };
            let dim2 = match b_t {
                Transpose::N => [k, n],
                Transpose::T => [n, k],
            };
            let a1 = gen_array::<$t1>(dim1);
            let t1 = Tensor2::<$t2>::from(gen_array(dim1))
                .into_device(device.clone())
                .await?;
            let (a1, t1) = match a_t {
                Transpose::N => (a1.view(), t1.view()),
                Transpose::T => (a1.t(), t1.t()),
            };
            let a2 = gen_array::<$t1>(dim2);
            let t2 = Tensor2::<$t2>::from(gen_array(dim2))
                .into_device(device.clone())
                .await?;
            let (a2, t2) = match b_t {
                Transpose::N => (a2.view(), t2.view()),
                Transpose::T => (a2.t(), t2.t()),
            };
            let a_true = a1.dot(&a2);
            let t_out = t1.dot(&t2)?;
            device.sync().await?;
            let a_out = t_out.read().await?.into_array();
            (a_true, a_out)
        }};
    }

    macro_rules! check_arrays {
        (f32 => ($a:expr, $b:expr)) => {
            assert_relative_eq!($a, $b);
        };
        (bf16 => ($a:expr, $b:expr)) => {
            let b = $b.map(|x| x.to_f32());
            assert_relative_eq!($a, b, epsilon = 0.01, max_relative = 0.01);
        };
        ($t:tt => ($a:expr, $b:expr)) => {
            assert_eq!($a, $b);
        };
    }

    macro_rules! test_dot {
        (bf16; $($name:ident => $args:expr,)+) => (
            $(
                #[allow(non_snake_case)]
                #[tokio::test]
                async fn $name () -> Result<()> {
                    let (a_true, a_out) = tensor_dot! { f32, bf16, $args };
                    check_arrays!(bf16 => (a_true, a_out));
                    Ok(())
                }
            )+
        );
        ($t:tt; $($name:ident => $args:expr,)+) => (
            $(
                #[allow(non_snake_case)]
                #[tokio::test]
                async fn $name () -> Result<()> {
                    let (a_true, a_out) = tensor_dot! { $t, $t, $args };
                    check_arrays!($t => (a_true, a_out));
                    Ok(())
                }
            )+
        );
    }

    test_dot!(
        u32;
        tensor_dot_u32_m21_k31_n41_N_N => (21, 31, 41, N, N),
        tensor_dot_u32_m121_k131_n141_N_N => (121, 131, 141, N, N),
        tensor_dot_u32_m121_k131_n141_T_N => (121, 131, 141, T, N),
        tensor_dot_u32_m121_k131_n141_N_T => (121, 131, 141, N, T),
        tensor_dot_u32_m121_k131_n141_T_T => (121, 131, 141, T, T),
    );

    test_dot!(
        i32;
        tensor_dot_i32_m21_k31_n41_N_N => (21, 31, 41, N, N),
        tensor_dot_i32_m121_k131_n141_N_N => (121, 131, 141, N, N),
        tensor_dot_i32_m121_k131_n141_T_N => (121, 131, 141, T, N),
        tensor_dot_i32_m121_k131_n141_N_T => (121, 131, 141, N, T),
        tensor_dot_i32_m121_k131_n141_T_T => (121, 131, 141, T, T),
    );

    test_dot!(
        f32;
        tensor_dot_f32_m21_k31_n41_N_N => (21, 31, 41, N, N),
        tensor_dot_f32_m121_k131_n141_N_N => (121, 131, 141, N, N),
        tensor_dot_f32_m121_k131_n141_T_N => (121, 131, 141, T, N),
        tensor_dot_f32_m121_k131_n141_N_T => (121, 131, 141, N, T),
        tensor_dot_f32_m121_k131_n141_T_T => (121, 131, 141, T, T),
        tensor_dot_f32_m25_k611_n6_N_N => (25, 611, 6, N, N),
    );

    test_dot!(
        bf16;
        tensor_dot_bf16_m21_k31_n41_N_N => (21, 31, 41, N, N),
        tensor_dot_bf16_m121_k131_n141_N_N => (121, 131, 141, N, N),
        tensor_dot_bf16_m121_k131_n141_T_N => (121, 131, 141, T, N),
        tensor_dot_bf16_m121_k131_n141_N_T => (121, 131, 141, N, T),
        tensor_dot_bf16_m121_k131_n141_T_T => (121, 131, 141, T, T),
    );

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
            tensor_dot_f32_m25_k57600_n6_N_N => (25, 57600, 6, N, N),
        );
    }
}
