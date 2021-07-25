use super::*;
use crate::ops::{Dot, DotAccumulate};
use bytemuck::{Pod, Zeroable};

#[derive(Copy)]
#[repr(C, packed)]
struct GemmPushConsts<T> {
    alpha: T,
    beta: T,
    a0: T, // ie relu negative slope
    m: u32,
    k: u32,
    n: u32,
    rsa: i32,
    csa: i32,
    rsb: i32,
    csb: i32,
    rsc: i32,
    csc: i32,
}

impl<T: Copy> Clone for GemmPushConsts<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T: Zeroable> Zeroable for GemmPushConsts<T> {}

unsafe impl<T: Pod> Pod for GemmPushConsts<T> {}

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn gemm_impl<T: Scalar>(
    a: TensorView2<T>,
    b: TensorView2<T>,
    bias: Option<TensorView1<T>>,
    acc: bool,
    mut c: TensorViewMut2<T>,
) -> Result<()> {
    use ScalarType::*;
    if !matches!(T::scalar_type(), U32 | I32 | F32 | BF16) {
        bail!("{} dot not implemented!", elem_type_name::<T>());
    }

    // Patch for custom 16 bit ops
    // If 16 bit is supported then this is unnecessary
    if size_eq::<T, u16>() && acc {
        todo!(); // c.fill(T::zero())?;
    }

    let name = format!(
        "gemm{}_{}",
        if bias.is_some() { "_bias_" } else { "" },
        elem_type_name::<T>(),
    );

    let module = crate::glsl_shaders::module(name)?;

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

    let [rsa, csa]: [isize; 2] = a.strides().try_into().unwrap();
    let [rsa, csa] = [rsa as i32, csa as i32];

    let [rsb, csb]: [isize; 2] = b.strides().try_into().unwrap();
    let [rsb, csb] = [rsb as i32, csb as i32];

    let [rsc, csc]: [isize; 2] = c.strides().try_into().unwrap();
    let [rsc, csc] = [rsc as i32, csc as i32];

    let one = match T::scalar_type() {
        F32 | BF16 => u32::from_le_bytes(1f32.to_le_bytes()),
        _ => 1u32,
    };
    let alpha = one;
    let beta = if acc { one } else { 0u32 };
    let a0 = 0u32;

    let push_consts = GemmPushConsts {
        alpha,
        beta,
        a0,
        m,
        k,
        n,
        rsa,
        csa,
        rsb,
        csb,
        rsc,
        csc,
    };

    let builder = module
        .compute_pass("main")?
        .slice(a.as_raw_slice())?
        .slice(b.as_raw_slice())?;
    let builder = if let Some(bias) = bias.as_ref() {
        builder.slice(bias.as_raw_slice())?
    } else {
        builder
    };
    let builder = builder.slice_mut(c.as_raw_slice_mut())?.push(push_consts)?;
    unsafe { builder.submit([m, n, 1]) }
}

impl<'b, T: Scalar, S: Data<Elem = T>> Dot<TensorView2<'b, T>> for TensorBase<S, Ix2> {
    type Output = Tensor2<T>;
    type Bias = TensorView1<'b, T>;
    fn dot_bias(self, rhs: TensorView2<'b, T>, bias: Option<Self::Bias>) -> Result<Self::Output> {
        let mut output = unsafe { Tensor::alloc(self.device(), [self.dim().0, rhs.dim().1])? };
        self.dot_bias_acc(rhs, bias, &mut output)?;
        Ok(output)
    }
}

impl<'b, T: Scalar, S: Data<Elem = T>> DotAccumulate<TensorView2<'b, T>> for TensorBase<S, Ix2> {
    fn dot_bias_acc(
        self,
        rhs: TensorView2<'b, T>,
        bias: Option<Self::Bias>,
        output: &mut Self::Output,
    ) -> Result<()> {
        gemm_impl(self.view(), rhs, bias, false, output.view_mut())
    }
}

#[cfg(test)]
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
            let t_out = t1.dot(t2)?;
            let a_out = t_out.read().await?.into_array();
            (a_true, a_out)
        }};
    }

    macro_rules! check_arrays {
        (f32 => ($a:expr, $b:expr)) => {
            assert_relative_eq!($a, $b, max_relative = 0.000_1);
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
                        for _ in 0 .. 10 {
                            a.view().dot(b.view()).unwrap();
                            smol::block_on(device.sync()).unwrap();
                        }
                        bencher.iter(|| {
                            a.view().dot(b.view()).unwrap();
                            smol::block_on(device.sync()).unwrap();
                        });
                    }
                )+
            );
        }

        bench_dot!(
            f32;
            tensor_dot_f32_m32_k32_n32_N_N => (32, 32, 32, N, N),
            tensor_dot_f32_m100_k100_n100_N_N => (100, 100, 100, N, N),
        );

        bench_dot!(
            bf16;
            tensor_dot_bf16_m32_k32_n32_N_N => (32, 32, 32, N, N),
            tensor_dot_bf16_m100_k100_n100_N_N => (100, 100, 100, N, N),
        );
    }
}
