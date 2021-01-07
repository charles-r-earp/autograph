#![allow(warnings)]
use approx::assert_relative_eq;
use autograph::backend::Device;
use autograph::tensor::{Dot, Num, Tensor, Tensor2, TensorView2};
use autograph::Result;
use half::bf16;
use ndarray::{linalg::Dot as ArrayDot, Array, Array2, ArrayView2, LinalgScalar};
use std::any::TypeId;
use std::fmt::Debug;

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
enum Transpose {
    N,
    T,
}

fn gen_array<T: Num>(dim: [usize; 2]) -> Array2<T> {
    let n = dim[0] * dim[1];
    let mut vec: Vec<T> = (1..=n)
        .into_iter()
        .map(|x| T::from_usize((x + 100) % 100).unwrap())
        .collect();
    Array2::from_shape_vec(dim, vec).unwrap()
}

macro_rules! tensor_dot {
    ($t1:ty, $t2:ty, $device:expr, $args:expr) => {{
        let device = $device;
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
        let t1 = Tensor2::<$t2>::from_array(device, gen_array(dim1))?;
        let (a1, t1) = match a_t {
            Transpose::N => (a1.view(), t1.view()),
            Transpose::T => (a1.t(), t1.t()),
        };
        let a2 = gen_array::<$t1>(dim2);
        let t2 = Tensor2::<$t2>::from_array(device, gen_array(dim2))?;
        let (a2, t2) = match b_t {
            Transpose::N => (a2.view(), t2.view()),
            Transpose::T => (a2.t(), t2.t()),
        };
        let a_true = a1.dot(&a2);
        let a_out = smol::block_on(t1.dot(&t2)?.to_array()?)?;
        (a_true, a_out)
    }};
}

macro_rules! check_arrays {
    (f32 => ($a:expr, $b:expr)) => {
        assert_relative_eq!($a, $b, max_relative = 0.000_1);
    };
    (bf16 => ($a:expr, $b:expr)) => {
        let b = $b.map(|x| x.to_f32());
        assert_relative_eq!($a, b, max_relative = 0.01);
    };
    ($t:tt => ($a:expr, $b:expr)) => {
        assert_eq!($a, $b);
    };
}

macro_rules! test_dot {
    (ignore bf16; $($name:ident => $args:expr,)+) => (
        $(
            #[ignore]
            #[test]
            fn $name () -> Result<()> {
                for device in Device::list() {
                    let (a_true, a_out) = tensor_dot! { f32, bf16, &device, $args };
                    check_arrays!(bf16 => (a_true, a_out));
                }
                Ok(())
            }
        )+
    );
    (ignore $t:tt; $($name:ident => $args:expr,)+) => (
        $(
            #[ignore]
            #[test]
            fn $name () -> Result<()> {
                for device in Device::list() {
                    let (a_true, a_out) = tensor_dot! { $t, $t, &device, $args };
                    check_arrays!($t => (a_true, a_out));
                }
                Ok(())
            }
        )+
    );
    ($t:tt; $($name:ident => $args:expr,)+) => (
        $(
            #[test]
            fn $name () -> Result<()> {
                for device in Device::list() {
                    let (a_true, a_out) = tensor_dot! { $t, $t, &device, $args };
                    check_arrays!($t => (a_true, a_out));
                }
                Ok(())
            }
        )+
    );
}

use Transpose::*;

// Note: ndarray is slow for dots not f32 / f64, so we can't test large sizes
// 16 and 64 bit types are not supported on all platforms, so ignored

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
    ignore bf16;
    tensor_dot_bf16_m21_k31_n41_N_N => (21, 31, 41, N, N),
    tensor_dot_bf16_m121_k131_n141_N_N => (121, 131, 141, N, N),
    tensor_dot_bf16_m121_k131_n141_T_N => (121, 131, 141, T, N),
    tensor_dot_bf16_m121_k131_n141_N_T => (121, 131, 141, N, T),
    tensor_dot_bf16_m121_k131_n141_T_T => (121, 131, 141, T, T),
);

test_dot!(
    ignore u64;
    tensor_dot_u64_m21_k31_n41_N_N => (21, 31, 41, N, N),
    tensor_dot_u64_m1021_k1031_n1041_N_N => (121, 131, 141, N, N),
    tensor_dot_u64_m1021_k1031_n1041_T_N => (121, 131, 141, T, N),
    tensor_dot_u64_m1021_k1031_n1041_N_T => (121, 131, 141, N, T),
    tensor_dot_u64_m1021_k1031_n1041_T_T => (121, 131, 141, T, T),
);

test_dot!(
    ignore i64;
    tensor_dot_i64_m21_k31_n41_N_N => (21, 31, 41, N, N),
    tensor_dot_i64_m1021_k1031_n1041_N_N => (121, 131, 141, N, N),
    tensor_dot_i64_m1021_k1031_n1041_T_N => (121, 131, 141, T, N),
    tensor_dot_i64_m1021_k1031_n1041_N_T => (121, 131, 141, N, T),
    tensor_dot_i64_m1021_k1031_n1041_T_T => (121, 131, 141, T, T),
);

test_dot!(
    ignore u64;
    tensor_dot_f64_m21_k31_n41_N_N => (21, 31, 41, N, N),
    tensor_dot_f64_m1021_k1031_n1041_N_N => (121, 131, 141, N, N),
    tensor_dot_f64_m1021_k1031_n1041_T_N => (121, 131, 141, T, N),
    tensor_dot_f64_m1021_k1031_n1041_N_T => (121, 131, 141, N, T),
    tensor_dot_f64_m1021_k1031_n1041_T_T => (121, 131, 141, T, T),
);
