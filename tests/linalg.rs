use approx::{assert_ulps_eq, AbsDiffEq, UlpsEq};
use autograph::backend::Device;
use autograph::tensor::{Dot, Num, Tensor, Tensor2};
use autograph::Result;
use half::bf16;
use ndarray::{linalg::Dot as ArrayDot, Array2};
use std::fmt::Debug;

fn tensor_dot_half<T: Num>(x1_array: Array2<f32>, x2_array: Array2<f32>) -> Result<()>
where
    Tensor2<T>: Dot<Tensor2<T>, Output = Tensor2<T>>,
{
    let y_vec = x1_array.dot(&x2_array).as_slice().unwrap().to_vec();
    let x1_array = x1_array.map(|&x| T::from_f32(x).unwrap());
    let x2_array = x2_array.map(|&x| T::from_f32(x).unwrap());
    tensor_dot_impl(x1_array, x2_array, move |y_vec_out| {
        let y_vec_out: Vec<f32> = y_vec_out.into_iter().map(|x| x.to_f32().unwrap()).collect();
        assert_ulps_eq!(y_vec_out.as_slice(), y_vec.as_slice());
    })
}

fn tensor_dot<T: Num>(x1_array: Array2<T>, x2_array: Array2<T>, f: fn(&[T], &[T])) -> Result<()>
where
    Array2<T>: ArrayDot<Array2<T>, Output = Array2<T>>,
    Tensor2<T>: Dot<Tensor2<T>, Output = Tensor2<T>>,
{
    let y_vec = x1_array.dot(&x2_array).as_slice().unwrap().to_vec();
    tensor_dot_impl(x1_array, x2_array, move |y_vec_out| {
        f(&y_vec_out, &y_vec);
    })
}

fn tensor_dot_impl<T: Num>(
    x1_array: Array2<T>,
    x2_array: Array2<T>,
    f: impl Fn(Vec<T>),
) -> Result<()>
where
    Tensor2<T>: Dot<Tensor2<T>, Output = Tensor2<T>>,
{
    for device in Device::list() {
        let x1_tensor = Tensor::from_array(&device, x1_array.view())?;
        let x2_tensor = Tensor::from_array(&device, x2_array.view())?;
        let y_tensor = x1_tensor.dot(&x2_tensor)?;
        let y_vec_out = smol::block_on(y_tensor.to_vec()?)?;
        f(y_vec_out);
    }

    Ok(())
}

fn check_ulps_eq<T: UlpsEq + Debug>(a: &[T], b: &[T])
where
    <T as AbsDiffEq>::Epsilon: Clone,
{
    assert_ulps_eq!(a, b);
}

fn check_eq<T: Num>(a: &[T], b: &[T]) {
    assert_eq!(a, b);
}

#[test]
fn tensor_dot_bf16() -> Result<()> {
    tensor_dot_half::<bf16>(
        ndarray::arr2(&[[1., 2.], [3., 4.]]),
        ndarray::arr2(&[[5., 6.], [7., 8.]]),
    )?;
    Ok(())
}

#[test]
fn tensor_dot_u32() -> Result<()> {
    tensor_dot::<u32>(
        ndarray::arr2(&[[1, 2], [3, 4]]),
        ndarray::arr2(&[[5, 6], [7, 8]]),
        check_eq,
    )?;
    Ok(())
}

#[test]
fn tensor_dot_i32() -> Result<()> {
    tensor_dot::<i32>(
        ndarray::arr2(&[[1, 2], [3, 4]]),
        ndarray::arr2(&[[5, 6], [7, 8]]),
        check_eq,
    )?;
    Ok(())
}

#[test]
fn tensor_dot_f32() -> Result<()> {
    tensor_dot::<f32>(
        ndarray::arr2(&[[1., 2.], [3., 4.]]),
        ndarray::arr2(&[[5., 6.], [7., 8.]]),
        check_ulps_eq,
    )?;
    Ok(())
}

#[test]
fn tensor_dot_u64() -> Result<()> {
    tensor_dot::<u64>(
        ndarray::arr2(&[[1, 2], [3, 4]]),
        ndarray::arr2(&[[5, 6], [7, 8]]),
        check_eq,
    )?;
    Ok(())
}

#[test]
fn tensor_dot_i64() -> Result<()> {
    tensor_dot::<i64>(
        ndarray::arr2(&[[1, 2], [3, 4]]),
        ndarray::arr2(&[[5, 6], [7, 8]]),
        check_eq,
    )?;
    Ok(())
}

#[test]
fn tensor_dot_f64() -> Result<()> {
    tensor_dot::<f64>(
        ndarray::arr2(&[[1., 2.], [3., 4.]]),
        ndarray::arr2(&[[5., 6.], [7., 8.]]),
        check_ulps_eq,
    )?;
    Ok(())
}
