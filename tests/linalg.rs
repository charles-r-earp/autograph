use approx::assert_ulps_eq;
use autograph::backend::Device;
use autograph::tensor::{Dot, Num, Tensor, Tensor2};
use autograph::Result;
use ndarray::{linalg::Dot as ArrayDot, Array2};

fn tensor_dot<T: Num>(x1_array: Array2<T>, x2_array: Array2<T>, f: fn(&[T], &[T])) -> Result<()>
where
    Array2<T>: ArrayDot<Array2<T>, Output = Array2<T>>,
    Tensor2<T>: Dot<Tensor2<T>, Output = Tensor2<T>>,
{
    let y_vec = x1_array.dot(&x2_array).as_slice().unwrap().to_vec();

    for device in Device::list() {
        let x1_tensor = Tensor::from_array(&device, x1_array.view())?;
        let x2_tensor = Tensor::from_array(&device, x2_array.view())?;
        let y_tensor = x1_tensor.dot(&x2_tensor)?;
        let y_vec_out = smol::block_on(y_tensor.to_vec()?)?;
        f(y_vec_out.as_slice(), y_vec.as_slice());
    }

    Ok(())
}

fn check_ulps_eq(a: &[f32], b: &[f32]) {
    assert_ulps_eq!(a, b);
}

fn check_eq<T: Num>(a: &[T], b: &[T]) {
    assert_eq!(a, b);
}

#[test]
fn test_dot_f32() -> Result<()> {
    tensor_dot::<f32>(
        ndarray::arr2(&[[1., 2.], [3., 4.]]),
        ndarray::arr2(&[[5., 6.], [7., 8.]]),
        check_ulps_eq,
    )?;
    Ok(())
}

#[test]
fn test_dot_u32() -> Result<()> {
    tensor_dot::<u32>(
        ndarray::arr2(&[[1, 2], [3, 4]]),
        ndarray::arr2(&[[5, 6], [7, 8]]),
        check_eq,
    )?;
    Ok(())
}

#[test]
fn test_dot_i32() -> Result<()> {
    tensor_dot::<i32>(
        ndarray::arr2(&[[1, 2], [3, 4]]),
        ndarray::arr2(&[[5, 6], [7, 8]]),
        check_eq,
    )?;
    Ok(())
}
