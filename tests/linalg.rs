use approx::assert_ulps_eq;
use autograph::backend::Device;
use autograph::tensor::Tensor;
use autograph::Result;
use ndarray::Array2;

#[test]
fn tensor_dot_f32() -> Result<()> {
    let x1_array = Array2::from_shape_vec([2, 2], vec![2., 3., 4., 5.])?;
    let x2_array = Array2::from_shape_vec([2, 2], vec![6., 7., 8., 9.])?;
    let y_vec = x1_array.dot(&x2_array.t()).as_slice().unwrap().to_vec();

    for device in Device::list() {
        let x1_tensor = Tensor::<f32, _>::from_array(&device, x1_array.view())?;
        let x2_tensor = Tensor::from_array(&device, x2_array.view())?;
        let y_tensor = x1_tensor.dot(&x2_tensor.t())?;
        let y_vec_out = smol::block_on(y_tensor.to_vec()?)?;
        assert_ulps_eq!(y_vec_out.as_slice(), y_vec.as_slice());
    }

    Ok(())
}

#[test]
fn tensor_dot_f64() -> Result<()> {
    let x1_array = Array2::from_shape_vec([2, 2], vec![2., 3., 4., 5.])?;
    let x2_array = Array2::from_shape_vec([2, 2], vec![6., 7., 8., 9.])?;
    let y_vec = x1_array.dot(&x2_array.t()).as_slice().unwrap().to_vec();

    for device in Device::list() {
        let x1_tensor = Tensor::<f64, _>::from_array(&device, x1_array.view())?;
        let x2_tensor = Tensor::from_array(&device, x2_array.view())?;
        let y_tensor = x1_tensor.dot(&x2_tensor.t())?;
        let y_vec_out = smol::block_on(y_tensor.to_vec()?)?;
        assert_ulps_eq!(y_vec_out.as_slice(), y_vec.as_slice());
    }

    Ok(())
}

#[test]
fn tensor_dot_i32() -> Result<()> {
    let x1_array = Array2::from_shape_vec([2, 2], vec![2, 3, 4, 5])?;
    let x2_array = Array2::from_shape_vec([2, 2], vec![6, 7, 8, 9])?;
    let y_vec = x1_array.dot(&x2_array.t()).as_slice().unwrap().to_vec();

    for device in Device::list() {
        let x1_tensor = Tensor::<i32, _>::from_array(&device, x1_array.view())?;
        let x2_tensor = Tensor::from_array(&device, x2_array.view())?;
        let y_tensor = x1_tensor.dot(&x2_tensor.t())?;
        let y_vec_out = smol::block_on(y_tensor.to_vec()?)?;
        assert_eq!(y_vec_out.as_slice(), y_vec.as_slice());
    }

    Ok(())
}
