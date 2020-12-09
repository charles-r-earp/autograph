use autograph::backend::Device;
use autograph::tensor::Tensor;
use autograph::Result;

#[test]
fn device_new_gpu() -> Result<()> {
    let mut i = 0;
    while let Some(gpu) = Device::new_gpu(i) {
        smol::block_on(gpu)?;
        i += 1;
    }
    Ok(())
}

#[test]
fn device_list_gpus() {
    Device::list_gpus();
}

#[test]
fn tensor_zeros_gpu() -> Result<()> {
    for gpu in Device::list_gpus() {
        Tensor::<f32, _>::zeros(&gpu, [64, 1, 28, 28])?;
    }
    Ok(())
}

#[test]
fn tensor_from_shape_cow_gpu() -> Result<()> {
    for gpu in Device::list_gpus() {
        Tensor::<f32, _>::from_shape_cow(&gpu, [64, 1, 28, 28], vec![1.; 64 * 1 * 28 * 28])?;
        let x = vec![1., 2., 3., 4.];
        let y = Tensor::<f32, _>::from_shape_cow(&gpu, x.len(), x.as_slice())?;
        let y = smol::block_on(y.to_vec()?)?;
        assert_eq!(x, y);
    }
    Ok(())
}
