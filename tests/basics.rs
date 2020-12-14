use autograph::backend::{Device, Buffer};
use autograph::tensor::Tensor;
use autograph::{Result, include_spirv};
use bytemuck::{Zeroable, Pod};

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

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct FillU32PushConsts {
    n: u32,
    x: u32
}


#[test]
fn compute_pass_fill_u32_gpu() -> Result<()> {
    let spirv = include_spirv!(env!("glsl::fill::fill_u32"));
    
    for gpu in Device::list_gpus() {
        let n = 10;
        let mut y = Buffer::<u32>::zeros(&gpu, n)?;
        gpu.compute_pass(spirv.as_ref(), "main")?
            .buffer_slice_mut(&y.as_buffer_slice_mut())?
            .push_constants(FillU32PushConsts { n: n as u32, x: 1 })?
            .work_groups(|_| [1, 1, 1])
            .enqueue()?;
        let y = smol::block_on(y.to_vec()?)?; 
        assert_eq!(y, vec![1u32; n]);
    }
    
    Ok(())
}

