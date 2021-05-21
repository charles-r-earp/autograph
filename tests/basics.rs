use autograph::{
    backend::{Buffer, Device, Scalar},
    include_spirv,
    tensor::{ArcTensor, Dimension, Tensor, Tensor2},
    Result,
};
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};
use ndarray::Array;

use std::iter::once;

#[test]
fn device_list() {
    Device::list();
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct FillPushConstsU32 {
    x: u32,
    n: u32,
}

#[test]
fn compute_pass() -> Result<()> {
    let spirv = include_spirv!("../src/shaders/glsl/fill_u32.spv");

    for device in Device::list() {
        let n = 10;
        let mut y = Buffer::<u32>::zeros(&device, n)?;
        device
            .compute_pass(spirv.as_ref(), "main")?
            .buffer_slice_mut(y.as_buffer_slice_mut())?
            .push_constants(bytemuck::cast_slice(&[FillPushConstsU32 {
                x: 1,
                n: n as u32,
            }]))?
            .global_size([n as u32, 1, 1])
            .enqueue()?;
        let y = smol::block_on(y.to_vec()?)?;
        assert_eq!(y, vec![1u32; n]);
    }

    Ok(())
}

#[test]
fn tensor_zeros() -> Result<()> {
    for device in Device::list() {
        Tensor::<f32, _>::zeros(&device, [64, 1, 28, 28])?;
    }
    Ok(())
}

#[test]
fn tensor_from_shape_cow() -> Result<()> {
    for device in Device::list() {
        Tensor::<f32, _>::from_shape_cow(&device, [64, 1, 28, 28], vec![1.; 64 * 28 * 28])?;
        let x = vec![1., 2., 3., 4.];
        let y = Tensor::<f32, _>::from_shape_cow(&device, x.len(), x.as_slice())?;
        smol::block_on(device.synchronize()?)?;
        let y = smol::block_on(y.to_vec()?)?;
        assert_eq!(x, y);
    }
    Ok(())
}

fn tensor_from_array<D: Dimension>(x: Array<u32, D>) -> Result<()> {
    for device in Device::list() {
        let y = smol::block_on(Tensor::from_array(&device, x.view())?.to_array()?)?;
        assert_eq!(x, y);
        let y_t = smol::block_on(Tensor::from_array(&device, x.t())?.to_array()?)?;
        assert_eq!(x.t(), y_t.view());
    }
    Ok(())
}

#[test]
fn tensor_from_array0() -> Result<()> {
    tensor_from_array(Array::from_elem((), 1))
}

#[test]
fn tensor_from_array1() -> Result<()> {
    tensor_from_array(Array::from_shape_vec(3, (1..=3).into_iter().collect())?)
}

#[test]
fn tensor_from_array2() -> Result<()> {
    tensor_from_array(Array::from_shape_vec(
        [2, 3],
        (1..=6).into_iter().collect(),
    )?)
}

#[test]
fn tensor_from_array3() -> Result<()> {
    tensor_from_array(Array::from_shape_vec(
        [2, 3, 4],
        (1..=24).into_iter().collect(),
    )?)
}

#[test]
fn tensor_from_array4() -> Result<()> {
    tensor_from_array(Array::from_shape_vec(
        [2, 3, 4, 5],
        (1..=120).into_iter().collect(),
    )?)
}

#[test]
fn test_from_array5() -> Result<()> {
    tensor_from_array(Array::from_shape_vec(
        [2, 3, 4, 5, 6],
        (1..=120 * 6).into_iter().collect(),
    )?)
}

#[test]
fn tensor_from_array6() -> Result<()> {
    tensor_from_array(Array::from_shape_vec(
        [2, 3, 4, 5, 6, 7],
        (1..=120 * 6 * 7).into_iter().collect(),
    )?)
}

#[allow(non_snake_case)]
#[test]
fn tensor_from_arrayD() -> Result<()> {
    tensor_from_array(Array::from_shape_vec(
        [2, 3, 4, 5, 6, 7, 8].as_ref(),
        (1..=120 * 6 * 7 * 8).into_iter().collect(),
    )?)
}

fn tensor_from_elem<T: Scalar>(xs: &[T]) -> Result<()> {
    let n = 1200;
    for device in Device::list() {
        for x in xs.iter().copied() {
            let y = Tensor::from_elem(&device, n, x)?;
            let y = smol::block_on(y.to_vec()?)?;
            assert_eq!(y, vec![x; n]);
        }
    }

    Ok(())
}

#[test]
fn tensor_from_elem_u8() -> Result<()> {
    tensor_from_elem::<u8>(&[1, 33, 255])
}

#[test]
fn tensor_from_elem_i8() -> Result<()> {
    tensor_from_elem::<i8>(&[1, -33, 127])
}

#[test]
fn tensor_from_elem_u16() -> Result<()> {
    tensor_from_elem::<u16>(&[1, 33, 1000])
}

#[test]
fn tensor_from_elem_i16() -> Result<()> {
    tensor_from_elem::<i16>(&[1, -33, 1000])
}

#[test]
fn tensor_from_elem_f16() -> Result<()> {
    tensor_from_elem::<f16>(&[f16::from_f32(1.), f16::from_f32(-33.), f16::from_f32(1000.)])
}

#[test]
fn tensor_from_elem_bf16() -> Result<()> {
    tensor_from_elem::<bf16>(&[
        bf16::from_f32(1.),
        bf16::from_f32(-33.),
        bf16::from_f32(1000.),
    ])
}

#[test]
fn tensor_from_elem_u32() -> Result<()> {
    tensor_from_elem::<u32>(&[1, 33, 1000])
}

#[test]
fn tensor_from_elem_i32() -> Result<()> {
    tensor_from_elem::<i32>(&[1, -33, 1000])
}

#[test]
fn tensor_from_elem_f32() -> Result<()> {
    tensor_from_elem::<f32>(&[1., -33., 0.1, 1000.])
}

#[test]
fn tensor_from_elem_u64() -> Result<()> {
    tensor_from_elem::<u64>(&[1, 33, 1000])
}

#[test]
fn tensor_from_elem_i64() -> Result<()> {
    tensor_from_elem::<i64>(&[1, -33, 1000])
}

#[test]
fn tensor_from_elem_f64() -> Result<()> {
    tensor_from_elem::<f64>(&[1., -33., 0.1, 1000.])
}

#[test]
fn tensor_copy_from_buffer_slice() -> Result<()> {
    for device in Device::list() {
        let x = Tensor::from_shape_cow(&device, 4, vec![1, 2, 3, 4])?;
        let mut y = Tensor::zeros(&device, 4)?;
        y.as_buffer_slice_mut()
            .unwrap()
            .copy_from_buffer_slice(x.as_buffer_slice().unwrap())?;
        let y = smol::block_on(y.to_vec()?)?;
        assert_eq!(y, vec![1, 2, 3, 4]);
    }
    Ok(())
}

#[test]
fn tensor_view_to_tensor() -> Result<()> {
    for device in Device::list().into_iter().chain(once(Device::new_cpu())) {
        let x = Tensor::from_shape_cow(&device, 4, vec![1, 2, 3, 4])?;
        let y = x.view().to_tensor()?;
        let y = smol::block_on(y.to_vec()?)?;
        assert_eq!(y, vec![1, 2, 3, 4]);
    }
    Ok(())
}

#[test]
fn arc_tensor_into_arc_tensor() -> Result<()> {
    for device in Device::list().into_iter().chain(once(Device::new_cpu())) {
        let x = ArcTensor::from_shape_cow(&device, 4, vec![1, 2, 3, 4])?;
        let y = x.into_arc_tensor()?;
        let y = smol::block_on(y.to_vec()?)?;
        assert_eq!(y, vec![1, 2, 3, 4]);
    }
    Ok(())
}

#[test]
fn tensor_serde() -> Result<()> {
    let shape = (100, 64);
    let vec = (0..(shape.0 * shape.1) as u32).into_iter().collect();
    let array = Array::from_shape_vec(shape, vec)?;
    for device in Device::list_gpus()
        .into_iter()
        .chain(once(Device::new_cpu()))
    {
        let tensor = Tensor2::<u32>::from_array(&device, array.view())?;
        let data = bincode::serialize(&tensor)?;
        let tensor: Tensor2<u32> = bincode::deserialize(&data)?;
        let array_out = smol::block_on(tensor.to_array()?)?;
        assert_eq!(array_out, array);
    }
    Ok(())
}
