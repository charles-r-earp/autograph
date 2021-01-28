use approx::assert_relative_eq;
use autograph::{
    backend::Device,
    tensor::{Axis, Dimension, IntoDimension, Num, RemoveAxis, Tensor},
    Result,
};
use half::bf16;
use ndarray::{Array, ArrayView};
use num_traits::Bounded;

fn array_argmin<T: PartialOrd + Copy + Bounded, D: RemoveAxis>(
    input: &ArrayView<T, D>,
    axis: Axis,
) -> Array<u32, D::Smaller> {
    let mut indices = Array::zeros(input.raw_dim().remove_axis(axis));
    let mut values = Array::from_elem(input.raw_dim().remove_axis(axis), T::max_value());
    for (u, x) in input.axis_iter(axis).enumerate() {
        for (x, (i, v)) in x.iter().zip(indices.iter_mut().zip(values.iter_mut())) {
            if x < v {
                *i = u as u32;
                *v = *x;
            }
        }
    }
    indices
}

fn array_argmax<T: PartialOrd + Copy + Bounded, D: RemoveAxis>(
    input: &ArrayView<T, D>,
    axis: Axis,
) -> Array<u32, D::Smaller> {
    let mut indices = Array::zeros(input.raw_dim().remove_axis(axis));
    let mut values = Array::from_elem(input.raw_dim().remove_axis(axis), T::min_value());
    for (u, x) in input.axis_iter(axis).enumerate() {
        for (x, (i, v)) in x.iter().zip(indices.iter_mut().zip(values.iter_mut())) {
            if x > v {
                *i = u as u32;
                *v = *x;
            }
        }
    }
    indices
}

fn gen_array<T: From<u8>, D: Dimension>(dim: D) -> Result<Array<T, D>> {
    let v1 = (1..=100)
        .cycle()
        .step_by(21)
        .take(dim.size())
        .map(T::from)
        .collect();
    Ok(Array::from_shape_vec(dim, v1)?)
}

fn tensor_sum<T: Num + From<u8> + num_traits::Zero, D: IntoDimension>(
    dim: D,
    axis: Axis,
) -> Result<()>
where
    D::Dim: RemoveAxis,
{
    let a1 = gen_array::<T, _>(dim.into_dimension())?;
    let y_true = a1.sum_axis(axis);
    for device in Device::list() {
        let t1 = Tensor::from_array(&device, a1.view())?;
        let t2 = t1.sum(axis)?;
        let y = smol::block_on(t2.to_array()?)?;
        assert_eq!(&y, &y_true);
    }
    Ok(())
}

fn tensor_sum_bf16<D: IntoDimension>(dim: D, axis: Axis) -> Result<()>
where
    D::Dim: RemoveAxis,
{
    let dim = dim.into_dimension();
    let a1_bf16 = gen_array::<bf16, _>(dim.clone())?;
    let a1_f32 = gen_array::<f32, _>(dim)?;
    let y_true = a1_f32.sum_axis(axis);
    for device in Device::list() {
        let t1 = Tensor::from_array(&device, a1_bf16.view())?;
        let t2 = t1.sum(axis)?;
        let y = smol::block_on(t2.to_array()?)?.map(|x| x.to_f32());
        assert_relative_eq!(&y, &y_true, epsilon = 0.01, max_relative = 0.01);
    }
    Ok(())
}

fn tensor_argmin<T: Num + From<u8> + PartialOrd + Bounded, D: IntoDimension>(
    dim: D,
    axis: Axis,
) -> Result<()>
where
    D::Dim: RemoveAxis,
{
    let a1 = gen_array::<T, _>(dim.into_dimension())?;
    let y_true = array_argmin(&a1.view(), axis);
    for device in Device::list() {
        let t1 = Tensor::from_array(&device, a1.view())?;
        let t2 = t1.argmin(axis)?;
        let y = smol::block_on(t2.to_array()?)?;
        assert_eq!(&y, &y_true);
    }
    Ok(())
}

fn tensor_argmin_bf16<D: IntoDimension>(dim: D, axis: Axis) -> Result<()>
where
    D::Dim: RemoveAxis,
{
    let dim = dim.into_dimension();
    let a1_bf16 = gen_array::<bf16, _>(dim.clone())?;
    let a1_f32 = gen_array::<f32, _>(dim)?;
    let y_true = array_argmin(&a1_f32.view(), axis);
    for device in Device::list() {
        let t1 = Tensor::from_array(&device, a1_bf16.view())?;
        let t2 = t1.argmin(axis)?;
        let y = smol::block_on(t2.to_array()?)?;
        assert_eq!(&y, &y_true);
    }
    Ok(())
}

fn tensor_argmax<T: Num + From<u8> + PartialOrd + Bounded, D: IntoDimension>(
    dim: D,
    axis: Axis,
) -> Result<()>
where
    D::Dim: RemoveAxis,
{
    let a1 = gen_array::<T, _>(dim.into_dimension())?;
    let y_true = array_argmax(&a1.view(), axis);
    for device in Device::list() {
        let t1 = Tensor::from_array(&device, a1.view())?;
        let t2 = t1.argmax(axis)?;
        let y = smol::block_on(t2.to_array()?)?;
        assert_eq!(&y, &y_true);
    }
    Ok(())
}

fn tensor_argmax_bf16<D: IntoDimension>(dim: D, axis: Axis) -> Result<()>
where
    D::Dim: RemoveAxis,
{
    let dim = dim.into_dimension();
    let a1_bf16 = gen_array::<bf16, _>(dim.clone())?;
    let a1_f32 = gen_array::<f32, _>(dim)?;
    let y_true = array_argmax(&a1_f32.view(), axis);
    for device in Device::list() {
        let t1 = Tensor::from_array(&device, a1_bf16.view())?;
        let t2 = t1.argmax(axis)?;
        let y = smol::block_on(t2.to_array()?)?;
        assert_eq!(&y, &y_true);
    }
    Ok(())
}

#[test]
fn tensor_sum_bf16_11x12_axis0() -> Result<()> {
    tensor_sum_bf16([11, 12], Axis(0))
}

#[test]
fn tensor_sum_bf16_22x23_axis1() -> Result<()> {
    tensor_sum_bf16([22, 23], Axis(1))
}

#[test]
fn tensor_sum_u32_11x12_axis0() -> Result<()> {
    tensor_sum::<u32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_sum_u32_22x23_axis1() -> Result<()> {
    tensor_sum::<u32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_sum_i32_11x12_axis0() -> Result<()> {
    tensor_sum::<i32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_sum_i32_22x23_axis1() -> Result<()> {
    tensor_sum::<i32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_sum_f32_11x12_axis0() -> Result<()> {
    tensor_sum::<f32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_sum_f32_22x23_axis1() -> Result<()> {
    tensor_sum::<f32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_argmin_bf16_11x12_axis0() -> Result<()> {
    tensor_argmin_bf16([11, 12], Axis(0))
}

#[test]
fn tensor_argmin_bf16_22x23_axis1() -> Result<()> {
    tensor_argmin_bf16([22, 23], Axis(1))
}

#[test]
fn tensor_argmin_u32_11x12_axis0() -> Result<()> {
    tensor_argmin::<u32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_argmin_u32_22x23_axis1() -> Result<()> {
    tensor_argmin::<u32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_argmin_i32_11x12_axis0() -> Result<()> {
    tensor_argmin::<i32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_argmin_i32_22x23_axis1() -> Result<()> {
    tensor_argmin::<i32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_argmin_f32_11x12_axis0() -> Result<()> {
    tensor_argmin::<f32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_argmin_f32_22x23_axis1() -> Result<()> {
    tensor_argmin::<f32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_argmax_bf16_11x12_axis0() -> Result<()> {
    tensor_argmax_bf16([11, 12], Axis(0))
}

#[test]
fn tensor_argmax_bf16_22x23_axis1() -> Result<()> {
    tensor_argmax_bf16([22, 23], Axis(1))
}

#[test]
fn tensor_argmax_u32_11x12_axis0() -> Result<()> {
    tensor_argmax::<u32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_argmax_u32_22x23_axis1() -> Result<()> {
    tensor_argmax::<u32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_argmax_i32_11x12_axis0() -> Result<()> {
    tensor_argmax::<i32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_argmax_i32_22x23_axis1() -> Result<()> {
    tensor_argmax::<i32, _>([22, 23], Axis(1))
}

#[test]
fn tensor_argmax_f32_11x12_axis0() -> Result<()> {
    tensor_argmax::<f32, _>([11, 12], Axis(0))
}

#[test]
fn tensor_argmax_f32_22x23_axis1() -> Result<()> {
    tensor_argmax::<f32, _>([22, 23], Axis(1))
}
