use super::*;
use anyhow::bail;
use ndarray::{Axis, Dimension, RemoveAxis};

#[allow(unused)]
#[derive(Debug, Copy, Clone)]
enum Reduction {
    Sum,
    Mean,
    Min,
    Max,
    Argmin,
    Argmax,
}

impl Reduction {
    fn as_str(&self) -> &'static str {
        use Reduction::*;
        match self {
            Sum => "sum",
            Mean => "mean",
            Min => "min",
            Max => "max",
            Argmin => "argmin",
            Argmax => "argmax",
        }
    }
}

fn reduce<T1, T2, D>(
    input: &TensorView<T1, D>,
    output: &mut TensorViewMut<T2, D::Smaller>,
    axis: Axis,
    reduction: Reduction,
    accumulate: bool,
) -> Result<()>
where
    T1: Scalar,
    T2: Scalar,
    D: Dimension,
{
    if size_eq::<T2, u16>() && !accumulate {
        output.as_raw_slice_mut().fill(T2::zero())?;
    }
    let name = format!("reduce_{}_final_{}", reduction.as_str(), T1::scalar_name());
    let module = glsl_shaders::module(name)?;
    let batch_size = output.len() as u32;
    let stride_x = if axis.0 > 0 {
        input.strides()[axis.0 - 1] as u32
    } else {
        1
    };
    let n = input.shape()[axis.0] as u32;
    let stride_y = input.strides()[axis.0] as u32;
    let accumulate = accumulate as u32;
    let builder = module
        .compute_pass("main")?
        .slice(input.as_raw_slice())?
        .slice_mut(output.as_raw_slice_mut())?
        .push([batch_size, stride_x, n, stride_y, accumulate])?;
    unsafe { builder.submit([batch_size, 1, 1]) }
}

/// Reductions
#[allow(unused)]
impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub(crate) fn sum(&self) -> Result<Tensor0<T>> {
        self.view().into_shape(self.len())?.sum_axis(Axis(0))
    }
    /// Computes the sum along the given axis
    pub(crate) fn sum_axis(&self, axis: Axis) -> Result<Tensor<T, D::Smaller>>
    where
        D: RemoveAxis,
    {
        if axis.0 >= self.shape().len() {
            bail!("Axis {:?} out of range for shape {:?}!", axis, self.shape());
        }
        let mut output =
            unsafe { Tensor::<T, D::Smaller>::alloc(self.device(), self.dim.remove_axis(axis))? };
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Sum,
            false,
        )?;
        Ok(output)
    }
    pub(crate) fn sum_with(&self, output: &mut TensorViewMut0<T>) -> Result<()> {
        self.view()
            .into_shape(self.len())?
            .sum_axis_with(Axis(0), output)
    }
    pub(crate) fn sum_axis_with(
        &self,
        axis: Axis,
        output: &mut TensorViewMut<T, D::Smaller>,
    ) -> Result<()>
    where
        D: RemoveAxis,
    {
        if axis.0 >= self.shape().len() {
            bail!("Axis {:?} out of range for shape {:?}!", axis, self.shape());
        }
        let output_dim = self.dim.remove_axis(axis);
        if output.raw_dim() != output_dim {
            bail!(
                "Output dim {:?} != input dim remove_axis {:?}!",
                output.shape(),
                output_dim.slice()
            );
        }
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Sum,
            true,
        )?;
        Ok(())
    }
    /// Computes the min value along the given axis
    pub(crate) fn min_axis(&self, axis: Axis) -> Result<Tensor<T, D::Smaller>>
    where
        D: RemoveAxis,
    {
        if axis.0 >= self.shape().len() {
            bail!("Axis {:?} out of range for shape {:?}!", axis, self.shape());
        }
        let mut output =
            unsafe { Tensor::<T, D::Smaller>::alloc(self.device(), self.dim.remove_axis(axis))? };
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Min,
            false,
        )?;
        Ok(output)
    }
    /// Computes the index of the min value along the given axis.
    ///
    /// For multiple min values, the first will be selected. NaN values are ignored, returns 0 if all values are NaN.
    pub(crate) fn argmin_axis<U: Uint>(&self, axis: Axis) -> Result<Tensor<U, D::Smaller>>
    where
        D: RemoveAxis,
    {
        if axis.0 >= self.shape().len() {
            bail!("Axis {:?} out of range for shape {:?}!", axis, self.shape());
        }
        let mut output =
            unsafe { Tensor::<u32, D::Smaller>::alloc(self.device(), self.dim.remove_axis(axis))? };
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Argmin,
            false,
        )?;
        output.cast_into()
    }
    /// Computes the index of the max value along the given axis.
    ///
    /// For multiple max values, the first will be selected. NaN values are ignored, returns 0 if all values are NaN.
    pub(crate) fn argmax_axis<U: Uint>(&self, axis: Axis) -> Result<Tensor<U, D::Smaller>>
    where
        D: RemoveAxis,
    {
        if axis.0 >= self.shape().len() {
            bail!("Axis {:?} out of range for shape {:?}!", axis, self.shape());
        }
        let mut output =
            unsafe { Tensor::<u32, D::Smaller>::alloc(self.device(), self.dim.remove_axis(axis))? };
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Argmax,
            false,
        )?;
        output.cast_into()
    }
    /// Indexes an `axis` with `indices`.
    ///
    /// **Errors**
    /// - The `axis` is out of range.
    /// - The operation could not be performed.
    pub(crate) fn index_select(
        &self,
        axis: Axis,
        indices: &TensorView<u32, D::Smaller>,
    ) -> Result<Tensor<T, D::Smaller>>
    where
        D: RemoveAxis,
    {
        if axis.0 >= self.shape().len() {
            bail!("Axis {:?} out of range for shape {:?}!", axis, self.shape());
        }
        let output_dim = self.dim.remove_axis(axis);
        if indices.raw_dim() != output_dim {
            bail!(
                "Indices shape {:?} != input dim remove_axis {:?}!",
                indices.shape(),
                output_dim.slice()
            );
        }
        let mut output = unsafe { Tensor::alloc(self.device(), output_dim)? };
        if size_eq::<T, u16>() {
            output.as_raw_slice_mut().fill(T::zero())?;
        }
        let name = format!("index_select_{}", T::scalar_name());
        let module = glsl_shaders::module(name)?;
        let batch_size = indices.dim.size() as u32;
        let n = self.dim[axis.0] as u32;
        let builder = module
            .compute_pass("main")?
            .slice(self.to_slice()?.as_slice())?
            .slice(indices.to_slice()?.as_slice())?
            .slice_mut(output.as_raw_slice_mut())?
            .push([batch_size, n])?;
        unsafe {
            builder.submit([batch_size, 1, 1])?;
        }
        Ok(output)
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::{device::Device, rust_shaders};
    use approx::assert_relative_eq;
    use half::bf16;
    use ndarray::{Array, ArrayView, IntoDimension};
    use num_traits::Bounded;
    use std::convert::TryFrom;

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

    fn tensor_sum<T: Scalar + From<u8> + num_traits::Zero, D: IntoDimension>(
        dim: D,
        axis: Axis,
    ) -> Result<()>
    where
        D::Dim: RemoveAxis,
    {
        smol::block_on(async {
            let a1 = gen_array::<T, _>(dim.into_dimension())?;
            let y_true = a1.sum_axis(axis);
            let device = Device::new()?;
            let _s = device.acquire().await;
            let t1 = TensorView::try_from(a1.view())?.into_device(device).await?;
            let t2 = t1.sum_axis(axis)?;
            let y = t2.read().await?;
            assert_eq!(&y.as_array(), &y_true.view());
            Ok(())
        })
    }

    fn tensor_sum_bf16<D: IntoDimension>(dim: D, axis: Axis) -> Result<()>
    where
        D::Dim: RemoveAxis,
    {
        smol::block_on(async {
            let dim = dim.into_dimension();
            let a1_bf16 = gen_array::<bf16, _>(dim.clone())?;
            let a1_f32 = gen_array::<f32, _>(dim)?;
            let y_true = a1_f32.sum_axis(axis);
            let device = Device::new()?;
            let _s = device.acquire().await;
            let t1 = Tensor::from(a1_bf16).into_device(device).await?;
            let t2 = t1.sum_axis(axis)?;
            let y = t2.read().await?.as_array().map(|x| x.to_f32());
            assert_relative_eq!(&y, &y_true, epsilon = 0.01, max_relative = 0.01);
            Ok(())
        })
    }

    fn tensor_argmin<T: Scalar + From<u8> + PartialOrd + Bounded, D: IntoDimension>(
        dim: D,
        axis: Axis,
    ) -> Result<()>
    where
        D::Dim: RemoveAxis,
    {
        smol::block_on(async {
            let a1 = gen_array::<T, _>(dim.into_dimension())?;
            let y_true = array_argmin(&a1.view(), axis);
            let device = Device::new()?;
            let _s = device.acquire().await;
            let t1 = TensorView::try_from(a1.view())?.into_device(device).await?;
            let t2 = t1.argmin_axis::<u32>(axis)?;
            let y = t2.read().await?;
            assert_eq!(&y.as_array(), &y_true.view());
            Ok(())
        })
    }

    fn tensor_argmin_bf16<D: IntoDimension>(dim: D, axis: Axis) -> Result<()>
    where
        D::Dim: RemoveAxis,
    {
        smol::block_on(async {
            let a1 = gen_array::<f32, _>(dim.into_dimension())?;
            let y_true = array_argmin(&a1.view(), axis);
            let device = Device::new()?;
            let _s = device.acquire().await;
            let t1 = Tensor::from(a1.view().map(|x| bf16::from_f32(*x)))
                .into_device(device)
                .await?;
            let t2 = t1.argmin_axis::<u32>(axis)?;
            let y = t2.read().await?;
            assert_eq!(&y.as_array(), &y_true.view());
            Ok(())
        })
    }

    fn tensor_argmax<T: Scalar + From<u8> + PartialOrd + Bounded, D: IntoDimension>(
        dim: D,
        axis: Axis,
    ) -> Result<()>
    where
        D::Dim: RemoveAxis,
    {
        smol::block_on(async {
            let a1 = gen_array::<T, _>(dim.into_dimension())?;
            let y_true = array_argmax(&a1.view(), axis);
            let device = Device::new()?;
            let _s = device.acquire().await;
            let t1 = TensorView::try_from(a1.view())?.into_device(device).await?;
            let t2 = t1.argmax_axis::<u32>(axis)?;
            let y = t2.read().await?;
            assert_eq!(&y.as_array(), &y_true.view());
            Ok(())
        })
    }

    fn tensor_argmax_bf16<D: IntoDimension>(dim: D, axis: Axis) -> Result<()>
    where
        D::Dim: RemoveAxis,
    {
        smol::block_on(async {
            let a1 = gen_array::<f32, _>(dim.into_dimension())?;
            let y_true = array_argmax(&a1.view(), axis);
            let device = Device::new()?;
            let _s = device.acquire().await;
            let t1 = Tensor::from(a1.view().map(|x| bf16::from_f32(*x)))
                .into_device(device)
                .await?;
            let t2 = t1.argmax_axis::<u32>(axis)?;
            let y = t2.read().await?;
            assert_eq!(&y.as_array(), &y_true.view());
            Ok(())
        })
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

    async fn atomic_add<T: Scalar + core::iter::Sum>(n: usize, entry: &str) -> Result<()> {
        let x_vec = (1..=n)
            .into_iter()
            .map(|x| T::from_usize(x).unwrap())
            .collect::<Vec<_>>();
        let y_true = x_vec.iter().copied().sum::<T>();
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = Buffer::from(x_vec).into_device(device.clone()).await?;
        let mut y = Buffer::<T>::zeros(device.clone(), 1)?;
        let entry = format!("atomic::tests::{}", entry);
        let builder = rust_shaders::compute_pass(entry)?
            .slice(x.as_slice())?
            .slice_mut(y.as_slice_mut())?;
        unsafe {
            builder.submit([x.len() as u32, 1, 1])?;
        }
        let y = y.read().await?[0];
        assert_eq!(y, y_true);
        Ok(())
    }

    #[cfg_attr(any(target_os = "ios", target_os = "macos"), ignore)]
    #[tokio::test]
    async fn atomic_add_f32() -> Result<()> {
        atomic_add::<f32>(4, "atomic_add_f32").await?;
        Ok(())
    }
}
