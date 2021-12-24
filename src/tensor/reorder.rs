use super::*;
use crate::rust_shaders;
use std::iter::repeat;

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Converts to standard layout.
    ///
    /// If in standard layout, borrows the tensor. Otherwise, copies into a new standard layout tensor.
    ///
    /// **Errors**
    /// See [`.into_standard_layout()`](TensorBase::into_standard_layout()).
    pub fn as_standard_layout(&self) -> Result<CowTensor<T, D>> {
        if self.is_standard_layout() {
            Ok(self.view().into())
        } else {
            self.view().into_standard_layout().map(Into::into)
        }
    }
    /// Converts into standard layout.
    ///
    /// If in standard layout, converts into an owned [`Tensor`]. Otherwise, copies the data into a new standard layout tensor.
    ///
    /// **Errors**
    /// - Supports up to 6 dimensional inputs.
    /// - Supports u32, i32, and f32.
    /// - The shader could not be executed.
    /// - The device disconnected or panicked.
    /// - See [`.into_owned()`](TensorBase::into_owned()).
    ///
    /// **Panics**
    /// Only u32, i32, and f32 are currently implemented.
    pub fn into_standard_layout(self) -> Result<Tensor<T, D>> {
        if self.is_standard_layout() {
            self.into_owned()
        } else {
            let ndim = if self.dim.ndim() <= 4 {
                4
            } else if self.dim.ndim() <= 6 {
                6
            } else {
                bail!("Up to 6 dimensions supported!");
            };
            let ty = match size_of::<T>() {
                4 => "u32",
                _ => bail!("{} not supported!", T::scalar_name()),
            };
            let mut output = unsafe { Tensor::alloc(self.device(), self.raw_dim())? };
            let name = format!("reorder::as_standard_layout_{}d_{}", ndim, ty);
            let mut builder = rust_shaders::core()?
                .compute_pass(&name)?
                .slice(self.as_raw_slice())?
                .slice_mut(output.as_raw_slice_mut())?;
            for (d, s) in self
                .shape()
                .iter()
                .copied()
                .zip(self.strides().iter().copied())
                .into_iter()
                .chain(repeat((1, 1)))
                .take(ndim)
            {
                builder = builder.push(d as u32)?.push(s as i32)?;
            }
            let n = self.len() as u32;
            unsafe {
                builder.submit([n, 1, 1])?;
            }
            Ok(output)
        }
    }
    /// Converts to an [`ArcTensor`] in standard layout.
    ///
    /// If in standard layout, converts to an [`ArcTensor`] (or clones the [`ArcTensor`]), otherwise copies the data into a new [`ArcTensor`].
    pub fn to_standard_layout_shared(&self) -> Result<ArcTensor<T, D>> {
        if self.is_standard_layout() {
            self.to_shared()
        } else {
            self.as_standard_layout()?.into_shared()
        }
    }
    #[allow(unused)]
    pub(super) fn as_reordered<T2, Sh>(&self, shape: Sh) -> Result<CowTensor<T2, D>>
    where
        T2: Scalar,
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        if self.dim == dim && self.strides == strides {
            self.cast_to()
        } else {
            if bytemuck::cast_slice(strides.slice())
                .iter()
                .copied()
                .any(|s: isize| s < 0)
            {
                bail!("Negative strides {:?} not supported!", strides);
            }
            let mut output = unsafe { Tensor::alloc(self.device(), dim)? };
            output.strides = strides;
            self.reorder_with(T2::zero(), &mut output.view_mut())?;
            Ok(output.into())
        }
    }
    #[allow(unused)]
    pub(super) fn reorder_with<T2: Scalar>(
        &self,
        beta: T2,
        output: &mut TensorViewMut<T2, D>,
    ) -> Result<()> {
        let ndim = output.ndim();
        let entry = format!(
            "reorder::reorder_{}d_{}_{}",
            ndim,
            T::scalar_name(),
            T2::scalar_name(),
        );
        let builder = rust_shaders::core()?.compute_pass(entry)?;
        let (builder, global_size) = match (
            (self.shape(), self.strides()),
            (output.shape(), output.strides()),
        ) {
            (([ih, iw], [rsx, csx]), ([oh, ow], [rsy, csy])) => {
                let bs = 1u32;
                let builder = builder
                    .push(bs)?
                    .push([*ih as u32, *iw as u32])?
                    .push([*rsx as u32, *csx as u32])?
                    .push(beta)?
                    .push([*oh as u32, *ow as u32])?
                    .push([*rsy as u32, *csy as u32])?;
                let groups_h = *oh / 16 + if *oh % 16 != 0 { 1 } else { 0 };
                let groups_w = *ow / 16 + if *ow % 16 != 0 { 1 } else { 0 };
                let global_size = groups_h * groups_w * 256;
                (builder, global_size)
            }
            _ => bail!("Unimplemented!"),
        };
        let builder = builder
            .slice(self.as_raw_slice())?
            .slice_mut(output.as_raw_slice_mut())?;
        unsafe { builder.submit([global_size as u32, 1, 1]) }
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;

    async fn into_standard_layout<T: Scalar, E: IntoDimension>(shape: E) -> Result<()> {
        let shape = shape.into_dimension();
        let x_vec = (0..shape.size())
            .into_iter()
            .map(|x| T::from_usize(x).unwrap())
            .collect();
        let x_array = Array::from_shape_vec(shape, x_vec)?;
        let axes = (0..x_array.ndim())
            .cycle()
            .skip(1)
            .take(x_array.ndim())
            .collect::<Vec<usize>>();
        let axes = E::Dim::from_dimension(&axes.into_dimension()).unwrap();
        let y_array = x_array
            .view()
            .permuted_axes(axes.clone())
            .as_standard_layout()
            .to_owned();
        let device = Device::new()?;
        let x = Tensor::from(x_array).into_device(device).await?;
        let y = x.permuted_axes(axes).into_standard_layout()?.read().await?;
        assert_eq!(y_array.view(), y.as_array());
        Ok(())
    }

    #[tokio::test]
    async fn into_standard_layout_4d_u32() -> Result<()> {
        into_standard_layout::<u32, _>([2, 3, 4, 5]).await?;
        Ok(())
    }

    #[tokio::test]
    async fn into_standard_layout_6d_u32() -> Result<()> {
        into_standard_layout::<u32, _>([2, 3, 4, 5, 6, 7]).await?;
        Ok(())
    }

    async fn reorder<T1, Sh1, T2, Sh2>(input_shape: Sh1, output_shape: Sh2) -> Result<()>
    where
        T1: Scalar,
        Sh1: ShapeBuilder,
        T2: Scalar,
        Sh2: ShapeBuilder<Dim = Sh1::Dim> + Clone,
    {
        let mut x_array = Array::<T1, _>::zeros(input_shape);
        for (x, i) in x_array.iter_mut().zip(1..) {
            *x = T1::from_usize(i).unwrap();
        }
        let mut y_array = Array::<T2, _>::zeros(output_shape.clone());
        for (indices, x) in x_array.indexed_iter() {
            if let Some(y) = y_array.get_mut(indices.into_dimension()) {
                *y = T2::from_usize(x.to_usize().unwrap()).unwrap();
            }
        }

        let device = Device::new()?;
        let x = CowTensor::from(x_array.view())
            .into_device(device.clone())
            .await?;
        let y = x
            .as_reordered::<T2, _>(output_shape)?
            .read()
            .await?
            .as_array()
            .to_owned();
        assert_eq!(y, y_array);
        Ok(())
    }

    #[tokio::test]
    async fn reorder_2d_f32_f32() -> Result<()> {
        reorder::<f32, _, f32, _>([3, 3], [4, 4].set_f(true)).await?;
        Ok(())
    }
}
