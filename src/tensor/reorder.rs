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
}
