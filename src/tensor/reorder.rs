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
            if self.dim.ndim() > 6 {
                bail!("Up to 6 dimensions supported!");
            }
            let ty = match size_of::<T>() {
                4 => "u32",
                _ => todo!(),
            };
            let mut output = unsafe { Tensor::alloc(self.device(), self.raw_dim())? };
            let name = format!("reorder::as_standard_layout_6d_{}", ty);
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
                .take(6)
            {
                builder = builder.push(d as u32)?.push(s as i32)?;
            }
            unsafe {
                builder.submit([self.len() as u32, 1, 1])?;
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
