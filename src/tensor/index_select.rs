use super::{Axis, Data, Dimension, Num, RemoveAxis, Result, Tensor, TensorBase, TensorView};
use crate::util::type_eq;
use anyhow::ensure;
use half::bf16;

impl<T: Num, S: Data<Elem = T>, D: RemoveAxis> TensorBase<S, D> {
    /// Indexes an axis with indices\
    ///
    /// Err: Errors if axis is out of range or self / indices is not standard layout
    pub(crate) fn index_select(
        &self,
        axis: Axis,
        indices: &TensorView<u32, D::Smaller>,
    ) -> Result<Tensor<T, D::Smaller>> {
        ensure!(axis.0 < self.shape().len());
        ensure!(self.shape().len() == indices.shape().len() + 1);
        ensure!(self.strides == self.dim.default_strides());
        ensure!(indices.strides == indices.dim.default_strides());
        let mut output = unsafe {
            Tensor::<T, D::Smaller>::uninitialized(&self.device(), self.dim.remove_axis(axis))?
        };
        let src = if type_eq::<T, bf16>() {
            include_shader!("glsl/index_select_bf16.spv")
        } else if type_eq::<T, u32>() {
            include_shader!("glsl/index_select_u32.spv")
        } else if type_eq::<T, i32>() {
            include_shader!("glsl/index_select_i32.spv")
        } else if type_eq::<T, f32>() {
            include_shader!("glsl/index_select_f32.spv")
        } else {
            unreachable!()
        };
        let device = self.device();
        let batch_size = indices.dim.size() as u32;
        let n = self.dim[axis.0] as u32;
        device
            .compute_pass(src, "main")?
            .buffer_slice(self.as_unordered_buffer_slice())?
            .buffer_slice(indices.as_unordered_buffer_slice())?
            .buffer_slice_mut(output.as_unordered_buffer_slice_mut())?
            .push_constants(bytemuck::cast_slice(&[batch_size, n]))?
            .global_size([batch_size, 1, 1])
            .enqueue()?;
        Ok(output)
    }
}
