use super::{
    Axis, Data, Dimension, Num, RemoveAxis, Result, Tensor, TensorBase, TensorView, TensorViewMut,
};
use crate::util::type_eq;
use anyhow::ensure;
use half::bf16;

impl<T: Num, S: Data<Elem = T>, D: RemoveAxis> TensorBase<S, D> {
    /// Computes the index of the max value along the given axis\
    pub fn sum(&self, axis: Axis) -> Result<Tensor<T, D::Smaller>> {
        ensure!(axis.0 < self.shape().len());
        let mut output = unsafe {
            Tensor::<T, D::Smaller>::uninitialized(&self.device(), self.dim.remove_axis(axis))?
        };
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Sum,
            false,
        )?;
        Ok(output)
    }
    pub(crate) fn sum_with(
        &self,
        axis: Axis,
        output: &mut TensorViewMut<T, D::Smaller>,
    ) -> Result<()> {
        ensure!(axis.0 < self.shape().len());
        ensure!(output.dim == self.dim.remove_axis(axis));
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Sum,
            true,
        )?;
        Ok(())
    }
}

impl<T: Num, S: Data<Elem = T>, D: RemoveAxis> TensorBase<S, D> {
    /// Computes the index of the min value along the given axis\
    ///
    /// For multiple max values, the first will be selected. NaN values are ignored, returns 0 if all values are NaN.
    pub fn argmin(&self, axis: Axis) -> Result<Tensor<u32, D::Smaller>> {
        ensure!(axis.0 < self.shape().len());
        let mut output = unsafe {
            Tensor::<u32, D::Smaller>::uninitialized(&self.device(), self.dim.remove_axis(axis))?
        };
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Argmin,
            false,
        )?;
        Ok(output)
    }
}

impl<T: Num, S: Data<Elem = T>, D: RemoveAxis> TensorBase<S, D> {
    /// Computes the index of the max value along the given axis\
    ///
    /// For multiple max values, the first will be selected. NaN values are ignored, returns 0 if all values are NaN.
    pub fn argmax(&self, axis: Axis) -> Result<Tensor<u32, D::Smaller>> {
        ensure!(axis.0 < self.shape().len());
        let mut output = unsafe {
            Tensor::<u32, D::Smaller>::uninitialized(&self.device(), self.dim.remove_axis(axis))?
        };
        reduce(
            &self.view(),
            &mut output.view_mut(),
            axis,
            Reduction::Argmax,
            false,
        )?;
        Ok(output)
    }
}

#[allow(unused)]
enum Reduction {
    Sum,
    Mean,
    Min,
    Max,
    Argmin,
    Argmax,
}

fn reduce<T1, T2, D>(
    input: &TensorView<T1, D>,
    output: &mut TensorViewMut<T2, D::Smaller>,
    axis: Axis,
    reduction: Reduction,
    accumulate: bool,
) -> Result<()>
where
    T1: Num,
    T2: Num,
    D: Dimension,
{
    let device = input.device();
    let src = {
        use Reduction::*;
        match reduction {
            Sum => {
                if type_eq::<T1, bf16>() {
                    include_shader!("glsl/reduce_sum_final_bf16.spv")
                } else if type_eq::<T1, u32>() {
                    include_shader!("glsl/reduce_sum_final_u32.spv")
                } else if type_eq::<T1, i32>() {
                    include_shader!("glsl/reduce_sum_final_i32.spv")
                } else if type_eq::<T1, f32>() {
                    include_shader!("glsl/reduce_sum_final_f32.spv")
                } else {
                    unreachable!()
                }
            }
            Mean => {
                if type_eq::<T1, bf16>() {
                    include_shader!("glsl/reduce_mean_final_bf16.spv")
                } else if type_eq::<T1, u32>() {
                    include_shader!("glsl/reduce_mean_final_u32.spv")
                } else if type_eq::<T1, i32>() {
                    include_shader!("glsl/reduce_mean_final_i32.spv")
                } else if type_eq::<T1, f32>() {
                    include_shader!("glsl/reduce_mean_final_f32.spv")
                } else {
                    unreachable!()
                }
            }
            Min => {
                if type_eq::<T1, bf16>() {
                    include_shader!("glsl/reduce_min_final_bf16.spv")
                } else if type_eq::<T1, u32>() {
                    include_shader!("glsl/reduce_min_final_u32.spv")
                } else if type_eq::<T1, i32>() {
                    include_shader!("glsl/reduce_min_final_i32.spv")
                } else if type_eq::<T1, f32>() {
                    include_shader!("glsl/reduce_min_final_f32.spv")
                } else {
                    unreachable!()
                }
            }
            Max => {
                if type_eq::<T1, bf16>() {
                    include_shader!("glsl/reduce_max_final_bf16.spv")
                } else if type_eq::<T1, u32>() {
                    include_shader!("glsl/reduce_max_final_u32.spv")
                } else if type_eq::<T1, i32>() {
                    include_shader!("glsl/reduce_max_final_i32.spv")
                } else if type_eq::<T1, f32>() {
                    include_shader!("glsl/reduce_max_final_f32.spv")
                } else {
                    unreachable!()
                }
            }
            Argmin => {
                if type_eq::<T1, bf16>() {
                    include_shader!("glsl/reduce_argmin_final_bf16.spv")
                } else if type_eq::<T1, u32>() {
                    include_shader!("glsl/reduce_argmin_final_u32.spv")
                } else if type_eq::<T1, i32>() {
                    include_shader!("glsl/reduce_argmin_final_i32.spv")
                } else if type_eq::<T1, f32>() {
                    include_shader!("glsl/reduce_argmin_final_f32.spv")
                } else {
                    unreachable!()
                }
            }
            Argmax => {
                if type_eq::<T1, bf16>() {
                    include_shader!("glsl/reduce_argmax_final_bf16.spv")
                } else if type_eq::<T1, u32>() {
                    include_shader!("glsl/reduce_argmax_final_u32.spv")
                } else if type_eq::<T1, i32>() {
                    include_shader!("glsl/reduce_argmax_final_i32.spv")
                } else if type_eq::<T1, f32>() {
                    include_shader!("glsl/reduce_argmax_final_f32.spv")
                } else {
                    unreachable!()
                }
            }
        }
    };
    let batch_size = output.len() as u32;
    let stride_x = if axis.0 > 0 {
        input.strides()[axis.0 - 1] as u32
    } else {
        1
    };
    let n = input.shape()[axis.0] as u32;
    let stride_y = input.strides()[axis.0] as u32;
    let accumulate = accumulate as u32;
    device
        .compute_pass(src, "main")?
        .buffer_slice(input.as_unordered_buffer_slice())?
        .buffer_slice_mut(output.as_buffer_slice_mut().unwrap())?
        .push_constants(bytemuck::cast_slice(&[
            batch_size, stride_x, n, stride_y, accumulate,
        ]))?
        .global_size([batch_size, 1, 1])
        .enqueue()
}
