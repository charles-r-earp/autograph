use super::{Data, DataMut, Dimension, Num, Result, TensorBase};
use crate::util::type_eq;
use anyhow::ensure;
use half::bf16;

impl<T: Num, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn scaled_add<S2>(&mut self, alpha: T, rhs: &TensorBase<S2, D>) -> Result<()>
    where
        S: DataMut,
        S2: Data<Elem = T>,
    {
        ensure!(self.dim == rhs.dim);
        ensure!(self.strides == rhs.strides);
        let src = if type_eq::<T, bf16>() {
            include_shader!("glsl/scaled_add_bf16.spv")
        } else if type_eq::<T, u32>() {
            include_shader!("glsl/scaled_add_u32.spv")
        } else if type_eq::<T, i32>() {
            include_shader!("glsl/scaled_add_i32.spv")
        } else if type_eq::<T, f32>() {
            include_shader!("glsl/scaled_add_f32.spv")
        } else {
            unreachable!()
        };
        let device = rhs.device();
        let n = self.len() as u32;
        let alpha = alpha.to_bits_u32().unwrap();
        device
            .compute_pass(src, "main")?
            .buffer_slice_mut(self.as_unordered_buffer_slice_mut())?
            .buffer_slice(rhs.as_unordered_buffer_slice())?
            .push_constants(bytemuck::cast_slice(&[n, alpha]))?
            .global_size([n, 1, 1])
            .enqueue()
    }
    pub fn add_assign<S2>(&mut self, rhs: &TensorBase<S2, D>) -> Result<()>
    where
        S: DataMut,
        S2: Data<Elem = T>,
    {
        self.scaled_add(T::one(), rhs)
    }
}
