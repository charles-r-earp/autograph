use super::{
    Data, Dimension, Ix1, Num, Result, Scalar, Tensor, Tensor2, TensorBase, TensorView,
    TensorViewMut, Unsigned,
};
use crate::util::type_eq;

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Scales the tensor to a new Tensor\
    ///
    /// Performs the operations y = alpha * (x as T2)\
    /// Err: Does not currently support non standard layout. Errors if the operation cannot be executed.
    pub fn scale_into<T2: Num>(self, alpha: T2) -> Result<Tensor<T2, D>> {
        let mut output = unsafe { Tensor::uninitialized(self.device(), self.raw_dim())? };
        output.strides = self.strides.clone();
        scaled_cast(&self.view(), &mut output.view_mut(), alpha)?;
        Ok(output)
    }
    /// Scales the tensor to a new Tensor\
    ///
    /// Performs the operations y = x as T2
    /// Err: Errors if the operation cannot be executed.
    pub fn cast_into<T2: Num>(self) -> Result<Tensor<T2, D>> {
        self.scale_into(T2::one())
    }
}

fn scaled_cast<T1, T2, D>(
    input: &TensorView<T1, D>,
    output: &mut TensorViewMut<T2, D>,
    alpha: T2,
) -> Result<()>
where
    T1: Scalar,
    T2: Num,
    D: Dimension,
{
    debug_assert!(input.len() == output.len());
    let device = input.device();
    let src = if type_eq::<T1, u8>() {
        if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_u8_f32.spv")
        } else {
            todo!()
        }
    } else if type_eq::<T1, f32>() {
        if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_f32_f32.spv")
        } else {
            todo!()
        }
    } else {
        todo!()
    };
    let n = input.len() as u32;
    let alpha = alpha.to_bits_u32().unwrap();
    device
        .compute_pass(src, "main")?
        .buffer_slice(input.as_unordered_buffer_slice())?
        .buffer_slice_mut(output.as_unordered_buffer_slice_mut())?
        .push_constants(bytemuck::cast_slice(&[n, alpha]))?
        .global_size([n, 1, 1])
        .enqueue()
}

impl<T: Unsigned, S: Data<Elem = T>> TensorBase<S, Ix1> {
    #[allow(unused_variables)]
    pub fn to_one_hot<T2: Num>(&self, nclasses: usize) -> Result<Tensor2<T2>> {
        todo!()
    }
}
