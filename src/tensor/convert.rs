use super::{
    CowRepr, CowTensor, Data, Dimension, Ix1, Num, OwnedRepr, Result, Scalar, Tensor, Tensor2,
    TensorBase, TensorView, TensorViewMut, Unsigned, ViewRepr,
};
use crate::util::type_eq;
use half::bf16;
use std::mem::transmute;

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Scales the tensor to a new Tensor\
    ///
    /// Performs the operations y = alpha * (x as T2)\
    /// Err: Does not currently support non standard layout. Errors if the operation cannot be executed.\
    /// Supports Scalar types u8, bf16, i32, u32, and f32
    pub fn scale_into<T2: Num>(self, alpha: T2) -> Result<Tensor<T2, D>> {
        // TODO: use implace operations when possible
        let mut output = unsafe { Tensor::uninitialized(self.device(), self.raw_dim())? };
        output.strides = self.strides.clone();
        scaled_cast(&self.view(), &mut output.view_mut(), alpha)?;
        Ok(output)
    }
    /// Scales the tensor to a new Tensor\
    ///
    /// Performs the operations y = x as T2
    /// Err: Does not currently support non standard layout. Errors if the operation cannot be executed.\
    /// Supports Scalar types u8, bf16, i32, u32, and f32
    pub fn cast_into<T2: Num>(self) -> Result<Tensor<T2, D>> {
        if type_eq::<T, T2>() {
            let TensorBase {
                device,
                dim,
                strides,
                data,
            } = self;
            let data: OwnedRepr<T> = OwnedRepr(data.into_buffer()?);
            let data: OwnedRepr<T2> = unsafe { transmute(data) };
            Ok(Tensor {
                device,
                dim,
                strides,
                data,
            })
        } else {
            self.scale_into(T2::one())
        }
    }
    pub fn cast_to<T2: Num>(&self) -> Result<CowTensor<T2, D>> {
        if type_eq::<T, T2>() {
            let TensorBase {
                device,
                dim,
                strides,
                data,
            } = self;
            let data: CowRepr<T> = CowRepr::from(ViewRepr(data.as_buffer_slice()));
            let data: CowRepr<T2> = unsafe { transmute(data) };
            Ok(CowTensor {
                device: device.clone(),
                dim: dim.clone(),
                strides: strides.clone(),
                data,
            })
        } else {
            Ok(self.view().scale_into(T2::one())?.into())
        }
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
        if type_eq::<T2, bf16>() {
            include_shader!("glsl/scaled_cast_u8_bf16.spv")
        } else if type_eq::<T2, u32>() {
            include_shader!("glsl/scaled_cast_u8_u32.spv")
        } else if type_eq::<T2, i32>() {
            include_shader!("glsl/scaled_cast_u8_i32.spv")
        } else if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_u8_f32.spv")
        } else {
            unreachable!()
        }
    } else if type_eq::<T1, u16>() {
        if type_eq::<T2, bf16>() {
            include_shader!("glsl/scaled_cast_u16_bf16.spv")
        } else if type_eq::<T2, u32>() {
            include_shader!("glsl/scaled_cast_u16_u32.spv")
        } else if type_eq::<T2, i32>() {
            include_shader!("glsl/scaled_cast_u16_i32.spv")
        } else if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_u16_f32.spv")
        } else {
            unreachable!()
        }
    } else if type_eq::<T1, bf16>() {
        if type_eq::<T2, bf16>() {
            include_shader!("glsl/scaled_cast_bf16_bf16.spv")
        } else if type_eq::<T2, u32>() {
            include_shader!("glsl/scaled_cast_bf16_u32.spv")
        } else if type_eq::<T2, i32>() {
            include_shader!("glsl/scaled_cast_bf16_i32.spv")
        } else if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_bf16_f32.spv")
        } else {
            unreachable!()
        }
    } else if type_eq::<T1, u32>() {
        if type_eq::<T2, bf16>() {
            include_shader!("glsl/scaled_cast_u32_bf16.spv")
        } else if type_eq::<T2, u32>() {
            include_shader!("glsl/scaled_cast_u32_u32.spv")
        } else if type_eq::<T2, i32>() {
            include_shader!("glsl/scaled_cast_u32_i32.spv")
        } else if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_u32_f32.spv")
        } else {
            unreachable!()
        }
    } else if type_eq::<T1, i32>() {
        if type_eq::<T2, bf16>() {
            include_shader!("glsl/scaled_cast_i32_bf16.spv")
        } else if type_eq::<T2, u32>() {
            include_shader!("glsl/scaled_cast_i32_u32.spv")
        } else if type_eq::<T2, i32>() {
            include_shader!("glsl/scaled_cast_i32_i32.spv")
        } else if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_i32_f32.spv")
        } else {
            unreachable!()
        }
    } else if type_eq::<T1, f32>() {
        if type_eq::<T2, bf16>() {
            include_shader!("glsl/scaled_cast_f32_bf16.spv")
        } else if type_eq::<T2, u32>() {
            include_shader!("glsl/scaled_cast_f32_u32.spv")
        } else if type_eq::<T2, i32>() {
            include_shader!("glsl/scaled_cast_f32_i32.spv")
        } else if type_eq::<T2, f32>() {
            include_shader!("glsl/scaled_cast_f32_f32.spv")
        } else {
            unreachable!()
        }
    } else {
        // i8, f16
        unimplemented!()
    };
    let n = input.len() as u32;
    let alpha = if let Some(alpha) = alpha.to_f32() {
        alpha.to_bits_u32().unwrap()
    } else {
        alpha.to_bits_u32().unwrap()
    };
    device
        .compute_pass(src, "main")?
        .buffer_slice(input.as_unordered_buffer_slice())?
        .buffer_slice_mut(output.as_unordered_buffer_slice_mut())?
        .push_constants(bytemuck::cast_slice(&[n, alpha]))?
        .global_size([n, 1, 1])
        .enqueue()
}

impl<T: Unsigned, S: Data<Elem = T>> TensorBase<S, Ix1> {
    pub fn to_one_hot<T2: Num>(&self, nclasses: usize) -> Result<Tensor2<T2>> {
        let device = self.device();
        let src = if type_eq::<T, u8>() {
            if type_eq::<T2, bf16>() {
                include_shader!("glsl/one_hot_u8_bf16.spv")
            } else if type_eq::<T2, u32>() {
                include_shader!("glsl/one_hot_u8_u32.spv")
            } else if type_eq::<T2, i32>() {
                include_shader!("glsl/one_hot_u8_i32.spv")
            } else if type_eq::<T2, f32>() {
                include_shader!("glsl/one_hot_u8_f32.spv")
            } else {
                unreachable!()
            }
        } else if type_eq::<T, u16>() {
            if type_eq::<T2, bf16>() {
                include_shader!("glsl/one_hot_u16_bf16.spv")
            } else if type_eq::<T2, u32>() {
                include_shader!("glsl/one_hot_u16_u32.spv")
            } else if type_eq::<T2, i32>() {
                include_shader!("glsl/one_hot_u16_i32.spv")
            } else if type_eq::<T2, f32>() {
                include_shader!("glsl/one_hot_u16_f32.spv")
            } else {
                unreachable!()
            }
        } else if type_eq::<T, u32>() {
            if type_eq::<T2, bf16>() {
                include_shader!("glsl/one_hot_u32_bf16.spv")
            } else if type_eq::<T2, u32>() {
                include_shader!("glsl/one_hot_u32_u32.spv")
            } else if type_eq::<T2, i32>() {
                include_shader!("glsl/one_hot_u32_i32.spv")
            } else if type_eq::<T2, f32>() {
                include_shader!("glsl/one_hot_u32_f32.spv")
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        };
        let n = self.dim();
        let mut output = unsafe { Tensor::uninitialized(device, [n, nclasses])? };
        device
            .compute_pass(src, "main")?
            .buffer_slice(self.as_unordered_buffer_slice())?
            .buffer_slice_mut(output.as_unordered_buffer_slice_mut())?
            .push_constants(bytemuck::cast_slice(&[n as u32, nclasses as u32]))?
            .global_size([n as u32, 1, 1])
            .enqueue()?;
        Ok(output)
    }
}
