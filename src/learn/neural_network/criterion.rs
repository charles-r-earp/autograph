use super::autograd::{Variable0, Variable2};
#[cfg(feature = "device")]
use crate::tensor::{ScalarTensor, ScalarTensorView, Tensor};
use crate::{
    device::Device,
    learn::criterion::{Criterion, CrossEntropyLoss},
    scalar::{Scalar, ScalarElem, ScalarType},
    tensor::{ScalarArcTensor, ScalarArcTensor1, Tensor2, TensorView1, TensorView2},
};
use anyhow::{bail, Result};
use dry::macro_for;
use half::bf16;
#[cfg(feature = "device")]
use krnl::macros::module;
use ndarray::Array2;
#[cfg(feature = "device")]
use num_traits::ToPrimitive;
use num_traits::{Float, Unsigned};
#[cfg(feature = "device")]
use paste::paste;

impl Criterion<Variable2, ScalarArcTensor1> for CrossEntropyLoss {
    type Output = Variable0;
    fn eval(&self, input: Variable2, target: ScalarArcTensor1) -> Result<Variable0> {
        if !matches!(input.scalar_type(), ScalarType::BF16 | ScalarType::F32)
            || !matches!(
                target.scalar_type(),
                ScalarType::U8 | ScalarType::U16 | ScalarType::U32
            )
        {
            bail!(
                "CrossEntropyLoss {:?} {:?} unimplemented!",
                input.scalar_type(),
                target.scalar_type()
            );
        }
        let mut builder = Variable0::builder();
        if let Some(node) = input.node() {
            let input = input.value().clone();
            let target = target.clone();
            builder.edge(node, move |output_grad| {
                macro_for!($X in [bf16, f32] {
                    macro_for!($T in [u8, u16, u32] {
                        if input.scalar_type() == $X::scalar_type() && target.scalar_type() == $T::scalar_type() {
                            let input = input.try_into_arc_tensor::<$X>().unwrap();
                            let target = target.try_into_arc_tensor::<$T>().unwrap();
                            let dy = output_grad
                                .into_device(Device::host())?
                                .cast_into_tensor::<$X>()?
                                .into_array()
                                .unwrap()
                                .into_scalar();
                            return Ok(
                                cross_entropy_loss_backward::<$X, $T>(input.view(), target.view(), dy.cast::<f32>())?
                                    .into_scalar_tensor()
                                    .into_shared()
                                    .unwrap(),
                            );
                        }
                    });
                });
                unreachable!()
            });
        }
        let loss = self.eval(input.into_value(), target)?;
        let value = ScalarArcTensor::from_elem(Device::host(), (), ScalarElem::F32(loss)).unwrap();
        Ok(builder.build(value))
    }
}

fn cross_entropy_loss_backward<T1: Scalar + Float, T2: Scalar + Unsigned>(
    x: TensorView2<T1>,
    t: TensorView1<T2>,
    mut dy: f32,
) -> Result<Tensor2<T1>> {
    dy /= x.dim().0 as f32;
    if let Some((x, t)) = x.as_array().zip(t.as_array()) {
        let mut dx = Array2::<T1>::zeros(x.raw_dim());
        for ((x, t), mut dx) in x
            .outer_iter()
            .zip(t.iter().copied())
            .zip(dx.outer_iter_mut())
        {
            let x_iter = x.iter().map(|x| x.cast::<f32>());
            let m = x_iter
                .clone()
                .fold(x_iter.clone().next().unwrap_or_default(), |m, x| m.max(x));
            let s: f32 = x_iter.clone().map(|x| (x - m).exp()).sum();
            for (i, (x, dx)) in x_iter.zip(dx.iter_mut()).enumerate() {
                let t = (i == t.to_usize().unwrap()) as u8 as f32;
                *dx = (dy * ((x - m).exp() / s - t)).cast();
            }
        }
        return Ok(dx.into());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let (batch_size, classes) = x.dim();
        macro_for!($X in [bf16, f32] {
            macro_for!($T in [u8, u16, u32] {
                if x.scalar_type() == $X::scalar_type() && t.scalar_type() == $T::scalar_type() {
                    let x = ScalarTensorView::from(x)
                        .try_into_tensor_view::<$X>()
                        .unwrap();
                    let t = ScalarTensorView::from(t)
                        .try_into_tensor_view::<$T>()
                        .unwrap();
                    let mut dx = unsafe { Tensor::<$X, _>::uninit(x.device(), x.raw_dim())? };
                    let kernel = paste! { kernels::[<cross_entropy_loss_backward_ $X _ $T>]::builder()?.build(dx.device())? };
                    kernel
                        .with_global_threads(batch_size.to_u32().unwrap())
                        .dispatch(
                            x.as_slice().unwrap(),
                            t.as_slice().unwrap(),
                            classes.to_u32().unwrap(),
                            dy,
                            dx.as_slice_mut().unwrap(),
                        )?;
                    return Ok(ScalarTensor::from(dx).try_into_tensor().unwrap());
                }
            });
        });
        unreachable!()
    }
}

#[cfg(feature = "device")]
#[module]
mod kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{
        buffer::UnsafeIndex,
        half::bf16,
        num_traits::{Float, ToPrimitive},
        scalar::Scalar,
    };
    use paste::paste;

    macro_for!($X in [bf16, f32] {
        macro_for!($T in [u8, u16, u32] {
            paste! {
                #[kernel(threads(256))]
                pub fn [<cross_entropy_loss_backward_ $X _ $T>](
                    #[global] x: Slice<$X>,
                    #[global] t: Slice<$T>,
                    classes: u32,
                    dy: f32,
                    #[global] dx: UnsafeSlice<$X>,
                ) {
                    let idx = kernel.global_id();
                    if idx as usize > t.len() {
                        return;
                    }
                    let mut m = x[(idx * classes) as usize].cast::<f32>();
                    for i in 1..classes {
                        let x = x[(idx * classes + i) as usize].cast::<f32>();
                        m = m.max(x);
                    }
                    let mut s = 0f32;
                    for i in 0..classes {
                        let x = x[(idx * classes + i) as usize].cast::<f32>();
                        s += (x - m).exp();
                    }
                    let t = t[idx as usize].to_u32().unwrap();
                    for i in 0..classes {
                        let x = x[(idx * classes + i) as usize].cast::<f32>();
                        let t = (i == t) as u8 as f32;
                        let dx = unsafe { dx.unsafe_index_mut((idx * classes + i) as usize) };
                        *dx = (dy * ((x - m).exp() / s - t)).cast();
                    }
                }
            }
        });
    });
}
