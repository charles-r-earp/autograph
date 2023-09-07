#[cfg(feature = "device")]
use crate::{
    buffer::Slice,
    tensor::{ScalarTensorView1, ScalarTensorView2, Tensor},
};
use crate::{
    buffer::{Data, ScalarData},
    scalar::Scalar,
    tensor::{ScalarTensorBase, ScalarTensorView, TensorBase, TensorView1, TensorView2},
};
use anyhow::{bail, Result};
use dry::macro_for;
use half::bf16;
#[cfg(feature = "device")]
use krnl::macros::module;
use ndarray::{ArrayBase, ArrayView1, ArrayView2, Data as ArrayData, Ix1, Ix2};
#[cfg(feature = "device")]
use num_traits::ToPrimitive;
use num_traits::{Float, Unsigned};
#[cfg(feature = "device")]
use paste::paste;

/// Accuracy.
pub trait Accuracy<T> {
    /// Accuracy of a prediction given `target`.
    ///
    /// Returns the number of correct predictions.
    fn accuracy(&self, target: T) -> Result<usize>;
}

impl<T1: Scalar, S1: ArrayData<Elem = T1>, T2: Scalar + Unsigned, S2: ArrayData<Elem = T2>>
    Accuracy<ArrayBase<S2, Ix1>> for ArrayBase<S1, Ix2>
{
    fn accuracy(&self, target: ArrayBase<S2, Ix1>) -> Result<usize> {
        let mut correct = 0;
        for (x, t) in self
            .outer_iter()
            .zip(target.iter().map(|x| x.to_usize().unwrap()))
        {
            let mut m = x[0];
            let mut mi = 0;
            for (i, x) in x.iter().copied().enumerate() {
                if let Some(x_f64) = x.to_f64() {
                    assert!(x_f64.is_finite());
                }
                if x > m {
                    m = x;
                    mi = i;
                }
            }
            if mi == t {
                correct += 1;
            }
        }
        Ok(correct)
    }
}

/// Implemented for:
/// - input: bf16, f32
/// - target: u8, u16, u32
impl<T1: Scalar, S1: Data<Elem = T1>, T2: Scalar + Unsigned, S2: Data<Elem = T2>>
    Accuracy<TensorBase<S2, Ix1>> for TensorBase<S1, Ix2>
{
    fn accuracy(&self, target: TensorBase<S2, Ix1>) -> Result<usize> {
        if let Some((input, target)) = self.as_array().zip(target.as_array()) {
            input.accuracy(target)
        } else {
            ScalarTensorView::from(self.view()).accuracy(ScalarTensorView::from(target.view()))
        }
    }
}

/// Implemented for:
/// - input: bf16, f32
/// - target: u8, u16, u32
impl<S1: ScalarData, S2: ScalarData> Accuracy<ScalarTensorBase<S2, Ix1>>
    for ScalarTensorBase<S1, Ix2>
{
    fn accuracy(&self, target: ScalarTensorBase<S2, Ix1>) -> Result<usize> {
        let device = self.device();
        if device.is_host() && target.device().is_host() {
            macro_for!($T1 in [bf16, f32] {
                if self.scalar_type() == $T1::scalar_type() {
                    macro_for!($T2 in [u8, u16, u32] {
                        if target.scalar_type() == $T2::scalar_type() {
                            let input = self.view().try_into_tensor_view::<$T1>().unwrap();
                            let target = target.view().try_into_tensor_view::<$T2>().unwrap();
                            return input.accuracy(target);
                        }
                    });
                }
            });
            bail!(
                "Accuracy {:?} {:?} not implemented!",
                self.scalar_type(),
                target.scalar_type()
            );
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            let (batch_size, classes) = self.dim();
            macro_for!($T1 in [bf16, f32] {
                macro_for!($T2 in [u8, u16, u32] {
                    if self.scalar_type() == $T1::scalar_type() && target.scalar_type() == $T2::scalar_type() {
                        let input = Slice::<$T1>::try_from(self.as_scalar_slice().unwrap()).unwrap();
                        let target = Slice::<$T2>::try_from(target.as_scalar_slice().unwrap()).unwrap();
                        let mut output = unsafe { Tensor::<u32, _>::uninit(input.device(), batch_size)? };
                        paste! {
                            kernels::[<accuracy_ $T1 _ $T2>]::builder()?
                                .build(device)?
                                .dispatch(
                                    input,
                                    target,
                                    classes.to_u32().unwrap(),
                                    output.as_slice_mut().unwrap(),
                                )?;
                        }
                        return output.sum().map(|x| x.try_into().unwrap());
                    }
                });
            });
            bail!(
                "Accuracy {:?} {:?} not implemented!",
                self.scalar_type(),
                target.scalar_type()
            );
        }
    }
}

/// Cross Entropy Loss.
pub trait CrossEntropyLoss<T> {
    /// Type of the output.
    type Output;
    /// Computes the loss given `target`.
    fn cross_entropy_loss(&self, target: T) -> Result<Self::Output>;
}

/// Implemented for:
/// - input: bf16, f32
/// - target: u8, u16, u32
impl<S1: ScalarData, S2: ScalarData> CrossEntropyLoss<ScalarTensorBase<S2, Ix1>>
    for ScalarTensorBase<S1, Ix2>
{
    type Output = f32;
    fn cross_entropy_loss(&self, target: ScalarTensorBase<S2, Ix1>) -> Result<Self::Output> {
        macro_for!($T1 in [bf16, f32] {
            if let Ok(input) = TensorView2::<$T1>::try_from(self.view()) {
                macro_for!($T2 in [u8, u16, u32] {
                    if let Ok(target) = TensorView1::<$T2>::try_from(target.view()) {
                        return input.cross_entropy_loss(target).map(Into::into);
                    }
                });
            }
        });
        bail!(
            "CrosEntropyLoss {:?} {:?} unimplemented!",
            self.scalar_type(),
            target.scalar_type()
        )
    }
}

/// Implemented for:
/// - input: bf16, f32
/// - target: u8, u16, u32
impl<T1: Scalar + Float, S1: Data<Elem = T1>, T2: Scalar + Unsigned, S2: Data<Elem = T2>>
    CrossEntropyLoss<TensorBase<S2, Ix1>> for TensorBase<S1, Ix2>
{
    type Output = f32;
    fn cross_entropy_loss(&self, target: TensorBase<S2, Ix1>) -> Result<Self::Output> {
        if let Some((input, target)) = self.as_array().zip(target.as_array()) {
            Ok(cross_entropy_loss_host(input, target))
        } else {
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
            #[cfg(feature = "device")]
            {
                cross_entropy_loss_device(self.view().into(), target.view().into())
            }
        }
    }
}

fn cross_entropy_loss_host<T1: Scalar + Float, T2: Scalar + Unsigned>(
    input: ArrayView2<T1>,
    target: ArrayView1<T2>,
) -> f32 {
    let x = input;
    let t = target;
    let mut y = 0f32;
    for (x, t) in x.outer_iter().zip(t.iter().copied()) {
        let m = x
            .iter()
            .map(|x| x.cast::<f32>())
            .fold(x[0].cast::<f32>(), f32::max);
        let s = x
            .iter()
            .copied()
            .map(|x| (x.cast::<f32>() - m).exp())
            .sum::<f32>();
        let x = x[t.to_usize().unwrap()];
        y += s.ln() - (x.cast::<f32>() - m);
    }
    y
}

#[cfg(feature = "device")]
fn cross_entropy_loss_device(input: ScalarTensorView2, target: ScalarTensorView1) -> Result<f32> {
    macro_for!($T1 in [bf16, f32] {
        if let Ok(input) = TensorView2::<$T1>::try_from(input.view()) {
            let (batch_size, classes) = input.dim();
            let input = input.as_slice().unwrap();
            macro_for!($T2 in [u8, u16, u32] {
                if let Ok(target) = TensorView1::<$T2>::try_from(target.view()) {
                    let target = target.as_slice().unwrap();
                    let mut output = unsafe { Tensor::<f32, _>::uninit(input.device(), batch_size)? };
                    let classes = classes.to_u32().unwrap();
                    let kernel = paste! {
                        kernels::[<cross_entropy_loss_ $T1 _ $T2>]::builder()?
                        .build(output.device())?
                    };
                    kernel.dispatch(input, target, classes, output.as_slice_mut().unwrap())?;
                    return output.sum();
                }
            });
        }
    });
    bail!(
        "CrossEntropyLoss {:?} {:?} unimplemented!",
        input.scalar_type(),
        target.scalar_type()
    )
}

#[cfg(feature = "device")]
#[module]
mod kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{half::bf16, num_traits::Float, scalar::Scalar};
    use paste::paste;

    macro_for!($T1 in [bf16, f32] {
        macro_for!($T2 in [u8, u16, u32] {
            paste! {
                #[kernel]
                pub fn [<accuracy_ $T1 _ $T2>](
                    #[global] x: Slice<$T1>,
                    #[global] t: Slice<$T2>,
                    classes: u32,
                    #[item] y: &mut u32,
                ) {
                    let classes = classes as usize;
                    let idx = kernel.item_id as usize;
                    let t = t[idx] as usize;
                    if t > classes {
                        *y = 0;
                        return;
                    }
                    let xt = x[idx * classes + t];
                    for i in 0..classes {
                        if i == t {
                            continue;
                        }
                        let x = x[idx * classes + i];
                        if !(xt > x) {
                            *y = 0;
                            return;
                        }
                    }
                    *y = 1;
                }

                #[kernel]
                pub fn [<cross_entropy_loss_ $T1 _ $T2>](
                    #[global] x: Slice<$T1>,
                    #[global] t: Slice<$T2>,
                    classes: u32,
                    #[item] y: &mut f32,
                ) {
                    let classes = classes as usize;
                    let idx = kernel.item_id as usize;
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
                    let t = t[idx as usize] as usize;
                    let x = x[idx * classes + t].cast::<f32>();
                    *y = s.ln() - (x - m);
                }
            }
        });
    });
}
