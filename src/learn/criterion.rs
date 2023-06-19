#[cfg(feature = "device")]
use crate::{
    buffer::Buffer,
    tensor::{ScalarTensorView1, ScalarTensorView2},
};
use crate::{
    buffer::{Data, ScalarData},
    scalar::Scalar,
    tensor::{ScalarTensorBase, TensorBase, TensorView1, TensorView2},
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

pub trait Criterion<X, T> {
    type Output;
    fn eval(&self, input: X, target: T) -> Result<Self::Output>;
}

// Accuracy.
#[derive(Default, Debug)]
pub struct Accuracy;

impl<T1: Scalar, S1: ArrayData<Elem = T1>, T2: Scalar + Unsigned, S2: ArrayData<Elem = T2>>
    Criterion<ArrayBase<S1, Ix2>, ArrayBase<S2, Ix1>> for Accuracy
{
    type Output = usize;
    fn eval(&self, input: ArrayBase<S1, Ix2>, target: ArrayBase<S2, Ix1>) -> Result<Self::Output> {
        Ok(accuracy_host(input.view(), target.view()))
    }
}

fn accuracy_host<X: Scalar, T: Scalar>(input: ArrayView2<X>, target: ArrayView1<T>) -> usize {
    let mut correct = 0;
    for (x, t) in input
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
    correct
}

impl<T1: Scalar, S1: Data<Elem = T1>, T2: Scalar + Unsigned, S2: Data<Elem = T2>>
    Criterion<TensorBase<S1, Ix2>, TensorBase<S2, Ix1>> for Accuracy
{
    type Output = usize;
    fn eval(
        &self,
        input: TensorBase<S1, Ix2>,
        target: TensorBase<S2, Ix1>,
    ) -> Result<Self::Output> {
        if let Some((input, target)) = input.as_array().zip(target.as_array()) {
            return Ok(accuracy_host(input, target));
        }
        todo!()
    }
}

impl<S1: ScalarData, S2: ScalarData> Criterion<ScalarTensorBase<S1, Ix2>, ScalarTensorBase<S2, Ix1>>
    for Accuracy
{
    type Output = usize;
    fn eval(
        &self,
        input: ScalarTensorBase<S1, Ix2>,
        target: ScalarTensorBase<S2, Ix1>,
    ) -> Result<Self::Output> {
        /*use crate::device::Device;
        let input = input.to_device(Device::host())?;
        let target = target.to_device(Device::host())?;*/
        if input.device().is_host() && target.device().is_host() {
            macro_for!($T1 in [bf16, f32] {
                if input.scalar_type() == $T1::scalar_type() {
                    macro_for!($T2 in [u8, u16, u32] {
                        if target.scalar_type() == $T2::scalar_type() {
                            let input = input.view().try_into_tensor_view::<$T1>().unwrap();
                            let target = target.view().try_into_tensor_view::<$T2>().unwrap();
                            return self.eval(input, target);
                        }
                    });
                }
            });
            bail!(
                "Accuracy {:?} {:?} not implemented!",
                input.scalar_type(),
                target.scalar_type()
            );
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            let input = input.view().try_into_tensor_view::<f32>().unwrap();
            let target = target.view().try_into_tensor_view::<u8>().unwrap();
            let (batch_size, classes) = input.dim();
            let mut output = unsafe { Buffer::<u8>::uninit(input.device(), batch_size)? };
            kernels::accuracy_f32_u8_u8::builder()?
                .build(input.device())?
                .dispatch(
                    input.as_slice().unwrap(),
                    target.as_slice().unwrap(),
                    classes.to_u32().unwrap(),
                    output.as_slice_mut(),
                )?;
            let count = output.to_vec()?.iter().map(|x| x.to_usize().unwrap()).sum();
            Ok(count)
        }
    }
}

/// Cross entropy loss.
#[derive(Default, Debug)]
pub struct CrossEntropyLoss {}

impl<S1: ScalarData, S2: ScalarData> Criterion<ScalarTensorBase<S1, Ix2>, ScalarTensorBase<S2, Ix1>>
    for CrossEntropyLoss
{
    type Output = f32;
    fn eval(
        &self,
        input: ScalarTensorBase<S1, Ix2>,
        target: ScalarTensorBase<S2, Ix1>,
    ) -> Result<Self::Output> {
        macro_for!($T1 in [bf16, f32] {
            if let Ok(input) = TensorView2::<$T1>::try_from(input.view()) {
                macro_for!($T2 in [u8, u16, u32] {
                    if let Ok(target) = TensorView1::<$T2>::try_from(target.view()) {
                        return self.eval(input, target).map(Into::into);
                    }
                });
            }
        });
        bail!(
            "CrosEntropyLoss {:?} {:?} unimplemented!",
            input.scalar_type(),
            target.scalar_type()
        )
    }
}

impl<T1: Scalar + Float, S1: Data<Elem = T1>, T2: Scalar + Unsigned, S2: Data<Elem = T2>>
    Criterion<TensorBase<S1, Ix2>, TensorBase<S2, Ix1>> for CrossEntropyLoss
{
    type Output = f32;
    fn eval(
        &self,
        input: TensorBase<S1, Ix2>,
        target: TensorBase<S2, Ix1>,
    ) -> Result<Self::Output> {
        if let Some((input, target)) = input.as_array().zip(target.as_array()) {
            Ok(cross_entropy_loss_host(input, target))
        } else {
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
            #[cfg(feature = "device")]
            {
                cross_entropy_loss_device(input.view().into(), target.view().into())
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
    let input = input.try_into_tensor_view::<f32>().unwrap();
    let (batch_size, classes) = input.dim();
    let input = input.as_slice().unwrap();
    let target = target.try_into_tensor_view::<u8>().unwrap();
    let target = target.as_slice().unwrap();
    let mut output = unsafe { Buffer::uninit(input.device(), batch_size)? };
    let classes = classes.to_u32().unwrap();
    kernels::cross_entropy_loss_f32_u8::builder()?
        .build(output.device())?
        .dispatch(input, target, classes, output.as_slice_mut())?;
    Ok(output.to_vec()?.iter().copied().sum())
}

#[cfg(feature = "device")]
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::num_traits::Float;

    #[kernel(threads(256))]
    pub fn accuracy_f32_u8_u8(
        #[global] x: Slice<f32>,
        #[global] t: Slice<u8>,
        classes: u32,
        #[item] y: &mut u8,
    ) {
        let classes = classes as usize;
        let idx = kernel.item_id() as usize;
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

    #[kernel(threads(256))]
    pub fn cross_entropy_loss_f32_u8(
        #[global] x: Slice<f32>,
        #[global] t: Slice<u8>,
        classes: u32,
        #[item] y: &mut f32,
    ) {
        let idx = kernel.item_id();
        let mut m = x[(idx * classes) as usize];
        for i in 1..classes {
            let x = x[(idx * classes + i) as usize];
            m = m.max(x);
        }
        let mut s = 0f32;
        for i in 0..classes {
            let x = x[(idx * classes + i) as usize];
            s += (x - m).exp();
        }
        let t = t[idx as usize] as u32;
        let x = x[(idx * classes + t) as usize];
        *y = s.ln() - (x - m);
    }
}
