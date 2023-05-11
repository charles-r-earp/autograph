use crate::{
    buffer::{Data, ScalarData},
    scalar::{Scalar, ScalarType},
    tensor::{ScalarTensor, ScalarTensorBase, ScalarTensorView, Tensor, TensorBase, TensorView},
};
use anyhow::{bail, Error, Result};
use dry::macro_for;
use half::{bf16, f16};
use ndarray::{
    Array, Array2, ArrayBase, ArrayView1, ArrayView2, Data as ArrayData, Dimension, Ix1, Ix2,
    RemoveAxis,
};
use num_traits::{Float, Unsigned};
use paste::paste;

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
    input
        .outer_iter()
        .zip(target.iter().map(|x| x.to_usize().unwrap()))
        .filter(|(input, class)| {
            let mut max = input[0];
            let mut max_index = 0;
            for (i, x) in input.iter().copied().enumerate() {
                if x > max {
                    max = x;
                    max_index = i;
                }
            }
            max_index == *class
        })
        .count()
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
        todo!()
    }
}

/// Cross entropy loss.
#[derive(Default, Debug)]
pub struct CrossEntropyLoss {}

impl<S1: ScalarData, S2: ScalarData, D: Dimension>
    Criterion<ScalarTensorBase<S1, D>, ScalarTensorBase<S2, D>> for CrossEntropyLoss
{
    type Output = ScalarTensor<D>;
    fn eval(
        &self,
        input: ScalarTensorBase<S1, D>,
        target: ScalarTensorBase<S2, D>,
    ) -> Result<Self::Output> {
        macro_for!($T in [bf16, f32] {
            if let Some((input, target)) = TensorView::<$T, D>::try_from(input.view())
                .ok()
                .zip(TensorView::<$T, D>::try_from(target.view()).ok())
            {
                return self.eval(input, target).map(Into::into);
            }
        });
        bail!(
            "CrosEntropyLoss {:?} {:?} unimplemented!",
            input.scalar_type(),
            target.scalar_type()
        )
    }
}

impl<T: Scalar + Float, S1: Data<Elem = T>, S2: Data<Elem = T>, D: Dimension>
    Criterion<TensorBase<S1, D>, TensorBase<S2, D>> for CrossEntropyLoss
{
    type Output = Tensor<T, D>;
    fn eval(&self, input: TensorBase<S1, D>, target: TensorBase<S2, D>) -> Result<Self::Output> {
        let input = input.into_dimensionality().map_err(Error::msg)?;
        let target = target.into_dimensionality().map_err(Error::msg)?;
        if let Some((input, target)) = input.as_array().zip(target.as_array()) {
            return Ok(cross_entropy_loss_host(input, target)
                .into_dimensionality()
                .map_err(Error::msg)?
                .into());
        }
        todo!()
    }
}

fn cross_entropy_loss_host<T: Scalar + Float>(
    input: ArrayView2<T>,
    target: ArrayView2<T>,
) -> Array2<T> {
    let x = input;
    let t = target;
    let mut y = Array2::<T>::zeros(x.raw_dim());
    for (mut y, (x, t)) in y.outer_iter_mut().zip(x.outer_iter().zip(t.outer_iter())) {
        let m = x
            .iter()
            .map(|x| x.cast::<f32>())
            .fold(x[0].cast::<f32>(), |m, x| if x > m { x } else { m });
        let s = x
            .iter()
            .copied()
            .map(|x| (x.cast::<f32>() - m).exp())
            .sum::<f32>();
        for (y, (x, t)) in y.iter_mut().zip(x.iter().zip(t.iter().copied())) {
            *y = ((s.ln() - (x.cast::<f32>() - m)) * t.cast::<f32>()).cast();
        }
    }
    y
}
