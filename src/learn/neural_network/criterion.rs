use super::autograd::Variable;
use crate::{
    learn::criterion::{Criterion, CrossEntropyLoss},
    scalar::{Scalar, ScalarType},
    tensor::{ScalarArcTensor, Tensor2, TensorView2},
};
use anyhow::{bail, Error, Result};
use half::bf16;
use ndarray::{Array2, Dimension, Ix2};

impl<D: Dimension + 'static> Criterion<Variable<D>, ScalarArcTensor<D>> for CrossEntropyLoss {
    type Output = Variable<D>;
    fn eval(&self, input: Variable<D>, target: ScalarArcTensor<D>) -> Result<Variable<D>> {
        if input.scalar_type() != target.scalar_type() {
            bail!("Expected target {target:?} scalar_type to match input {input:?}!");
        }
        let scalar_type = input.scalar_type();
        if !matches!(scalar_type, ScalarType::BF16 | ScalarType::F32) {
            bail!("CrossEntropyLoss {scalar_type:?} unimplemented!");
        }
        let mut builder = Variable::<D>::builder();
        if let Some(node) = input.node() {
            let input = input
                .value()
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(Error::msg)?;
            let target = target
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(Error::msg)?;
            builder.edge(node, move |output_grad| {
                let output_grad = output_grad.into_dimensionality().map_err(Error::msg)?;
                if let Some(((x, t), dy)) = input
                    .view()
                    .try_into()
                    .ok()
                    .zip(target.view().try_into().ok())
                    .zip(output_grad.view().try_into().ok())
                {
                    cross_entropy_loss_backward::<bf16>(x, t, dy)?
                        .into_scalar_tensor()
                        .into_shared()
                        .unwrap()
                        .into_dimensionality::<D>()
                        .map_err(Error::msg)
                } else if let Some(((x, t), dy)) = input
                    .view()
                    .try_into()
                    .ok()
                    .zip(target.view().try_into().ok())
                    .zip(output_grad.view().try_into().ok())
                {
                    cross_entropy_loss_backward::<f32>(x, t, dy)?
                        .into_scalar_tensor()
                        .into_shared()
                        .unwrap()
                        .into_dimensionality::<D>()
                        .map_err(Error::msg)
                } else {
                    unreachable!()
                }
            })
        }
        let value = self
            .eval(input.into_value(), target)?
            .into_shared()
            .unwrap();
        Ok(builder.build(value))
    }
}

fn cross_entropy_loss_backward<T: Scalar>(
    x: TensorView2<T>,
    t: TensorView2<T>,
    dy: TensorView2<T>,
) -> Result<Tensor2<T>> {
    if let Some(((x, t), dy)) = x.as_array().zip(t.as_array()).zip(dy.as_array()) {
        let mut dx = Array2::<T>::zeros(x.raw_dim());
        let scale = 1. / x.dim().0 as f32;
        for ((x, t), (dy, mut dx)) in x
            .outer_iter()
            .zip(t.outer_iter())
            .zip(dy.outer_iter().zip(dx.outer_iter_mut()))
        {
            let x_iter = x.iter().map(|x| x.cast::<f32>());
            let m = x_iter
                .clone()
                .fold(x_iter.clone().next().unwrap_or_default(), |m, x| m.max(x));
            let s: f32 = x_iter.clone().map(|x| (x - m).exp()).sum();
            for ((x, t), (dy, dx)) in x_iter
                .zip(t.iter().copied())
                .zip(dy.iter().copied().zip(dx.iter_mut()))
            {
                let t = t.cast::<f32>();
                let dy = dy.cast::<f32>();
                *dx = (dx.cast::<f32>() + scale * dy * ((x - m).exp() / s - t)).cast();
            }
        }
        return Ok(dx.into());
    }
    todo!()
}
