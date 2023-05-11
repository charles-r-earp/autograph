use super::autograd::Variable;
use crate::{
    buffer::{Data, ScalarData},
    learn::criterion::{Criterion, CrossEntropyLoss},
    tensor::{ArcTensor, ScalarArcTensor},
};
use anyhow::Result;
use ndarray::{Array, Array2, ArrayView2, Dimension};

impl<D: Dimension + 'static> Criterion<Variable<D>, ScalarArcTensor<D>> for CrossEntropyLoss {
    type Output = Variable<D>;
    fn eval(&self, input: Variable<D>, target: ScalarArcTensor<D>) -> Result<Variable<D>> {
        let mut builder = Variable::builder();
        if let Some(node) = input.node() {
            let input = input.value().clone();
            let target = target.clone();
            builder.edge(node, |output_grad| {
                let input = ArcTensor::<f32, D>::try_from(input).unwrap();
                let target = ArcTensor::try_from(target).unwrap();
                let input = input.as_array().unwrap().into_dimensionality().unwrap();
                let target = target.as_array().unwrap().into_dimensionality().unwrap();
                let output_grad = ArcTensor::<f32, D>::try_from(output_grad).unwrap();
                let output_grad = output_grad
                    .as_array()
                    .unwrap()
                    .into_dimensionality()
                    .unwrap();
                Ok(ArcTensor::from(
                    cross_entropy_loss_backward_host(input, target, output_grad)
                        .into_dimensionality()
                        .unwrap(),
                )
                .into())
            })
        }
        let value = self.eval(input.into_value(), target)?.into();
        Ok(builder.build(value))
    }
}

fn cross_entropy_loss_backward_host(
    x: ArrayView2<f32>,
    t: ArrayView2<f32>,
    dy: ArrayView2<f32>,
) -> Array2<f32> {
    let mut dx = Array::zeros(x.raw_dim());
    let scale = 1. / x.dim().0 as f32;
    for ((x, t), (dy, mut dx)) in x
        .outer_iter()
        .zip(t.outer_iter())
        .zip(dy.outer_iter().zip(dx.outer_iter_mut()))
    {
        let m = x.iter().copied().fold(x[0], |m, x| m.max(x));
        let s: f32 = x.iter().map(|x| (x - m).exp()).sum();
        for ((x, t), (dy, mut dx)) in x
            .iter()
            .copied()
            .zip(t.iter().copied())
            .zip(dy.iter().copied().zip(dx.iter_mut()))
        {
            *dx += scale * dy * ((x - m).exp() / s - t);
        }
    }
    dx
}
