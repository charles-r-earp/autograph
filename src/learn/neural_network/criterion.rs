use super::autograd::{Variable, Variable0};
use crate::{
    learn::criterion::{Criterion, CrossEntropyLoss},
    result::Result,
    scalar::Uint,
    tensor::{Data, TensorBase},
};
use anyhow::anyhow;
use ndarray::RemoveAxis;

impl<T: Uint, S: Data<Elem = T>, D: RemoveAxis> Criterion<Variable<D>, TensorBase<S, D::Smaller>>
    for CrossEntropyLoss
{
    type Output = Variable0;
    fn loss(
        &self,
        prediction: Variable<D>,
        target: TensorBase<S, D::Smaller>,
    ) -> Result<Variable0> {
        let nclasses = prediction
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| anyhow!("Expected  at least 2 dimensions!"))?;
        let target = target
            .into_dimensionality()?
            .to_one_hot_float(prediction.float_type(), nclasses)?;
        prediction
            .into_dimensionality()?
            .cross_entropy_loss(target.into())
    }
}
