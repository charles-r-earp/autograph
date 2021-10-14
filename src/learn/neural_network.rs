//!
/*!
# Examples
A [LeNet-5](<http://yann.lecun.com/exdb/lenet/>) network might look like this:
```no_run
use autograph::{
    result::Result,
    learn::neural_network::layer::{Layer, Forward, Conv, Dense, Relu, MaxPool}
};
use serde::{Serialize, Deserialize};

#[derive(Layer, Forward, Clone, Serialize, Deserialize, Debug)]
struct Lenet5 {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    pool1: MaxPool,
    #[autograph(layer)]
    conv2: Conv,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    pool2: MaxPool,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu3: Relu,
    #[autograph(layer)]
    dense2: Dense,
    #[autograph(layer)]
    relu4: Relu,
    #[autograph(layer)]
    dense3: Dense,
}

impl Lenet5 {
    fn new() -> Result<Self> {
        let conv1 = Conv::from_inputs_outputs_kernel(1, 6, [5, 5]);
        let relu1 = Relu::default();
        let pool1 = MaxPool::from_kernel([2, 2])
            .with_strides(2)?;
        let conv2 = Conv::from_inputs_outputs_kernel(6, 16, [5, 5]);
        let relu2 = Relu::default();
        let pool2 = MaxPool::from_kernel([2, 2])
            .with_strides(2)?;
        let dense1 = Dense::from_inputs_outputs(256, 120);
        let relu3 = Relu::default();
        let dense2 = Dense::from_inputs_outputs(120, 84);
        let relu4 = Relu::default();
        let dense3 = Dense::from_inputs_outputs(84, 10)
            .with_bias(true);
        Ok(Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            dense1,
            relu3,
            dense2,
            relu4,
            dense3,
        })
    }
}
```
*/
use super::{Infer, Stats, Summarize, Summary, Test, Train};
use crate::{
    learn::criterion::{Criterion, CrossEntropyLoss},
    ops::AddAssign,
    result::Result,
    scalar::Uint,
    tensor::{
        float::{FloatData, FloatTensor0, FloatTensorBase, FloatTensorD},
        Data, Tensor0, TensorBase,
    },
};
use ndarray::{Axis, Dimension};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// Variables and Parameters
pub mod autograd;
use autograd::{Variable, Variable0};

mod criterion;

/// Layers
pub mod layer;
use layer::Layer;

/// Optimizers
pub mod optimizer;
use optimizer::{Optimizer, Sgd};

/// A neural network.
///
/// Provides an [`Infer`] implementation for [`Layer`]'s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network<L>(L);

impl<L> Network<L> {
    /// Returns the layer.
    pub fn into_inner(self) -> L {
        self.0
    }
}

impl<L> From<L> for Network<L> {
    fn from(layer: L) -> Self {
        Self(layer)
    }
}

impl<L: Layer, O> From<NetworkTrainer<L, O>> for Network<L> {
    fn from(trainer: NetworkTrainer<L, O>) -> Self {
        let mut network = trainer.network;
        for parameter in network.parameters_mut() {
            parameter.require_grad_mut(false);
        }
        network
    }
}

impl<L> Deref for Network<L> {
    type Target = L;
    fn deref(&self) -> &L {
        &self.0
    }
}

impl<L> DerefMut for Network<L> {
    fn deref_mut(&mut self) -> &mut L {
        &mut self.0
    }
}

impl<L: Layer, S: FloatData, D: Dimension> Infer<FloatTensorBase<S, D>> for Network<L> {
    fn infer(&self, input: &FloatTensorBase<S, D>) -> Result<FloatTensorD> {
        self.0
            .forward(input.to_shared()?.into_dyn().into())?
            .into_value()
            .into_owned()
    }
}

/// A neural network trainer.
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkTrainer<L, C = CrossEntropyLoss, O = Sgd> {
    network: Network<L>,
    criterion: C,
    optimizer: O,
    summary: Summary,
}

impl<L, C, O> NetworkTrainer<L, C, O> {
    /// Sets the criterion.
    pub fn with_criterion<C2>(self, criterion: C2) -> NetworkTrainer<L, C2, O> {
        NetworkTrainer {
            network: self.network,
            criterion,
            optimizer: self.optimizer,
            summary: self.summary,
        }
    }
    /// Sets the optimizer.
    pub fn with_optimizer<O2>(self, optimizer: O2) -> NetworkTrainer<L, C, O2> {
        NetworkTrainer {
            network: self.network,
            criterion: self.criterion,
            optimizer,
            summary: self.summary,
        }
    }
}

impl<L: Layer> From<Network<L>> for NetworkTrainer<L> {
    fn from(mut network: Network<L>) -> Self {
        for parameter in network.parameters_mut() {
            parameter.require_grad_mut(true);
        }
        Self {
            network,
            criterion: CrossEntropyLoss::default(),
            optimizer: Sgd::default(),
            summary: Summary::default(),
        }
    }
}

impl<L, C, O> Summarize for NetworkTrainer<L, C, O> {
    fn summarize(&self) -> Summary {
        self.summary.clone()
    }
}

impl<L, C, O> Deref for NetworkTrainer<L, C, O> {
    type Target = Network<L>;
    fn deref(&self) -> &Self::Target {
        &self.network
    }
}

impl<
        L: Layer,
        O,
        S1: FloatData,
        D1: Dimension,
        U: Uint,
        S2: Data<Elem = U>,
        D2: Dimension,
        C: Criterion<Variable<D2::Larger>, TensorBase<S2, D2>, Output = Variable0>,
    > Test<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)> for NetworkTrainer<L, C, O>
{
    fn test<I>(&self, test_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = Result<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)>>,
    {
        smol::block_on(async {
            let mut count = 0;
            let mut loss: Option<FloatTensor0> = None;
            let mut correct: Option<Tensor0<u32>> = None;
            for x in test_iter {
                let (x, t) = x?;
                count += x.shape().first().copied().unwrap_or(0);
                let y = self.forward(x.into_shared()?.into_dyn().into())?;
                let pred = y.value().argmax_axis(Axis(1))?.into_dimensionality()?;
                if let Some(correct) = correct.as_mut() {
                    pred.accuracy_with(&t.view(), &mut correct.view_mut())?;
                } else {
                    correct.replace(pred.accuracy(&t.view())?);
                };
                let batch_loss = self.criterion.eval(y.into_dimensionality()?, t)?;
                if let Some(loss) = loss.as_mut() {
                    loss.add_assign(batch_loss.value())?;
                } else {
                    loss.replace(batch_loss.value().view().into_owned()?);
                }
            }
            let loss_fut = if let Some(loss) = loss {
                Some(loss.scale_into(1. / count as f32)?.read())
            } else {
                None
            };
            let correct_fut = correct.map(|correct| correct.read());

            let loss = if let Some(loss_fut) = loss_fut {
                loss_fut.await?.as_array().first().copied()
            } else {
                None
            };
            let correct = if let Some(correct_fut) = correct_fut {
                correct_fut
                    .await?
                    .as_array()
                    .first()
                    .map(|correct| *correct as usize)
            } else {
                None
            };

            Ok(Stats {
                count,
                loss,
                correct,
            })
        })
    }
}

impl<
        L: Layer,
        O: Optimizer,
        S1: FloatData,
        D1: Dimension,
        U: Uint,
        S2: Data<Elem = U>,
        D2: Dimension,
        C: Criterion<Variable<D2::Larger>, TensorBase<S2, D2>, Output = Variable0>,
    > Train<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)> for NetworkTrainer<L, C, O>
{
    fn init<F, I>(&mut self, _: F) -> Result<()>
    where
        F: FnOnce(usize) -> I,
        I: IntoIterator,
        <I as IntoIterator>::Item:
            IntoIterator<Item = Result<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)>>,
    {
        self.summary = Summary::default();
        Ok(())
    }
    fn train_test<I1, I2>(&mut self, train_iter: I1, test_iter: I2) -> Result<(Stats, Stats)>
    where
        I1: IntoIterator<Item = Result<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)>>,
        I2: IntoIterator<Item = Result<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)>>,
    {
        let network = &mut self.network;
        let criterion = &self.criterion;
        let optimizer = &mut self.optimizer;
        self.summary.run_epoch(|_| {
            smol::block_on(async {
                let mut train_count = 0;
                let mut train_loss: Option<FloatTensor0> = None;
                let mut train_correct: Option<Tensor0<u32>> = None;
                let mut train_iter = train_iter.into_iter();
                while let Some((x, t)) = train_iter.next().transpose()? {
                    train_count += x.shape().first().copied().unwrap_or(0);
                    let x = Variable::from(x.into_shared()?.into_dyn()).with_training(true);
                    let y = network.forward(x)?;
                    let pred = y.value().argmax_axis(Axis(1))?.into_dimensionality()?;
                    if let Some(train_correct) = train_correct.as_mut() {
                        pred.accuracy_with(&t.view(), &mut train_correct.view_mut())?;
                    } else {
                        train_correct.replace(pred.accuracy(&t.view())?);
                    };
                    let batch_loss = criterion.eval(y.into_dimensionality()?, t)?;
                    if let Some(train_loss) = train_loss.as_mut() {
                        train_loss.add_assign(batch_loss.value())?;
                    } else {
                        train_loss.replace(batch_loss.value().view().into_owned()?);
                    }
                    batch_loss.backward()?;
                    network.update(optimizer)?;
                }

                let mut test_count = 0;
                let mut test_loss: Option<FloatTensor0> = None;
                let mut test_correct: Option<Tensor0<u32>> = None;
                let mut test_iter = test_iter.into_iter();
                while let Some((x, t)) = test_iter.next().transpose()? {
                    test_count += x.shape().first().copied().unwrap_or(0);
                    let y = network.forward(x.into_shared()?.into_dyn().into())?;
                    let pred = y.value().argmax_axis(Axis(1))?.into_dimensionality()?;
                    if let Some(test_correct) = test_correct.as_mut() {
                        pred.accuracy_with(&t.view(), &mut test_correct.view_mut())?;
                    } else {
                        test_correct.replace(pred.accuracy(&t.view())?);
                    };
                    let batch_loss = criterion.eval(y.into_dimensionality()?, t)?;
                    if let Some(test_loss) = test_loss.as_mut() {
                        test_loss.add_assign(batch_loss.value())?;
                    } else {
                        test_loss.replace(batch_loss.value().view().into_owned()?);
                    }
                }

                let train_loss_fut = if let Some(train_loss) = train_loss {
                    Some(train_loss.scale_into(1. / train_count as f32)?.read())
                } else {
                    None
                };
                let train_correct_fut = train_correct.map(|train_correct| train_correct.read());
                let test_loss_fut = if let Some(test_loss) = test_loss {
                    Some(test_loss.scale_into(1. / test_count as f32)?.read())
                } else {
                    None
                };
                let test_correct_fut = test_correct.map(|test_correct| test_correct.read());

                let train_loss = if let Some(train_loss_fut) = train_loss_fut {
                    train_loss_fut.await?.as_array().first().copied()
                } else {
                    None
                };
                let train_correct = if let Some(train_correct_fut) = train_correct_fut {
                    train_correct_fut
                        .await?
                        .as_array()
                        .first()
                        .map(|correct| *correct as usize)
                } else {
                    None
                };
                let test_loss = if let Some(test_loss_fut) = test_loss_fut {
                    test_loss_fut.await?.as_array().first().copied()
                } else {
                    None
                };
                let test_correct = if let Some(test_correct_fut) = test_correct_fut {
                    test_correct_fut
                        .await?
                        .as_array()
                        .first()
                        .map(|correct| *correct as usize)
                } else {
                    None
                };

                let train_stats = Stats {
                    count: train_count,
                    loss: train_loss,
                    correct: train_correct,
                };
                let test_stats = Stats {
                    count: test_count,
                    loss: test_loss,
                    correct: test_correct,
                };
                Ok((train_stats, test_stats))
            })
        })
    }
}
