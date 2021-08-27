//!
/*!
# Examples
A [LeNet-5](<http://yann.lecun.com/exdb/lenet/>) network might look like this:
```no_run
use autograph::{
    result::Result,
    learn::neural_network::layer::{Layer, Forward, Conv, Dense, Relu, MaxPool}
};

#[derive(Layer, Forward, Clone, Debug)]
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
            .with_bias(true)?;
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
    device::Device,
    result::Result,
    scalar::{FloatType, Uint},
    tensor::{
        float::{FloatData, FloatTensor, FloatTensorBase, FloatTensorD},
        Data, Tensor, TensorBase,
    },
};
use ndarray::{Axis, Dimension};
use std::ops::{Deref, DerefMut};

/// Variables and Parameters
pub mod autograd;
use autograd::Variable;

/// Layers
pub mod layer;
use layer::Layer;

/// Optimizers
pub mod optimizer;
use optimizer::{Optimizer, Sgd};

/// A neural network.
///
/// Provides an [`Infer`] implementation for [`Layer`]'s.
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

impl<L, O> From<NetworkTrainer<L, O>> for Network<L> {
    fn from(trainer: NetworkTrainer<L, O>) -> Self {
        trainer.network
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

impl<L: Layer, S1: FloatData, D1: Dimension, U: Uint, S2: Data<Elem = U>, D2: Dimension>
    Test<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)> for Network<L>
{
    fn test<I>(&self, test_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = Result<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)>>,
    {
        smol::block_on(async {
            let mut count = 0;
            let mut loss = FloatTensor::zeros(FloatType::F32, Device::host(), ()).unwrap();
            for x in test_iter {
                let (x, t) = x?;
                count += x.shape().first().copied().unwrap_or(0);
                let y = self.forward(x.into_shared()?.into_dyn().into())?;
                let nclasses = y.shape().get(1).copied().unwrap_or(1);
                let t = t
                    .into_dimensionality()?
                    .to_one_hot_float(y.float_type(), nclasses)?
                    .into_dyn();
                if loss.device() == Device::host() {
                    loss = FloatTensor::zeros(y.float_type(), y.device(), ())?;
                }
                y.cross_entropy_loss(t.into())?
                    .value()
                    .sum_with(&mut loss.view_mut())?;
            }
            let loss = loss
                .scale_into(1. / count as f32)?
                .read()
                .await?
                .as_array()
                .as_slice()
                .unwrap()[0];
            Ok(Stats {
                count,
                loss: Some(loss),
                correct: None,
            })
        })
    }
}

/// A neural network trainer.
///
/// Loss Function:
/// - Classification: Cross Entroy Loss
pub struct NetworkTrainer<L, O = Sgd> {
    network: Network<L>,
    optimizer: O,
    summary: Summary,
}

impl<L, O> NetworkTrainer<L, O> {
    /// Sets the optimizer.
    pub fn with_optimizer<O2>(self, optimizer: O2) -> NetworkTrainer<L, O2> {
        NetworkTrainer {
            network: self.network,
            optimizer,
            summary: self.summary,
        }
    }
}

impl<L: Layer> From<Network<L>> for NetworkTrainer<L, Sgd> {
    fn from(mut network: Network<L>) -> Self {
        for parameter in network.parameters_mut() {
            parameter.require_grad_mut(true);
        }
        Self {
            network,
            optimizer: Sgd::default(),
            summary: Summary::default(),
        }
    }
}

impl<L, O> Summarize for NetworkTrainer<L, O> {
    fn summarize(&self) -> Summary {
        self.summary.clone()
    }
}

impl<L, O> Deref for NetworkTrainer<L, O> {
    type Target = Network<L>;
    fn deref(&self) -> &Self::Target {
        &self.network
    }
}

impl<L: Layer, O, S1: FloatData, D1: Dimension, U: Uint, S2: Data<Elem = U>, D2: Dimension>
    Test<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)> for NetworkTrainer<L, O>
{
    fn test<I>(&self, test_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = Result<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)>>,
    {
        self.network.test(test_iter)
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
    > Train<(FloatTensorBase<S1, D1>, TensorBase<S2, D2>)> for NetworkTrainer<L, O>
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
        let optimizer = &mut self.optimizer;
        self.summary.run_epoch(|_| {
            smol::block_on(async {
                let mut train_count = 0;
                let mut train_loss =
                    FloatTensor::zeros(FloatType::F32, Device::host(), ()).unwrap();
                let mut train_correct = Tensor::zeros(Device::host(), ()).unwrap();
                for x in train_iter {
                    let (x, t) = x?;
                    train_count += x.shape().first().copied().unwrap_or(0);
                    let mut loss = {
                        let x = Variable::from(x.into_shared()?)
                            .into_dyn()
                            .with_name("input")?
                            .with_training(true);
                        let y = network.forward(x)?;
                        let nclasses = y.shape().get(1).copied().unwrap_or(1);
                        let t = t.into_dimensionality()?.into_device(y.device()).await?;
                        if train_correct.device() == Device::host() {
                            train_correct = Tensor::zeros(y.device(), ())?;
                        }
                        y.value()
                            .argmax_axis(Axis(1))?
                            .accuracy_with(&t.view(), &mut train_correct.view_mut())?;
                        let t = t
                            .into_dimensionality()?
                            .to_one_hot_float(y.float_type(), nclasses)?
                            .into_dyn();
                        if train_loss.device() == Device::host() {
                            train_loss = FloatTensor::zeros(y.float_type(), y.device(), ())?;
                        }
                        y.cross_entropy_loss(t.into())?
                    };
                    loss.backward()?;
                    loss.value().sum_with(&mut train_loss.view_mut())?;
                    network.update(optimizer)?;
                }
                let mut test_count = 0;
                let mut test_loss = FloatTensor::zeros(FloatType::F32, Device::host(), ()).unwrap();
                let mut test_correct = Tensor::zeros(Device::host(), ()).unwrap();
                for x in test_iter {
                    let (x, t) = x?;
                    test_count += x.shape().first().copied().unwrap_or(0);
                    let x = Variable::from(x.into_shared()?)
                        .into_dyn()
                        .with_name("input")?;
                    let y = network.forward(x)?;
                    let nclasses = y.shape().get(1).copied().unwrap_or(1);
                    let t = t.into_dimensionality()?.into_device(y.device()).await?;
                    if test_correct.device() == Device::host() {
                        test_correct = Tensor::zeros(y.device(), ())?;
                    }
                    y.value()
                        .argmax_axis(Axis(1))?
                        .accuracy_with(&t.view(), &mut test_correct.view_mut())?;
                    let t = t
                        .into_dimensionality()?
                        .into_device(y.device())
                        .await?
                        .to_one_hot_float(y.float_type(), nclasses)?
                        .into_dyn();
                    if test_loss.device() == Device::host() {
                        test_loss = FloatTensor::zeros(y.float_type(), y.device(), ())?;
                    }
                    y.cross_entropy_loss(t.into())?
                        .value()
                        .sum_with(&mut test_loss.view_mut())?;
                }
                let train_fut = if train_count > 0 {
                    Some(async {
                        let loss_fut = train_loss.scale_into(1. / train_count as f32)?.read();
                        let correct_fut = train_correct.read();
                        Ok((loss_fut.await?, correct_fut.await?))
                    })
                } else {
                    None
                };
                let test_fut = if test_count > 0 {
                    Some(async {
                        let loss_fut = test_loss.scale_into(1. / train_count as f32)?.read();
                        let correct_fut = test_correct.read();
                        Ok((loss_fut.await?, correct_fut.await?))
                    })
                } else {
                    None
                };
                let train_stats = if let Some(train_fut) = train_fut {
                    match train_fut.await {
                        Ok((train_loss, train_correct)) => {
                            let loss = train_loss.as_array().as_slice().unwrap().first().copied();
                            let correct = train_correct
                                .as_array()
                                .as_slice()
                                .unwrap()
                                .first()
                                .map(|x| *x as usize);
                            Stats {
                                count: train_count,
                                loss,
                                correct,
                            }
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                } else {
                    Stats::default()
                };
                let test_stats = if let Some(test_fut) = test_fut {
                    match test_fut.await {
                        Ok((test_loss, test_correct)) => {
                            let loss = test_loss.as_array().as_slice().unwrap().first().copied();
                            let correct = test_correct
                                .as_array()
                                .as_slice()
                                .unwrap()
                                .first()
                                .map(|x| *x as usize);
                            Stats {
                                count: test_count,
                                loss,
                                correct,
                            }
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                } else {
                    Stats::default()
                };
                Ok((train_stats, test_stats))
            })
        })
    }
}
