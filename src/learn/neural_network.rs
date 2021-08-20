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
use super::Infer;
use crate::{
    float_tensor::{FloatData, FloatTensorBase, FloatTensorD},
    result::Result,
};
use ndarray::Dimension;

/// Variables and Parameters
pub mod autograd;

/// Layers
pub mod layer;
use layer::Layer;

/// Optimizers
pub mod optimizer;

/// A neural network.
///
/// Provides an [`Infer`] implementation for [`Layer`]'s.
pub struct Network<L>(pub L);

impl<L: Layer, S: FloatData, D: Dimension> Infer<FloatTensorBase<S, D>> for Network<L> {
    fn infer(&self, input: &FloatTensorBase<S, D>) -> Result<FloatTensorD> {
        self.0
            .forward(input.to_shared()?.into_dyn().into())?
            .into_value()
            .into_owned()
    }
}
