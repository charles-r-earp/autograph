//!
/*!
# Examples
A [LeNet-5](<http://yann.lecun.com/exdb/lenet/>) network might look like this:
```
use autograph::learn::neural_network::layer::{Layer, Forward, Conv, Dense, Relu, MeanPool};

#[derive(Layer, Forward)]
struct Lenet5 {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    pool1: MeanPool,
    #[autograph(layer)]
    conv2: Conv,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    pool2: MeanPool,
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
