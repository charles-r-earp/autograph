/*!

Neural networks can be constructed from [`layers`](neural_network::layer). Some layers, such as [`Conv`](neural_network::layer::Conv)
and [`Dense`](neural_network::layer::Dense), have parameters that can be trained. [`MaxPool`](neural_network::layer::MaxPool) is a functional
layer that applies the pooling function to its input, while [`Flatten`](neural_network::layer::Flatten) reshapes the input into 2 dimensions.
Activations like [`Relu`](neural_network::layer::Relu) are often applied after trainable layers.

Layers implement [`Layer`](neural_network::layer::Layer) which allows initializing training and access to the [`parameters`](neural_network::autograd::Parameter),
and [`Forward`](neural_network::layer::Forward) which applies a function to one or more [`variables`](neural_network::autograd::Variable).

[`Parameters`](neural_network::autograd::Parameter) store the trainable weights and biases of the network. They are composed of 3 parts, the value [`tensor`](crate::tensor),
the gradient tensor, and the [`state`](neural_network::optimizer::State) of the optimizer. Parameters can be converted to variables via
[`.to_variable()`](neural_network::autograd::Parameter::to_variable), so that they can be used in variable functions.

[`Variables`](neural_network::autograd::Variable) are the inputs, outputs, and parameters of the network. In addition to the [`tensor`](crate::tensor) value, variables may have a
[`node`](neural_network::autograd::Node) which stores the gradient. Each node can have [edges](neural_network::autograd::builder::VariableBuilder::edge) that compute the gradient of the input
given the output gradient.

Typically the forward pass will conclude with a loss function such as [`.cross_entropy_loss()`](criterion::CrossEntropyLoss), which
evaluates the model against the target, for example a set of correctly labeled images.

During training, nodes and edges form a backward graph, connecting the node of a single variable (ie the loss) to the parameter gradients.
[`.backward()`](neural_network::autograd::Node::backward) executes the backward pass, computing all of the gradients.

The [`optimizer`](neural_network::optimizer), such as [`SGD`](neural_network::optimizer::SGD), [`updates`](neural_network::optimizer::Optimizer::update) the parameters.

Example:
```
# use anyhow::Result;
# use autograph::{krnl::{device::Device, scalar::ScalarType}, tensor::{Tensor, ScalarArcTensor}};
# use autograph::learn::neural_network::{
#   autograd::{Variable2, Variable4},
#   layer::{Layer, Forward, Conv2, Dense, MaxPool2, Flatten, Relu},
#   optimizer::{Optimizer, SGD},
# };
# use autograph::learn::criterion::CrossEntropyLoss;
#[derive(Layer, Forward)]
#[autograph(forward(Variable4, Output=Variable2))]
struct LeNet5 {
    conv1: Conv2,
    relu1: Relu,
    pool1: MaxPool2,
    conv2: Conv2,
    relu2: Relu,
    pool2: MaxPool2,
    flatten: Flatten,
    dense1: Dense,
    relu3: Relu,
    dense2: Dense,
    relu4: Relu,
    dense3: Dense,
}

impl LeNet5 {
    fn new(device: Device, scalar_type: ScalarType) -> Result<Self> {
        let conv1 = Conv2::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(1)
            .outputs(6)
            .filter([5, 5])
            .build()?;
        let relu1 = Relu;
        let pool1 = MaxPool2::builder().filter([2, 2]).build();
        let conv2 = Conv2::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(6)
            .outputs(16)
            .filter([5, 5])
            .build()?;
        let relu2 = Relu;
        let pool2 = MaxPool2::builder().filter([2, 2]).build();
        let flatten = Flatten;
        let dense1 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(16 * 4 * 4)
            .outputs(128)
            .build()?;
        let relu3 = Relu;
        let dense2 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(128)
            .outputs(84)
            .build()?;
        let relu4 = Relu;
        let dense3 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(84)
            .outputs(10)
            .bias(true)
            .build()?;
        Ok(Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            flatten,
            dense1,
            relu3,
            dense2,
            relu4,
            dense3,
        })
    }
}

# fn main() -> Result<()> {
# let device = Device::host();
let mut model = LeNet5::new(device.clone(), ScalarType::F32)?;
# let x = Variable4::from(Tensor::<f32, _>::zeros(device.clone(), [1, 1, 28, 28])?);
# let t = ScalarArcTensor::zeros(device.clone(), [1], ScalarType::U8)?;
# let optimizer = SGD::builder().build();
# let learning_rate = 0.01;
model.set_training(true)?;
let y = model.forward(x)?;
let loss = y.cross_entropy_loss(t)?;
loss.backward()?;
for parameter in model.make_parameter_iter_mut()? {
    optimizer.update(learning_rate, parameter)?;
}
# Ok(())
# }
*/

/// Autograd.
pub mod autograd;
// public for testing
#[doc(hidden)]
pub mod criterion;
/// Layers.
pub mod layer;
/// Optimizers.
pub mod optimizer;
