/*!
A machine learning library for Rust.

GPGPU kernels implemented with [krnl].
- Host / Device execution.
- Tensors emulate [ndarray].
   - Host tensors can be borrowed as arrays.
- Tensors / Models / Optimizers can be serialized with [serde].
   - Portable between platforms.
   - Save / resume training progress.
- Fully extensible, in Rust.

# Getting Started
- See [krnl::device] for creating devices.
- See [tensor] for creating tensors.
- See [`learn::neural_network`] for Neural Networks.

# Example
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
model.try_for_each_parameter_view_mut(|parameter| {
    optimizer.update(learning_rate, parameter)
})?;
# Ok(())
# }
*/
#![cfg_attr(doc_cfg, feature(doc_auto_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// anyhow
pub extern crate anyhow;
/// half
pub extern crate half;
/// krnl
pub extern crate krnl;
/// ndarray
pub extern crate ndarray;
/// num-traits
pub extern crate num_traits;

/// Datasets.
#[cfg(feature = "dataset")]
pub mod dataset;
/// Machine Learning.
#[cfg(feature = "learn")]
pub mod learn;
/// Ops.
pub mod ops;
/// Tensors.
pub mod tensor;
