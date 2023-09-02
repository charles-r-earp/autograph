/*!

```rust
#[derive(Layer, Forward, Debug)]
#[autograph(forward(Variable4, Output=Variable2))]
struct LeNet5 {
    #[layer]
    conv1: Conv2<Relu>,
    #[layer]
    pool1: MaxPool2,
    #[layer]
    conv2: Conv2<Relu>,
    #[layer]
    pool2: MaxPool2,
    #[layer]
    flatten: Flatten,
    #[layer]
    dense1: Dense<Relu>,
    #[layer]
    dense2: Dense<Relu>,
    #[layer]
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
            .activation(Relu)
            .build()?;
        let pool1 = MaxPool2::builder().size([2, 2]).strides([2, 2]).build();
        let conv2 = Conv2::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(6)
            .outputs(16)
            .filter([5, 5])
            .activation(Relu)
            .build()?;
        let pool2 = MaxPool2::builder().size([2, 2]).strides([2, 2]).build();
        let flatten = Flatten;
        let dense1 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(16 * 4 * 4)
            .outputs(128)
            .activation(Relu)
            .build()?;
        let dense2 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(128)
            .outputs(84)
            .activation(Relu)
            .build()?;
        let dense3 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(84)
            .outputs(10)
            .bias(true)
            .build()?;
        Ok(Self {
            conv1,
            pool1,
            conv2,
            pool2,
            flatten,
            dense1,
            dense2,
            dense3,
        })
    }
}

let y = model.forward(x)?;
let loss = CrossEntropyLoss::default().eval(y, t)?;
loss.backward()?;
for parameter in model.parameters_mut()? {
    optimizer.update(learning_rate, parameter)?;
}
*/

pub mod autograd;
mod criterion;
pub mod layer;
pub mod optimizer;
