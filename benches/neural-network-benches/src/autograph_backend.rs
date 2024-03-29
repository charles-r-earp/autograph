use anyhow::Result;
use autograph::{
    half::bf16,
    krnl::{device::Device, scalar::ScalarType},
    learn::{
        criterion::CrossEntropyLoss,
        neural_network::{
            autograd::{Variable2, Variable4},
            layer::{Conv2, Dense, Flatten, Forward, Layer, MaxPool2, Relu},
            optimizer::SGD,
        },
    },
    tensor::ScalarArcTensor,
};

pub struct LeNet5Classifier {
    device: Device,
    scalar_type: ScalarType,
    model: LeNet5,
    optimizer: Option<SGD>,
}

impl LeNet5Classifier {
    pub fn new(device: Device, scalar_type: ScalarType) -> Result<Self> {
        let model = LeNet5::new(device.clone(), scalar_type)?;
        Ok(Self {
            device,
            scalar_type,
            model,
            optimizer: None,
        })
    }
    pub fn with_sgd(self, momentum: bool) -> Self {
        let mut builder = SGD::builder();
        if momentum {
            builder = builder.momentum(0.01);
        }
        let optimizer = builder.build();
        Self {
            optimizer: Some(optimizer),
            ..self
        }
    }
    pub fn infer(&self, batch_size: usize) -> Result<()> {
        let x = ScalarArcTensor::zeros(
            self.device.clone(),
            [batch_size, 1, 28, 28],
            self.scalar_type,
        )?;
        let y = self.model.forward(x.into())?.into_value();
        match y.scalar_type() {
            ScalarType::BF16 => {
                let _ = y
                    .try_into_arc_tensor::<bf16>()
                    .unwrap()
                    .into_array()?
                    .into_raw_vec();
            }
            ScalarType::F32 => {
                let _ = y
                    .try_into_arc_tensor::<f32>()
                    .unwrap()
                    .into_array()?
                    .into_raw_vec();
            }
            _ => unimplemented!(),
        }
        Ok(())
    }
    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        self.model.init_parameter_grads()?;
        let x = ScalarArcTensor::zeros(
            self.device.clone(),
            [batch_size, 1, 28, 28],
            self.scalar_type,
        )?;
        let t = ScalarArcTensor::zeros(self.device.clone(), batch_size, ScalarType::U8)?;
        let y = self.model.forward(x.into())?;
        let loss = y.cross_entropy_loss(t)?;
        loss.backward()?;
        let optimizer = self.optimizer.as_ref().unwrap();
        let learning_rate = 0.01;
        self.model.update(learning_rate, optimizer)?;
        Ok(())
    }
}

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
