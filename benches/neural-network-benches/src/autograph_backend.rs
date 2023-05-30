use anyhow::Result;
use autograph::{
    device::Device,
    learn::{
        criterion::{Criterion, CrossEntropyLoss},
        neural_network::{
            autograd::{ParameterViewMutD, Variable2, Variable4},
            layer::{Conv2, Dense, Forward, Layer, MaxPool2, Relu},
            optimizer::{Optimizer, SGD},
        },
    },
    scalar::ScalarType,
    tensor::ScalarArcTensor,
};

pub struct Lenet5Classifier {
    device: Device,
    scalar_type: ScalarType,
    model: Lenet5,
    optimizer: Option<SGD>,
}

impl Lenet5Classifier {
    pub fn new(device: Device, scalar_type: ScalarType) -> Result<Self> {
        let model = Lenet5::new(device.clone(), scalar_type)?;
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
        let y = self.model.forward(x.into())?;
        let _ = y
            .into_value()
            .try_into_arc_tensor::<f32>()
            .unwrap()
            .into_array()?
            .into_raw_vec();
        Ok(())
    }
    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        self.model.set_training(true)?;
        let x = ScalarArcTensor::zeros(
                self.device.clone(),
                [batch_size, 1, 28, 28],
                self.scalar_type,
            )?;
        let t = ScalarArcTensor::zeros(self.device.clone(), batch_size, ScalarType::U8)?;
        let y = self.model.forward(x.into())?;
        let loss = CrossEntropyLoss::default().eval(y, t)?;
        loss.backward()?;
        let optimizer = self.optimizer.as_ref().unwrap();
        let learning_rate = 0.01;
        for parameter in self.model.parameters_mut()? {
            optimizer.update(learning_rate, parameter)?;
        }
        self.model.set_training(false)?;
        Ok(())
    }
}

struct Lenet5 {
    conv1: Conv2<Relu>,
    pool1: MaxPool2,
    conv2: Conv2<Relu>,
    pool2: MaxPool2,
    dense1: Dense<Relu>,
    dense2: Dense<Relu>,
    dense3: Dense,
}

impl Lenet5 {
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
            dense1,
            dense2,
            dense3,
        })
    }
}

impl Layer for Lenet5 {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.conv1.set_training(training)?;
        self.conv2.set_training(training)?;
        self.dense1.set_training(training)?;
        self.dense2.set_training(training)?;
        self.dense3.set_training(training)?;
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        Ok(self
            .conv1
            .parameters_mut()?
            .into_iter()
            .chain(self.conv2.parameters_mut()?)
            .chain(self.dense1.parameters_mut()?)
            .chain(self.dense2.parameters_mut()?)
            .chain(self.dense3.parameters_mut()?)
            .collect())
    }
}

impl Forward<Variable4> for Lenet5 {
    type Output = Variable2;
    fn forward(&self, input: Variable4) -> Result<Self::Output> {
        let Self {
            conv1,
            pool1,
            conv2,
            pool2,
            dense1,
            dense2,
            dense3,
        } = self;
        input
            .forward(conv1)?
            .forward(pool1)?
            .forward(conv2)?
            .forward(pool2)?
            .flatten()?
            .forward(dense1)?
            .forward(dense2)?
            .forward(dense3)
    }
}
