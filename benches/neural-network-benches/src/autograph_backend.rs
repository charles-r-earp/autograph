use anyhow::Result;
use autograph::{
    device::Device,
    learn::{
        criterion::{Criterion, CrossEntropyLoss},
        neural_network::{
            layer::{Dense, Forward, Layer},
            optimizer::{self, Optimizer, SGD},
        },
    },
    scalar::ScalarType,
    tensor::{ScalarArcTensor, ScalarTensor},
};

pub struct LinearClassifier {
    device: Device,
    scalar_type: ScalarType,
    inputs: usize,
    outputs: usize,
    dense: Dense,
    optimizer: Option<SGD>,
}

impl LinearClassifier {
    pub fn new(
        device: Device,
        scalar_type: ScalarType,
        inputs: usize,
        outputs: usize,
    ) -> Result<Self> {
        let dense = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(inputs)
            .outputs(outputs)
            .bias(true)
            .build()?;
        Ok(Self {
            device,
            scalar_type,
            inputs,
            outputs,
            dense,
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
        let x = ScalarTensor::zeros(
            self.device.clone(),
            [batch_size, self.inputs],
            self.scalar_type,
        )?;
        let _ = self.dense.forward(x.into())?;
        Ok(())
    }
    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        self.dense.set_training(true)?;
        let x = ScalarArcTensor::zeros(
            self.device.clone(),
            [batch_size, self.inputs],
            self.scalar_type,
        )?;
        let t = ScalarArcTensor::zeros(
            self.device.clone(),
            [batch_size, self.outputs],
            self.scalar_type,
        )?;
        let y = self.dense.forward(x.into())?;
        let loss = CrossEntropyLoss::default().eval(y, t)?;
        loss.backward()?;
        let optimizer = self.optimizer.as_ref().unwrap();
        let learning_rate = 0.01;
        for parameter in self.dense.parameters_mut()? {
            optimizer.update(learning_rate, parameter)?;
        }
        self.dense.set_training(false)?;
        Ok(())
    }
}
