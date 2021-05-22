use super::{autograd::Parameter, Dense, Identity, Sgd};
use crate::{
    backend::Device,
    tensor::{float_tensor::FloatTensor, Tensor},
    Result,
};
use rand_distr::{Distribution, Normal};

pub struct SgdBuilder {
    learning_rate: f32,
    momentum: f32, // Not implemented yet
}

impl Default for SgdBuilder {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.,
        }
    }
}

impl SgdBuilder {
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    pub fn build(self) -> Sgd {
        Sgd {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            //velocities: Default::default(),
        }
    }
}

pub struct DenseBuilder<A> {
    device: Device,
    inputs: usize,
    outputs: usize,
    weight_data: Vec<f32>,
    bias_data: Option<Vec<f32>>,
    activation: A,
}

impl Default for DenseBuilder<Identity> {
    fn default() -> Self {
        Self {
            device: Device::new_cpu(),
            inputs: 0,
            outputs: 0,
            weight_data: Vec::new(),
            bias_data: None,
            activation: Identity,
        }
    }
}

impl<A> DenseBuilder<A> {
    pub fn device(mut self, device: &Device) -> Self {
        self.device = device.clone();
        self
    }
    pub fn inputs(mut self, inputs: usize) -> Self {
        self.inputs = inputs;
        self
    }
    pub fn outputs(mut self, outputs: usize) -> Self {
        self.outputs = outputs;
        self
    }
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias_data = if bias {
            Some(vec![0.; self.outputs])
        } else {
            None
        };
        self
    }
    pub fn activation<A2>(self, activation: A2) -> DenseBuilder<A2> {
        DenseBuilder {
            device: self.device,
            inputs: self.inputs,
            outputs: self.outputs,
            weight_data: self.weight_data,
            bias_data: self.bias_data,
            activation,
        }
    }
    pub fn build(mut self) -> Result<Dense<A>> {
        if self.weight_data.len() != self.outputs * self.inputs {
            self.weight_data = he_normal((self.outputs, self.inputs));
        }
        if let Some(bias_data) = self.bias_data.as_mut() {
            if bias_data.len() != self.outputs {
                *bias_data = vec![0.; self.outputs]
            }
        }
        let weight =
            Tensor::from_shape_cow(&self.device, [self.outputs, self.inputs], self.weight_data)?;
        let weight = Parameter::from(FloatTensor::from(weight));
        let bias = if let Some(bias_data) = self.bias_data {
            let bias = Tensor::from_shape_cow(&self.device, self.outputs, bias_data)?;
            Some(Parameter::from(FloatTensor::from(bias)))
        } else {
            None
        };
        let activation = self.activation;
        // activation.to_device_mut(&device)?;
        Ok(Dense {
            weight,
            bias,
            activation,
        })
    }
}

fn he_normal((outputs, inputs): (usize, usize)) -> Vec<f32> {
    let std_dev = f32::sqrt(2. / inputs as f32);
    let normal = Normal::new(0., std_dev).unwrap();
    normal
        .sample_iter(&mut rand::thread_rng())
        .take(outputs * inputs)
        .collect()
}
