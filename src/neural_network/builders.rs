use super::{
    autograd::{Parameter, Parameter2},
    Dense, Identity, Network, Sgd,
};
use crate::{
    backend::Device,
    tensor::{
        float::{FloatTensor, FloatType},
        Tensor,
    },
    Result,
};
use anyhow::bail;
use half::bf16;
use rand::Rng;
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
    /// Set the device, defaults to cpu.
    pub fn device(mut self, device: &Device) -> Self {
        self.device = device.clone();
        self
    }
    /// Set the number of inputs, defaults to 0 which will lazily initialize on Dense::forward_mut.
    pub fn inputs(mut self, inputs: usize) -> Self {
        self.inputs = inputs;
        self
    }
    /// Set the number of outputs.
    pub fn outputs(mut self, outputs: usize) -> Self {
        self.outputs = outputs;
        self
    }
    /// Whether to include a bias, defaults to false.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias_data = if bias {
            Some(vec![0.; self.outputs])
        } else {
            None
        };
        self
    }
    /// Sets the activation, defaults to Identity.\
    //
    // The activation A2 should impl Network + 'static
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
    pub fn build(mut self) -> Result<Dense<A>>
    where
        A: Network,
    {
        if self.outputs == 0 {
            bail!("Outputs are required, greater than 0!")
        }
        if self.weight_data.len() != self.outputs * self.inputs {
            self.weight_data =
                he_normal(&mut rand::thread_rng(), (self.outputs, self.inputs)).collect();
        }
        if let Some(bias_data) = self.bias_data.as_mut() {
            if bias_data.len() != self.outputs {
                *bias_data = vec![0.; self.outputs]
            }
        }
        let weight = if !self.weight_data.is_empty() {
            // TODO: impl for FloatType::BF16
            let weight = Tensor::from_shape_vec(
                &self.device,
                [self.outputs, self.inputs],
                self.weight_data,
            )?;
            Parameter::from(FloatTensor::from(weight))
        } else if self.inputs == 0 {
            Parameter::from(FloatTensor::zeros(
                &self.device,
                FloatType::F32,
                [self.outputs, 0],
            )?)
        } else {
            dense_weight(&self.device, FloatType::F32, (self.outputs, self.inputs))?
        };
        let bias = if let Some(bias_data) = self.bias_data {
            let bias = Tensor::from_shape_cow(&self.device, self.outputs, bias_data)?;
            Some(Parameter::from(FloatTensor::from(bias)))
        } else {
            None
        };
        let activation = self.activation.into_device(&self.device)?;
        Ok(Dense {
            weight,
            bias,
            activation,
        })
    }
}

pub(super) fn dense_weight(
    device: &Device,
    float_type: FloatType,
    (outputs, inputs): (usize, usize),
) -> Result<Parameter2> {
    let mut rng = rand::thread_rng();
    let weight_iter = he_normal(&mut rng, (outputs, inputs));
    let weight = match float_type {
        FloatType::BF16 => {
            let weight_iter = weight_iter.map(bf16::from_f32);
            let weight = Tensor::from_shape_vec(device, [outputs, inputs], weight_iter.collect())?;
            FloatTensor::from(weight)
        }
        FloatType::F32 => {
            let weight = Tensor::from_shape_vec(device, [outputs, inputs], weight_iter.collect())?;
            FloatTensor::from(weight)
        }
    };
    Ok(weight.into())
}

fn he_normal<R: Rng>(
    rng: &mut R,
    (outputs, inputs): (usize, usize),
) -> impl Iterator<Item = f32> + '_ {
    let std_dev = f32::sqrt(2. / inputs as f32);
    let normal = Normal::new(0., std_dev).unwrap();
    normal.sample_iter(rng).take(outputs * inputs)
}
