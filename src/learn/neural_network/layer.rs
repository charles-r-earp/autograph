use super::{
    autograd::{
        Parameter, Parameter1, Parameter2, ParameterViewMut1, ParameterViewMut2, ParameterViewMutD,
        Variable, Variable2,
    },
    optimizer::Optimizer,
};
use crate::{
    buffer::Data,
    buffer::{Buffer, ScalarBuffer},
    device::Device,
    krnl::krnl_core::half::{bf16, f16},
    ops::AddAssign,
    scalar::ScalarType,
    tensor::{ArcTensor, ArcTensor2, ScalarTensor, Tensor, Tensor2, TensorBase},
};
use anyhow::{bail, Result};
use ndarray::{linalg::Dot, Array1, Array2, Dimension, Ix2};
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};
use std::{any::Any, marker::PhantomData};

pub mod builder {
    use super::*;

    pub struct DenseBuilder<A = Identity> {
        inputs: usize,
        outputs: usize,
        bias: bool,
        scalar_type: ScalarType,
        device: Device,
        activation: A,
    }

    impl DenseBuilder {
        pub(super) fn new() -> Self {
            Self {
                inputs: 0,
                outputs: 0,
                bias: false,
                scalar_type: ScalarType::F32,
                device: Device::host(),
                activation: Identity,
            }
        }
    }

    impl<A> DenseBuilder<A> {
        pub fn inputs(self, inputs: usize) -> Self {
            Self { inputs, ..self }
        }
        pub fn outputs(self, outputs: usize) -> Self {
            Self { outputs, ..self }
        }
        pub fn bias(self, bias: bool) -> Self {
            Self { bias, ..self }
        }
        pub fn activation<A2>(self, activation: A2) -> DenseBuilder<A2> {
            let Self {
                inputs,
                outputs,
                bias,
                activation: _,
                scalar_type,
                device,
            } = self;
            DenseBuilder {
                inputs,
                outputs,
                bias,
                activation,
                scalar_type,
                device,
            }
        }
        pub fn scalar_type(self, scalar_type: ScalarType) -> Self {
            Self {
                scalar_type,
                ..self
            }
        }
        pub fn device(self, device: Device) -> Self {
            Self { device, ..self }
        }
        pub fn build(self) -> Result<Dense<A>> {
            let Self {
                inputs,
                outputs,
                bias,
                activation,
                scalar_type,
                device,
            } = self;
            if !matches!(scalar_type, ScalarType::BF16 | ScalarType::F32) {
                bail!("Dense {scalar_type:?} not supported!");
            }
            let a = if inputs > 0 {
                f32::sqrt(2. / inputs as f32)
            } else {
                0.
            };
            let mut rng = thread_rng();
            let weight_iter = Uniform::new(-a, a)
                .sample_iter(&mut rng)
                .take(inputs * outputs);
            let weight = match scalar_type {
                ScalarType::BF16 => ScalarBuffer::from(Buffer::from(
                    weight_iter.map(bf16::from_f32).collect::<Vec<_>>(),
                )),
                ScalarType::F32 => {
                    ScalarBuffer::from(Buffer::from(weight_iter.collect::<Vec<_>>()))
                }
                _ => unreachable!(),
            };
            let weight = weight.into_device(device.clone())?;
            let weight = Parameter::from(
                ScalarTensor::from(weight)
                    .into_shape([outputs, inputs])
                    .unwrap(),
            );
            let bias = if bias {
                let bias_iter = Uniform::new(-a, a).sample_iter(rng).take(outputs);
                let bias = match scalar_type {
                    ScalarType::BF16 => ScalarBuffer::from(Buffer::from(
                        bias_iter.map(bf16::from_f32).collect::<Vec<_>>(),
                    )),
                    ScalarType::F32 => {
                        ScalarBuffer::from(Buffer::from(bias_iter.collect::<Vec<_>>()))
                    }
                    _ => unreachable!(),
                };
                let bias = bias.into_device(device)?;
                Some(Parameter::from(ScalarTensor::from(bias)))
            } else {
                None
            };
            Ok(Dense {
                weight,
                bias,
                activation,
            })
        }
    }
}
use builder::DenseBuilder;

pub struct LayerMut<'a> {
    inner: &'a mut dyn Layer,
}

impl<'a> LayerMut<'a> {
    pub fn new(layer: &'a mut dyn Layer) -> Self {
        Self { inner: layer }
    }
}

impl Layer for LayerMut<'_> {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.inner.set_training(training)
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        self.inner.parameters_mut()
    }
    fn layers_mut(&mut self) -> Result<Vec<LayerMut>> {
        self.inner.layers_mut()
    }
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        self.inner.to_device_mut(device)
    }
}

pub trait Layer {
    fn set_training(&mut self, training: bool) -> Result<()> {
        for mut layer in self.layers_mut()? {
            layer.set_training(training)?;
        }
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        todo!()
    }
    fn layers_mut(&mut self) -> Result<Vec<LayerMut>> {
        Ok(Vec::new())
    }
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        for layer in self.layers_mut()?.iter_mut() {
            layer.to_device_mut(device.clone())?;
        }
        Ok(())
    }
}

pub trait Forward<X> {
    type Output;
    fn forward(&self, input: X) -> Result<Self::Output>;
}

///```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let device = Device::host();
/// let dense = Dense::builder()
///    .inputs(1)
///    .outputs(1)
///    .bias(true))
///    .activation(Relu)
///    .scalar_type(ScalarType::BF16)
///    .device(device.clone())
///    .build()?;
/// # }
///```
pub struct Dense<A = Identity> {
    weight: Parameter2,
    bias: Option<Parameter1>,
    activation: A,
}

impl Dense {
    pub fn builder() -> DenseBuilder {
        DenseBuilder::new()
    }
}

impl<A> Dense<A> {
    pub fn weight_view_mut(&mut self) -> ParameterViewMut2 {
        todo!()
    }
    pub fn bias_view_mut(&mut self) -> Result<Option<ParameterViewMut1>> {
        todo!()
    }
    pub fn into_device(self, device: Device) -> Result<Self> {
        todo!()
    }
    pub fn to_device_mut(&mut self, device: Device) -> Result<()> {
        todo!()
    }
}

impl Layer for Dense {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.weight.set_training(training);
        if let Some(bias) = self.bias.as_mut() {
            bias.set_training(training);
        }
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        let mut parameters = Vec::new();
        parameters.push(self.weight.make_view_mut()?.into_dyn());
        if let Some(bias) = self.bias.as_mut() {
            parameters.push(bias.make_view_mut()?.into_dyn());
        }
        Ok(parameters)
    }
}

impl<A: Forward<Variable2, Output = Variable2> + Any> Forward<Variable2> for Dense<A> {
    type Output = Variable2;
    fn forward(&self, input: Variable2) -> Result<Self::Output> {
        let mut output = input.dot(&self.weight.to_variable().t())?;
        if let Some(bias) = self.bias.as_ref() {
            output.add_assign(&bias.to_variable())?;
        }
        self.activation.forward(output)
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Identity;

impl<X> Forward<X> for Identity {
    type Output = X;
    fn forward(&self, input: X) -> Result<Self::Output> {
        Ok(input)
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Relu;
