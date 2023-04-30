use super::autograd::{ParameterD, VariableD};
use crate::device::Device;
use anyhow::Result;

/// A trait for networks and layers.
pub trait Layer: Forward + Send + Sync + 'static {
    // impl if has layers

    fn try_for_each_layer<'a>(
        &'a self,
        f: &mut dyn FnMut(&'a dyn Layer) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }
    fn try_for_each_layer_mut<'a>(
        &'a mut self,
        f: &mut dyn FnMut(&'a mut dyn Layer) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }

    // impl if has parameters

    fn try_for_each_parameter<'a>(
        &'a self,
        f: &mut dyn FnMut(&'a ParameterD) -> Result<()>,
    ) -> Result<()> {
        self.try_for_each_layer(&mut |layer| layer.try_for_each_parameter(f))
    }
    fn try_for_each_parameter_mut<'a>(
        &'a mut self,
        f: &mut dyn FnMut(&'a mut ParameterD) -> Result<()>,
    ) -> Result<()> {
        self.try_for_each_layer_mut(&mut |layer| layer.try_for_each_parameter_mut(f))
    }

    // impl if has layers and parameters

    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        // TODO: rayon?
        let mut has_layer_parameters = false;
        self.try_for_each_layer_mut(&mut |layer| {
            has_layer_parameters |= layer.parameter_count() > 0;
            layer.to_device_mut(device.clone())
        })?;
        if !has_layer_parameters {
            self.try_for_each_parameter_mut(&mut |parameter| {
                parameter.to_device_mut(device.clone())
            })?;
        }
        Ok(())
    }

    // provided methods

    fn for_each_layer<'a>(&'a self, f: &mut dyn FnMut(&'a dyn Layer)) {
        self.try_for_each_layer(&mut |layer| {
            f(layer);
            Ok(())
        })
        .unwrap();
    }
    fn for_each_layer_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut dyn Layer)) {
        self.try_for_each_layer_mut(&mut |layer| {
            f(layer);
            Ok(())
        })
        .unwrap();
    }
    fn layer_count(&self) -> usize {
        let mut count = 0;
        self.for_each_layer(&mut |_| count += 1);
        count
    }
    fn layers(&self) -> Vec<&dyn Layer> {
        let mut layers = Vec::with_capacity(self.layer_count());
        self.for_each_layer(&mut |layer| layers.push(layer));
        layers
    }
    fn layers_mut(&mut self) -> Vec<&mut dyn Layer> {
        let mut layers = Vec::with_capacity(self.layer_count());
        self.for_each_layer_mut(&mut |layer| layers.push(layer));
        layers
    }

    fn for_each_parameter<'a>(&'a self, f: &mut dyn FnMut(&'a ParameterD)) {
        self.try_for_each_parameter(&mut |parameter| {
            f(parameter);
            Ok(())
        })
        .unwrap();
    }
    fn for_each_parameter_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut ParameterD)) {
        self.try_for_each_parameter_mut(&mut |parameter| {
            f(parameter);
            Ok(())
        })
        .unwrap();
    }
    fn parameter_count(&self) -> usize {
        let mut count = 0;
        self.for_each_parameter(&mut |_| count += 1);
        count
    }
    fn parameters(&self) -> Vec<ParameterD> {
        let mut parameters = Vec::with_capacity(self.parameter_count());
        self.for_each_parameter(&mut |parameter| parameters.push(parameter.clone()));
        parameters
    }
    fn parameters_mut(&mut self) -> Vec<&mut ParameterD> {
        let mut parameters = Vec::with_capacity(self.parameter_count());
        self.for_each_parameter_mut(&mut |parameter| parameters.push(parameter));
        parameters
    }
}

/// A trait for the forward pass.
///
/// [`Layer`]'s implement [`Forward`], which computes the output as a function of the input.
///
/// # Derive
/// [`Forward`] can be [derived](autograph_derive) for sequential layers (ie typical feed-foward networks).
pub trait Forward {
    /// Computes the forward pass.
    ///
    /// # Autograd
    /// Operations on [`Variable`](super::autograd::Variable) are expected to apply backward ops via [`Variable::with_backward()`].
    ///
    /// **Errors**
    ///
    /// Returns an error if the operation could not be performed. Generally the implemenation should return an error instead of panicking.
    fn forward(&self, input: VariableD) -> Result<VariableD>;
}

/// Dense / fully connected layer.
#[derive(Clone)]
pub struct Dense {
    weight: ParameterD,
    bias: Option<ParameterD>,
}

impl Dense {
    /// Creates a new [`Dense`] for `inputs` and `outputs`.
    ///
    /// The weight is initialized with a uniform distribution of (-a, a) where a = sqrt(1 / inputs).
    pub fn from_inputs_outputs(inputs: usize, outputs: usize) -> Self {
        todo!()
        /*
        let a = f32::sqrt(2. / inputs as f32);
        let data = Uniform::new(-a, a)
            .sample_iter(&mut rand::thread_rng())
            .take(inputs * outputs)
            .collect::<Vec<_>>();
        let buffer = FloatBuffer::from(Buffer::from(data));
        let weight = Parameter::from(
            FloatArcTensor::from(buffer)
                .into_shape([outputs, inputs].as_ref())
                .unwrap(),
        );
        Self { weight, bias: None }*/
    }
    /// Adds a bias to the layer.
    ///
    /// The bias is initialized with a uniform distribution of (-a, a) where a = sqrt(1 / inputs).
    pub fn with_bias(mut self, bias: bool) -> Result<Self> {
        todo!()
        /*
        if bias {
            let inputs = self.weight.shape()[1];
            let outputs = self.weight.shape()[0];
            let a = f32::sqrt(2. / inputs as f32);
            let data = Uniform::new(-a, a)
                .sample_iter(&mut rand::thread_rng())
                .take(outputs)
                .collect::<Vec<_>>();
            let device = self.weight.device();
            let buffer = FloatBuffer::from(smol::block_on(Buffer::from(data).into_device(device))?);
            self.bias.replace(Parameter::from(
                FloatArcTensor::from(buffer)
                    .into_shape([outputs].as_ref())
                    .unwrap(),
            ));
        } else {
            self.bias = None;
        }
        Ok(self)*/
    }
}

impl Layer for Dense {
    fn try_for_each_parameter<'a>(
        &'a self,
        f: &mut dyn FnMut(&'a ParameterD) -> Result<()>,
    ) -> Result<()> {
        f(&self.weight)?;
        if let Some(bias) = self.bias.as_ref() {
            f(bias)?;
        }
        Ok(())
    }
    fn try_for_each_parameter_mut<'a>(
        &'a mut self,
        f: &mut dyn FnMut(&'a mut ParameterD) -> Result<()>,
    ) -> Result<()> {
        f(&mut self.weight)?;
        if let Some(bias) = self.bias.as_mut() {
            f(bias)?;
        }
        Ok(())
    }
}

impl Forward for Dense {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        todo!()
        /*let input = smol::block_on(input.into_device(self.weight.device()))?.flatten()?;
        // TODO: convert to float type of weight
        let weight = self.weight.clone().into_dimensionality()?;
        let bias = if let Some(bias) = self.bias.as_ref().map(Parameter::clone) {
            Some(bias.into_dimensionality()?)
        } else {
            None
        };
        Ok(input.dot_bias(&weight.t(), bias.as_ref())?.into_dyn())*/
    }
}
