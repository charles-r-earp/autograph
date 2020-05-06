use crate::{AutographResult, Buffer, Variable, Parameter, Activation, init};
use super::{Dense, DenseOp};
use std::cell::Cell;

pub trait LayerBuilder: Sized {
  type Layer;
  fn input(self, input: &Variable) -> Self;
  fn train(self) -> Self;
  fn build(self) -> AutographResult<Self::Layer>;
}

pub struct DenseBuilder {
  input: Option<Variable>,
  train: bool,
  units: usize,
  use_bias: bool,
  act: Option<Activation>
}

impl Dense {
  pub fn builder() -> DenseBuilder {
    DenseBuilder {
      input: None,
      train: false,
      units: 0,
      use_bias: false,
      act: None
    }
  }
}

impl DenseBuilder {
  pub fn units(mut self, units: usize) -> Self {
    self.units = units;
    self
  }
  pub fn bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn act(mut self, act: Activation) -> Self {
    self.act.replace(act);
    self
  }
}

impl LayerBuilder for DenseBuilder {
  type Layer = super::Dense;
  fn input(mut self, input: &Variable) -> Self {
    self.input.replace(input.clone());
    self
  }
  fn train(mut self) -> Self {
    self.train = true;
    self
  }
  fn build(self) -> AutographResult<Dense> {
    use num_traits::AsPrimitive;
    let input = self.input.unwrap();
    let devices = input.devices();
    let batch_size = input.dims()[0];
    let input_channels = input.dims()[1..].iter().product();
    let weight = Parameter::zeros(devices, [self.units, input_channels], self.train)?;
    let gain = self.act.map_or(1., |act| init::calculate_gain(act));
    let factor = std::f32::consts::SQRT_2 / AsPrimitive::<f32>::as_(self.units);
    init::normal(weight.value(), 0., gain * factor, &mut init::thread_rng())?;
    let bias = if self.use_bias {
      let bias = Parameter::zeros(devices, [1, self.units], self.train)?;
      Some(bias)
    } else { None };
    let act = self.act;
    let output = Variable::zeros(devices, [batch_size, self.units], self.train || input.grad().is_some())?;
    let backward_mode = Cell::new(None);
    let ops = {
      let mut batch_sizes = input.value()
        .vertices()
        .iter()
        .map(|v| v.dims()[0]);
      let mut input_duals = input.duals();
      let mut weight_duals = weight.duals();
      let mut bias_duals = bias.as_ref()
        .map(|b| b.duals()); 
      let mut output_duals = output.duals();
      let mut ops = Vec::with_capacity(devices.len());
      for device in devices {
        let batch_size = batch_sizes.next().unwrap();
        let input = input_duals.next().unwrap();
        let weight = weight_duals.next().unwrap();
        let bias = bias_duals.as_mut()
          .map(|b| b.next().unwrap());
        let output = output_duals.next().unwrap();
        let mkn = [batch_size, input_channels, self.units];
        ops.push(DenseOp::new(device, mkn, input, weight, bias, act, output)?);  
      }
      ops
    };
    Ok(Dense{input, weight, bias, act, output, backward_mode, ops}) 
  }
}


