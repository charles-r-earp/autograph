use crate::{Device, Into2d, Conv2dArgs, Pool2dArgs};
use super::{Dense, Conv2d, MaxPool2d};

pub struct DenseBuilder {
  pub(super) device: Device,
  pub(super) inputs: Option<usize>,
  pub(super) outputs: Option<usize>,
  pub(super) weight_data: Option<Vec<f32>>,
  pub(super) use_bias: bool
}

impl DenseBuilder {
  pub fn inputs(mut self, inputs: usize) -> Self {
    debug_assert_ne!(inputs, 0);
    self.inputs.replace(inputs);
    self
  }
  pub fn outputs(mut self, outputs: usize) -> Self {
    debug_assert_ne!(outputs, 0);
    self.outputs.replace(outputs);
    self
  }
  pub fn init_weight_from_iter(mut self, weight_iter: impl IntoIterator<Item=f32>) -> Self {
    let inputs = self.inputs.expect("DenseBuilder inputs must be set before init_weight_from_iter()!");
    let outputs = self.outputs.expect("DenseBuilder outputs must be set before init_weight_from_iter()!");
    let weight_data = weight_iter.into_iter()
      .take(outputs * inputs)
      .collect();
    self.weight_data.replace(weight_data);
    self
  }
  pub fn bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn build(self) -> Dense {
    self.into()
  }
}

pub struct Conv2dBuilder {
  pub(super) device: Device,
  pub(super) inputs: Option<usize>,
  pub(super) outputs: Option<usize>,
  pub(super) kernel: Option<[usize; 2]>,
  pub(super) weight_data: Option<Vec<f32>>,
  pub(super) use_bias: bool,
  pub(super) args: Conv2dArgs
}

impl Conv2dBuilder {
  pub fn inputs(mut self, inputs: usize) -> Self {
    debug_assert_ne!(inputs, 0);
    self.inputs.replace(inputs);
    self
  }
  pub fn outputs(mut self, outputs: usize) -> Self {
    debug_assert_ne!(outputs, 0);
    self.outputs.replace(outputs);
    self
  }
  pub fn kernel(mut self, kernel: impl Into2d) -> Self {
    let kernel = kernel.into_2d();
    debug_assert_ne!(kernel[0] * kernel[1], 0);
    self.kernel.replace(kernel);
    self
  }
  pub fn init_weight_from_iter(mut self, weight_iter: impl IntoIterator<Item=f32>) -> Self {
    let inputs = self.inputs.expect("Conv2dBuilder inputs must be set before init_weight_from_iter()!");
    let outputs = self.outputs.expect("Conv2dBuilder outputs must be set before init_weight_from_iter()!");
    let [kh, kw] = self.kernel.expect("Conv2dBuilder kernel must be set before init_weight_from_iter()!");
    let weight_data = weight_iter.into_iter()
      .take(outputs * inputs * kh * kw)
      .collect();
    self.weight_data.replace(weight_data);
    self
  }
  pub fn bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn args(mut self, args: Conv2dArgs) -> Self {
    self.args = args;
    self
  }
  pub fn build(self) -> Conv2d {
    self.into()
  }
}

pub struct MaxPool2dBuilder {
  pub(super) args: Pool2dArgs
}

impl MaxPool2dBuilder {
  pub fn args(mut self, args: Pool2dArgs) -> Self {
    self.args = args;
    self
  }
  pub fn build(self) -> MaxPool2d {
    self.into()
  }
}

