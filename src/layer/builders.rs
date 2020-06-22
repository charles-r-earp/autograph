use crate::{Device, Into2d, Conv2dArgs, Pool2dArgs};
#[cfg(feature="cuda")]
use crate::CudaGpu;
use super::{
  Layer, 
  Dense, 
  Conv2d, 
  MaxPool2d,
  Relu, 
  //DataParallel
};
use std::sync::Arc;
use ndarray::{Dimension, IntoDimension, Ix1, Ix2, Ix4};

pub trait LayerBuilder: Default + Clone {
  type Layer;
  fn device(self, device: &Device) -> Self { self }
  fn build(self) -> Self::Layer;
}

#[derive(Default, Clone)]
pub struct DenseBuilder {
  pub(super) device: Option<Device>,
  pub(super) inputs: Option<usize>,
  pub(super) outputs: Option<usize>,
  pub(super) weight_data: Option<Vec<f32>>,
  pub(super) use_bias: bool,
  pub(super) bias_data: Option<Vec<f32>>
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
  pub fn weight_data(mut self, mut f: impl FnMut(Ix2) -> Vec<f32>) -> Self {
    let inputs = self.inputs.expect("DenseBuilder inputs must be set before weight_data()!");
    let outputs = self.outputs.expect("DenseBuilder outputs must be set before weight_data()!");
    let weight_dim = [outputs, inputs].into_dimension();
    let weight_data = f(weight_dim);
    debug_assert_eq!(weight_data.len(), weight_dim.size());
    self.weight_data.replace(weight_data);
    self
  }
  pub fn bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn bias_data(mut self, mut f: impl FnMut(Ix1) -> Vec<f32>) -> Self {
    let outputs = self.outputs.expect("DenseBuilder outputs must be set before bias_data()!");
    let bias_dim = outputs.into_dimension();
    let bias_data = f(bias_dim);
    debug_assert_eq!(bias_data.len(), bias_dim.size());
    self.bias_data.replace(bias_data);
    self
  }
}

impl LayerBuilder for DenseBuilder {
  type Layer = Dense;
  fn device(mut self, device: &Device) -> Self {
    self.device.replace(device.clone());
    self
  }
  fn build(self) -> Dense {
    self.into()
  }
}

#[derive(Default, Clone)]
pub struct Conv2dBuilder {
  pub(super) device: Option<Device>,
  pub(super) inputs: Option<usize>,
  pub(super) outputs: Option<usize>,
  pub(super) kernel: Option<[usize; 2]>,
  pub(super) weight_data: Option<Vec<f32>>,
  pub(super) use_bias: bool,
  pub(super) bias_data: Option<Vec<f32>>,
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
  pub fn weight_data(mut self, mut f: impl FnMut(Ix4) -> Vec<f32>) -> Self {
    let inputs = self.inputs.expect("Conv2dBuilder inputs must be set before weight_data()!");
    let outputs = self.outputs.expect("Conv2dBuilder outputs must be set before weight_data()!");
    let [kh, kw] = self.kernel.expect("Conv2dBuilder kernel must be set before weight_data()!");
    let weight_dim = [outputs, inputs, kh, kw].into_dimension();
    let weight_data = f(weight_dim);
    debug_assert_eq!(weight_data.len(), weight_dim.size());
    self.weight_data.replace(weight_data);
    self
  }
  pub fn bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn bias_data(mut self, mut f: impl FnMut(Ix1) -> Vec<f32>) -> Self {
    let outputs = self.outputs.expect("DenseBuilder outputs must be set before bias_data()!");
    let bias_dim = outputs.into_dimension();
    let bias_data = f(bias_dim);
    debug_assert_eq!(bias_data.len(), bias_dim.size());
    self.bias_data.replace(bias_data);
    self
  }
  pub fn args(mut self, args: Conv2dArgs) -> Self {
    self.args = args;
    self
  }
}

impl LayerBuilder for Conv2dBuilder {
  type Layer = Conv2d;
  fn device(mut self, device: &Device) -> Self {
    self.device.replace(device.clone());
    self
  }
  fn build(self) -> Conv2d {
    self.into()
  }
}

#[derive(Default, Clone)]
pub struct ReluBuilder{}

impl LayerBuilder for ReluBuilder {
  type Layer = Relu;
  fn build(self) -> Relu { 
    self.into()
  }
}

#[derive(Default, Clone)]
pub struct MaxPool2dBuilder {
  pub(super) args: Pool2dArgs
}

impl MaxPool2dBuilder {
  pub fn args(mut self, args: Pool2dArgs) -> Self {
    self.args = args;
    self
  }
}

impl LayerBuilder for MaxPool2dBuilder {
  type Layer = MaxPool2d;
  fn build(self) -> MaxPool2d {
    self.into()
  }
}

/* Prototype 
pub struct DataParallelBuilder<L: Layer> {
  pub(super) device: Option<Device>,
  #[cfg(feature="cuda")]
  pub(super) cuda_gpus: Vec<Arc<CudaGpu>>,
  pub(super) layer_builder: L::Builder
}

impl<L: Layer> Default for DataParallelBuilder<L> {
  fn default() -> Self {
    Self {
      device: None,
      #[cfg(feature="cuda")]
      cuda_gpus: Vec::new(),
      layer_builder: L::builder()
    }
  }
}

impl<L: Layer> Clone for DataParallelBuilder<L> {
  fn clone(&self) -> Self {
    Self {
      device: self.device.clone(),
      #[cfg(feature="cuda")]
      cuda_gpus: self.cuda_gpus.clone(),
      layer_builder: self.layer_builder.clone()
    }
  }
}

impl<L: Layer> DataParallelBuilder<L> {
  #[cfg(feature="cuda")]
  fn cuda_gpus(mut self, cuda_gpus: impl AsRef<[CudaGpu]>) -> Self {
    self.cuda_gpus.replace(cuda_gpus.as_ref().to_vec());
    self
  } 
} 


impl<L: Layer> LayerBuilder for DataParallelBuilder<L> {
  type Layer = DataParallel<L>;
  fn device(mut self, device: &Device) -> Self {
    self.device.replace(device.clone());
    self
  }
  fn build(self) -> DataParallel<L> {
    self.into()
  }
} 
*/
