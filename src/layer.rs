use crate::{
  Num,
  Device,
  Tensor,
  Tensor2,
  Tensor4,
  TensorView,
  TensorView2,
  TensorView4,
  RwTensor,
  Conv2dArgs,
  Pool2dArgs,
};
use crate::autograd::{
  Parameter,
  Parameter1,
  Parameter2,
  Parameter4,
  ParameterD,
  Variable,
  Variable2,
  Variable4
};
use ndarray::{
  Dimension,
  Ix2,
  Ix4
};

pub mod builders;
use builders::{
  LayerBuilder,
  DenseBuilder,
  Conv2dBuilder, 
  ReluBuilder,
  MaxPool2dBuilder,
  //DataParallelBuilder,
};

#[cfg(test)]
mod tests;

pub trait Layer: Sized {
  type Builder: LayerBuilder<Layer=Self>;
  fn builder() -> Self::Builder { Self::Builder::default() } 
  fn parameters(&self) -> Vec<ParameterD> { Vec::new() }
  fn init_training(&mut self) {}
  fn to_builder(&self, with_data: bool) -> Self::Builder;
}

pub trait Inference<D: Dimension> {
  type OutputDim: Dimension;
  fn infer(&self, input: &TensorView<f32, D>) -> Tensor<f32, Self::OutputDim>;
}

pub trait Forward<D: Dimension>: Inference<D> {
  fn forward(&self, input: &Variable<D>, train: bool) -> Variable<Self::OutputDim>;
}

pub struct Dense {
  weight: Parameter2,
  bias: Option<Parameter1>
}

impl From<DenseBuilder> for Dense {
  fn from(builder: DenseBuilder) -> Self {
    let device = builder.device.expect("DenseBuilder requires device to be specified!");
    let inputs = builder.inputs.expect("DenseBuilder requires inputs to be specified!");
    let outputs = builder.outputs.expect("DenseBuilder requires outputs to specified!");
    let weight_value = if let Some(weight_data) = builder.weight_data {
      RwTensor::from_shape_vec(&device, [outputs, inputs], weight_data)
    }
    else {
      RwTensor::zeros(&device, [outputs, inputs]) 
    };
    let weight = Parameter::new(weight_value, None);
    let bias = if builder.use_bias {
      let bias_value = if let Some(bias_data) = builder.bias_data {
        RwTensor::from_shape_vec(&device, outputs, bias_data)
      }
      else {
        RwTensor::zeros(
          &device,
          outputs
        )
      };
      Some(Parameter::new(bias_value, None))
    } else { None };
    Self{weight, bias}
  }
}

impl Layer for Dense {
  type Builder = DenseBuilder;
  fn parameters(&self) -> Vec<ParameterD> {
    let weight = self.weight.clone()
      .into_dyn();
    if let Some(bias) = &self.bias {
      let bias = bias.clone()
        .into_dyn();
      vec![weight, bias]
    }
    else {
      vec![weight]
    }
  }
  fn init_training(&mut self) {
    self.weight.init_grad();
    if let Some(bias) = &mut self.bias {
      bias.init_grad();
    }
  } 
  fn to_builder(&self, with_data: bool) -> DenseBuilder {
    let device = Some(self.weight.value().device().clone());
    let (outputs, inputs) = self.weight.value().dim();
    let (outputs, inputs) = (Some(outputs), Some(inputs));
    let weight_data = if with_data {
      let weight_data = self.weight.value()
        .read()
        .unwrap()
        .as_slice()
        .into_owned();
      Some(weight_data)
    } else { None };
    let use_bias = self.bias.is_some();
    let bias_data = if with_data {
      if let Some(bias) = &self.bias {
        let bias_data = bias.value()
          .read()
          .unwrap()
          .as_slice()
          .into_owned();
        Some(bias_data)
      } else { None }
    } else { None };
    DenseBuilder {
      device,
      inputs,
      outputs,
      weight_data,
      use_bias,
      bias_data
    }
  }
}

impl Inference<Ix2> for Dense {
  type OutputDim = Ix2;
  fn infer(&self, input: &TensorView2<f32>) -> Tensor2<f32> {
    let weight = self.weight.value()
      .read()
      .unwrap();
    if let Some(bias) = &self.bias {
      let bias = bias.value()
        .read()
        .unwrap();
      let outputs = bias.dim();
      input.dense(&weight.view(), Some(&bias.view())) 
    }
    else {
      input.dense(&weight.view(), None)
    }
  }
}

impl Forward<Ix2> for Dense {
  fn forward(&self, input: &Variable2, train: bool) -> Variable2 {
    if train {
      debug_assert!(self.weight.grad().is_some());
      if let Some(bias) = &self.bias {
        debug_assert!(bias.grad().is_some());
        input.dense(&self.weight, Some(&bias))
      }
      else {
        input.dense(&self.weight, None)
      }
    }
    else {
      let weight = Parameter::new(self.weight.value().clone(), None);
      if let Some(bias) = &self.bias {
        let bias = Parameter::new(bias.value().clone(), None);
        input.dense(&weight, Some(&bias))
      }
      else {
        input.dense(&weight, None)
      }
    }
  }
}

pub struct Conv2d {
  weight: Parameter4,
  bias: Option<Parameter1>,
  args: Conv2dArgs
}

impl From<Conv2dBuilder> for Conv2d {
  fn from(builder: Conv2dBuilder) -> Self {
    let device = builder.device.expect("Conv2dBuilder requires device to be specified!");
    let inputs = builder.inputs.expect("Conv2dBuilder requires inputs to be specified!");
    let outputs = builder.outputs.expect("Conv2dBuilder requires outputs to specified!");
    let [kh, kw] = builder.kernel.expect("Conv2dBuilder requires kernel to specified!");
    let weight_value = if let Some(weight_data) = builder.weight_data {
      RwTensor::from_shape_vec(&device, [outputs, inputs, kh, kw], weight_data)
    }
    else {
      RwTensor::zeros(&device, [outputs, inputs, kh, kw]) 
    };
    let weight = Parameter::new(weight_value, None);
    let bias = if builder.use_bias {
      let bias_value = if let Some(bias_data) = builder.bias_data {
        RwTensor::from_shape_vec(&device, outputs, bias_data)
      }
      else {
        RwTensor::zeros(
          &device,
          outputs
        )
      };
      Some(Parameter::new(bias_value, None))
    } else { None };
    Self {
      weight, 
      bias, 
      args: builder.args
    }
  }
}

impl Layer for Conv2d {
  type Builder = Conv2dBuilder;
  fn parameters(&self) -> Vec<ParameterD> {
    let weight = self.weight.clone()
      .into_dyn();
    if let Some(bias) = &self.bias {
      let bias = bias.clone()
        .into_dyn();
      vec![weight, bias]
    }
    else {
      vec![weight]
    }
  }
  fn init_training(&mut self) {
    self.weight.init_grad();
    if let Some(bias) = &mut self.bias {
      bias.init_grad();
    }
  } 
  fn to_builder(&self, with_data: bool) -> Conv2dBuilder {
    let device = Some(self.weight.value().device().clone());
    let (outputs, inputs, kh, kw) = self.weight.value().dim();
    let (outputs, inputs) = (Some(outputs), Some(inputs));
    let kernel = Some([kh, kw]);
    let weight_data = if with_data {
      let weight_data = self.weight.value()
        .read()
        .unwrap()
        .as_slice()
        .into_owned();
      Some(weight_data)
    } else { None };
    let use_bias = self.bias.is_some();
    let bias_data = if with_data {
      if let Some(bias) = &self.bias {
        let bias_data = bias.value()
          .read()
          .unwrap()
          .as_slice()
          .into_owned();
        Some(bias_data)
      } else { None }
    } else { None };
    let args = self.args.clone();
    Conv2dBuilder {
      device,
      inputs,
      outputs,
      kernel,
      weight_data,
      use_bias,
      bias_data,
      args
    }
  }
}

impl Inference<Ix4> for Conv2d {
  type OutputDim = Ix4;
  fn infer(&self, input: &TensorView4<f32>) -> Tensor4<f32> {
    let weight = self.weight.value()
      .read()
      .unwrap();
    if let Some(bias) = &self.bias {
      let bias = bias.value()
        .read()
        .unwrap();
      let outputs = bias.dim();
      input.conv2d(&weight.view(), Some(&bias.view()), &self.args) 
    }
    else {
      input.conv2d(&weight.view(), None, &self.args)
    }
  }
}

impl Forward<Ix4> for Conv2d {
  fn forward(&self, input: &Variable4, train: bool) -> Variable4 {
    if train {
      debug_assert!(self.weight.grad().is_some());
      if let Some(bias) = &self.bias {
        debug_assert!(bias.grad().is_some());
        input.conv2d(&self.weight, Some(&bias), &self.args)
      }
      else {
        input.conv2d(&self.weight, None, &self.args)
      }
    }
    else {
      let weight = Parameter::new(self.weight.value().clone(), None);
      if let Some(bias) = &self.bias {
        let outputs = bias.value().dim();
        let bias = Parameter::new(bias.value().clone(), None);
        input.conv2d(&weight, Some(&bias), &self.args)
      }
      else {
        input.conv2d(&weight, None, &self.args)
      }
    }
  }
}

#[derive(Default)]
pub struct Relu {}

impl Layer for Relu {
  type Builder = ReluBuilder;
  fn to_builder(&self, with_data: bool) -> ReluBuilder {
    ReluBuilder::default()
  }
}

impl From<ReluBuilder> for Relu {
  fn from(builder: ReluBuilder) -> Self {
    Self{}
  }
}

impl<D: Dimension> Inference<D> for Relu {
  type OutputDim = D;
  fn infer(&self, input: &TensorView<f32, D>) -> Tensor<f32, D> {
    input.relu()
  }
}

impl<D: Dimension> Forward<D> for Relu {
  fn forward(&self, input: &Variable<D>, train: bool) -> Variable<D> {
    input.relu()
  }
}

pub struct MaxPool2d {
  args: Pool2dArgs
}

impl Layer for MaxPool2d {
  type Builder = MaxPool2dBuilder;
  fn to_builder(&self, with_data: bool) -> MaxPool2dBuilder {
    MaxPool2dBuilder {
      args: self.args
    }
  }
} 
  
impl From<MaxPool2dBuilder> for MaxPool2d {
  fn from(builder: MaxPool2dBuilder) -> Self {
    Self{args: builder.args}
  }
}

impl Inference<Ix4> for MaxPool2d {
  type OutputDim = Ix4;
  fn infer(&self, input: &TensorView4<f32>) -> Tensor4<f32> {
    input.max_pool2d(&self.args)
  }
}

impl Forward<Ix4> for MaxPool2d {
  fn forward(&self, input: &Variable4, train: bool) -> Variable4 {
    input.max_pool2d(&self.args)
  }
}

/* Prototype
pub struct DataParallel<L: Layer> {
  layers: Vec<L>
}

impl<L: Layer> Layer for DataParallel<L> {
  type Builder = DataParallelBuilder<L>;
  fn parameters(&self) -> Vec<ParameterD> {
    self.layers.iter()
      .flat_map(|layer| layer.parameters())
      .collect()
  }
  fn init_training(&mut self) {
    self.layers.iter_mut()
      .for_each(|l| l.init_training());
  }
  fn to_builder(&self, with_data: bool) -> DataParallelBuilder<L> {
    unimplemented!()
  }
}

impl<L: Layer> From<DataParallelBuilder<L>> for DataParallel<L> {
  fn from(builder: DataParallelBuilder<L>) -> Self {
    unimplemented!();
  }
}
*/
