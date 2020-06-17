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
  DenseBuilder,
  Conv2dBuilder
};

pub trait Layer {
  fn params(&self) -> Vec<ParameterD> { Vec::new() }
  fn init_training(&mut self) {}
}

pub trait Inference<D: Dimension> {
  fn infer(&self, input: &TensorView<f32, D>) -> Tensor<f32, D>;
}

pub trait Forward<D: Dimension> {
  fn forward(&self, input: &Variable<D>, train: bool) -> Variable<D>;
}

pub struct Dense {
  weight: Parameter2,
  bias: Option<Parameter1>
}

impl Dense {
  pub fn builder(device: &Device) -> DenseBuilder {
    DenseBuilder {
      device: device.clone(),
      inputs: None,
      outputs: None,
      weight_data: None,
      use_bias: false
    }
  }
}

impl From<DenseBuilder> for Dense {
  fn from(builder: DenseBuilder) -> Self {
    let inputs = builder.inputs.expect("DenseBuilder requires inputs to be specified!");
    let outputs = builder.outputs.expect("DenseBuilder requires outputs to specified!");
    let weight_value = if let Some(weight_data) = builder.weight_data {
      RwTensor::from_shape_vec(&builder.device, [outputs, inputs], weight_data)
    }
    else {
      RwTensor::zeros(&builder.device, [outputs, inputs]) 
    };
    let weight = Parameter::new(weight_value, None);
    let bias = if builder.use_bias {
      let bias_value = RwTensor::zeros(
        &builder.device,
        outputs
      );
      Some(Parameter::new(bias_value, None))
    } else { None };
    Self{weight, bias}
  }
}

impl Layer for Dense {
  fn params(&self) -> Vec<ParameterD> {
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
}

impl Inference<Ix2> for Dense {
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

impl Conv2d {
  pub fn builder(device: &Device) -> Conv2dBuilder {
    Conv2dBuilder {
      device: device.clone(),
      inputs: None,
      outputs: None,
      kernel: None,
      weight_data: None,
      use_bias: false,
      args: Conv2dArgs::default()
    }
  }
}

impl From<Conv2dBuilder> for Conv2d {
  fn from(builder: Conv2dBuilder) -> Self {
    let inputs = builder.inputs.expect("Conv2dBuilder requires inputs to be specified!");
    let outputs = builder.outputs.expect("Conv2dBuilder requires outputs to specified!");
    let [kh, kw] = builder.kernel.expect("Conv2dBuilder requires kernel to specified!");
    let weight_value = if let Some(weight_data) = builder.weight_data {
      RwTensor::from_shape_vec(&builder.device, [outputs, inputs, kh, kw], weight_data)
    }
    else {
      RwTensor::zeros(&builder.device, [outputs, inputs, kh, kw]) 
    };
    let weight = Parameter::new(weight_value, None);
    let bias = if builder.use_bias {
      let bias_value = RwTensor::zeros(
        &builder.device,
        outputs
      );
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
  fn params(&self) -> Vec<ParameterD> {
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
}

impl Inference<Ix4> for Conv2d {
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

impl Layer for Relu {}

impl<D: Dimension> Inference<D> for Relu {
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
