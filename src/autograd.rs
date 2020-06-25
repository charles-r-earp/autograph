use crate::{
  Num, 
  DataOwned,
  Device, 
  Transpose, 
  Buffer,
  RwRepr,
  Tensor, Tensor2, 
  ArcTensor, ArcTensor2, ArcTensor4, ArcTensorD,
  RwTensor, RwTensor0, RwTensor1, RwTensor2, RwTensor4, RwTensorD,
  RwReadTensor,
  RwWriteTensor,
  Conv2dArgs, Pool2dArgs
};
use std::sync::{Arc, Weak, Mutex, LockResult, PoisonError};
use ndarray::{IntoDimension, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, IxDyn, RemoveAxis};
use num_traits::ToPrimitive;

#[derive(Clone)]
pub struct Gradient<D: Dimension> {
  tensor: RwTensor<f32, D>
}

pub type Gradient0 = Gradient<Ix0>;
pub type Gradient1 = Gradient<Ix1>;
pub type Gradient2 = Gradient<Ix2>;
pub type Gradient3 = Gradient<Ix3>;
pub type Gradient4 = Gradient<Ix4>;
pub type GradientD = Gradient<IxDyn>;

impl<D: Dimension> Gradient<D> {
  fn new(device: &Device, shape: impl IntoDimension<Dim=D>) -> Self {
    let device = device.clone();
    let dim = shape.into_dimension();
    let buffer = unsafe { Buffer::uninitialized(&device, 0) };
    let data = RwRepr::from_buffer(buffer);
    let tensor = RwTensor {
      device,
      dim,
      data
    };
    Self {
      tensor
    }
  }
  pub fn read(&self) -> Option<LockResult<RwReadTensor<f32, D>>> {
    match self.tensor.read() {
      Ok(x) => {
        if x.data.buffer.len() != 0 {
          Some(Ok(x))
        }
        else {
          None
        }
      },
      Err(poison_error) => {
        let x = poison_error.into_inner();
        if x.data.buffer.len() != 0 {
          Some(Err(PoisonError::new(x)))
        }
        else {
          None
        }
      }
    }  
  }
  pub fn write(&self) -> LockResult<RwWriteTensor<f32, D>> {
    self.tensor.write()
      .map(|mut x| {
        if x.data.buffer.len() == 0 {
          let device = &x.device;
          let len = x.dim.size();
          *x.data.buffer = Buffer::zeros(device, len);
        }
        x
      })
      .map_err(|poison_error| {
        let mut x = poison_error.into_inner();
        if x.data.buffer.len() == 0 {
          let device = &x.device;
          let len = x.dim.size();
          *x.data.buffer = Buffer::zeros(device, len);
        }
        PoisonError::new(x)
      })
  }
  fn into_dyn(self) -> Gradient<IxDyn> {
    Gradient{tensor: self.tensor.into_dyn()}
  }
  fn into_dimensionality<D2: Dimension>(self) -> Option<Gradient<D2>> {
    self.tensor.clone().into_dimensionality()
      .map(|tensor| Gradient{tensor})
  }
  fn into_shape<D2: Dimension>(self, shape: impl IntoDimension<Dim=D2>) -> Option<Gradient<D2>> {
    self.tensor.into_shape(shape)
      .map(|tensor| Gradient{tensor})
  }
}

#[doc(hidden)]
#[proxy_enum::proxy(BackwardVariableOp)]
pub mod backward_variable_op_proxy {
  use super::{
    DenseBackwardInput,
    CrossEntropyBackward,
    Conv2dBackwardInput,
    MaxPool2dBackward,
    ReluBackward
  };
  
  pub enum BackwardVariableOp {
    DenseBackwardInput(DenseBackwardInput),
    CrossEntropyBackward(CrossEntropyBackward),
    Conv2dBackwardInput(Conv2dBackwardInput),
    MaxPool2dBackward(MaxPool2dBackward),
    ReluBackward(ReluBackward)
  }
  
  impl BackwardVariableOp {
    #[implement]
    pub(super) fn exec(&self) {}
  }
}
use backward_variable_op_proxy::BackwardVariableOp;

#[doc(hidden)]
#[proxy_enum::proxy(BackwardParameterOp)]
pub mod backward_parameter_op_proxy {
  use super::{
    DenseBackwardWeight,
    DenseBackwardBias,
    Conv2dBackwardWeightBias
  };
  
  pub enum BackwardParameterOp {
    DenseBackwardWeight(DenseBackwardWeight),
    DenseBackwardBias(DenseBackwardBias),
    Conv2dBackwardWeightBias(Conv2dBackwardWeightBias)
  }
  
  impl BackwardParameterOp {
    #[implement]
    pub(super) fn exec(&self) {}
  }
}
use backward_parameter_op_proxy::BackwardParameterOp;

#[doc(hidden)]
pub struct DenseBackwardInput {
  input_grad: Gradient2,
  weight: RwTensor2<f32>,
  output_grad: Gradient2
}

impl DenseBackwardInput {
  fn exec(&self) {
    let mut input_grad = self.input_grad.write()
      .unwrap();
    let weight = self.weight.read()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    crate::gemm(1., &output_grad, Transpose::No, &weight, Transpose::No, 1., &mut input_grad);
  }
}

impl From<DenseBackwardInput> for BackwardVariableOp {
  fn from(op: DenseBackwardInput) -> Self {
    BackwardVariableOp::DenseBackwardInput(op)
  }
}

#[doc(hidden)]
pub struct DenseBackwardWeight {
  input: ArcTensor2<f32>,
  weight_grad: Gradient2,
  output_grad: Gradient2
}

impl DenseBackwardWeight {
  fn exec(&self) {
    let (batch_size, inputs) = self.input.dim();
    let mut weight_grad = self.weight_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    crate::gemm(1., &output_grad, Transpose::Yes, &self.input, Transpose::No, 1., &mut weight_grad); 
  }
}

impl From<DenseBackwardWeight> for BackwardParameterOp {
  fn from(op: DenseBackwardWeight) -> Self {
    BackwardParameterOp::DenseBackwardWeight(op)
  }
}

#[doc(hidden)]
pub struct DenseBackwardBias {
  bias_grad: Gradient1,
  output_grad: Gradient2
}

impl DenseBackwardBias {
  fn exec(&self) {
    let mut bias_grad = self.bias_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    crate::broadcast_backward(&mut bias_grad, &output_grad);
  }
}

impl From<DenseBackwardBias> for BackwardParameterOp {
  fn from(op: DenseBackwardBias) -> Self {
    BackwardParameterOp::DenseBackwardBias(op)
  }
}

#[doc(hidden)]
pub struct CrossEntropyBackward {
  input: ArcTensor2<f32>,
  input_grad: Gradient2,
  target: ArcTensor2<f32>,
  output_grad: Gradient0
}

impl CrossEntropyBackward {
  fn exec(&self) {
    let mut input_grad = self.input_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    crate::cross_entropy_backward(&self.input, &mut input_grad, &self.target, &output_grad);
  }
}

impl From<CrossEntropyBackward> for BackwardVariableOp {
  fn from(op: CrossEntropyBackward) -> Self {
    BackwardVariableOp::CrossEntropyBackward(op)
  }
}

#[doc(hidden)]
pub struct Conv2dBackwardInput {
  input_grad: Gradient4,
  weight: RwTensor4<f32>,
  args: Conv2dArgs,
  output_grad: Gradient4,
}

impl Conv2dBackwardInput {
  fn exec(&self) {
    let mut input_grad = self.input_grad.write()
      .unwrap();
    let weight = self.weight.read()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    crate::conv2d_backward_input(
      &mut input_grad, 
      &weight.view(),
      &self.args, 
      &output_grad.view()
    );
  }
}

impl From<Conv2dBackwardInput> for BackwardVariableOp {
  fn from(op: Conv2dBackwardInput) -> Self {
    BackwardVariableOp::Conv2dBackwardInput(op)
  }
}

#[doc(hidden)]
pub struct Conv2dBackwardWeightBias {
  input: ArcTensor4<f32>,
  weight_grad: Gradient4,
  bias_grad: Option<Gradient1>,
  args: Conv2dArgs,
  output_grad: Gradient4,
}

impl Conv2dBackwardWeightBias {
  fn exec(&self) {
    let mut weight_grad = self.weight_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    if let Some(bias_grad) = &self.bias_grad {
      let mut bias_grad = bias_grad.write()
        .unwrap();
      crate::conv2d_backward_weight_bias(
        &self.input, 
        &mut weight_grad.view_mut(), 
        Some(&mut bias_grad.view_mut()), 
        &self.args, 
        &output_grad.view()
      );
    }
    else {
      crate::conv2d_backward_weight_bias(
        &self.input, 
        &mut weight_grad.view_mut(), 
        None, 
        &self.args, 
        &output_grad.view()
      );
    }
  }
}

impl From<Conv2dBackwardWeightBias> for BackwardParameterOp {
  fn from(op: Conv2dBackwardWeightBias) -> Self {
    BackwardParameterOp::Conv2dBackwardWeightBias(op)
  }
}

#[doc(hidden)]
pub struct MaxPool2dBackward {
  input: ArcTensor4<f32>,
  input_grad: Gradient4,
  args: Pool2dArgs,
  workspace: Option<Buffer<u8>>,
  output_grad: Gradient4
}

impl MaxPool2dBackward {
  fn exec(&self) {
    let mut input_grad = self.input_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    crate::max_pool2d_backward(&self.input, &mut input_grad, &self.args, self.workspace.as_ref(), &output_grad);
  }
}

impl From<MaxPool2dBackward> for BackwardVariableOp {
  fn from(op: MaxPool2dBackward) -> Self {
    BackwardVariableOp::MaxPool2dBackward(op)
  }
}

#[doc(hidden)]
pub struct ReluBackward {
  input: ArcTensorD<f32>,
  input_grad: GradientD,
  output_grad: GradientD
}

impl ReluBackward {
  fn exec(&self) {
    let mut input_grad = self.input_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap()
      .unwrap();
    crate::relu_backward(&self.input, &mut input_grad, &output_grad);
  }
}

impl From<ReluBackward> for BackwardVariableOp {
  fn from(op: ReluBackward) -> Self {
    BackwardVariableOp::ReluBackward(op)
  }
}

pub struct Graph {
  backward_variable_ops: Mutex<Vec<BackwardVariableOp>>,
  backward_parameter_ops: Mutex<Vec<BackwardParameterOp>>
}

impl Graph {
  pub fn new() -> Arc<Self> {
    Arc::new(Self {
      backward_variable_ops: Mutex::new(Vec::new()),
      backward_parameter_ops: Mutex::new(Vec::new())  
    })
  }
  fn backward_variable_op(&self, op: impl Into<BackwardVariableOp>) {
    self.backward_variable_ops.lock()
      .unwrap()
      .push(op.into());
  }
  fn backward_parameter_op(&self, op: impl Into<BackwardParameterOp>) {
    self.backward_parameter_ops.lock()
      .unwrap()
      .push(op.into());
  }
  fn exec_backward_variable_ops(&self) {
    self.backward_variable_ops.lock()
      .unwrap()
      .iter()
      .rev()
      .for_each(|op| op.exec());
  } 
  fn exec_backward_parameter_ops(&self) {
    self.backward_parameter_ops.lock()
      .unwrap()
      .iter()
      .rev()
      .for_each(|op| op.exec());
  }
}

#[derive(Clone)]
pub struct Variable<D: Dimension> {
  graph: Weak<Graph>,
  value: ArcTensor<f32, D>,
  grad: Option<Gradient<D>>
}

pub type Variable0 = Variable<Ix0>;
pub type Variable2 = Variable<Ix2>;
pub type Variable4 = Variable<Ix4>;
pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> Variable<D> {
  pub fn new(graph: Option<&Arc<Graph>>, value: impl Into<ArcTensor<f32, D>>, requires_grad: bool) -> Self {
    let value = value.into();
    let (graph, grad) = if let Some(graph) = graph {
      let graph = Arc::downgrade(graph);
      let grad = if requires_grad {
        Some(Gradient::new(value.device(), value.raw_dim()))
      } else { None };
      (graph, grad)
    }
    else {
      (Weak::new(), None)
    };
    Self {
      graph, 
      value, 
      grad
    }
  }
  pub fn device(&self) -> &Device {
    self.value.device()
  }
  pub fn value(&self) -> &ArcTensor<f32, D> {
    &self.value
  }
  pub fn grad(&self) -> Option<&Gradient<D>> {
    self.grad.as_ref()
  }
  pub fn detach(&self) -> Self {
    Self {
      graph: Weak::new(),
      value: self.value.clone(),
      grad: None
    }
  }
  pub fn into_dyn(self) -> VariableD {
    let Variable{graph, value, grad} = self;
    Variable {
      graph,
      value: value.into_dyn(),
      grad: grad.map(|grad| grad.into_dyn())
    }
  }
  pub fn into_dimensionality<D2: Dimension>(self) -> Option<Variable<D2>> {
    let Variable{graph, value, grad} = self;
    value.into_dimensionality()
      .map(|value| {
        let grad = grad.map(|grad| { 
          grad.into_dimensionality()
            .unwrap()
        });
        Variable {
          graph,
          value,
          grad
        }
      })
  }
  pub fn into_shape<D2: Dimension>(self, shape: impl IntoDimension<Dim=D2>) -> Option<Variable<D2>> {
    let Variable{graph, value, grad} = self;
    value.into_shape(shape)
      .map(|value| {
        let grad = grad.map(|grad| {
          grad.into_shape(value.raw_dim())
            .unwrap()
        });
        Variable {
          graph,
          value,
          grad
        }
      })
  }
  pub fn flatten(&self) -> Variable2
    where D: RemoveAxis {
    let dims = self.value.dim.slice();
    let batch_size = dims[0];
    let inputs = dims[1..].iter().product();
    self.clone()
      .into_shape([batch_size, inputs])
      .unwrap()
  }
  pub fn relu(&self) -> Self {
    let graph = Weak::upgrade(&self.graph);
    let output = Self::new(
      graph.as_ref(),
      self.value.relu(),
      self.grad().is_some()
    );
    if let Some(output_grad) = output.grad() {
      let graph = graph.unwrap();
      let output_grad = output_grad.clone()
        .into_dyn();
      let input = self.value.clone()
        .into_dyn();
      let input_grad = self.grad()
        .unwrap()
        .clone()
        .into_dyn(); 
      graph.backward_variable_op(ReluBackward {
        input,
        input_grad,
        output_grad
      });
    }
    output
  }
}

impl Variable0 {
  pub fn backward(&self, graph: Arc<Graph>) {
    debug_assert!(
      Arc::ptr_eq(&graph, &Weak::upgrade(&self.graph).unwrap())
    );
    self.grad()
      .unwrap()
      .write()
      .unwrap()
      .fill(1.);
    let graph = Arc::try_unwrap(graph)
      .ok()
      .unwrap();
    graph.exec_backward_variable_ops();
    graph.exec_backward_parameter_ops();
  }
}

impl Variable2 {
  pub fn dense(&self, weight: &Parameter2, bias: Option<&Parameter1>) -> Self {
    let graph = Weak::upgrade(&self.graph);
    let output_value = {
      let weight_value = weight.value.read().unwrap();
      let bias_value = bias.map(|b| b.value.read().unwrap());
      self.value.dense(&weight_value.view(), bias_value.as_ref().map(|b| b.view()).as_ref())
    };
    let output = Self::new(
      graph.as_ref(), 
      output_value,
      self.grad.is_some() || weight.grad.is_some() 
        || bias.as_ref().map_or(false, |b| b.grad.is_some())
    );
    if let Some(output_grad) = output.grad() {
      let graph = graph.unwrap();
      if let Some(weight_grad) = weight.grad() {
        graph.backward_parameter_op(DenseBackwardWeight {
          input: self.value.clone(),
          weight_grad: weight_grad.clone(),
          output_grad: output_grad.clone()
        });
      }
      if let Some(bias) = &bias {
        if let Some(bias_grad) = bias.grad() {
          graph.backward_parameter_op(DenseBackwardBias {
            bias_grad: bias_grad.clone(), 
            output_grad: output_grad.clone()
          });
        } 
      }
      if let Some(input_grad) = self.grad() {
        graph.backward_variable_op(DenseBackwardInput {
          input_grad: input_grad.clone(),
          weight: weight.value.clone(),
          output_grad: output_grad.clone()
        });
      }
    }
    output
  }
  pub fn cross_entropy_loss(&self, target: &ArcTensor2<f32>) -> Variable0 {
    let graph = Weak::upgrade(&self.graph);
    let output = Variable::new(
      graph.as_ref(),
      self.value.cross_entropy_loss(&target.view()),
      self.grad.is_some()
    );
    if let Some(output_grad) = output.grad() {
      let graph = graph.unwrap();
      let input = self.value.clone();
      let input_grad = self.grad().unwrap().clone();
      let target = target.clone();
      let output_grad = output_grad.clone();
      graph.backward_variable_op(CrossEntropyBackward {
        input,
        input_grad,
        target,
        output_grad,
      });
    }
    output
  }
}

impl Variable4 {
  pub fn conv2d(&self, weight: &Parameter4, bias: Option<&Parameter1>, args: &Conv2dArgs) -> Self {
    let graph = Weak::upgrade(&self.graph);
    let output_value = {
      let weight_value = weight.value.read().unwrap();
      let bias_value = bias.map(|b| b.value.read().unwrap());
      self.value.conv2d(
        &weight_value.view(), 
        bias_value.as_ref().map(|b| b.view()).as_ref(), 
        args
      )
    };
    let output = Self::new(
      graph.as_ref(),
      output_value,
      self.grad.is_some() || weight.grad.is_some() 
    );
    if let Some(output_grad) = output.grad() {
      let graph = graph.unwrap();
      if let Some(weight_grad) = weight.grad() {
        let bias_grad = bias.map(|bias| 
          bias.grad().unwrap().clone()
        );
        graph.backward_parameter_op(Conv2dBackwardWeightBias {
          input: self.value.clone(),
          weight_grad: weight_grad.clone(),
          bias_grad,
          args: args.clone(),
          output_grad: output_grad.clone()
        });
      }
      if let Some(input_grad) = self.grad() {
        graph.backward_variable_op(Conv2dBackwardInput {
          input_grad: input_grad.clone(),
          weight: weight.value.clone(),
          args: args.clone(),
          output_grad: output_grad.clone()
        });
      }
    }
    output
  }
  pub fn max_pool2d(&self, args: &Pool2dArgs) -> Self {
    let graph = Weak::upgrade(&self.graph);
    let train = self.grad.is_some(); 
    let (output_value, workspace) = crate::max_pool2d_forward(
      &self.value, 
      args, 
      train
    );
    let output = Self::new(
      graph.as_ref(),
      output_value,
      train
    );
    if let Some(output_grad) = output.grad() {
      let graph = graph.unwrap();
      let input = self.value.clone();
      let input_grad = self.grad().unwrap().clone();
      let args = args.clone();
      let output_grad = output_grad.clone();
      graph.backward_variable_op(MaxPool2dBackward {
        input,
        input_grad,
        args,
        workspace,
        output_grad
      })
    }
    output
  }
}

#[derive(Clone)]
pub struct Parameter<D: Dimension> {
  value: RwTensor<f32, D>,
  grad: Option<Gradient<D>>
} 

pub type Parameter1 = Parameter<Ix1>;
pub type Parameter2 = Parameter<Ix2>;
pub type Parameter4 = Parameter<Ix4>;
pub type ParameterD = Parameter<IxDyn>;

impl<D: Dimension> Parameter<D> {
  pub fn new(value: impl Into<RwTensor<f32, D>>) -> Self {
    let value = value.into();
    let grad = None;
    Self{value, grad}
  }
  pub fn value(&self) -> &RwTensor<f32, D> {
    &self.value
  }
  pub fn grad(&self) -> Option<&Gradient<D>> {
    self.grad.as_ref()
  }
  pub fn set_training(&mut self, training: bool) {
    if training {
      self.grad.replace(
        Gradient::new(
          self.value.device(),
          self.value.raw_dim()
        )
      );
    }
    else {
      self.grad = None;
    }
  }
  pub fn into_dyn(self) -> ParameterD {
    let Parameter{value, grad} = self;
    Parameter {
      value: value.into_dyn(),
      grad: grad.map(|grad| grad.into_dyn())
    }
  }
  pub fn into_dimensionality<D2: Dimension>(self) -> Option<Parameter<D2>> {
    let Parameter{value, grad} = self;
    value.into_dimensionality()
      .map(|value| {
        let grad = grad.map(|grad| { 
          grad.into_dimensionality()
            .unwrap()
        });
        Parameter {
          value,
          grad
        }
      })
  }
  pub fn into_shape<D2: Dimension>(self, shape: impl IntoDimension<Dim=D2>) -> Option<Parameter<D2>> {
    let Parameter{value, grad} = self;
    value.into_shape(shape)
      .map(|value| {
        let grad = grad.map(|grad| {
          grad.into_shape(value.raw_dim())
            .unwrap()
        });
        Parameter {
          value,
          grad
        }
      })
  }
}
