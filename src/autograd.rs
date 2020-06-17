use crate::{
  Num, 
  Device, 
  Transpose, 
  Tensor, Tensor2, 
  ArcTensor, ArcTensor2, 
  RwTensor, RwTensor0, RwTensor1, RwTensor2,
  Conv2dArgs
};
use std::sync::{Arc, Weak, Mutex};
use ndarray::{IntoDimension, Dimension, Ix0, Ix1, Ix2, Ix4, IxDyn, RemoveAxis};
use num_traits::ToPrimitive;

#[doc(hidden)]
#[proxy_enum::proxy(BackwardVariableOp)]
pub mod backward_variable_op_proxy {
  use super::{
    DenseBackwardInput,
    CrossEntropyBackward
  };
  
  pub enum BackwardVariableOp {
    DenseBackwardInput(DenseBackwardInput),
    CrossEntropyBackward(CrossEntropyBackward)
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
    DenseBackwardBias
  };
  
  pub enum BackwardParameterOp {
    DenseBackwardWeight(DenseBackwardWeight),
    DenseBackwardBias(DenseBackwardBias)
  }
  
  impl BackwardParameterOp {
    #[implement]
    pub(super) fn exec(&self) {}
  }
}
use backward_parameter_op_proxy::BackwardParameterOp;

#[doc(hidden)]
pub struct DenseBackwardInput {
  input_grad: RwTensor2<f32>,
  weight: RwTensor2<f32>,
  output_grad: RwTensor2<f32>
}

impl DenseBackwardInput {
  fn exec(&self) {
    unimplemented!()
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
  weight_grad: RwTensor2<f32>,
  output_grad: RwTensor2<f32>
}

impl DenseBackwardWeight {
  fn exec(&self) {
    let (batch_size, inputs) = self.input.dim();
    let alpha = batch_size.to_f32()
      .unwrap()
      .recip();
    let mut weight_grad = self.weight_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap();
    crate::gemm(alpha, &output_grad, Transpose::Yes, &self.input, Transpose::No, 1., &mut weight_grad); 
  }
}

impl From<DenseBackwardWeight> for BackwardParameterOp {
  fn from(op: DenseBackwardWeight) -> Self {
    BackwardParameterOp::DenseBackwardWeight(op)
  }
}

#[doc(hidden)]
pub struct DenseBackwardBias {
  bias_grad: RwTensor1<f32>,
  output_grad: RwTensor2<f32>
}

impl DenseBackwardBias {
  fn exec(&self) {
    let mut bias_grad = self.bias_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap();
    crate::broadcast_backward(&mut bias_grad, &output_grad);
  }
}

impl From<DenseBackwardBias> for BackwardParameterOp {
  fn from(op: DenseBackwardBias) -> Self {
    return BackwardParameterOp::DenseBackwardBias(op)
  }
}

#[doc(hidden)]
pub struct CrossEntropyBackward {
  input: ArcTensor2<f32>,
  input_grad: RwTensor2<f32>,
  target: ArcTensor2<f32>,
  output_grad: RwTensor0<f32>
}

impl CrossEntropyBackward {
  fn exec(&self) {
    let mut input_grad = self.input_grad.write()
      .unwrap();
    let output_grad = self.output_grad.read()
      .unwrap();
    crate::cross_entropy_backward(&self.input, &mut input_grad, &self.target, &output_grad);
  }
}

impl From<CrossEntropyBackward> for BackwardVariableOp {
  fn from(op: CrossEntropyBackward) -> Self {
    BackwardVariableOp::CrossEntropyBackward(op)
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
      .for_each(|op| op.exec());
  } 
  fn exec_backward_parameter_ops(&self) {
    self.backward_parameter_ops.lock()
      .unwrap()
      .iter()
      .for_each(|op| op.exec());
  }
}

#[derive(Clone)]
pub struct Variable<D: Dimension> {
  graph: Weak<Graph>,
  value: ArcTensor<f32, D>,
  grad: Option<RwTensor<f32, D>>
}

pub type Variable0 = Variable<Ix0>;
pub type Variable2 = Variable<Ix2>;
pub type Variable4 = Variable<Ix4>;
pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> Variable<D> {
  pub fn new(graph: &Arc<Graph>, value: impl Into<ArcTensor<f32, D>>, grad: Option<RwTensor<f32, D>>) -> Self {
    let value = value.into();
    #[cfg(debug_assertions)]
    {
      if let Some(grad) = grad.as_ref() {
        assert_eq!(value.device(), grad.device());
        assert_eq!(value.raw_dim(), grad.raw_dim());
      }
    }
    let graph = Arc::downgrade(graph);
    Self{graph, value, grad}
  }
  pub fn device(&self) -> &Device {
    self.value.device()
  }
  pub fn value(&self) -> &ArcTensor<f32, D> {
    &self.value
  }
  pub fn grad(&self) -> Option<&RwTensor<f32, D>> {
    self.grad.as_ref()
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
    let graph = Weak::upgrade(&self.graph)
      .unwrap();
    let output_value = ArcTensor::from(
      self.value.relu()
    );
    let output_grad = if self.grad.is_some() {
      Some(RwTensor::zeros(&output_value.device, output_value.raw_dim()))
    } else { None };
    if let Some(output_grad) = &output_grad {
      unimplemented!();
    }
    Self::new(&graph, output_value, output_grad)
  }
}

impl Variable0 {
  pub fn backward(&self, graph: Arc<Graph>) {
    debug_assert!(
      Arc::ptr_eq(&graph, &Weak::upgrade(&self.graph).unwrap())
    );
    self.grad.as_ref()
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
    let graph = Weak::upgrade(&self.graph)
      .unwrap();
    let output_value = ArcTensor::from(
      {
        let weight_value = weight.value.read().unwrap();
        let bias_value = bias.map(|b| b.value.read().unwrap());
        self.value.dense(&weight_value.view(), bias_value.as_ref().map(|b| b.view()).as_ref())
      }
    );
    let output_grad = if self.grad.is_some() || weight.grad.is_some() {
      Some(RwTensor::zeros(self.device(), output_value.raw_dim()))
    } else { None };
    if let Some(output_grad) = &output_grad {
      if let Some(weight_grad) = &weight.grad {
        graph.backward_parameter_op(DenseBackwardWeight {
          input: self.value.clone(),
          weight_grad: weight_grad.clone(),
          output_grad: output_grad.clone()
        });
      }
      if let Some(bias) = &bias {
        if let Some(bias_grad) = &bias.grad {
          graph.backward_parameter_op(DenseBackwardBias {
            bias_grad: bias_grad.clone(), 
            output_grad: output_grad.clone()
          });
        } 
      }
      if let Some(input_grad) = &self.grad {
        graph.backward_variable_op(DenseBackwardInput {
          input_grad: input_grad.clone(),
          weight: weight.value.clone(),
          output_grad: output_grad.clone()
        });
      }
    }
    Self::new(&graph, output_value, output_grad)
  }
  pub fn cross_entropy_loss(&self, target: impl Into<ArcTensor2<f32>>) -> Variable0 {
    let graph = Weak::upgrade(&self.graph)
      .unwrap();
    let target = target.into();
    let output_value = ArcTensor::from(
      self.value.cross_entropy_loss(&target.view())
    );
    let output_grad = self.grad.as_ref()
      .map(|input_grad| {
      let output_grad = RwTensor::zeros(self.device(), ());
      graph.backward_variable_op(CrossEntropyBackward {
        input: self.value.clone(),
        input_grad: input_grad.clone(),
        target,
        output_grad: output_grad.clone()
      });
      output_grad
    });
    Variable::new(&graph, output_value, output_grad)
  }
}

impl Variable4 {
  pub fn conv2d(&self, weight: &Parameter4, bias: Option<&Parameter1>, args: &Conv2dArgs) -> Self {
    let graph = Weak::upgrade(&self.graph)
      .unwrap();
    let output_value = ArcTensor::from(
      {
        let weight_value = weight.value.read().unwrap();
        let bias_value = bias.map(|b| b.value.read().unwrap());
        self.value.conv2d(
          &weight_value.view(), 
          bias_value.as_ref().map(|b| b.view()).as_ref(), 
          args
        )
      }
    );
    let output_grad = if self.grad.is_some() || weight.grad.is_some() {
      Some(RwTensor::zeros(self.device(), output_value.raw_dim()))
    } else { None };
    if let Some(output_grad) = &output_grad {
      if let Some(weight_grad) = &weight.grad {
        unimplemented!()
        /*graph.backward_parameter_op(Conv2dBackwardWeight {
          input: self.value.clone(),
          weight_grad: weight_grad.clone(),
          output_grad: output_grad.clone()
        });*/
      }
      if let Some(bias) = &bias {
        if let Some(bias_grad) = &bias.grad {
          unimplemented!()
          /*graph.backward_parameter_op(DenseBackwardBias {
            bias_grad: bias_grad.clone(), 
            output_grad: output_grad.clone()
          });*/
        } 
      }
      if let Some(input_grad) = &self.grad {
        unimplemented!()
        /*graph.backward_variable_op(DenseBackwardInput {
          input_grad: input_grad.clone(),
          weight: weight.value.clone(),
          output_grad: output_grad.clone()
        });*/
      }
    }
    Variable::new(&graph, output_value, output_grad)
  }
}



#[derive(Clone)]
pub struct Parameter<D: Dimension> {
  value: RwTensor<f32, D>,
  grad: Option<RwTensor<f32, D>>
} 

pub type Parameter1 = Parameter<Ix1>;
pub type Parameter2 = Parameter<Ix2>;
pub type Parameter4 = Parameter<Ix4>;
pub type ParameterD = Parameter<IxDyn>;

impl<D: Dimension> Parameter<D> {
  pub fn new(value: impl Into<RwTensor<f32, D>>, grad: Option<RwTensor<f32, D>>) -> Self {
    let value = value.into();
    #[cfg(debug_assertions)]
    {
      if let Some(grad) = grad.as_ref() {
        assert_eq!(value.device(), grad.device());
        assert_eq!(value.raw_dim(), grad.raw_dim());
      }
    }
    Self{value, grad}
  }
  pub fn value(&self) -> &RwTensor<f32, D> {
    &self.value
  }
  pub fn grad(&self) -> Option<&RwTensor<f32, D>> {
    self.grad.as_ref()
  }
  pub fn init_grad(&mut self) {
    self.grad = Some(RwTensor::zeros(&self.value.device, self.value.raw_dim()));
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
