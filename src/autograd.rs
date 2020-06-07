use crate::{Num, Device, Transpose, Tensor, Tensor2, ArcTensor, ArcTensor2, RwTensor, RwTensor0, RwTensor2};
use std::sync::{Arc, Weak, Mutex};
use ndarray::{Dimension, Ix0, Ix2};
use num_traits::ToPrimitive;

#[doc(hidden)]
#[proxy_enum::proxy(BackwardOp)]
pub mod backward_op_proxy {
  use super::{
    DenseBackwardInput,
    DenseBackwardWeight,
    DenseBackwardBias,
    CrossEntropyBackward
  };
  
  pub enum BackwardOp {
    DenseBackwardInput(DenseBackwardInput),
    DenseBackwardWeight(DenseBackwardWeight),
    DenseBackwardBias(DenseBackwardBias),
    CrossEntropyBackward(CrossEntropyBackward)   
  }
  
  impl BackwardOp {
    #[implement]
    pub(super) fn exec(&self) {}
  }
}
#[cfg(not(feature="dyn_graph"))]
#[doc(inline)]
pub use backward_op_proxy::BackwardOp;

#[cfg(feature="dyn_graph")]
pub type BackwardOp = Box<dyn Fn()>;

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

impl From<DenseBackwardInput> for BackwardOp {
  fn from(op: DenseBackwardInput) -> Self {
    #[cfg(not(feature="dyn_graph"))]
    { return BackwardOp::DenseBackwardInput(op); }
    #[cfg(feature="dyn_graph")]
    { return Box::new(move || op.exec()); }
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

impl From<DenseBackwardWeight> for BackwardOp {
  fn from(op: DenseBackwardWeight) -> Self {
    #[cfg(not(feature="dyn_graph"))]
    { return BackwardOp::DenseBackwardWeight(op); }
    #[cfg(feature="dyn_graph")]
    { return Box::new(move || op.exec()); }
  }
}

#[doc(hidden)]
pub struct DenseBackwardBias {
  bias_grad: RwTensor2<f32>,
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

impl From<DenseBackwardBias> for BackwardOp {
  fn from(op: DenseBackwardBias) -> Self {
    #[cfg(not(feature="dyn_graph"))]
    { return BackwardOp::DenseBackwardBias(op); }
    #[cfg(feature="dyn_graph")]
    { return Box::new(move || op.exec()); }
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

impl From<CrossEntropyBackward> for BackwardOp {
  fn from(op: CrossEntropyBackward) -> Self {
    #[cfg(not(feature="dyn_graph"))]
    { return BackwardOp::CrossEntropyBackward(op); }
    #[cfg(feature="dyn_graph")]
    { return Box::new(move || op.exec()); }
  }
}

pub struct Graph {
  backward_ops: Mutex<Vec<BackwardOp>>
}

impl Graph {
  pub fn new() -> Arc<Self> {
    Arc::new(Self{backward_ops: Mutex::new(Vec::new())})
  }
  fn backward_op(&self, op: impl Into<BackwardOp>) {
    self.backward_ops.lock()
      .unwrap()
      .push(op.into());
  }
  fn exec(self) {
    self.backward_ops.into_inner()
      .unwrap()
      .iter()
      .rev()
      .for_each(|op| {
        #[cfg(not(feature="dyn_graph"))]
        { op.exec(); }
        #[cfg(feature="dyn_graph")]
        { panic!("dyn_graph"); op(); }
      });
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
}

impl Variable2 {
  pub fn dense(&self, weight: &Parameter2, bias: Option<&Parameter2>) -> Self {
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
        graph.backward_op(DenseBackwardWeight {
          input: self.value.clone(),
          weight_grad: weight_grad.clone(),
          output_grad: output_grad.clone()
        });
      }
      if let Some(bias) = &bias {
        if let Some(bias_grad) = &bias.grad {
          graph.backward_op(DenseBackwardBias {
            bias_grad: bias_grad.clone(), 
            output_grad: output_grad.clone()
          });
        } 
      }
      if let Some(input_grad) = &self.grad {
        graph.backward_op(DenseBackwardInput {
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
      graph.backward_op(CrossEntropyBackward {
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
    Arc::try_unwrap(graph)
      .ok()
      .unwrap()
      .exec();
  }
}

#[derive(Clone)]
pub struct Parameter<D: Dimension> {
  value: RwTensor<f32, D>,
  grad: Option<RwTensor<f32, D>>
} 

pub type Parameter2 = Parameter<Ix2>;

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
}
