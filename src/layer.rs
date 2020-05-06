use super::{AutographResult, Element, Variable, Parameter, Autograd, ForwardMode, BackwardMode, Activation};
use super::backend::DenseOp;
use std::cell::Cell;

pub mod builders;

pub trait Layer: Autograd {
  fn output(&self) -> &Variable;
  fn collect_parameters(&self, parameters: &mut Vec<Parameter>);
  fn parameters(&self) -> Vec<Parameter> {
    let mut parameters = Vec::new();
    self.collect_parameters(&mut parameters);
    parameters
  }
}

pub struct Dense {
  input: Variable,
  weight: Parameter,
  bias: Option<Parameter>,
  act: Option<Activation>,
  output: Variable,
  backward_mode: Cell<Option<BackwardMode>>,
  ops: Vec<DenseOp>
}

impl Autograd for Dense {
  fn forward(&self, mode: ForwardMode) -> AutographResult<()> {
    for op in &self.ops {
      op.forward(mode)?;
    }
    self.backward_mode.set(mode.backward_mode());
    Ok(())
  } 
  fn backward(&self) -> AutographResult<()> {
    let mode = self.backward_mode.replace(None)
      .unwrap();
    for op in &self.ops {
      op.backward(mode)?;
    }
    Ok(())
  }
}

impl Layer for Dense {
  fn output(&self) -> &Variable { &self.output }
  fn collect_parameters(&self, parameters: &mut Vec<Parameter>) {
    parameters.push(self.weight.clone());
    if let Some(bias) = self.bias.as_ref() {
      parameters.push(bias.clone());
    }
  }
}
