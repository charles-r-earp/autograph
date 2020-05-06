use super::{AutographResult, Unsigned, Tensor, TensorRepr, Variable, Autograd, ForwardMode, BackwardMode};
use super::backend::CrossEntropyOp;
use std::cell::Cell;

pub struct CrossEntropy<U: Unsigned> {
  input: Variable,
  target: Tensor<U>,
  output: Tensor<f32>,
  backward_mode: Cell<Option<BackwardMode>>,
  avg_loss: Cell<Option<f32>>,
  ops: Vec<CrossEntropyOp<U>>
}

impl<U: Unsigned> CrossEntropy<U> {
  pub fn new(input: &Variable, target: &Tensor<U>) -> AutographResult<Self> {
    let input = input.clone();
    let target = target.clone();
    let devices = input.devices();
    let output = Tensor::zeros(devices, input.dims(), TensorRepr::Batched)?;
    let ops = {
      let batch_size = input.dims()[0];
      let nclasses = input.dims()[1];
      let mut input_duals = input.duals();
      let mut target = target.buffers();
      let mut output = output.buffers();
      let mut ops = Vec::with_capacity(devices.len());
      for device in devices {
        let input = input_duals.next().unwrap();
        let target = target.next().unwrap();
        let output = output.next().unwrap();
        ops.push(CrossEntropyOp::new(device, batch_size, nclasses, input, target, output)?);
      }
      ops
    };
    let backward_mode = Cell::new(None);
    let avg_loss = Cell::new(Some(0.));
    Ok(Self{input, target, output, backward_mode, avg_loss, ops})
  }
  pub fn average_loss(&self) -> AutographResult<f32> {
    use num_traits::AsPrimitive;
    if let Some(loss) = self.avg_loss.get() {
      Ok(loss)
    }
    else {
      let loss: f32 = self.output.to_vec()?
        .iter()
        .sum(); 
      let loss = loss / AsPrimitive::<f32>::as_(self.input.dims()[0]);
      self.avg_loss.set(Some(loss));
      Ok(loss)
    }
  }
}

impl<U: Unsigned> Autograd for CrossEntropy<U> {
  fn forward(&self, mode: ForwardMode) -> AutographResult<()> {
    self.backward_mode.set(mode.backward_mode());
    self.avg_loss.set(None);
    for op in &self.ops {
      op.forward(mode)?;
    }
    Ok(())
  }
  fn backward(&self) -> AutographResult<()> {
    let mode = self.backward_mode.replace(None).unwrap();
    for op in &self.ops {
      op.backward(mode)?;
    }
    Ok(())
  }
}
