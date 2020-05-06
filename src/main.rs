#![allow(warnings)]
use autograph::{Device, Tensor, TensorRepr, Variable, Autograd, ForwardMode};
use autograph::layer::{Layer, builders::LayerBuilder, Dense};
use autograph::loss::CrossEntropy;
use std::time::Instant;
use num_traits::AsPrimitive;

fn main() {
  let batch_size = 100;
  let devices = vec![Device::cpu()];
  let x = Variable::zeros(&devices, [batch_size, 28*28], false).unwrap();
  let model = Dense::builder()
    .input(&x)
    .units(10)
    .bias()
    .build()
    .unwrap();
  let t = Tensor::zeros(&devices, [batch_size], TensorRepr::Batched).unwrap();
  let criterion = CrossEntropy::new(model.output(), &t).unwrap();
  let mut loss = 0f32;
  let mut correct = 0;
  let start = Instant::now();
  for _ in 0 .. 10_000 / batch_size {
    x.value().write(&vec![1.; batch_size*28*28]).unwrap();
    t.write(&vec![4; batch_size]).unwrap();
    model.forward(ForwardMode::Infer).unwrap();
    criterion.forward(ForwardMode::Infer).unwrap();
    let y = model.output().value().to_vec().unwrap();
    loss += criterion.average_loss().unwrap();
    // accuracy(y, t) 
  }
  let loss: f32 = loss / AsPrimitive::<f32>::as_(10_000 / batch_size);
  let end = Instant::now();
  println!("loss: {:?} elapsed: {:?}", loss, end - start);
}


 
