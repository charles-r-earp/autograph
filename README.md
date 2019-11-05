[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/autograph/LICENSE)
[![Build Status](https://api.travis-ci.com/charles-r-earp/autograph.svg?branch=master)](https://travis-ci.com/charles-r-earp/autograph)

# autograph
Machine Learning Library for Rust

# Concept
High performance Machine Learning Library inspired by Pytorch, built on top of ndarray. Flexible design allows for parallel and incremental execution, enabling dynamic composition of layers and functions, instead of purely statically defined models. 

# Features
- Layers
  - Dense
- Functions
  - Dense
  - Softmax
  - CrossEntropyLoss
  - ClassificationMatches
- Initializers
  - Zeros (Default)
  - Random(rand::Distribution)
  - HeNormal
- Optimizers
  - Default (Learning Rate only)
  - SGD (with Momentum, Dampening, Weight Decay, and Nesterov)

# Example
"/examples/mnist_dense.rs"  
```
use autograph as ag;
use ndarray as nd;
use std::{rc::Rc, time::Instant};

fn main() {
  use ag::{layer, layer::{Forward, Layer}, functional::{CrossEntropyLoss, ClassificationMatches}, optim};
  use num_traits::ToPrimitive;
  let dataset = ag::datasets::Mnist::new();
  let mut model = layer::Dense::builder()
    .units(10)
    .use_bias()
    .build();
  let optimizer = optim::SGD::builder()
    .momentum(0.5)
    .weight_decay(0.1)
    .nesterov()
    .build();
  model.param_iter_mut()
    .for_each(|p| p.set_optimizer(optimizer.clone()));
  println!("{:#?}", &model); 
  model.build(&nd::Array::zeros([1, 1, 28, 28]).into_dyn());
  let nparams = model.param_iter()
    .map(|p| p.view().len())
    .sum::<usize>();
  println!("{} trainable parameters.", nparams);
  let lr = 0.01;
  let now = Instant::now();
  for epoch in 1 ..= 10 {
    let mut train_total = 0;
    let mut train_loss = 0f32;
    let mut train_correct = 0;
    dataset.train(64)
      .for_each(|(x, t)| {
      model.param_iter_mut()
        .for_each(|p| p.zero_grad());
      let (y, loss) = {
        let tape = Rc::new(ag::autograd::Tape::new());
        let x = ag::autograd::Var::new(&tape, x, false);
        let y = model.forward(&x);
        let loss = y.cross_entropy_loss(&t);
        loss.backward();
        (y, loss)
      };
      model.param_iter_mut()
        .for_each(|p| p.step(lr));
      train_total += t.shape()[0];
      train_loss += loss.into_array()
        .into_dimensionality()
        .unwrap()
        .into_scalar();
      train_correct += y.value()
        .classification_matches(&t);
    });
    train_loss /= train_total.to_f32().unwrap();
    let train_accuracy = train_correct.to_f32().unwrap() / train_total.to_f32().unwrap();
    let mut test_total = 0;
    let mut test_loss = 0f32;
    let mut test_correct = 0;
    dataset.test(512)
      .for_each(|(x, t)| {
      let y = model.forward(&x.into_dyn());
      let loss = y.cross_entropy_loss(&t);
      test_total += t.shape()[0];
      test_loss += loss.into_scalar();
      test_correct += y.classification_matches(&t);
    });
    test_loss /= test_total.to_f32().unwrap();
    let test_accuracy = test_correct.to_f32().unwrap() / test_total.to_f32().unwrap();
    println!("[{}] train_loss = {:.5} train_accuracy = {:.2}% test_loss = {:.5} test_accuracy = {:.2}% elapsed = {:.0?}", 
      epoch, train_loss, 100.*train_accuracy, test_loss, 100.*test_accuracy, now.elapsed());
  }
}     
```
output:
```
loading 60,000 train_images: "/../autograph/datasets/mnist/train-images-idx3-ubyte.gz"
loading 60,000 train_labels: "/../autograph/datasets/mnist/train-labels-idx1-ubyte.gz"
loading 10,000 test_images: "/../autograph/datasets/mnist/t10k-images-idx3-ubyte.gz"
loading 10,000 test_labels: "/../autograph/datasets/mnist/t10k-labels-idx1-ubyte.gz"
Dense {
  units: 10,
  kernel: Param {
      value: [] shape=[0], strides=[0], layout=C | F (0x3), dynamic ndim=1,
      grad: None,
      initializer: Some(
          HeNormal,
      ),
      optimizer: Some(
          SGD {
              velocity: [] shape=[0], strides=[0], layout=C | F (0x3), dynamic ndim=1,
              momentum: 0.5,
              dampening: 0.0,
              weight_decay: 0.1,
              nesterov: true,
          },
      ),
  },
  bias: Some(
      Param {
          value: [] shape=[0], strides=[0], layout=C | F (0x3), dynamic ndim=1,
          grad: None,
          initializer: None,
          optimizer: Some(
              SGD {
                  velocity: [] shape=[0], strides=[0], layout=C | F (0x3), dynamic ndim=1,
                  momentum: 0.5,
                  dampening: 0.0,
                  weight_decay: 0.1,
                  nesterov: true,
              },
          ),
      },
  ),
}
7850 trainable parameters.
[1] train_loss = 0.36672 train_accuracy = 89.48% test_loss = 0.30128 test_accuracy = 91.30% elapsed = 223ms
[2] train_loss = 0.30406 train_accuracy = 91.31% test_loss = 0.28703 test_accuracy = 91.79% elapsed = 442ms
[3] train_loss = 0.29374 train_accuracy = 91.79% test_loss = 0.28419 test_accuracy = 92.07% elapsed = 661ms
[4] train_loss = 0.28524 train_accuracy = 91.99% test_loss = 0.28332 test_accuracy = 92.10% elapsed = 879ms
[5] train_loss = 0.28162 train_accuracy = 92.08% test_loss = 0.28571 test_accuracy = 92.30% elapsed = 1s
[6] train_loss = 0.27868 train_accuracy = 92.18% test_loss = 0.29969 test_accuracy = 91.60% elapsed = 1s
[7] train_loss = 0.27721 train_accuracy = 92.26% test_loss = 0.28449 test_accuracy = 91.96% elapsed = 2s
[8] train_loss = 0.27339 train_accuracy = 92.33% test_loss = 0.28589 test_accuracy = 91.96% elapsed = 2s
[9] train_loss = 0.27412 train_accuracy = 92.41% test_loss = 0.28943 test_accuracy = 92.24% elapsed = 2s
[10] train_loss = 0.27002 train_accuracy = 92.42% test_loss = 0.27964 test_accuracy = 92.31% elapsed = 2s
```

# Next Steps
Pull Requests welcome!
- Relu
- Conv 
- Saving / Loading of Model Structure and Parameters
  



