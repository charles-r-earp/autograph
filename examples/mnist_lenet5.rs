#![allow(warnings)]
use autograph::{
  Device, Cpu, 
  Tensor, Tensor2, Tensor4,
  TensorView4, 
  ArcTensor,
  Pool2dArgs
};
use autograph::autograd::{
  Graph, 
  Variable, Variable2, Variable4,
  ParameterD
};
use autograph::layer::{
  Layer, Inference, Forward, 
  Conv2d, Dense
};
use autograph::datasets::Mnist; // requires feature "datasets"
#[cfg(feature="cuda")]
use autograph::CudaGpu;
use ndarray::{Ix2, Ix4};
use std::time::Instant;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use argmm::ArgMinMax;
use num_traits::ToPrimitive;

struct Lenet5 {
  conv1: Conv2d,
  conv2: Conv2d,
  dense1: Dense,
  dense2: Dense,
  dense3: Dense
}

impl Lenet5 {
  fn new<R: Rng>(device: &Device, mut rng: &mut R) -> Self {
    let normal = Normal::new(0., 0.01).unwrap();
  
    let conv1 = Conv2d::builder(&device)
      .inputs(1)
      .outputs(6)
      .kernel(5)
      .init_weight_from_iter(normal.sample_iter(&mut rng))
      .build();
    let conv2 = Conv2d::builder(&device)
      .inputs(6)
      .outputs(16)
      .kernel(5)
      .init_weight_from_iter(normal.sample_iter(&mut rng))
      .build();
    let dense1 = Dense::builder(&device)
      .inputs(256)
      .outputs(120)
      .init_weight_from_iter(normal.sample_iter(&mut rng))
      .build();
    let dense2 = Dense::builder(&device)
      .inputs(120)
      .outputs(84)
      .init_weight_from_iter(normal.sample_iter(&mut rng))
      .build();
    let dense3 = Dense::builder(&device)
      .inputs(84)
      .outputs(10)
      .init_weight_from_iter(normal.sample_iter(&mut rng))
      .bias()
      .build();
    Self {
      conv1,
      conv2,
      dense1,
      dense2,
      dense3
    }
  }
}

impl Layer for Lenet5 {
  fn parameters(&self) -> Vec<ParameterD> {
    self.conv1.parameters()
      .into_iter()
      .chain(self.conv2.parameters())
      .chain(self.dense1.parameters())
      .chain(self.dense2.parameters())
      .chain(self.dense3.parameters())
      .collect()
  }
  fn init_training(&mut self) {
    self.conv1.init_training();
    self.conv2.init_training();
    self.dense1.init_training();
    self.dense2.init_training();
    self.dense3.init_training();
  }
}

impl Inference<Ix4> for Lenet5 {
  type OutputDim = Ix2;
  fn infer(&self, input: &TensorView4<f32>) -> Tensor2<f32> {
    let x = self.conv1.infer(&input.view())
      .relu()
      .max_pool2d(&Pool2dArgs::default().kernel(2).strides(2));
    let x = self.conv2.infer(&x.view())
      .relu()
      .max_pool2d(&Pool2dArgs::default().kernel(2).strides(2));
    let x = self.dense1.infer(&x.view().into_flatten())
      .relu();
    let x = self.dense2.infer(&x.view())
      .relu();
    let y = self.dense3.infer(&x.view());
    y
  } 
}

impl Forward<Ix4> for Lenet5 {
  type OutputDim = Ix2;
  fn forward(&self, input: &Variable4, train: bool) -> Variable2 {
    let x = self.conv1.forward(&input, train)
      .relu()
      .max_pool2d(&Pool2dArgs::default().kernel(2).strides(2));
    let x = self.conv2.forward(&x, train)
      .relu()
      .max_pool2d(&Pool2dArgs::default().kernel(2).strides(2));
    let x = self.dense1.forward(&x.flatten(), train)
      .relu();
    let x = self.dense2.forward(&x, train)
      .relu();
    let y = self.dense3.forward(&x, train);
    y
  }
}

fn main() {
  println!("MNIST Lenet5 Example");
  
  #[cfg(not(feature="cuda"))]
  let device = Device::from(Cpu::new());
  #[cfg(feature="cuda")]
  let device = Device::from(CudaGpu::new(0));
  
  let mut rng = SmallRng::seed_from_u64(0); 
  
  let mut model = Lenet5::new(&device, &mut rng);
  model.init_training();
  
  let lr = 0.01;
  let train_batch_size: usize = 60;
  let eval_batch_size: usize = 1000;
  
  let dataset = Mnist::new();

  let start = Instant::now();
  for epoch in 1 ..= 20 {
    let mut train_loss = 0.;
    let mut train_correct: usize = 0;
    dataset.train(train_batch_size)
      .for_each(|(x_arr, t_arr)| {
        let graph = Graph::new();
        let x = Variable::new(
          &graph,
          Tensor::from_array(&device, x_arr).to_f32(),
          None
        );
        let t = ArcTensor::from_array(&device, t_arr)
          .to_one_hot_f32(10);
        let y = model.forward(&x, true);
        model.parameters()
          .iter()
          .for_each(|w| {
            w.grad()
              .unwrap()
              .write()
              .unwrap()
              .fill(0.);
          });
        let loss = y.cross_entropy_loss(t);
        loss.backward(graph);
        model.parameters()
          .iter()
          .for_each(|w| {
            let mut w_value = w.value()
              .write()
              .unwrap();
            let w_grad = w.grad()
              .unwrap()
              .read()
              .unwrap();
            w_value.scaled_add(-lr, &w_grad);
          });
        y.value()
          .as_slice()
          .chunks_exact(10)
          .zip(t_arr.as_slice().unwrap())
          .for_each(|(y, &t)| {
            if y.argmax() == Some(t as usize) {
              train_correct += 1;
            }  
          });
        train_loss += loss.value()
          .as_slice()[0];
      });
    train_loss /= 60_000f32;
    let train_acc = train_correct.to_f32().unwrap() * 100f32 / 60_000f32; 
    
    let mut eval_loss = 0.;
    let mut eval_correct: usize = 0;
    dataset.eval(eval_batch_size)
      .for_each(|(x_arr, t_arr)| {
        let x = Tensor::from_array(&device, x_arr)
          .to_f32();
        let t = Tensor::from_array(&device, t_arr)
          .to_one_hot_f32(10);
        let y = model.infer(&x.view());
        let loss = y.cross_entropy_loss(&t.view());
        y.as_slice()
          .chunks_exact(10)
          .zip(t_arr.as_slice().unwrap())
          .for_each(|(y, &t)| {
            if y.argmax() == Some(t as usize) {
              eval_correct += 1;
            }  
          });
        eval_loss += loss.as_slice()[0];
      });
    eval_loss /= 10_000f32;
    let eval_acc = eval_correct.to_f32().unwrap() * 100f32 / 10_000f32;
    let elapsed = Instant::now() - start;
    println!("epoch: {} elapsed {:.0?} train_loss: {:.5} train_acc: {:.2}% eval_loss: {:.5} eval_acc: {:.2}%", 
      epoch, elapsed, train_loss, train_acc, eval_loss, eval_acc);
  }
}
