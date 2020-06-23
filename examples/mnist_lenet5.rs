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
  Conv2d, Dense,
  builders::{LayerBuilder, Conv2dBuilder, DenseBuilder}
};
use autograph::datasets::Mnist; // requires feature "datasets"
#[cfg(feature="cuda")]
use autograph::CudaGpu;
use ndarray::{Dimension, Ix2, Ix4};
use std::time::Instant;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal, Uniform};
use argmm::ArgMinMax;
use num_traits::ToPrimitive;

#[derive(Clone)]
struct Lenet5Builder {
  device: Option<Device>,
  conv1: Conv2dBuilder,
  conv2: Conv2dBuilder,
  dense1: DenseBuilder,
  dense2: DenseBuilder,
  dense3: DenseBuilder
}

impl Default for Lenet5Builder {
  fn default() -> Self {
    let conv1 = Conv2d::builder()
      .inputs(1)
      .outputs(6)
      .kernel(5);
    let conv2 = Conv2d::builder()
      .inputs(6)
      .outputs(16)
      .kernel(5);
    let dense1 = Dense::builder()
      .inputs(256)
      .outputs(120);
    let dense2 = Dense::builder()
      .inputs(120)
      .outputs(84);
    let dense3 = Dense::builder()
      .inputs(84)
      .outputs(10)
      .bias();
    Self {
      device: None,
      conv1,
      conv2,
      dense1,
      dense2,
      dense3
    }
  }
}

impl Lenet5Builder {
  fn init(mut self, mut rng: &mut impl Rng) -> Self {
    fn he_normal(inputs: usize) -> Normal<f32> {
      let std_dev = f32::sqrt(2. / inputs.to_f32().unwrap());
      Normal::new(0., std_dev).unwrap()
    }
    fn xavier_uniform(inputs: usize, outputs: usize) -> Uniform<f32> {
      let range = f32::sqrt(6. / (inputs + outputs).to_f32().unwrap());
      Uniform::new(-range, range)
    }
    self.conv1 = self.conv1.weight_data(|d| {
      let (outputs, inputs, kh, kw) = d.into_pattern(); 
      xavier_uniform(inputs, outputs)
        .sample_iter(&mut rng)
        .take(d.size())
        .collect()
    });
    self.conv2 = self.conv2.weight_data(|d| {
      let (outputs, inputs, kh, kw) = d.into_pattern(); 
      xavier_uniform(inputs, outputs)
        .sample_iter(&mut rng)
        .take(d.size())
        .collect()
    });
    self.dense1 = self.dense1.weight_data(|d| {
      let (outputs, inputs) = d.into_pattern(); 
      he_normal(inputs)
        .sample_iter(&mut rng)
        .take(d.size())
        .collect()
    });
    self.dense2 = self.dense2.weight_data(|d| {
      let (outputs, inputs) = d.into_pattern(); 
      he_normal(inputs)
        .sample_iter(&mut rng)
        .take(d.size())
        .collect()
    });
    self.dense3 = self.dense3.weight_data(|d| {
      let (outputs, inputs) = d.into_pattern(); 
      he_normal(inputs)
        .sample_iter(&mut rng)
        .take(d.size())
        .collect()
    });
    self
  }
}

impl LayerBuilder for Lenet5Builder {
  type Layer = Lenet5;
  fn device(mut self, device: &Device) -> Self {
    self.device.replace(device.clone());
    self
  }
  fn build(self) -> Lenet5 {
    self.into()
  }
}

struct Lenet5 {
  conv1: Conv2d,
  conv2: Conv2d,
  dense1: Dense,
  dense2: Dense,
  dense3: Dense
}

impl Layer for Lenet5 {
  type Builder = Lenet5Builder;
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
  fn to_builder(&self, with_data: bool) -> Lenet5Builder {
    let device = None;
    let conv1 = self.conv1.to_builder(with_data);
    let conv2 = self.conv2.to_builder(with_data);
    let dense1 = self.dense1.to_builder(with_data);
    let dense2 = self.dense2.to_builder(with_data);
    let dense3 = self.dense3.to_builder(with_data);
    Lenet5Builder {
      device,
      conv1,
      conv2,
      dense1,
      dense2,
      dense3
    }
  }
}

impl From<Lenet5Builder> for Lenet5 {
  fn from(builder: Lenet5Builder) -> Self {
    let device = builder.device.unwrap();
    let conv1 = builder.conv1.device(&device)
      .build();
    let conv2 = builder.conv2.device(&device)
      .build();
    let dense1 = builder.dense1.device(&device)
      .build();
    let dense2 = builder.dense2.device(&device)
      .build();
    let dense3 = builder.dense3.device(&device)
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
  
  let mut model = Lenet5::builder()
    .device(&device)
    .init(&mut rng)
    .build();
  model.init_training();
  
  let train_batch_size: usize = 256;
  let eval_batch_size: usize = 1024;
  
  let lr = 0.001;
  
  println!("train_batch_size: {}", train_batch_size);
  println!("lr: {}", lr);
  
  let dataset = Mnist::new();

  let start = Instant::now();
  for epoch in 1 ..= 200 {
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
        let t = ArcTensor::from(
          Tensor::from_array(&device, t_arr)
            .to_one_hot_f32(10)
        );
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
        let loss = y.cross_entropy_loss(&t);
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
