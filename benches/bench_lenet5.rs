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
#[cfg(feature="cuda")]
use autograph::CudaGpu;
use ndarray::{Dimension, Ix2, Ix4};
use std::time::Instant;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal, Uniform};
use num_traits::ToPrimitive;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn bench_autograph_lenet5(c: &mut Criterion, device: &Device, batch_size: usize) {
  let mut rng = SmallRng::seed_from_u64(0); 
  let mut model = Lenet5::builder()
    .device(&device)
    .init(&mut rng)
    .build();
  model.init_training();
  
  let lr = 0.001;
  
  let x = ArcTensor::ones(&device, [batch_size, 1, 28, 28]);
  let t = Tensor::<u8, _>::zeros(&device, batch_size);
  c.bench_function(&format!("autograph_lenet5_train_{}_{:?}", batch_size, device), |b| b.iter(|| {
    let graph = Graph::new();
    let x = Variable::new(
      &graph,
      x.clone(),
      None
    );
    let y = model.forward(&x, true);
    let t = ArcTensor::from(t.to_one_hot_f32(10));
    let loss = y.cross_entropy_loss(&t);
    loss.backward(graph);
    model.parameters()
      .iter()
      .for_each(|w| {
        let mut w_value = w.value()
          .write()
          .unwrap();
        let mut w_grad = w.grad()
          .unwrap()
          .write()
          .unwrap();
        w_value.scaled_add(-lr, &w_grad);
        w_grad.fill(0.);
      });
    device.synchronize();
  }));
  let x = ArcTensor::ones(&device, [batch_size, 1, 28, 28]);
  let t = Tensor::<u8, _>::zeros(&device, batch_size);
  c.bench_function(&format!("autograph_lenet5_eval_{}_{:?}", batch_size, device), |b| b.iter(|| {
    let y = model.infer(&x.view());
    let t = t.to_one_hot_f32(10);
    let loss = y.cross_entropy_loss(&t.view());
    device.synchronize();
  }));
}

#[derive(Debug)]
struct TchLenet5 {
  conv1: tch::nn::Conv2D,
  conv2: tch::nn::Conv2D,
  dense1: tch::nn::Linear,
  dense2: tch::nn::Linear,
  dense3: tch::nn::Linear
}

impl TchLenet5 {
  fn new(vs: &tch::nn::Path) -> Self {
    let mut conv_config = tch::nn::ConvConfig::default();
    conv_config.bias = false;
    let conv1 = tch::nn::conv2d(vs, 1, 6, 5, conv_config);
    let conv2 = tch::nn::conv2d(vs, 6, 16, 5, conv_config);
    let mut linear_config = tch::nn::LinearConfig::default();
    linear_config.bias = false;
    let dense1 = tch::nn::linear(vs, 256, 120, linear_config);
    let dense2 = tch::nn::linear(vs, 120, 84, linear_config);
    linear_config.bias = true;
    let dense3 = tch::nn::linear(vs, 84, 10, linear_config);
    Self {
      conv1,
      conv2,
      dense1,
      dense2,
      dense3
    }
  }
}

impl tch::nn::ModuleT for TchLenet5 {
  fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
    xs.apply(&self.conv1)
      .max_pool2d_default(2)
      .apply(&self.conv2)
      .max_pool2d_default(2)
      .view([-1, 256])
      .apply(&self.dense1)
      .relu()
      .apply(&self.dense2)
      .relu()
      .apply(&self.dense3)
  }
}

fn bench_tch_lenet5(c: &mut Criterion, device: tch::Device, batch_size: usize) {
  use tch::nn::ModuleT;
  use tch::nn::OptimizerConfig;
  let vs = tch::nn::VarStore::new(device);
  let model = TchLenet5::new(&vs.root());
  let lr = 0.001;
  let mut optim = tch::nn::Sgd::default()
    .build(&vs, lr)
    .unwrap();
  
  let device_name = match device {
    tch::Device::Cpu => String::from("cpu"),
    tch::Device::Cuda(i) => format!("cuda:{}", i)
  };
  let x = tch::Tensor::ones(&[batch_size as i64, 1, 28, 28], (tch::Kind::Float, device));
  let t = tch::Tensor::zeros(&[batch_size as i64], (tch::Kind::Int64, device));
  c.bench_function(&format!("tch_lenet5_train_{}_{}", batch_size, &device_name), |b| b.iter(|| {
    let y = model.forward_t(&x, true);
    let loss = y.cross_entropy_for_logits(&t);
    optim.backward_step(&loss);
  }));
  let x = tch::Tensor::ones(&[batch_size as i64, 1, 28, 28], (tch::Kind::Float, device));
  let t = tch::Tensor::zeros(&[batch_size as i64], (tch::Kind::Int64, device));
  c.bench_function(&format!("tch_lenet5_eval_{}_{}", batch_size, &device_name), |b| b.iter(|| {
    let y = model.forward_t(&x, false);
    let loss = y.cross_entropy_for_logits(&t);
  }));
} 

pub fn criterion_benchmark(c: &mut Criterion) {
  let cpu = Device::from(Cpu::new());
  bench_autograph_lenet5(c, &cpu, 256);
  #[cfg(feature="cuda")] 
  {
    let gpu = Device::from(CudaGpu::new(0));
    bench_autograph_lenet5(c, &gpu, 256);
  }
  bench_tch_lenet5(c, tch::Device::Cpu, 256);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
