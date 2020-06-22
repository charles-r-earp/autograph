use super::*;
use crate::Cpu;
use crate::autograd::Graph;
#[cfg(feature="cuda")]
use crate::CudaGpu;
use ndarray::{IntoDimension, Dimension, Ix2, Ix4};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal, Uniform};
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

#[cfg(feature="cuda")]
#[test]
fn test_lenet5_cuda() {
  let cpu = Device::from(Cpu::new());
  let cuda_gpu = Device::from(CudaGpu::new(0));
  
  let mut rng = SmallRng::seed_from_u64(100);
  
  let model_builder = Lenet5::builder()
    .init(&mut rng);
  let mut model_cpu = model_builder.clone()
    .device(&cpu)
    .build();
  let mut model_cuda = model_builder.clone()
    .device(&cuda_gpu)
    .build();
  
  let batch_size = 60;
  let x_dim = [batch_size, 1, 28, 28].into_dimension();
  
  model_cpu.init_training();
  model_cuda.init_training();
  
 {
  
    let x_vec: Vec<f32> = Normal::new(0., 1.)
      .unwrap()
      .sample_iter(&mut rng)
      .take(x_dim.size())
      .collect();
    let x_cpu = Tensor::from_shape_vec(&cpu, x_dim, x_vec.as_slice());
    let x_cuda = Tensor::from_shape_vec(&cuda_gpu, x_cpu.raw_dim(), x_vec.as_slice());
    let y_cpu = model_cpu.infer(&x_cpu.view());
    let y_cuda = model_cuda.infer(&x_cuda.view());
    approx::assert_relative_eq!(&*y_cpu.as_slice(), &*y_cuda.as_slice(), max_relative=0.1);  
    
    let t_vec = vec![0u8; batch_size];
    let t_cpu = Tensor::from_shape_vec(&cpu, batch_size, t_vec.as_slice())
      .to_one_hot_f32(10);
    let t_cuda = Tensor::from_shape_vec(&cuda_gpu, batch_size, t_vec.as_slice())
      .to_one_hot_f32(10);
    
    approx::assert_relative_eq!(&*t_cpu.as_slice(), &*t_cuda.as_slice(), max_relative=0.001);
     
    let loss_cpu = y_cpu.cross_entropy_loss(&t_cpu.view());
    let loss_cuda = y_cuda.cross_entropy_loss(&t_cuda.view());
    approx::assert_relative_eq!(&*loss_cpu.as_slice(), &*loss_cuda.as_slice(), max_relative=0.1);
  
    let graph_cpu = Graph::new();
    let graph_cuda = Graph::new();
    
    let x_cpu = Variable::new(
      &graph_cpu,
      x_cpu,
      None
    );
    let x_cuda = Variable::new(
      &graph_cuda,
      x_cuda,
      None
    );
    let pool_args = Pool2dArgs::default().kernel(2).strides(2);
    let x1_cpu = model_cpu.conv1.forward(&x_cpu, true);
    let x1_cuda = model_cuda.conv1.forward(&x_cuda, true);
    let x2_cpu = x1_cpu.relu();
    let x2_cuda = x1_cuda.relu();
    let x3_cpu = x2_cpu.max_pool2d(&pool_args);
    let x3_cuda = x2_cuda.max_pool2d(&pool_args);
    let x4_cpu = model_cpu.conv2.forward(&x3_cpu, true);
    let x4_cuda = model_cuda.conv2.forward(&x3_cuda, true);
    let x5_cpu = x4_cpu.relu();
    let x5_cuda = x4_cuda.relu();
    let x6_cpu = x5_cpu.max_pool2d(&pool_args);
    let x6_cuda = x5_cuda.max_pool2d(&pool_args);
    let x7_cpu = model_cpu.dense1.forward(&x6_cpu.flatten(), true);
    let x7_cuda = model_cuda.dense1.forward(&x6_cuda.flatten(), true);
    let x8_cpu = x7_cpu.relu();
    let x8_cuda = x7_cuda.relu();
    let x9_cpu = model_cpu.dense2.forward(&x7_cpu, true);
    let x9_cuda = model_cuda.dense2.forward(&x7_cuda, true);
    let x10_cpu = x9_cpu.relu();
    let x10_cuda = x9_cuda.relu();
    let x11_cpu = model_cpu.dense3.forward(&x10_cpu, true);
    let x11_cuda = model_cuda.dense3.forward(&x10_cuda, true);
    let y_cpu = x11_cpu.relu();
    let y_cuda = x11_cuda.relu();
    approx::assert_relative_eq!(&*y_cpu.value().as_slice(), &*y_cuda.value().as_slice(), max_relative=0.001);  
    let loss_cpu = y_cpu.cross_entropy_loss(t_cpu);
    let loss_cuda = y_cuda.cross_entropy_loss(t_cuda);
    approx::assert_relative_eq!(&*loss_cpu.value().as_slice(), &*loss_cuda.value().as_slice(), max_relative=0.001);
    loss_cpu.backward(graph_cpu);
    loss_cuda.backward(graph_cuda);
    let vars_cpu = vec![
      x1_cpu.into_dyn(), 
      x2_cpu.into_dyn(),
      x3_cpu.into_dyn(), 
      x4_cpu.into_dyn(),
      x5_cpu.into_dyn(),
      x6_cpu.into_dyn(),
      x7_cpu.into_dyn(),
      x8_cpu.into_dyn(),
      x9_cpu.into_dyn(),
      x10_cpu.into_dyn(),
      x11_cpu.into_dyn(),
      y_cpu.into_dyn()
    ];
    let vars_cuda = vec![
      x1_cuda.into_dyn(),
      x2_cuda.into_dyn(),
      x3_cuda.into_dyn(),
      x4_cuda.into_dyn(),
      x5_cuda.into_dyn(),
      x6_cuda.into_dyn(),
      x7_cuda.into_dyn(),
      x8_cuda.into_dyn(),
      x9_cuda.into_dyn(),
      x10_cuda.into_dyn(),
      x11_cuda.into_dyn(),
      y_cuda.into_dyn()
    ];
    vars_cpu.into_iter()
      .zip(vars_cuda)
      .enumerate()
      .for_each(|(u, (x_cpu, x_cuda))| {
        let x_grad_cpu = x_cpu.grad()
          .unwrap()
          .read()
          .unwrap();
        let x_grad_cpu = &*x_grad_cpu.as_slice();
        let x_grad_cuda = x_cuda.grad()
          .unwrap()
          .read()
          .unwrap();
        let x_grad_cuda = &*x_grad_cuda.as_slice();
        let info: Vec<_> = x_grad_cpu.iter()
          .zip(x_grad_cuda.iter())
          .enumerate()
          .filter_map(|(i, (a, b))| {
            if approx::relative_ne!(a, b, max_relative=0.1) {
              Some(format!("[{}] {} != {}", i, a, b))
            } else { None }
          })
          .collect();
        assert!(
          info.is_empty(),
          "x{}.grad()\n{:#?}",
          u, info
        );
      });
    model_cpu.parameters()
      .into_iter()
      .zip(model_cuda.parameters())
      .enumerate()
      .for_each(|(u, (w_cpu, w_cuda))| {
        let w_grad_cpu = w_cpu.grad()
          .unwrap()
          .read()
          .unwrap();
        let w_grad_cpu = &*w_grad_cpu.as_slice();
        let w_grad_cuda = w_cuda.grad()
          .unwrap()
          .read()
          .unwrap();
        let w_grad_cuda = &*w_grad_cuda.as_slice();
        let info: Vec<_> = w_grad_cpu.iter()
          .zip(w_grad_cuda.iter())
          .enumerate()
          .filter_map(|(i, (a, b))| {
            if approx::relative_ne!(a, b, max_relative=0.1) {
              Some(format!("[{}] {} != {}", i, a, b))
            } else { None }
          })
          .collect();
        assert!(
          info.is_empty(),
          "w{}.grad()\n{:#?}",
          u, info
        );
        /*let w_cpu = w_cpu.value()
          .read()
          .unwrap();
        let w_cpu = &*w_cpu.as_slice();
        let w_cuda = w_cuda.value()
          .read()
          .unwrap();
        let w_cuda = &*w_cuda.as_slice();
        assert!(
          approx::relative_eq!(w_cpu, w_cuda, max_relative=0.001), 
          "w{}.value() \n{:.0?}\n !=\n{:.0?}",
          u, &w_cpu, &w_cuda
        );*/
      });
  }
}
