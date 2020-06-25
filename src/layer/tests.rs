use super::*;
use crate::Cpu;
use crate::autograd::Graph;
#[cfg(feature="cuda")]
use crate::CudaGpu;
use ndarray::{IntoDimension, Dimension, Ix2, Ix4};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use num_traits::ToPrimitive;

struct Lenet5 {
  conv1: Conv2d,
  conv2: Conv2d,
  dense1: Dense,
  dense2: Dense,
  dense3: Dense
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
  fn set_training(&mut self, training: bool) {
    self.conv1.set_training(training);
    self.conv2.set_training(training);
    self.dense1.set_training(training);
    self.dense2.set_training(training);
    self.dense3.set_training(training);
  }
}

impl Forward<Ix4> for Lenet5 {
  type OutputDim = Ix2;
  fn forward(&self, input: &Variable4) -> Variable2 {
    let pool_args = Pool2dArgs::default()
      .kernel(2)
      .strides(2);
    input.forward(&self.conv1)
      .relu()
      .max_pool2d(&pool_args)
      .forward(&self.conv2)
      .relu()
      .max_pool2d(&pool_args)
      .flatten()
      .forward(&self.dense1)
      .relu()
      .forward(&self.dense2)
      .relu()
      .forward(&self.dense3)
  }
}

#[cfg(feature="cuda")]
#[test]
fn test_lenet5_cuda() {
  let cpu = Device::from(Cpu::new());
  let cuda_gpu = Device::from(CudaGpu::new(0));
  
  let mut rng = SmallRng::seed_from_u64(100);
  
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
