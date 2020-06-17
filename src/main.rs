#![allow(warnings)]
use autograph::{Device, Cpu, Tensor, Conv2dArgs, Pool2dArgs};
use autograph::layer::{Conv2d, MaxPool2d, Relu, Dense, Inference};
#[cfg(feature="cuda")]
use autograph::CudaGpu;

use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use timeit::*;


fn main() {  
  #[cfg(not(feature="cuda"))]
  let device = Device::from(Cpu::new());
  #[cfg(feature="cuda")]
  let device = Device::from(CudaGpu::new());
  
  let mut rng = SmallRng::seed_from_u64(0);
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
  
  let x = Tensor::ones(&device, [100, 1, 28, 28]);
  
  timeit!({
    let x = conv1.infer(&x.view())
      .relu()
      .max_pool2d(&Pool2dArgs::default().kernel(2).strides(2));
    let x = conv2.infer(&x.view())
      .relu()
      .max_pool2d(&Pool2dArgs::default().kernel(2).strides(2));
    let x = dense1.infer(&x.view().into_flatten())
      .relu();
    let x = dense2.infer(&x.view())
      .relu();
    let y = dense3.infer(&x.view());
  });
}


