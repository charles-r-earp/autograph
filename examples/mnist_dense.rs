#![allow(warnings)]
use autograph::{Device, Cpu, Tensor, ArcTensor, RwTensor};
use autograph::autograd::{Graph, Variable, Parameter};
use autograph::datasets::Mnist;
#[cfg(feature="cuda")]
use autograph::CudaGpu;
use std::time::Instant;
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use argmm::ArgMinMax;
use num_traits::ToPrimitive;

fn main() {
  println!("MNIST Dense Example");
  
  #[cfg(not(feature="cuda"))]
  let device = Device::from(Cpu::new());
  #[cfg(feature="cuda")]
  let device = Device::from(CudaGpu::new(0));
  
  let mut rng = SmallRng::seed_from_u64(0); 
  
  let w_data: Vec<f32> = Normal::new(0., 0.01)
    .unwrap()
    .sample_iter(&mut rng)
    .take(10*28*28)
    .collect();
  let w = Parameter::new(
    RwTensor::from_shape_vec(&device, [10, 28*28], w_data),
    Some(RwTensor::zeros(&device, [10, 28*28]))
  );
  let b = Parameter::new(
    RwTensor::zeros(&device, [1, 10]),
    Some(RwTensor::zeros(&device, [1, 10]))
  );
  
  let lr = 0.01;
  let train_batch_size: usize = 100;
  let eval_batch_size: usize = 1000;
  
  let dataset = Mnist::new();

  let start = Instant::now();
  for epoch in 1 ..= 5 {
    let mut train_loss = 0.;
    let mut train_correct: usize = 0;
    dataset.train(train_batch_size)
      .for_each(|(x_arr, t_arr)| {
        let graph = Graph::new();
        let x = Variable::new(
          &graph,
          Tensor::from_shape_vec(&device, [train_batch_size, 28*28], x_arr.as_slice().unwrap())
            .to_f32(),
          None
        );
        let t = ArcTensor::from_shape_vec(&device, train_batch_size, t_arr.as_slice().unwrap())
          .to_one_hot_f32(10);
        let y = x.dense(&w, Some(&b));
        w.grad()
          .unwrap()
          .write()
          .unwrap()
          .fill(0.);
        b.grad()
          .unwrap()
          .write()
          .unwrap()
          .fill(0.);
        let loss = y.cross_entropy_loss(t);
        loss.backward(graph);
        let mut w_value = w.value()
          .write()
          .unwrap();
        let w_grad = w.grad()
          .unwrap()
          .read()
          .unwrap();
        w_value.scaled_add(-lr, &w_grad);
        let mut b_value = b.value()
          .write()
          .unwrap();
        let b_grad = b.grad()
          .unwrap()
          .read()
          .unwrap();
        b_value.scaled_add(-lr, &b_grad);
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
        let x = Tensor::from_shape_vec(&device, [eval_batch_size, 28*28], x_arr.as_slice().unwrap())
          .to_f32();
        let t = Tensor::from_shape_vec(&device, eval_batch_size, t_arr.as_slice().unwrap())
          .to_one_hot_f32(10);
        let w = w.value()
          .read()
          .unwrap();
        let b = b.value()
          .read()
          .unwrap();
        let y = x.dense(&w.view(), Some(&b.view()));
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
