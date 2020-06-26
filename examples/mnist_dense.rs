#![allow(warnings)]
use autograph::{Device, Cpu, Tensor, ArcTensor, RwTensor};
use autograph::autograd::{Graph, Variable, Parameter};
use autograph::datasets::Mnist; // requires feature "datasets"
use autograph::utils::classification_accuracy;
#[cfg(feature="cuda")]
use autograph::CudaGpu;
use std::time::Instant;
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use num_traits::ToPrimitive;
use argparse::{ArgumentParser, Store, StoreTrue};

fn main() {
  
  let (epochs, lr, train_batch_size, eval_batch_size, no_cuda) = {
    let mut epochs = 10;
    let mut lr = 0.001;
    let mut train_batch_size: usize = 100;
    let mut eval_batch_size: usize = 1000;
    let mut no_cuda = false;
    {
      let mut ap = ArgumentParser::new();
      ap.set_description("MNIST Dense Example");
      ap.refer(&mut epochs)
        .add_option(&["-e", "--epochs"], Store, "Number of epochs to train for.");
      ap.refer(&mut lr)
        .add_option(&["--learning-rate"], Store, "Learning Rate");
      ap.refer(&mut train_batch_size)
        .add_option(&["--train-batch_size"], Store, "Training Batch Size");
      ap.refer(&mut eval_batch_size)
        .add_option(&["--eval-batch-size"], Store, "Evaluation Batch Size");
      ap.refer(&mut no_cuda)
        .add_option(&["--no-cuda"], StoreTrue, "Uses cpu even if cuda feature is enabled.");
      ap.parse_args_or_exit();
    }
    (epochs, lr, train_batch_size, eval_batch_size, no_cuda)
  };
  
  #[cfg(not(feature="cuda"))]
  let device = Device::from(Cpu::new());
  #[cfg(feature="cuda")]
  let device = if no_cuda {
    Device::from(Cpu::new())  
  } 
  else { 
    Device::from(CudaGpu::new(0)) 
  };
  
  println!("epochs: {}", epochs);
  println!("lr: {}", lr);
  println!("train_batch_size: {}", train_batch_size);
  println!("eval_batch_size: {}", eval_batch_size);
  println!("no_cuda: {}", no_cuda);
  println!("device: {:?}", &device);
  
  let mut rng = SmallRng::seed_from_u64(0); 
  
  let mut w = Parameter::new(
    RwTensor::random(&device, [10, 28*28], &Normal::new(0., 0.01).unwrap(), &mut rng)
  );
  let mut b = Parameter::new(
    RwTensor::zeros(&device, 10)
  );
  
  let dataset = Mnist::new();

  let start = Instant::now();
  for epoch in 1 ..= epochs {
    let mut train_loss = 0.;
    let mut train_correct: usize = 0;
    dataset.train(train_batch_size)
      .for_each(|(x_arr, t_arr)| {
        w.set_training(true);
        b.set_training(true);
        let graph = Graph::new();
        let x = Variable::new(
          Some(&graph),
          Tensor::from_array(&device, x_arr.view())
            .to_f32(),
          false
        );
        let t = ArcTensor::from(
          Tensor::from_array(&device, t_arr.view())
            .to_one_hot_f32(10)
        );
        let y = x.flatten().dense(&w, Some(&b));
        let loss = y.cross_entropy_loss(&t);
        loss.backward(graph);
        let lr = lr / x_arr.shape()[0].to_f32().unwrap();
        let mut w_value = w.value()
          .write()
          .unwrap();
        let w_grad = w.grad()
          .unwrap()
          .read()
          .unwrap()
          .unwrap();
        w_value.scaled_add(-lr, &w_grad);
        let mut b_value = b.value()
          .write()
          .unwrap();
        let b_grad = b.grad()
          .unwrap()
          .read()
          .unwrap()
          .unwrap();
        b_value.scaled_add(-lr, &b_grad);
        train_correct += classification_accuracy(&y.value().as_array().view(), &t_arr);
        train_loss += loss.value()
          .as_slice()[0];
      });
    train_loss /= 60_000f32;
    let train_acc = train_correct.to_f32().unwrap() * 100f32 / 60_000f32; 
    
    let mut eval_loss = 0.;
    let mut eval_correct: usize = 0;
    dataset.eval(eval_batch_size)
      .for_each(|(x_arr, t_arr)| {
        w.set_training(false);
        b.set_training(false);
        let x = Variable::new(
          None,
          Tensor::from_array(&device, x_arr.view())
            .to_f32(),
          false
        );
        let t = ArcTensor::from(
          Tensor::from_array(&device, t_arr.view())
            .to_one_hot_f32(10)
        );
        let y = x.flatten().dense(&w, Some(&b));
        let loss = y.cross_entropy_loss(&t);
        eval_correct += classification_accuracy(&y.value().as_array().view(), &t_arr);
        eval_loss += loss.value().as_slice()[0];
      });
    eval_loss /= 10_000f32;
    let eval_acc = eval_correct.to_f32().unwrap() * 100f32 / 10_000f32;
    let elapsed = Instant::now() - start;
    println!("epoch: {} elapsed {:.0?} train_loss: {:.5} train_acc: {:.2}% eval_loss: {:.5} eval_acc: {:.2}%", 
      epoch, elapsed, train_loss, train_acc, eval_loss, eval_acc);
  }
}
