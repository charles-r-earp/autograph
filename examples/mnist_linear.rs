use autograph as ag;
use std::{rc::Rc, time::Instant};
use rand_distr::{Distribution, Normal}; 

fn main() {
  use ag::{Linear, LinearBackward, CrossEntropy, iter::MeanExt};
  let dataset = ag::datasets::Mnist::new();
  let mut w = ag::Param::<f32>::new(ag::Tensor::from_shape_iter([28*28, 10], Normal::new(0., 0.001).unwrap().sample_iter(&mut rand::thread_rng())));
  let mut b = ag::Param::<f32>::new(ag::Tensor::zeros([10])); 
  let lr = 0.001;
  let mut epoch = 1;
  loop {
    let now = Instant::now();
    let loss = dataset.train(64)
      .map(|(x, t)| {
      let graph = Rc::new(ag::Graph::default());
      let x = ag::Var::new(&graph, x);
      w.zero_grad();
      b.zero_grad();
      let (loss, loss_grad) = x.linear(&w, Some(&b))
        .cross_entropy_loss(&t);
      loss_grad.linear_backward(&w, Some(&b));
      w.step(lr);
      b.step(lr);
      loss
    }).mean::<f32>();
    if epoch % 10 == 0 {
      println!("[{}] loss = {:.5} duration = {:0.?}", epoch, loss, now.elapsed());
    }
    epoch += 1;
  }
}
      
