use autograph as ag;
use ndarray as nd;
use std::{rc::Rc, time::Instant};

fn main() {
  use ag::{layer, layer::{Forward, Layer}, functional::{CrossEntropyLoss, ClassificationMatches}, optim};
  use num_traits::ToPrimitive;
  let dataset = ag::datasets::Mnist::new();
  let mut model = layer::Sequential::default();
  model.push(layer::Conv::builder().units(6).kernel_size([5; 2]).build());
  model.push(layer::Relu);
  model.push(layer::MaxPool::new([2; 2]));
  model.push(layer::Conv::builder().units(16).kernel_size([5; 2]).build());
  model.push(layer::Relu);
  model.push(layer::MaxPool::new([2; 2]));
  model.push(layer::Dense::builder().units(120).build());
  model.push(layer::Relu);
  model.push(layer::Dense::builder().units(84).build());
  model.push(layer::Relu);
  model.push(layer::Dense::builder().units(10).use_bias().build());
  model.param_iter_mut()
    .for_each(|p| { p.optimizer.replace(Box::new(optim::SGD::builder().build())); });
  println!("{:#?}", &model); 
  model.build(&nd::Array::zeros([1, 1, 28, 28]).into_dyn());
  let nparams = model.param_iter()
    .map(|p| p.view().len())
    .sum::<usize>();
  println!("{} trainable parameters.", nparams);
  let lr = 0.1;
  let now = Instant::now();
  for epoch in 1 ..= 10 {
    let mut bar = progress::Bar::new();
    bar.set_job_title("train...");
    let mut train_total = 0;
    let mut train_loss = 0f32;
    let mut train_correct = 0;
    dataset.train(256)
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
        .for_each(|p| p.step(lr / 256.));
      train_total += t.shape()[0];
      train_loss += loss.into_array()
        .into_dimensionality()
        .unwrap()
        .into_scalar();
      train_correct += y.value()
        .classification_matches(&t);
      bar.reach_percent(train_total as i32 * 100 / 60_000);
    });
    train_loss /= train_total.to_f32().unwrap();
    let train_accuracy = train_correct.to_f32().unwrap() / train_total.to_f32().unwrap();
    bar.reach_percent(100);
    bar.jobs_done();
    bar.set_job_title("test...");
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
      bar.reach_percent(test_total as i32 * 100 / 10_000);
    });
    test_loss /= test_total.to_f32().unwrap();
    let test_accuracy = test_correct.to_f32().unwrap() / test_total.to_f32().unwrap();
    bar.reach_percent(100);
    bar.jobs_done();
    println!("[{}] train_loss = {:.5} train_accuracy = {:.2}% test_loss = {:.5} test_accuracy = {:.2}% elapsed = {:.0?}", 
      epoch, train_loss, 100.*train_accuracy, test_loss, 100.*test_accuracy, now.elapsed());
  }
}
      
