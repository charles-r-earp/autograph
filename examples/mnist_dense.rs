use std::time::Instant;
use autograph as ag;

fn main() {
  use ag::{Layer, iter::MeanExt};
  use num_traits::ToPrimitive;
  let dataset = ag::datasets::Mnist::new();
  let mut net = ag::Dense::builder()
    .units(10)
    .use_bias()
    .build();
  let mut optimizer = ag::optim::LearningRate(0.01);
  for epoch in 0 .. 1 {
    let now = Instant::now();
    let loss: f32 = dataset.train(32)
      .map(|(x, t)| {
      let (batch_size, channels, height, width) = x.dim();
      let y = net.forward(x.into_shape([batch_size, channels*height*width])
        .unwrap()
        .into_dyn())
        .into_dimensionality()
        .unwrap();
      let (loss, dy) = ag::cross_entropy_loss(&y, &t);
      net.backward(&dy.into_dyn());
      net.params_mut()
        .iter_mut()
        .for_each(|p| p.step(&mut optimizer));
      loss
    }).mean();
    let (correct, total) = dataset.test(512)
      .map(|(x, t)| {
      let (batch_size, channels, height, width) = x.dim();
      let y = net.forward(x.into_shape([batch_size, channels*height*width])
        .unwrap()
        .into_dyn())
        .into_dimensionality()
        .unwrap();
      (ag::correct(&y, &t), batch_size)
    }).fold((0, 0), |(a_c, a_t), (c, n)| (a_c + c, a_t + n));
    let acc = correct.to_f32().unwrap() * 100. / total.to_f32().unwrap();
    println!("[{}] epoch_loss = {:.5}, accuracy = {} / {} = {:.2?}%, duration = {:.0?}", epoch, loss, correct, total, acc, now.elapsed());
  }
}
      
