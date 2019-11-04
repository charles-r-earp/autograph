use autograph as ag;
use ndarray as nd;
use std::{rc::Rc, time::Instant};
use rand_distr::{Distribution, Normal}; 

fn main() {
  use ag::{layer, layer::{Forward, Layer}, functional::{Softmax, CrossEntropyLoss, ClassificationMatches}, iter_ext::MeanExt};
  use nd::ShapeBuilder;
  use num_traits::ToPrimitive;
  let dataset = ag::datasets::Mnist::new();
  let mut model = layer::dense_builder::<f32>()
    .units(10)
    .use_bias()
    .build();
  println!("{:#?}", &model); 
  model.build(&nd::Array::zeros([1, 1, 28, 28]).into_dyn());
  let lr = 0.01;
  let now = Instant::now();
  for epoch in 1 ..= 10 {
    let mut train_total = 0;
    let mut train_loss = 0f32;
    let mut train_correct = 0;
    dataset.train(64)
      .for_each(|(x, t)| {
      model.train();
      let (y, loss) = {
        let graph = Rc::new(ag::Graph::new());
        let x = ag::Var::new(&graph, x, false);
        let y = model.forward(&x);
        let loss = y.cross_entropy_loss(&t);
        loss.backward();
        (y, loss)
      };
      model.step(lr);
      train_total += t.shape()[0];
      train_loss += loss.into_array()
        .into_dimensionality()
        .unwrap()
        .into_scalar();
      train_correct += y.value()
        .classification_matches(&t);
    });
    train_loss /= train_total.to_f32().unwrap();
    let train_accuracy = train_correct.to_f32().unwrap() / train_total.to_f32().unwrap();
    let mut test_total = 0;
    let mut test_loss = 0f32;
    let mut test_correct = 0;
    model.eval();
    dataset.test(512)
      .for_each(|(x, t)| {
      let y = model.forward(&x.into_dyn());
      let loss = y.cross_entropy_loss(&t);
      test_total += t.shape()[0];
      test_loss += loss.into_scalar();
      test_correct += y.classification_matches(&t);
    });
    test_loss /= test_total.to_f32().unwrap();
    let test_accuracy = test_correct.to_f32().unwrap() / test_total.to_f32().unwrap();
    println!("[{}] train_loss = {:.5} train_accuracy = {:.2}% test_loss = {:.5} test_accuracy = {:.2}% elapsed = {:.0?}", 
      epoch, train_loss, 100.*train_accuracy, test_loss, 100.*test_accuracy, now.elapsed());
  }
}
      
