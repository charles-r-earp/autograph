use autograph as ag;
use ndarray as nd;
use args::Args;
use getopts::Occur;
use std::{rc::Rc, time::Instant};

fn main() {
  use ag::{layer, layer::{Forward, Layer}, functional::{CrossEntropyLoss, ClassificationMatches}, optim};
  use num_traits::ToPrimitive;
  let mut args = Args::new("Mnist Lenet5", "Trains and evaluates a CNN.");
  args.option("", "epochs", "number of epochs to train for", "", Occur::Optional, Some(1.to_string()));
  args.parse_from_cli()
    .unwrap();
  println!("{}", &args);
  let epochs: usize = args.value_of("epochs")
    .unwrap();
  let dataset = ag::datasets::Mnist::new();
  let mut model = layer::Sequential::<f32>::default();
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
  for epoch in 1 ..= epochs {
    let mut bar = progress::Bar::new();
    bar.set_job_title("train: ");
    let mut train_total = 0;
    dataset.train(64)
      .for_each(|(x, t)| {
      model.param_iter_mut()
        .for_each(|p| p.zero_grad());
      {
        let tape = Rc::new(ag::autograd::Tape::new());
        let x = ag::autograd::Var::new(&tape, x, false);
        model.forward(&x)
          .cross_entropy_loss(&t)
          .backward();
      }
      model.param_iter_mut()
        .for_each(|p| p.step(lr / t.len().to_f32().unwrap()));
      train_total += t.len();
      bar.reach_percent(((train_total * 100) / 60_000).to_i32().unwrap());
    });
    bar.set_job_title("test: ");
    bar.reach_percent(0);
    let mut test_total = 0;
    let test_correct: usize = dataset.test(1024)
      .map(|(x, t)| {
        test_total += t.len();
        let correct = model.forward(&x.into_dyn())
          .classification_matches(&t);
        bar.reach_percent(((test_total * 100) / 10_000).to_i32().unwrap());
        correct
      }).sum();
    let test_accuracy = test_correct.to_f32().unwrap() / 10_000f32;
    bar.reach_percent(100);
    if let Some((terminal_size::Width(w), _)) = terminal_size::terminal_size() {
      let s = (0 .. w).into_iter()
        .map(|_| " ")
        .fold(String::new(), |acc, s| acc + &s);
      print!("\r{}", s);
    }
    println!("\r[{}] test_accuracy = {:.2}% elapsed = {:.0?}", epoch, 100.*test_accuracy, now.elapsed());
  }
}
      
