#![allow(warnings)]
use argparse::{ArgumentParser, Store, StoreTrue};
use autograph::autograd::{Graph, ParameterD, Variable, Variable2, Variable4};
use autograph::datasets::Mnist; // requires feature "datasets"
use autograph::layer::{Forward, Layer, Conv2d, MaxPool2d, Relu, Flatten, Dense};
use autograph::utils::classification_accuracy;
#[macro_use]
extern crate autograph_derive; // for #[derive(Layer)] and #[impl_forward(_, _)] 
#[cfg(feature = "cuda")]
use autograph::CudaGpu;
use autograph::{ArcTensor, Cpu, Device, Pool2dArgs, Tensor, Tensor2, Tensor4, TensorView4};
use ndarray::{Dimension, Ix2, Ix4};
use num_traits::ToPrimitive;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

// A version of the LeNet5 Model
// impl_forward is a macro that generates a sequential 
#[impl_forward(Ix4, Ix2)]
#[derive(Layer)]
struct Lenet5 (
    Conv2d,
    Relu,
    MaxPool2d,
    Conv2d,
    Relu,
    MaxPool2d,
    Flatten,
    Dense,
    Relu,
    Dense,
    Relu,
    Dense
);

impl Lenet5 {
    // new is the primary constructor for a struct
    // Here we construct the model on the given device
    // Note that currently Conv2d and Dense layers fill their parameters with zeros, so the model must be manually initialized
    pub fn new(device: &Device) -> Self {
        Lenet5 (
            Conv2d::builder()
                .device(&device)
                .inputs(1)
                .outputs(6)
                .kernel(5)
                .build(),
            Relu::default(),
            MaxPool2d::builder()
                .args(
                    Pool2dArgs::default()
                        .kernel(2)
                        .strides(2)
                )
                .build(),
            Conv2d::builder()
                .device(&device)
                .inputs(6)
                .outputs(16)
                .kernel(5)
                .build(),
            Relu::default(),
            MaxPool2d::builder()
                .args(
                    Pool2dArgs::default()
                        .kernel(2)
                        .strides(2)
                )
                .build(),
            Flatten::default(),
            Dense::builder()
                .device(&device)
                .inputs(256)
                .outputs(120)
                .build(),
            Relu::default(),
            Dense::builder()
                .device(&device)
                .inputs(120)
                .outputs(84)
                .build(),
            Relu::default(),
            Dense::builder()
                .device(&device)
                .inputs(84)
                .outputs(10)
                .bias()
                .build()
        )
    }
}

// Layer is a core trait for Layers and Models
/* This is generated by #[derive(Layer)]
impl Layer for Lenet5 {
    // Gathers all the parameters in the model
    fn parameters(&self) -> Vec<ParameterD> {
        self.0
            .parameters()
            .into_iter()
            .chain(self.1.parameters())
            .chain(self.2.parameters())
            .chain(self.3.parameters())
            .chain(self.4.parameters())
            .chain(self.5.parameters())
            .chain(self.6.parameters())
            .chain(self.7.parameters())
            .chain(self.8.parameters())
            .chain(self.9.parameters())
            .chain(self.10.parameters())
            .chain(self.11.parameters())
            .collect()
    }
    // Prepares the model for training (or evaluation)
    fn set_training(&mut self, training: bool) {
        self.0.set_training(training);
        self.1.set_training(training);
        self.2.set_training(training);
        self.3.set_training(training);
        self.4.set_training(training);
        self.5.set_training(training);
        self.6.set_training(training);
        self.7.set_training(training);
        self.8.set_training(training);
        self.9.set_training(training);
        self.10.set_training(training);
        self.11.set_training(training);
    }
}
*/

// Forward is a trait for Layers and Models
// Forward executes the forward pass, returning the prediction of the model
/* This is generated by #[impl_forward(Ix4, Ix2)]
impl Forward<Ix4> for Lenet5 {
    type OutputDim = Ix2;
    fn forward(&self, input: &Variable<Ix4>) -> Variable<Ix2> {
        // Variable has a forward(layer: impl Forward) method
        // This makes it easy to chain several operations
        input
            .forward(&self.0)
            .forward(&self.1)
            .forward(&self.2)
            .forward(&self.3)
            .forward(&self.4)
            .forward(&self.5)
            .forward(&self.6)
            .forward(&self.7)
            .forward(&self.8)
            .forward(&self.9)
            .forward(&self.10)
            .forward(&self.11)
    }
}
*/

fn main() {
    let (epochs, lr, train_batch_size, eval_batch_size, no_cuda) = {
        let mut epochs = 50;
        let mut lr = 0.001;
        let mut train_batch_size: usize = 256;
        let mut eval_batch_size: usize = 1024;
        let mut no_cuda = false;
        {
            let mut ap = ArgumentParser::new();
            ap.set_description("MNIST Lenet5 Example");
            ap.refer(&mut epochs).add_option(
                &["-e", "--epochs"],
                Store,
                "Number of epochs to train for.",
            );
            ap.refer(&mut lr)
                .add_option(&["--learning-rate"], Store, "Learning Rate");
            ap.refer(&mut train_batch_size).add_option(
                &["--train-batch_size"],
                Store,
                "Training Batch Size",
            );
            ap.refer(&mut eval_batch_size).add_option(
                &["--eval-batch-size"],
                Store,
                "Evaluation Batch Size",
            );
            ap.refer(&mut no_cuda).add_option(
                &["--no-cuda"],
                StoreTrue,
                "Uses cpu even if cuda feature is enabled.",
            );
            ap.parse_args_or_exit();
        }
        (epochs, lr, train_batch_size, eval_batch_size, no_cuda)
    };

    #[cfg(not(feature = "cuda"))]
    let device = Device::from(Cpu::new());
    #[cfg(feature = "cuda")]
    let device = if no_cuda {
        Device::from(Cpu::new())
    } else {
        Device::from(CudaGpu::new(0))
    };

    println!("epochs: {}", epochs);
    println!("lr: {}", lr);
    println!("train_batch_size: {}", train_batch_size);
    println!("eval_batch_size: {}", eval_batch_size);
    println!("no_cuda: {}", no_cuda);
    println!("device: {:?}", &device);

    let mut rng = SmallRng::seed_from_u64(0);

    let mut model = Lenet5::new(&device);
    model.parameters().into_iter().for_each(|w| {
        let dim = w.value().raw_dim();
        if dim.ndim() > 1 {
            // Leave biases as zeros
            w.value()
                .write()
                .unwrap()
                .fill_random(&Normal::new(0., 0.01).unwrap(), &mut rng)
        }
    });

    let dataset = Mnist::new();

    let start = Instant::now();
    for epoch in 1..=epochs {
        let mut train_loss = 0.;
        let mut train_correct: usize = 0;
        dataset.train(train_batch_size).for_each(|(x_arr, t_arr)| {
            model.set_training(true);
            let graph = Graph::new();
            let x = Variable::new(
                Some(&graph),
                Tensor::from_array(&device, x_arr).to_f32(),
                false,
            );
            let t = ArcTensor::from(Tensor::from_array(&device, t_arr).to_one_hot_f32(10));
            let y = model.forward(&x);
            let loss = y.cross_entropy_loss(&t);
            loss.backward(graph);
            model.parameters().iter().for_each(|w| {
                let mut w_value = w.value().write().unwrap();
                let w_grad = w.grad().unwrap().read().unwrap().unwrap();
                w_value.scaled_add(-lr, &w_grad);
            });
            train_correct += classification_accuracy(&y.value().as_array().view(), &t_arr);
            train_loss += loss.value().as_slice()[0];
        });
        train_loss /= 60_000f32;
        let train_acc = train_correct.to_f32().unwrap() * 100f32 / 60_000f32;

        let mut eval_loss = 0.;
        let mut eval_correct: usize = 0;
        dataset.eval(eval_batch_size).for_each(|(x_arr, t_arr)| {
            model.set_training(false);
            let x = Variable::new(None, Tensor::from_array(&device, x_arr).to_f32(), false);
            let t = ArcTensor::from(Tensor::from_array(&device, t_arr).to_one_hot_f32(10));
            let y = model.forward(&x);
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
