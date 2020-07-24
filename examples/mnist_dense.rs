#![allow(warnings)]
use argparse::{ArgumentParser, Store, StoreTrue};
use autograph::autograd::{Graph, Parameter, Variable};
use autograph::datasets::Mnist; // requires feature "datasets"
use autograph::utils::classification_accuracy;
use autograph::{ArcTensor, Cpu, Device, RwTensor, Tensor};
use num_traits::ToPrimitive;
use rand::{rngs::SmallRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

fn main() {
    // Use argparse to get command line arguments
    let (epochs, learning_rate, train_batch_size, eval_batch_size) = {
        let mut epochs = 10;
        let mut learning_rate = 0.001;
        let mut train_batch_size: usize = 100;
        let mut eval_batch_size: usize = 1000;
        {
            let mut ap = ArgumentParser::new();
            ap.set_description("MNIST Dense Example");
            ap.refer(&mut epochs).add_option(
                &["-e", "--epochs"],
                Store,
                "Number of epochs to train for.",
            );
            ap.refer(&mut learning_rate)
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
            ap.parse_args_or_exit();
        }
        (epochs, learning_rate, train_batch_size, eval_batch_size)
    };

    // Construct a device that will be used to create Tensors
    // Devices can be created with the From trait
    // ie Device::from(Cpu::new()) 
    // or Device::from(CudaGpu::new(index))
    // Default returns a CudaGpu if cuda is enabled, otherwise a Cpu
    let device = Device::default();

    println!("epochs: {}", epochs);
    println!("learning_rate: {}", learning_rate);
    println!("train_batch_size: {}", train_batch_size);
    println!("eval_batch_size: {}", eval_batch_size);
    println!("device: {:?}", &device);

    // Construct a Random Number Generator to initialize our model.
    // We could use any Rng, but SmallRng is simple, fast, lightweight, and deterministic
    // By seeding with 0 the model parameters will be the same on each program run
    let mut rng = SmallRng::seed_from_u64(0);

    // Create a trainable weight Parameter that wraps a RwTensor.
    // A RwTensor is a special tensor that uses an Arc<RwLock<Vec<T>>> to allow data to be safely shared and modified.
    // Here we are initializing from a normal distribution with mean 0. and standard deviation 0.01
    // The shape of our weight is [10, 28*28], because we are mapping a 28x28 image to 10 digits
    let mut w = Parameter::new(RwTensor::random(
        &device,
        [10, 28 * 28],
        &Normal::new(0., 0.01).unwrap(),
        &mut rng,
    ));
    // Create a trainable bias Parameter filled with zeros
    let mut b = Parameter::new(RwTensor::zeros(&device, 10));

    // Create a Mnist dataset
    // See http://yann.lecun.com/exdb/mnist/
    // Loads the dataset into memory, potentially downloading and saving it to datasets/mnist/*
    let dataset = Mnist::new();

    // Record the time
    let start = Instant::now();

    // Iterate for epochs
    for epoch in 1..=epochs {
        let mut train_loss = 0.;
        let mut train_correct: usize = 0;
        // Training loop: iterate over the training set
        // Mnist::train() returns an iterator of (images: ArrayView4<u8>, labels: ArrayView1<u8>)
        dataset.train(train_batch_size).for_each(|(x_arr, t_arr)| {
            // Set the parameters to training mode
            // Note that this also releases any gradients from a previous backward pass
            w.set_training(true);
            b.set_training(true);
            // Construct a graph. A graph is used to create Variables, and to enqueue operations that will be used to compute gradients
            let graph = Graph::new(); // This is an Arc<Graph> so that it can be shared
                                      // Construct a Variable with the graph and data from the dataset
            let x = Variable::new(
                Some(&graph),
                // Construct a tensor from the input array view
                // Tensor::to_f32() converts the u8 data to f32 on the device
                Tensor::from_array(&device, x_arr.view()).to_f32(),
                // requires_grad is false because this is an input, we don't need to compute its gradient
                false,
            );
            // Construct an ArcTensor for our target
            // t_arr is a batch of input labels that are converted to a matrix of shape [batch_size, nclasses=10]
            // ArcTensor::from(Tensor) consumes (ie moves) the tensor, the data is not copied
            // t must be an ArcTensor because it will be stored in the graph to compute the gradient of the model prediction y
            let t = ArcTensor::from(Tensor::from_array(&device, t_arr.view()).to_one_hot_f32(10));
            // View x [batch_size, 1, 28, 28] as [batch_size, 28*28] then perform the operation y = x.mm(w.t()) + b
            // Operations on Variable enqueue backward ops into the graph, if provided, to compute gradients
            let y = x.flatten().dense(&w, Some(&b));
            // Compute the total loss of our model prediction. Cross entropy is used for classification.
            let loss = y.cross_entropy_loss(&t);
            // Compute the backward pass
            // Here graph is consumed (ie dropped). Variables x, y, and loss will no longer be tied to this graph and any further operations will not compute gradients.
            // The gradient of y will be computed, then the gradients of w and b
            loss.backward(graph);
            let lr = learning_rate / x_arr.shape()[0].to_f32().unwrap();
            // Lock the weight value, acquiring exclusive write access
            let mut w_value = w.value().write().unwrap(); // Unwraps the LockResult
                                                          // Lock the weight gradient, acquiring shared read access
            let w_grad = w
                .grad()
                .unwrap()
                .read()
                .unwrap() // Unwraps the Option, would be None if backward was not called
                .unwrap(); // Unwraps the LockResult
                           // Update the weight
            w_value.scaled_add(-lr, &w_grad);
            let mut b_value = b.value().write().unwrap();
            let b_grad = b.grad().unwrap().read().unwrap().unwrap();
            // Update the bias
            b_value.scaled_add(-lr, &b_grad);
            train_correct += classification_accuracy(&y.value().as_array().view(), &t_arr);
            train_loss += loss.value().as_slice()[0];
        });
        train_loss /= 60_000f32;
        let train_acc = train_correct.to_f32().unwrap() * 100f32 / 60_000f32;

        let mut eval_loss = 0.;
        let mut eval_correct: usize = 0;
        dataset.eval(eval_batch_size).for_each(|(x_arr, t_arr)| {
            // Here we set training to false, which prevents gradients from being computed
            w.set_training(false);
            b.set_training(false);
            // Construct a variable, without a graph
            let x = Variable::new(
                None,
                Tensor::from_array(&device, x_arr.view()).to_f32(),
                false,
            );
            let t = ArcTensor::from(Tensor::from_array(&device, t_arr.view()).to_one_hot_f32(10));
            // Perform the same operation as before, but only execute the forward pass (ie inference)
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
