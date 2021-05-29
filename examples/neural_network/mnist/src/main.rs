use autograph::{
    backend::Device,
    tensor::{Tensor4},
    dataset::{Dataset, mnist},
    learn::{Fit, FitOptions},
    neural_network::{ClassificationTrainer, Dense, Sgd},
    Result,
};

// Returning a Result from main allows using the ? operator
fn main() -> Result<()> {
    // Create a device for the first Gpu
    // Cpu execution is currently unsupported.
    let device = Device::new_gpu(0).expect("No gpu!");
    // The mnist function is imported from autograph::dataset, and loads the data as a pair of
    // arrays.
    // X is a set of 70_000 28 x 28 u8 images ie shape = (70_000, 1, 28, 28)
    // Y is a set of 70_000 u8 labels ie shape = 70_000
    let (x_array, y_array) = mnist()?;
    // Map the 4d u8 images into 2d floats
    let x_data = Dataset::map(&x_array, |x: Tensor4<u8>| {
        let batch_size = x.shape()[0];
        x
            .into_shape([batch_size, 28 * 28])?
            .scale_into(1. / 255.)
    });
    // Create dense model with weight and bias
    let model = Dense::builder()
        .device(&device) // defaults to cpu
        .inputs(28 * 28) // defaults to 0, which will lazily initialize
        .outputs(10)
        .bias(true) // defaults to false
        .build()?;
    // Stochastic Gradient Descent with a learning rate (default is 0.001)
    let optim = Sgd::builder().learning_rate(0.001).build();
    // Construct a trainer for Classification. This implements Fit as well as Preduct.
    let mut trainer = ClassificationTrainer::from_network_optimizer(model, optim);
    // Fit the model to the dataset. Note that &(Array, Array) implements Dataset which loads the data
    // into Tensors.
    // The last argument to fit is a callback which takes a reference to &mut self (ie the trainer)
    // and the stats for each epoch and returns whether to continue training, ie like a while loop.
    trainer.fit(
        &device,
        &(x_data, y_array.view()),
        FitOptions::default().train_batch_size(64),
        |_trainer, stats| {
            println!(
                "epoch: {} elapsed: {:.2?} loss: {:.5} accuracy: {:.2}%",
                stats.get_epoch(),
                stats.get_elapsed(),
                stats.get_train_loss(),
                100. * stats
                    .get_train_accuracy()
                    .expect("train_accuracy not computed!"),
            );
            Ok(stats.get_epoch() < 10)
        },
    )?;
    Ok(())
}
