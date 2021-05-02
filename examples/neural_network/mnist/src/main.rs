use autograph::{
    backend::Device,
    dataset::{mnist, Dataset},
    learn::{Fit, FitOptions, Predict},
    neural_network::{ClassificationTrainer, Dense, Sgd},
    Result,
};

// Returning a Result from main allows using the ? operator
fn main() -> Result<()> {
    // Create a device for the first Gpu
    let device = Device::new_gpu(0).expect("No gpu!")?;
    // The mnist function is imported from autograph::dataset, and loads the data as a pair of
    // arrays.
    let (x_array, y_array) = mnist()?;
    // TODO: replace this with dataset adapter?
    let x_array = x_array
        .map(|x| *x as f32 / 255.)
        .into_shape([70_000, 28 * 28])?;
    // Create dense model with weight and bias
    let model = Dense::builder()
        .device(&device)
        .inputs(28 * 28)
        .outputs(10)
        .bias(true)
        .build()?;
    // Stochastic Gradient Descent with a learning rate (default is 0.001)
    let optim = Sgd::builder().learning_rate(0.001).build();
    // Construct a trainer for Classification. This implements Fit as well as Preduct.
    let mut trainer = ClassificationTrainer::from_network_optimizer(model, optim);
    // Fit the model to the dataset. Note that &(Array, Array) implements Dataset which loads the data
    // into Tensors.
    // The last argument to fit is a callback which takes a reference to &mut self (ie the trainer)
    // and the stats for each epoch and returns whether to continue training.
    trainer.fit(
        &device,
        &(x_array.view(), y_array.view()),
        FitOptions::default().train_batch_size(64),
        |_trainer, stats| {
            println!(
                "epoch: {} elapsed: {:?} loss: {:.5}",
                stats.get_epoch(),
                stats.get_elapsed(),
                stats.get_train_loss()
            );
            Ok(stats.get_epoch() < 10)
        },
    )?;
    // TODO: Implement accuracy for Tensors and in Fit impl
    let mut correct = 0;
    let mut total = 0;
    for xy_future in (&(x_array, y_array)).batches(&device, 64, false) {
        let (x, y) = smol::block_on(xy_future)?;
        let pred = trainer.predict(x)?;
        let pred = smol::block_on(pred.to_array()?)?;
        let y = smol::block_on(y.to_array()?)?;
        for (x, y) in pred.iter().copied().zip(y.iter().copied()) {
            if x == y as u32 {
                correct += 1;
            }
            total += 1;
        }
    }
    println!("accuracy: {:.2}%", 100. * correct as f32 / total as f32);
    Ok(())
}
