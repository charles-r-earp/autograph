use autograph::{
    dataset::mnist::Mnist,
    device::{Device, Api, DeviceType},
    learn::{
        neural_network::{
            layer::{Dense, Layer},
            Network, NetworkTrainer,
        },
        Summarize, Train,
    },
    result::Result,
    tensor::{float::FloatTensor4, Tensor1, TensorView},
};
use ndarray::{s, ArrayView1, ArrayView4, Axis};
use std::convert::TryFrom;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a device.
    let device = Device::new()?;
    dbg!(device.info());
    
    // Define the model.
    let dense = Dense::from_inputs_outputs(28 * 28, 10)
        .with_bias(true)?
        // Transfer the layer into the device. Most operations (ie compute shaders) only
        // run on a device.
        .into_device(device.clone())
        // Note that Host -> Device transfers do not block / the future resolves immediately.
        .await?;
    // Wrap the layer in a Network. Network is simply a new type wrapper that provides an
    // implementation for the Infer trait, and NetworkTrainer stores the model in a Network.
    let network = Network::from(dense);

    // Construct a trainer to train the model.
    let mut trainer = NetworkTrainer::from(network);
    dbg!(&trainer);

    // Load the dataset.
    println!("Loading dataset...");
    let mnist = Mnist::builder().download(true).build()?;
    // Use the first 60_000 images as the training set.
    let train_images = mnist
        .images()
        .slice(s![..60_000, .., .., ..]);
    let train_classes = mnist.classes().slice(s![..60_000]);
    let train_batch_size = 100;
    // Use the last 10_000 images as the test set.
    let test_images = mnist
        .images()
        .slice(s![60_000.., .., .., ..]);
    let test_classes = mnist.classes().slice(s![60_000..]);
    let test_batch_size = 1_000;

    // Stream the data to the device, converting arrays to tensors.
    fn batch_iter<'a>(
        device: &'a Device,
        images: &'a ArrayView4<u8>,
        classes: &'a ArrayView1<u8>,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<(FloatTensor4, Tensor1<u8>)>> + 'a {
        images
            .axis_chunks_iter(Axis(0), batch_size)
            .into_iter()
            .zip(classes.axis_chunks_iter(Axis(0), batch_size))
            .map(move |(x, t)| {
                let x = smol::block_on(TensorView::try_from(x)?.into_device(device.clone()))?
                    // normalize the bytes to f32
                    .scale_into::<f32>(1. / 255.)?
                    .into_float();
                let t = smol::block_on(TensorView::try_from(t)?.into_device(device.clone()))?;
                Ok((x, t))
            })
    }

    println!("Training...");
    // Run the training for 10 epochs
    while trainer.summarize().epoch < 10 {
        let train_iter = batch_iter(&device, &train_images, &train_classes, train_batch_size);
        let test_iter = batch_iter(&device, &test_images, &test_classes, test_batch_size);
        trainer.train_test(train_iter, test_iter)?;
        println!("{:#?}", trainer.summarize());
    }

    Ok(())
}
