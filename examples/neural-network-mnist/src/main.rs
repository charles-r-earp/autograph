use autograph::{
    dataset::mnist::Mnist,
    device::Device,
    learn::{
        neural_network::{
            layer::{Dense, Layer},
            optimizer::Sgd,
            Network, NetworkTrainer,
        },
        Summarize, Train,
    },
    result::Result,
    tensor::{float::FloatTensor2, Tensor1, TensorView},
};
use ndarray::{s, ArrayView1, ArrayView2, Axis};
use std::convert::TryFrom;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a device.
    let device = Device::new()?;
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

    // Define and construct an Sgd optimizer.
    // Note that Sgd is the default optimizer and doesn't need to be specified or configured.
    let learning_rate = 0.001;
    // The new method checks that learning rate is between 0. and 1.
    // Momentum is not implemented yet, defaults to 0.
    let optimizer = Sgd::new(learning_rate)?;

    // Construct a trainer to train the model.
    let mut trainer = NetworkTrainer::from(network).with_optimizer(optimizer);

    // Load the dataset. This will look in the downloads folder for the zipped image and label
    // files, or download them if not found.
    let mnist = Mnist::builder().download(true).build()?;
    // Use the first 60_000 images as the training set.
    // Currently the imput must be flattened to 2d.
    let train_images = mnist
        .images()
        .slice(s![..60_000, .., .., ..])
        .into_shape([60_000, 28 * 28])?;
    let train_classes = mnist.classes().slice(s![..60_000]);
    let train_batch_size = 100;
    // Use the last 10_000 images as the test set.
    let test_images = mnist
        .images()
        .slice(s![60_000.., .., .., ..])
        .into_shape([10_000, 28 * 28])?;
    let test_classes = mnist.classes().slice(s![60_000..]);
    let test_batch_size = 1_000;

    // Stream the data to the device, converting arrays to tensors.
    fn batch_iter<'a>(
        device: &'a Device,
        images: &'a ArrayView2<u8>,
        classes: &'a ArrayView1<u8>,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<(FloatTensor2, Tensor1<u8>)>> + 'a {
        // Stream the images in batches to the device.
        let images_iter = images
            .axis_chunks_iter(Axis(0), batch_size)
            .into_iter()
            .map(move |x| {
                smol::block_on(async {
                    Result::<_, autograph::error::Error>::Ok(
                        TensorView::try_from(x)?
                            .into_device(device.clone())
                            .await?
                            // normalize the bytes to f32
                            .scale_into::<f32>(1. / 255.)?
                            .into_float(),
                    )
                })
            });
        // Stream the classes in batches to the device.
        let classes_iter = classes
            .axis_chunks_iter(Axis(0), batch_size)
            .into_iter()
            .map(move |t| {
                smol::block_on(async {
                    Result::<_, autograph::error::Error>::Ok(
                        TensorView::try_from(t)?.into_device(device.clone()).await?,
                    )
                })
            });
        // Zip together the image and class iterators.
        images_iter
            .zip(classes_iter)
            .map(|(x, t)| Result::<_, autograph::error::Error>::Ok((x?, t?)))
    }

    // Run the training for 10 epochs
    while trainer.summarize().epoch < 10 {
        let train_iter = batch_iter(&device, &train_images, &train_classes, train_batch_size);
        let test_iter = batch_iter(&device, &test_images, &test_classes, test_batch_size);
        trainer.train_test(train_iter, test_iter)?;
        println!("{:#?}", trainer.summarize());
    }

    Ok(())
}
