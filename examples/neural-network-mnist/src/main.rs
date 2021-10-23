use autograph::{
    dataset::mnist::Mnist,
    device::Device,
    learn::{
        neural_network::{
            layer::{Layer, Forward, Conv, Relu, Dense, MaxPool},
            Network, NetworkTrainer,
        },
        Summarize, Train,
    },
    result::Result,
    tensor::{float::FloatTensor4, Tensor1, TensorView},
};
use ndarray::{s, ArrayView1, ArrayView4, Axis};
use argparse::{ArgumentParser, StoreConst};
use std::{
    fmt::Debug,
    convert::TryFrom,
};

#[derive(Clone, Copy)]
enum NetworkKind {
    Linear,
    CNN,
    Lenet5,
}

#[derive(Layer, Forward, Clone, Debug)]
struct CNN {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    dense2: Dense,
}

impl CNN {
    fn new() -> Result<Self> {
        let conv1 = Conv::from_inputs_outputs_kernel(1, 6, [5, 5]);
        let relu1 = Relu::default();
        let dense1 = Dense::from_inputs_outputs(6 * 24 * 24, 84);
        let relu2 = Relu::default();
        let dense2 = Dense::from_inputs_outputs(84, 10)
            .with_bias(true)?;
        Ok(Self {
            conv1,
            relu1,
            dense1,
            relu2,
            dense2,
        })
    }
}

#[derive(Layer, Forward, Clone, Debug)]
struct Lenet5 {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    pool1: MaxPool,
    #[autograph(layer)]
    conv2: Conv,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    pool2: MaxPool,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu3: Relu,
    #[autograph(layer)]
    dense2: Dense,
    #[autograph(layer)]
    relu4: Relu,
    #[autograph(layer)]
    dense3: Dense,
}

impl Lenet5 {
    fn new() -> Result<Self> {
        let conv1 = Conv::from_inputs_outputs_kernel(1, 6, [5, 5]);
        let relu1 = Relu::default();
        let pool1 = MaxPool::from_kernel([2, 2])
            .with_strides(2)?;
        let conv2 = Conv::from_inputs_outputs_kernel(6, 16, [5, 5]);
        let relu2 = Relu::default();
        let pool2 = MaxPool::from_kernel([2, 2])
            .with_strides(2)?;
        let dense1 = Dense::from_inputs_outputs(16 * 4 * 4, 120);
        let relu3 = Relu::default();
        let dense2 = Dense::from_inputs_outputs(120, 84);
        let relu4 = Relu::default();
        let dense3 = Dense::from_inputs_outputs(84, 10)
            .with_bias(true)?;
        Ok(Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            dense1,
            relu3,
            dense2,
            relu4,
            dense3,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut kind = NetworkKind::Linear;
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Neural Network MNIST Example");
        ap.refer(&mut kind)
            .add_option(&["--linear"], StoreConst(NetworkKind::Linear), "A linear network.")
            .add_option(&["--cnn"], StoreConst(NetworkKind::CNN), "A convolutional network.")
            .add_option(&["--lenet5"], StoreConst(NetworkKind::Lenet5), "The LeNet5 network.");
         ap.parse_args_or_exit();
    }

    // Create a device.
    let device = Device::new()?;
    println!("{:#?}", device.info());

    match kind {
        NetworkKind::Linear => {
            let dense = Dense::from_inputs_outputs(28 * 28, 10)
                .with_bias(true)?;
            train(device, dense).await
        }
        NetworkKind::CNN => {
            train(device, CNN::new()?).await
        }
        NetworkKind::Lenet5 => {
            train(device, Lenet5::new()?).await
        }
    }
}

async fn train<L: Layer + Debug>(device: Device, layer: L) -> Result<()> {
    // Transfer the layer into the device. Most operations (ie compute shaders) only
    // run on a device.
    let layer = layer.into_device(device.clone())
    // Note that Host -> Device transfers do not block / the future resolves immediately.
    .await?;

    // Wrap the layer in a Network. Network is simply a new type wrapper that provides an
    // implementation for the Infer trait, and NetworkTrainer stores the model in a Network.
    let network = Network::from(layer);

    // Construct a trainer to train the model.
    let mut trainer = NetworkTrainer::from(network);
    println!("{:#?}", &trainer);

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
    let test_batch_size = 1000;

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
    // Run the training for 100 epochs
    while trainer.summarize().epoch < 100 {
        let train_iter = batch_iter(&device, &train_images, &train_classes, train_batch_size);
        let test_iter = batch_iter(&device, &test_images, &test_classes, test_batch_size);
        trainer.train_test(train_iter, test_iter)?;
        println!("{:#?}", trainer.summarize());
    }

    Ok(())
}
