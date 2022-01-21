use autograph::{
    dataset::mnist::Mnist,
    device::Device,
    learn::{
        neural_network::{
            layer::{Layer, Forward, Conv, Relu, Dense, MaxPool},
            NetworkTrainer,
        },
        Summarize, Train, Test,
    },
    result::Result,
    tensor::{Tensor, float::FloatTensor4, Tensor1, TensorView},
};
use ndarray::{s, Array1, ArrayView1, ArrayView4, Axis};
use argparse::{ArgumentParser, Store, StoreConst, StoreTrue};
use indicatif::{ProgressStyle, ProgressBar, ProgressIterator};
use serde::{Serialize, Deserialize};
use std::{
    fmt::Debug,
    convert::TryFrom,
    path::PathBuf,
    fs,
};
use rand::seq::SliceRandom;

#[derive(Clone, Copy)]
enum NetworkKind {
    Linear,
    CNN,
    Lenet5,
}

#[derive(Layer, Forward, Clone, Debug, Serialize, Deserialize)]
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

#[derive(Layer, Forward, Clone, Debug, Serialize, Deserialize)]
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
    let mut save = false;
    let mut epochs = 100;
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Neural Network MNIST Example");
        ap.refer(&mut kind)
            .add_option(&["--linear"], StoreConst(NetworkKind::Linear), "A linear network.")
            .add_option(&["--cnn"], StoreConst(NetworkKind::CNN), "A convolutional network.")
            .add_option(&["--lenet5"], StoreConst(NetworkKind::Lenet5), "The LeNet5 network.");
        ap.refer(&mut save)
            .add_option(&["--save"], StoreTrue, "Load / save the trainer.");
        ap.refer(&mut epochs)
            .add_option(&["--epochs"], Store, "The number of epochs to train for.");
        ap.parse_args_or_exit();
    }

    // Create a device.
    let device = Device::new()?;
    //println!("{:#?}", device.info());


    fn save_path(name: &str, save: bool) -> Option<PathBuf> {
        if save {
            Some(format!("{}_trainer.bincode", name).into())
        } else {
            None
        }
    }

    match kind {
        NetworkKind::Linear => {
            let dense = Dense::from_inputs_outputs(28 * 28, 10)
                .with_bias(true)?;
            train(device, dense, save_path("linear", save), epochs).await
        }
        NetworkKind::CNN => {
            train(device, CNN::new()?, save_path("cnn", save), epochs).await
        }
        NetworkKind::Lenet5 => {
            train(device, Lenet5::new()?, save_path("lenet5", save), epochs).await
        }
    }
}

async fn train<L: Layer + Debug + Serialize + for<'de> Deserialize<'de>>(device: Device, layer: L, save_path: Option<PathBuf>, epochs: usize) -> Result<()> {

    // Construct a trainer to train the network.
    let mut trainer = match save_path.as_ref() {
        // Load the trainer from a file.
        Some(save_path) if save_path.exists() => bincode::deserialize(& fs::read(save_path)?)?,
        // Use the provided layer.
        _ => NetworkTrainer::from_network(layer.into())
    };

    // Transfer the trainer to the device. Most operations (ie compute shaders) only
    // run on a device.
    trainer.to_device_mut(device.clone()).await?;
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
    ) -> impl ExactSizeIterator<Item = Result<(FloatTensor4, Tensor1<u8>)>> + 'a {
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

    // Shuffled training data iterator
    fn shuffled_batch_iter<'a>(
        device: &'a Device,
        images: &'a ArrayView4<'a, u8>,
        classes: &'a ArrayView1<'a, u8>,
        batch_size: usize,
    ) -> impl ExactSizeIterator<Item = Result<(FloatTensor4, Tensor1<u8>)>> + 'a {
        let mut indices = (0 .. images.shape()[0]).into_iter().collect::<Vec<usize>>();
        indices.shuffle(&mut rand::thread_rng());
        (0 .. indices.len())
            .into_iter()
            .step_by(batch_size)
            .map(move |index| {
                let batch_indices = &indices[index..(index+batch_size).min(indices.len())];
                let x = batch_indices.iter()
                    .copied()
                    .flat_map(|i| images.index_axis(Axis(0), i))
                    .copied()
                    .collect::<Array1<u8>>()
                    .into_shape([batch_indices.len(), images.dim().1, images.dim().2, images.dim().3])?;
                let t = batch_indices.iter()
                    .copied()
                    .map(|i| classes[i])
                    .collect::<Array1<u8>>();
                let x = smol::block_on(Tensor::from(x).into_device(device.clone()))?
                    // normalize the bytes to f32
                    .scale_into::<f32>(1. / 255.)?
                    .into_float();
                let t = smol::block_on(Tensor::from(t).into_device(device.clone()))?;
                Ok((x, t))
            })
    }

    // Show a progress bar
    fn progress_iter<X>(iter: impl ExactSizeIterator<Item=X>, epoch: usize, name: &str) -> impl ExactSizeIterator<Item=X> {
        let style = ProgressStyle::default_bar()
            .template(&format!("[epoch: {} elapsed: {{elapsed}}] {} [{{bar}}] {{pos:>7}}/{{len:7}} [eta: {{eta}}]", epoch, name))
            .progress_chars("=> ");
        let bar = ProgressBar::new(iter.len() as u64)
            .with_style(style);
        iter.progress_with(bar)
    }

    println!("Training...");
    // Run the training for the specified epochs
    while trainer.summarize().epoch < epochs {
        let epoch = trainer.summarize().epoch;
        let train_iter = progress_iter(shuffled_batch_iter(&device, &train_images, &train_classes, train_batch_size), epoch, "training");
        let test_iter = progress_iter(batch_iter(&device, &test_images, &test_classes, test_batch_size), epoch, "testing");
        trainer.train_test(train_iter, test_iter)?;
        println!("{:#?}", trainer.summarize());

        // Save the trainer at each epoch.
        if let Some(save_path) = save_path.as_ref() {
            fs::write(save_path, bincode::serialize(&trainer)?)?;
        }
    }

    println!("Evaluating...");
    let test_iter = progress_iter(batch_iter(&device, &test_images, &test_classes, test_batch_size), trainer.summarize().epoch, "evaluating");
    let stats = trainer.test(test_iter)?;
    println!("{:#?}", stats);

    Ok(())
}
