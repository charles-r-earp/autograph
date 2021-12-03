use crate::{Library, TrainerDescriptor, TrainerStats};
use autograph::{
    dataset::mnist::Mnist,
    device::Device,
    learn::{
        neural_network::{
            layer::{Conv, Dense, Forward, Layer, MaxPool, Relu},
            NetworkTrainer,
        },
        Summarize, Train,
    },
    result::Result,
    tensor::{float::FloatTensor4, Tensor, Tensor1, TensorView},
};
use ndarray::{s, Array1, ArrayView1, ArrayView4, Axis};
use rand::seq::SliceRandom;

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
        let pool1 = MaxPool::from_kernel([2, 2]).with_strides(2)?;
        let conv2 = Conv::from_inputs_outputs_kernel(6, 16, [5, 5]);
        let relu2 = Relu::default();
        let pool2 = MaxPool::from_kernel([2, 2]).with_strides(2)?;
        let dense1 = Dense::from_inputs_outputs(16 * 4 * 4, 120);
        let relu3 = Relu::default();
        let dense2 = Dense::from_inputs_outputs(120, 84);
        let relu4 = Relu::default();
        let dense3 = Dense::from_inputs_outputs(84, 10).with_bias(true)?;
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
    let mut indices = (0..images.shape()[0]).into_iter().collect::<Vec<usize>>();
    indices.shuffle(&mut rand::thread_rng());
    (0..indices.len())
        .into_iter()
        .step_by(batch_size)
        .map(move |index| {
            let batch_indices = &indices[index..(index + batch_size).min(indices.len())];
            let x = batch_indices
                .iter()
                .copied()
                .flat_map(|i| images.index_axis(Axis(0), i))
                .copied()
                .collect::<Array1<u8>>()
                .into_shape([
                    batch_indices.len(),
                    images.dim().1,
                    images.dim().2,
                    images.dim().3,
                ])?;
            let t = batch_indices
                .iter()
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

pub enum Autograph {}

impl Library for Autograph {
    fn name() -> &'static str {
        "autograph"
    }
    fn benchmark(
        trainer_descriptor: &TrainerDescriptor,
        mut epoch_cb: impl FnMut(usize),
    ) -> Result<TrainerStats> {
        let dataset_descriptor = &trainer_descriptor.dataset;
        let train_batch_size = dataset_descriptor.train_batch_size;
        let test_batch_size = dataset_descriptor.test_batch_size;
        let epochs = trainer_descriptor.epochs;

        let mnist = Mnist::builder().download(true).build()?;
        // Use the first 60_000 images as the training set.
        let train_images = mnist.images().slice(s![..60_000, .., .., ..]);
        let train_classes = mnist.classes().slice(s![..60_000]);
        // Use the last 10_000 images as the test set.
        let test_images = mnist.images().slice(s![60_000.., .., .., ..]);
        let test_classes = mnist.classes().slice(s![60_000..]);

        let device = Device::new()?;

        let mut trainer = NetworkTrainer::from_network(Lenet5::new()?.into());
        smol::block_on(trainer.to_device_mut(device.clone()))?;

        let mut total_time = vec![0f32; epochs];
        let mut test_accuracy = vec![0f32; epochs];

        for epoch in 0..epochs {
            let train_iter =
                shuffled_batch_iter(&device, &train_images, &train_classes, train_batch_size);
            let test_iter = batch_iter(&device, &test_images, &test_classes, test_batch_size);
            trainer.train_test(train_iter, test_iter)?;
            let summary = trainer.summarize();
            total_time[epoch] = summary.total_time.as_secs_f32();
            test_accuracy[epoch] = summary.test_stats.accuracy().unwrap();

            epoch_cb(epoch);
        }

        Ok(TrainerStats {
            total_time,
            test_accuracy,
        })
    }
}
