use autograph::{
    anyhow::Result,
    dataset::mnist::{Mnist, MnistKind},
    krnl::{
        device::Device,
        krnl_core::half::bf16,
        scalar::{ScalarElem, ScalarType},
    },
    learn::{
        criterion::{Accuracy, CrossEntropyLoss},
        neural_network::{
            autograd::{Variable, Variable2, Variable4},
            layer::{Conv2, Dense, Flatten, Forward, Layer, MaxPool2, Relu},
            optimizer::{Optimizer, SGD},
        },
    },
    ndarray::{self, ArcArray, ArcArray1, Axis, Dimension, Ix4},
    tensor::{CowTensor, ScalarTensor, Tensor, Tensor1, Tensor4},
};
use clap::{Parser, ValueEnum};
use num_format::{Locale, ToFormattedString};
use rand::{seq::index::sample, thread_rng};
use std::{fmt::Debug, time::Instant};

#[derive(Layer, Forward, Debug)]
#[autograph(forward(Variable4, Output=Variable2))]
struct LeNet5 {
    conv1: Conv2,
    relu1: Relu,
    pool1: MaxPool2,
    conv2: Conv2,
    relu2: Relu,
    pool2: MaxPool2,
    flatten: Flatten,
    dense1: Dense,
    relu3: Relu,
    dense2: Dense,
    relu4: Relu,
    dense3: Dense,
}

impl LeNet5 {
    fn new(device: Device, scalar_type: ScalarType) -> Result<Self> {
        let conv1 = Conv2::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(1)
            .outputs(6)
            .filter([5, 5])
            .build()?;
        let relu1 = Relu;
        let pool1 = MaxPool2::builder().filter([2, 2]).build();
        let conv2 = Conv2::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(6)
            .outputs(16)
            .filter([5, 5])
            .build()?;
        let relu2 = Relu;
        let pool2 = MaxPool2::builder().filter([2, 2]).build();
        let flatten = Flatten;
        let dense1 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(16 * 4 * 4)
            .outputs(128)
            .build()?;
        let relu3 = Relu;
        let dense2 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(128)
            .outputs(84)
            .build()?;
        let relu4 = Relu;
        let dense3 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(84)
            .outputs(10)
            .bias(true)
            .build()?;
        Ok(Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            flatten,
            dense1,
            relu3,
            dense2,
            relu4,
            dense3,
        })
    }
}

#[derive(Clone, Copy, derive_more::Display, Debug, ValueEnum)]
enum Dataset {
    #[display(fmt = "mnist")]
    MNIST,
    #[display(fmt = "fashion-mnist")]
    FashionMNIST,
}

#[derive(Clone, Copy, derive_more::Display, Debug, ValueEnum)]
enum ScalarKind {
    #[display(fmt = "bf16")]
    BF16,
    #[display(fmt = "f32")]
    F32,
}

impl From<ScalarKind> for ScalarType {
    fn from(kind: ScalarKind) -> Self {
        match kind {
            ScalarKind::BF16 => ScalarType::BF16,
            ScalarKind::F32 => ScalarType::F32,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author)]
struct Options {
    #[arg(long)]
    device: Option<usize>,
    #[arg(long, default_value_t = Dataset::MNIST)]
    dataset: Dataset,
    #[arg(long, default_value_t = ScalarKind::F32)]
    scalar_type: ScalarKind,
    #[arg(short, long, default_value_t = 100)]
    epochs: usize,
    #[arg(long, default_value_t = 100)]
    train_batch_size: usize,
    #[arg(long, default_value_t = 1000)]
    test_batch_size: usize,
    #[arg(long, default_value_t = 0.1)]
    learning_rate: f32,
    #[arg(long)]
    momentum: Option<f32>,
}

fn main() -> Result<()> {
    let options = Options::parse();
    println!("{options:#?}");
    let mnist_kind = match options.dataset {
        Dataset::MNIST => MnistKind::MNIST,
        Dataset::FashionMNIST => MnistKind::FashionMNIST,
    };
    let Mnist {
        train_images,
        train_classes,
        test_images,
        test_classes,
        ..
    } = Mnist::builder()
        .kind(mnist_kind)
        .download(true)
        .verbose(true)
        .build()?;
    let (train_images, train_classes) =
        (ArcArray::from(train_images), ArcArray::from(train_classes));
    let (test_images, test_classes) = (ArcArray::from(test_images), ArcArray::from(test_classes));
    let device = if let Some(index) = options.device {
        Device::builder().index(index).build()?
    } else {
        Device::host()
    };
    if let Some(info) = device.info() {
        println!("{info:#?}");
    }
    let scalar_type = ScalarType::from(options.scalar_type);
    let mut model = LeNet5::new(device.clone(), scalar_type)?;
    let optimizer = {
        let mut builder = SGD::builder();
        if let Some(momentum) = options.momentum {
            builder = builder.momentum(momentum);
        }
        builder.build()
    };
    println!("model: {model:#?}");
    let parameter_count = model
        .parameter_iter()
        .map(|x| x.raw_dim().size())
        .sum::<usize>();
    println!(
        "{} parameters",
        parameter_count.to_formatted_string(&Locale::en)
    );
    println!("optimizer: {optimizer:#?}");
    let image_scale = 1f32 / 255f32;
    let image_scale = match options.scalar_type {
        ScalarKind::BF16 => ScalarElem::BF16(bf16::from_f32(image_scale)),
        ScalarKind::F32 => ScalarElem::F32(image_scale),
    };
    let start = Instant::now();
    for epoch in 1..=options.epochs {
        let epoch_start = Instant::now();
        let train_iter = shuffled_batches(
            train_images.clone(),
            train_classes.clone(),
            device.clone(),
            options.train_batch_size,
        );
        let train_stats = train(
            &mut model,
            image_scale,
            &optimizer,
            options.learning_rate,
            train_iter,
        )?;
        let train_count = train_stats.count;
        let train_correct = train_stats.correct;
        let train_loss = train_stats.mean_loss();
        let train_acc = train_stats.accuracy();
        let test_iter = batches(
            test_images.clone(),
            test_classes.clone(),
            device.clone(),
            options.test_batch_size,
        );
        let test_stats = test(&model, image_scale, test_iter)?;
        let test_count = test_stats.count;
        let test_correct = test_stats.correct;
        let test_loss = test_stats.mean_loss();
        let test_acc = test_stats.accuracy();
        let epoch_elapsed = epoch_start.elapsed();
        println!(
            "[{epoch}] train_loss: {train_loss:.5} train_acc: {train_acc:.2}% {train_correct}/{train_count} test_loss: {test_loss:.5} test_acc: {test_acc:.2}% {test_correct}/{test_count} elapsed: {epoch_elapsed:.2?}"
        );
    }
    println!("Finished in {:?}.", start.elapsed());
    Ok(())
}

fn batches(
    images: ArcArray<u8, Ix4>,
    classes: ArcArray1<u8>,
    device: Device,
    batch_size: usize,
) -> impl Iterator<Item = Result<(Tensor4<u8>, Tensor1<u8>)>> {
    let (count, _inputs, _height, _width) = images.dim();
    (0..count).step_by(batch_size).map(move |index| {
        let end = (index + batch_size).min(count);
        let images = images.slice_axis(
            Axis(0),
            ndarray::Slice::new(index as isize, Some(end as isize), 1),
        );
        let classes = classes.slice_axis(
            Axis(0),
            ndarray::Slice::new(index as isize, Some(end as isize), 1),
        );
        let images = CowTensor::from(images).to_device(device.clone());
        let classes = CowTensor::from(classes).to_device(device.clone());
        images.and_then(|images| Ok((images, classes?)))
    })
}

fn shuffled_batches(
    images: ArcArray<u8, Ix4>,
    classes: ArcArray1<u8>,
    device: Device,
    batch_size: usize,
) -> impl Iterator<Item = Result<(Tensor4<u8>, Tensor1<u8>)>> {
    let (count, inputs, height, width) = images.dim();
    let mut index_iter = sample(&mut thread_rng(), count, count).into_iter();
    (0..count).step_by(batch_size).map(move |index| {
        let batch_size = (index..count).take(batch_size).len();
        let mut output_images = Vec::<u8>::with_capacity(batch_size * inputs * height * width);
        let mut output_classes = Vec::<u8>::with_capacity(batch_size);
        for index in index_iter.by_ref().take(batch_size) {
            output_images.extend_from_slice(images.index_axis(Axis(0), index).as_slice().unwrap());
            output_classes.push(classes[index]);
        }
        let images = Tensor::from(output_images)
            .into_shape([batch_size, inputs, height, width])
            .unwrap()
            .into_device(device.clone());
        let classes = Tensor::from(output_classes).into_device(device.clone());
        images.and_then(|images| Ok((images, classes?)))
    })
}

#[derive(Default)]
struct Stats {
    count: usize,
    loss: f32,
    correct: usize,
}

impl Stats {
    fn mean_loss(&self) -> f32 {
        self.loss / self.count as f32
    }
    fn accuracy(&self) -> f32 {
        (self.correct * 100) as f32 / self.count as f32
    }
}

fn train<I: Iterator<Item = Result<(Tensor4<u8>, Tensor1<u8>)>>>(
    model: &mut LeNet5,
    image_scale: ScalarElem,
    optimizer: &SGD,
    learning_rate: f32,
    mut iter: I,
) -> Result<Stats> {
    let mut stats = Stats::default();
    while let Some((x, t)) = iter.by_ref().next().transpose()? {
        stats.count += x.shape().first().unwrap();
        model.set_training(true)?;
        let x = Variable::from(ScalarTensor::from(x).scaled_cast(image_scale)?);
        let t = ScalarTensor::from(t).into_shared()?;
        let y = model.forward(x)?;
        stats.correct += y.value().accuracy(t.view())?;
        let loss = y.cross_entropy_loss(t)?;
        stats.loss += loss
            .value()
            .clone()
            .cast_into_tensor::<f32>()?
            .into_array()?
            .into_scalar();
        loss.backward()?;
        for parameter in model.make_parameter_iter_mut()? {
            optimizer.update(learning_rate, parameter)?;
        }
        model.set_training(false)?;
    }
    Ok(stats)
}

fn test<I: Iterator<Item = Result<(Tensor4<u8>, Tensor1<u8>)>>>(
    model: &LeNet5,
    image_scale: ScalarElem,
    mut iter: I,
) -> Result<Stats> {
    let mut stats = Stats::default();
    while let Some((x, t)) = iter.by_ref().next().transpose()? {
        stats.count += x.shape().first().unwrap();
        let x = Variable::from(ScalarTensor::from(x).scaled_cast(image_scale)?);
        let t = ScalarTensor::from(t).into_shared()?;
        let y = model.forward(x)?;
        stats.correct += y.value().accuracy(t.view())?;
        let loss = y.cross_entropy_loss(t)?;
        stats.loss += loss
            .into_value()
            .cast_into_tensor::<f32>()?
            .into_array()?
            .into_scalar();
    }
    Ok(stats)
}
