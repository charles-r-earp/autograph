use autograph::{
    anyhow::{Error, Result},
    buffer::Buffer,
    dataset::mnist::{Mnist, MnistKind},
    krnl::krnl_core::half::bf16,
    learn::{
        criterion::{Accuracy, Criterion, CrossEntropyLoss},
        neural_network::{
            autograd::{ParameterViewMutD, Variable2, Variable4},
            layer::{Dense, Forward, Layer},
            optimizer::{Optimizer, SGD},
        },
    },
    ndarray::Axis,
    scalar::{ScalarElem, ScalarType},
    tensor::{CowTensor, ScalarCowTensor, ScalarCowTensor1, ScalarTensor4, Tensor},
};
use clap::{Parser, ValueEnum};
use parking_lot::Mutex;
use rand::{seq::index::sample, thread_rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{fmt::Debug, sync::Arc, time::Instant};

#[derive(Clone, Copy, ValueEnum, Debug)]
enum DatasetKind {
    Mnist,
    FashionMnist,
}

impl DatasetKind {
    fn mnist_kind(self) -> MnistKind {
        match self {
            Self::Mnist => MnistKind::Digits,
            Self::FashionMnist => MnistKind::Fashion,
        }
    }
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum ModelKind {
    Linear,
}

struct Linear {
    dense: Dense,
}

impl Linear {
    fn new(scalar_type: ScalarType) -> Result<Self> {
        let dense = Dense::builder()
            .inputs(28 * 28)
            .outputs(10)
            .bias(true)
            .scalar_type(scalar_type)
            .build()?;
        Ok(Self { dense })
    }
}

impl Layer for Linear {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.dense.set_training(training)?;
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        self.dense.parameters_mut()
    }
}

impl Forward<Variable4> for Linear {
    type Output = Variable2;
    fn forward(&self, input: Variable4) -> Result<Self::Output> {
        input.flatten().map_err(Error::msg)?.forward(&self.dense)
    }
}

enum Model {
    Linear(Linear),
}

impl Model {
    fn new(kind: ModelKind, scalar_type: ScalarType) -> Result<Self> {
        match kind {
            ModelKind::Linear => Ok(Self::Linear(Linear::new(scalar_type)?)),
        }
    }
}

impl Layer for Model {
    fn set_training(&mut self, training: bool) -> Result<()> {
        match self {
            Self::Linear(x) => {
                x.set_training(training)?;
            }
        }
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        match self {
            Self::Linear(x) => x.parameters_mut(),
        }
    }
}

impl Forward<Variable4> for Model {
    type Output = Variable2;
    fn forward(&self, input: Variable4) -> Result<Self::Output> {
        match self {
            Self::Linear(linear) => linear.forward(input),
        }
    }
}

#[derive(Default)]
struct Stats {
    count: usize,
    loss: f32,
    correct: usize,
}

impl std::ops::AddAssign for Stats {
    fn add_assign(&mut self, rhs: Self) {
        self.count += rhs.count;
        self.loss += rhs.loss;
        self.correct += rhs.correct;
    }
}

#[derive(Parser, Debug)]
#[command(author)]
struct Options {
    #[arg(long, value_enum, default_value_t = DatasetKind::Mnist)]
    dataset: DatasetKind,
    #[arg(long)]
    offline: bool,
    #[arg(long, value_enum, default_value_t = ModelKind::Linear)]
    model: ModelKind,
    #[arg(long)]
    bf16: bool,
    #[arg(short, long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 100)]
    train_batch_size: usize,
    #[arg(long, default_value_t = 1000)]
    test_batch_size: usize,
    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,
    #[arg(long)]
    parallel: bool,
}

fn main() -> Result<()> {
    let options = Options::parse();
    eprintln!("{options:#?}");
    let ((train_images, train_classes), (test_images, test_classes)) = match options.dataset {
        DatasetKind::Mnist | DatasetKind::FashionMnist => {
            let Mnist {
                train_images,
                train_classes,
                test_images,
                test_classes,
                ..
            } = Mnist::builder()
                .kind(options.dataset.mnist_kind())
                .download(!options.offline)
                .verbose(true)
                .build()?;
            ((train_images, train_classes), (test_images, test_classes))
        }
    };
    let train_images = train_images.into_shared();
    let train_classes = train_classes.into_shared();
    let scalar_type = if options.bf16 {
        ScalarType::BF16
    } else {
        ScalarType::F32
    };
    let mut model = Model::new(options.model, scalar_type)?;
    let optimizer = SGD::default();

    let image_scale = 1f32 / 255f32;
    let image_scale = if options.bf16 {
        ScalarElem::BF16(bf16::from_f32(image_scale))
    } else {
        ScalarElem::F32(image_scale)
    };

    let train_batch_size = options.train_batch_size;
    let train_iter_fn = || {
        let count = train_classes.len();
        let index_vec = sample(&mut thread_rng(), count, count);
        batches_indexed(
            train_batch_size,
            train_images.as_slice().unwrap(),
            train_classes.as_slice().unwrap(),
            index_vec.into_iter(),
        )
        .map(|(x, t)| -> Result<_> {
            let x = Tensor::from(Buffer::from(x))
                .into_shape([options.train_batch_size, 1, 28, 28])
                .unwrap()
                .into_scalar_tensor()
                .scaled_cast(image_scale)?;
            let t = ScalarCowTensor::from(Tensor::from(Buffer::from(t)).into_scalar_tensor());
            Ok((x, t))
        })
    };

    let train_par_iter_fn = || {
        let train_images_classes = (train_images.clone(), train_classes.clone());
        let (sender, receiver) = crossbeam_channel::bounded(1);
        rayon::spawn(move || {
            let (train_images, train_classes) = train_images_classes;
            let count = train_classes.len();
            let index_vec = sample(&mut thread_rng(), count, count);
            let iter = batches_indexed(
                options.train_batch_size,
                train_images.as_slice().unwrap(),
                train_classes.as_slice().unwrap(),
                index_vec.into_iter(),
            )
            .map(|(x, t)| -> Result<_> {
                let x = Tensor::from(Buffer::from(x))
                    .into_shape([options.train_batch_size, 1, 28, 28])
                    .unwrap()
                    .into_scalar_tensor()
                    .scaled_cast(image_scale)?;
                let t = ScalarCowTensor::from(Tensor::from(Buffer::from(t)).into_scalar_tensor());
                Ok((x, t))
            });
            for batch in iter {
                let result = sender.send(batch);
                if result.is_err() {
                    return;
                }
            }
        });
        receiver.into_iter()
    };

    let test_iter_fn = || {
        test_images
            .axis_chunks_iter(Axis(0), options.test_batch_size)
            .zip(test_classes.axis_chunks_iter(Axis(0), options.test_batch_size))
            .map(|(x, t)| -> Result<_> {
                let x = CowTensor::from(x)
                    .into_scalar_cow_tensor()
                    .scaled_cast(image_scale)?;
                let t = CowTensor::from(t).into_scalar_cow_tensor();
                Ok((x, t))
            })
    };

    let test_par_iter_fn = || {
        test_images
            .axis_chunks_iter(Axis(0), options.test_batch_size)
            .into_par_iter()
            .zip(test_classes.axis_chunks_iter(Axis(0), options.test_batch_size))
            .map(|(x, t)| -> Result<_> {
                let x = CowTensor::from(x)
                    .into_scalar_cow_tensor()
                    .scaled_cast(image_scale)?;
                let t = CowTensor::from(t).into_scalar_cow_tensor();
                Ok((x, t))
            })
    };

    model.set_training(true)?;

    for epoch in 1..=options.epochs {
        let epoch_start = Instant::now();
        fn train<'a, I: Iterator<Item = Result<(ScalarTensor4, ScalarCowTensor1<'a>)>>>(
            model: &mut Model,
            optimizer: &SGD,
            learning_rate: f32,
            mut train_iter: I,
        ) -> Result<Stats> {
            let mut train_stats = Stats::default();
            while let Some((x, t)) = train_iter.next().transpose()? {
                train_stats.count += x.shape().first().unwrap();
                let y = model.forward(x.into())?;
                train_stats.correct += Accuracy.eval(y.value().view(), t.view())?;
                let t_one_hot = t.to_one_hot(10, y.scalar_type())?;
                let loss = CrossEntropyLoss::default().eval(y, t_one_hot.into())?;
                loss.backward()?;
                for parameter in model.parameters_mut()? {
                    optimizer.update(learning_rate, parameter)?;
                }
                train_stats.loss += loss
                    .into_value()
                    .cast_into_tensor::<f32>()?
                    .into_array()?
                    .sum();
            }
            Ok(train_stats)
        }
        let train_stats = if options.parallel {
            train(
                &mut model,
                &optimizer,
                options.learning_rate,
                train_par_iter_fn(),
            )?
        } else {
            train(
                &mut model,
                &optimizer,
                options.learning_rate,
                train_iter_fn(),
            )?
        };
        let train_count = train_stats.count;
        let train_correct = train_stats.correct;
        let train_loss = train_stats.loss / train_count as f32;
        let train_acc = (train_correct * 100) as f32 / train_count as f32;

        let test_stats = if options.parallel {
            let test_stats = Arc::new(Mutex::new(Stats::default()));
            test_par_iter_fn().try_for_each_with(
                test_stats.clone(),
                |test_stats, batch| -> Result<()> {
                    let (x, t) = batch?;
                    let mut stats = Stats::default();
                    stats.count = x.shape().first().copied().unwrap();
                    let y = model.forward(x.into())?.into_value();
                    stats.correct = Accuracy.eval(y.view(), t.view())?;
                    let t_one_hot = t.to_one_hot(10, y.scalar_type())?;
                    stats.loss += CrossEntropyLoss::default()
                        .eval(y, t_one_hot)?
                        .cast_into_tensor::<f32>()?
                        .into_array()?
                        .sum();
                    *test_stats.lock() += stats;
                    Ok(())
                },
            )?;
            let test_stats = std::mem::take(&mut *test_stats.lock());
            test_stats
        } else {
            let mut test_stats = Stats::default();
            let mut test_iter = test_iter_fn();
            while let Some((x, t)) = test_iter.next().transpose()? {
                test_stats.count += x.shape().first().copied().unwrap();
                let y = model.forward(x.into())?.into_value();
                test_stats.correct += Accuracy.eval(y.view(), t.view())?;
                let t_one_hot = t.to_one_hot(10, y.scalar_type())?;
                test_stats.loss += CrossEntropyLoss::default()
                    .eval(y, t_one_hot)?
                    .cast_into_tensor::<f32>()?
                    .into_array()?
                    .sum();
            }
            test_stats
        };
        let test_count = test_stats.count;
        let test_correct = test_stats.count;
        let test_loss = test_stats.loss / test_count as f32;
        let test_acc = (test_stats.correct * 100) as f32 / test_count as f32;
        let epoch_elapsed = epoch_start.elapsed();
        println!(
                "[{epoch}] train_loss: {train_loss} train_acc: {train_acc}% {train_count}/{train_count} test_loss: {test_loss} test_acc: {test_acc}% {test_correct}/{test_count} elapsed: {epoch_elapsed:?}"
            );
    }
    Ok(())
}

pub fn batches_indexed<'a, I: Iterator<Item = usize> + 'a>(
    batch_size: usize,
    images: &'a [u8],
    classes: &'a [u8],
    mut index_iter: I,
) -> impl Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a {
    let image_size = 28 * 28;
    (0..classes.len() / batch_size).map(move |_| {
        let mut output_images = Vec::with_capacity(batch_size * image_size);
        let mut output_classes = Vec::with_capacity(batch_size);
        for index in index_iter.by_ref().take(batch_size) {
            output_images.extend(&images[index * image_size..(index + 1) * image_size]);
            output_classes.push(classes[index]);
        }
        (output_images, output_classes)
    })
}

/*
fn load_dataset(options: &Options) -> [ScalarArcTensorD; 4] {
    let Mnist {
        images, classes, ..
    } = Mnist::builder()
        .kind(options.dataset.kind())
        .download(true)
        .build()
        .unwrap();
    // Use the first 60_000 images as the training set.
    let train_images = ScalarArcTensor::from(Tensor::from(
        images.slice(s![..60_000, .., .., ..]).to_owned(),
    ))
    .into_dyn();
    let train_classes =
        ScalarArcTensor::from(Tensor::from(classes.slice(s![..60_000]).to_owned())).into_dyn();
    // Use the last 10_000 images as the test set.
    let test_images = ScalarArcTensor::from(Tensor::from(
        images.slice(s![60_000.., .., .., ..]).to_owned(),
    ))
    .into_dyn();
    let test_classes =
        ScalarArcTensor::from(Tensor::from(classes.slice(s![60_000..]).to_owned())).into_dyn();
    [train_images, train_classes, test_images, test_classes]
}

fn preprocess_images(input: ScalarArcTensorD) -> Result<ScalarArcTensorD> {
    let scale = 1f32 / 255f32;
    let output = TensorViewD::<u8>::try_from(input.view())
        .unwrap()
        .into_array()
        .unwrap()
        .map(|x| scale * x.cast::<f32>());
    Ok(ScalarArcTensor::from(Tensor::from(output)))

    //input.scaled_cast(scale.into())
}

fn accuracy(input: ScalarTensorView2, classes: ScalarTensorView1) -> usize {
    let input = TensorView2::<f32>::try_from(input).unwrap();
    let input = input.as_array().unwrap();
    let classes = TensorView1::<u8>::try_from(classes).unwrap();
    let classes = classes.as_array().unwrap();
    input
        .outer_iter()
        .zip(classes.iter().map(|x| x.to_usize().unwrap()))
        .filter(|(input, class)| {
            let mut max = input[0];
            let mut max_index = 0;
            for (i, x) in input.iter().copied().enumerate() {
                if x > max {
                    max = x;
                    max_index = i;
                }
            }
            max_index == *class
        })
        .count()
}

fn run<M>(options: Options, mut model: M, dataset: [ScalarArcTensorD; 4])
where
    M: Layer + Debug,
{
    println!("{model:#?}");
    println!("Training...");
    let nclasses = 10;
    let [train_images, train_classes, test_images, test_classes] = dataset;

    let criterion = CrossEntropyLoss::default();
    for epoch in 0..options.epochs {
        let epoch_start = Instant::now();
        model.set_training(true);
        let mut train_count = 0;
        let mut train_loss = 0f32;
        let mut train_correct = 0;
        let mut train_iter = Batches::new(
            (train_images.clone(), train_classes.clone()),
            options.train_batch_size,
        )
        .into_iter()
        .map(|batch| batch.and_then(|(input, classes)| Ok((preprocess_images(input)?, classes))));
        while let Some((x, t)) = train_iter.next().transpose().unwrap() {
            train_count += x.shape().first().copied().unwrap_or_default();
            let y = model.forward(x.into()).unwrap();
            train_correct += accuracy(
                y.value().view().into_dimensionality().unwrap(),
                t.view().into_dimensionality().unwrap(),
            );
            let t_one_hot = t.to_one_hot(nclasses, y.scalar_type()).unwrap();
            let loss = criterion.eval(y, t_one_hot.into()).unwrap();
            loss.backward().unwrap();
            //rayon::scope(|scope| {
            model.for_each_parameter_mut(&mut |parameter| {
                //scope.spawn(|_| {
                let Gradient::Dense(gradient) = parameter.grad().unwrap().load().unwrap();
                parameter
                    .make_view_mut()
                    .unwrap()
                    .scaled_add((-options.learning_rate).into(), &gradient)
                    .unwrap();
                // });
            });
            //});
            let loss = ArcTensorD::<f32>::try_from(loss.into_value())
                .unwrap()
                .into_array()
                .unwrap();
            train_loss += loss.sum();
        }
        let train_loss = train_loss / train_count as f32;
        let train_acc = (train_correct * 100) as f32 / train_count as f32;
        model.set_training(false);
        let mut test_count = 0;
        let mut test_loss = 0f32;
        let mut test_correct = 0;
        let mut test_iter = Batches::new(
            (test_images.clone(), test_classes.clone()),
            options.test_batch_size,
        )
        .into_iter()
        .map(|batch| batch.and_then(|(input, classes)| Ok((preprocess_images(input)?, classes))));
        while let Some((x, t)) = test_iter.next().transpose().unwrap() {
            test_count += x.shape().first().copied().unwrap_or_default();
            let y = model.forward(x.into()).unwrap();
            test_correct += accuracy(
                y.value().view().into_dimensionality().unwrap(),
                t.view().into_dimensionality().unwrap(),
            );
            let t_one_hot = t.to_one_hot(nclasses, y.scalar_type()).unwrap();
            let loss = criterion.eval(y.into_value(), t_one_hot).unwrap();
            let loss = TensorD::<f32>::try_from(loss)
                .unwrap()
                .into_array()
                .unwrap();
            test_loss += loss.sum();
        }
        let test_loss = test_loss / test_count as f32;
        let test_acc = (test_correct * 100) as f32 / test_count as f32;
        let epoch_elapsed = epoch_start.elapsed();
        println!(
            "[{epoch}] train_loss: {train_loss} train_acc: {train_acc}% {train_count}/{train_count} test_loss: {test_loss} test_acc: {test_acc}% {test_correct}/{test_count} elapsed: {epoch_elapsed:?}"
        );
    }
}
*/
