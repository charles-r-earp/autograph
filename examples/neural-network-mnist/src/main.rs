use autograph::{
    anyhow::{Error, Result},
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
    ndarray::{Array, Array1, Array4, ArrayView1, ArrayView4, Axis},
    scalar::{ScalarElem, ScalarType},
    tensor::{CowTensor, ScalarCowTensor, ScalarCowTensor1, ScalarTensor4, Tensor},
};
use clap::{Parser, ValueEnum};
use parking_lot::Mutex;
use rand::{seq::index::sample, thread_rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, ValueEnum, Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
struct Checkpoint {
    epoch: usize,
    time: Duration,
    dataset: DatasetKind,
    model: Model,
    optimizer: SGD,
}

impl Checkpoint {
    fn path_for(options: &Options, scalar_type: ScalarType) -> PathBuf {
        let Options {
            dataset,
            model,
            momentum,
            ..
        } = options;
        let mut name = format!("{dataset:?}_{model:?}_{scalar_type:?}_sgd").to_lowercase();
        if let Some(momentum) = momentum {
            name.push_str(&format!("_momentum_{momentum}"));
        }
        PathBuf::from(name).with_extension("checkpoint")
    }
    fn load(path: &Path) -> Result<Option<Self>> {
        if path.exists() {
            let string = std::fs::read_to_string(path)?;
            Ok(Some(serde_json::from_str(&string)?))
        } else {
            Ok(None)
        }
    }
    fn save(&self, path: &Path) -> Result<()> {
        let string = serde_json::to_string(&self)?;
        std::fs::write(path, string)?;
        Ok(())
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
    #[arg(short, long, default_value_t = 100)]
    epochs: usize,
    #[arg(long, default_value_t = 100)]
    train_batch_size: usize,
    #[arg(long, default_value_t = 1000)]
    test_batch_size: usize,
    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,
    #[arg(long)]
    momentum: Option<f32>,
    #[arg(long)]
    checkpoint: bool,
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
    let scalar_type = if options.bf16 {
        ScalarType::BF16
    } else {
        ScalarType::F32
    };
    let checkpoint_path = Checkpoint::path_for(&options, scalar_type);
    let checkpoint = if options.checkpoint {
        Checkpoint::load(&checkpoint_path)?
    } else {
        None
    };
    let Checkpoint {
        mut epoch,
        mut time,
        dataset: _,
        mut model,
        mut optimizer,
    } = if let Some(checkpoint) = checkpoint {
        checkpoint
    } else {
        let model = Model::new(options.model, scalar_type)?;
        let optimizer = {
            let mut builder = SGD::builder();
            if let Some(momentum) = options.momentum {
                builder = builder.momentum(momentum);
            }
            builder.build()
        };
        Checkpoint {
            epoch: 1,
            time: Duration::default(),
            dataset: options.dataset,
            model,
            optimizer,
        }
    };

    let image_scale = 1f32 / 255f32;
    let image_scale = if options.bf16 {
        ScalarElem::BF16(bf16::from_f32(image_scale))
    } else {
        ScalarElem::F32(image_scale)
    };

    while epoch <= options.epochs {
        let epoch_start = Instant::now();
        let train_iter = shuffled_batches(
            options.train_batch_size,
            train_images.view(),
            train_classes.view(),
        )
        .map(|(x, t)| -> Result<_> {
            let x = Tensor::from(x)
                .into_scalar_tensor()
                .scaled_cast(image_scale)?;
            let t = ScalarCowTensor::from(Tensor::from(t).into_scalar_tensor());
            Ok((x, t))
        });
        let train_stats = train(&mut model, &optimizer, options.learning_rate, train_iter)?;
        let train_count = train_stats.count;
        let train_correct = train_stats.correct;
        let train_loss = train_stats.loss / train_count as f32;
        let train_acc = (train_correct * 100) as f32 / train_count as f32;
        let test_iter = test_images
            .axis_chunks_iter(Axis(0), options.test_batch_size)
            .into_par_iter()
            .zip(test_classes.axis_chunks_iter(Axis(0), options.test_batch_size))
            .map(|(x, t)| -> Result<_> {
                let x = CowTensor::from(x)
                    .into_scalar_cow_tensor()
                    .scaled_cast(image_scale)?;
                let t = CowTensor::from(t).into_scalar_cow_tensor();
                Ok((x, t))
            });
        let test_stats = test(&model, test_iter)?;
        let test_count = test_stats.count;
        let test_correct = test_stats.count;
        let test_loss = test_stats.loss / test_count as f32;
        let test_acc = (test_stats.correct * 100) as f32 / test_count as f32;
        let epoch_elapsed = epoch_start.elapsed();
        println!(
                "[{epoch}] train_loss: {train_loss} train_acc: {train_acc}% {train_count}/{train_count} test_loss: {test_loss} test_acc: {test_acc}% {test_correct}/{test_count} elapsed: {epoch_elapsed:?}"
            );
        time += epoch_elapsed;
        if options.checkpoint {
            let checkpoint = Checkpoint {
                epoch,
                time,
                dataset: options.dataset,
                model,
                optimizer,
            };
            checkpoint.save(&checkpoint_path)?;
            model = checkpoint.model;
            optimizer = checkpoint.optimizer;
        }
        epoch += 1;
    }
    println!("Finished in {time:?}.");
    let start = Instant::now();
    let test_iter = test_images
        .axis_chunks_iter(Axis(0), options.test_batch_size)
        .into_par_iter()
        .zip(test_classes.axis_chunks_iter(Axis(0), options.test_batch_size))
        .map(|(x, t)| -> Result<_> {
            let x = CowTensor::from(x)
                .into_scalar_cow_tensor()
                .scaled_cast(image_scale)?;
            let t = CowTensor::from(t).into_scalar_cow_tensor();
            Ok((x, t))
        });
    let test_stats = test(&model, test_iter)?;
    let test_count = test_stats.count;
    let test_correct = test_stats.count;
    let test_loss = test_stats.loss / test_count as f32;
    let test_acc = (test_stats.correct * 100) as f32 / test_count as f32;
    let elapsed = start.elapsed();
    println!(
        "[inference] loss: {test_loss} acc: {test_acc}% {test_correct}/{test_count} elapsed: {elapsed:?}"
    );
    Ok(())
}

pub fn shuffled_batches<'a>(
    batch_size: usize,
    images: ArrayView4<'a, u8>,
    classes: ArrayView1<'a, u8>,
) -> impl Iterator<Item = (Array4<u8>, Array1<u8>)> + 'a {
    let (count, depth, height, width) = images.dim();
    let mut index_iter = sample(&mut thread_rng(), count, count).into_iter();
    (0..count / batch_size).map(move |_| {
        let mut output_images = Vec::<u8>::with_capacity(batch_size * depth * height * width);
        let mut output_classes = Vec::<u8>::with_capacity(batch_size);
        for index in index_iter.by_ref().take(batch_size) {
            output_images.extend_from_slice(images.index_axis(Axis(0), index).as_slice().unwrap());
            output_classes.push(classes[index]);
        }
        let output_images = Array::from(output_images)
            .into_shape([batch_size, depth, height, width])
            .unwrap();
        let output_classes = output_classes.into();
        (output_images, output_classes)
    })
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

fn train<'a, I: Iterator<Item = Result<(ScalarTensor4, ScalarCowTensor1<'a>)>>>(
    model: &mut Model,
    optimizer: &SGD,
    learning_rate: f32,
    mut train_iter: I,
) -> Result<Stats> {
    let mut train_stats = Stats::default();
    while let Some((x, t)) = train_iter.next().transpose()? {
        train_stats.count += x.shape().first().unwrap();
        model.set_training(true)?;
        let y = model.forward(x.into())?;
        train_stats.correct += Accuracy.eval(y.value().view(), t.view())?;
        let t_one_hot = t.to_one_hot(10, y.scalar_type())?;
        let loss = CrossEntropyLoss::default().eval(y, t_one_hot.into())?;
        loss.backward()?;
        for parameter in model.parameters_mut()? {
            optimizer.update(learning_rate, parameter)?;
        }
        model.set_training(false)?;
        train_stats.loss += loss
            .into_value()
            .cast_into_tensor::<f32>()?
            .into_array()?
            .sum();
    }
    Ok(train_stats)
}

fn test<'a, I: IntoParallelIterator<Item = Result<(ScalarTensor4, ScalarCowTensor1<'a>)>>>(
    model: &Model,
    test_iter: I,
) -> Result<Stats> {
    let test_stats = Arc::new(Mutex::new(Stats::default()));
    test_iter.into_par_iter().try_for_each_with(
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
    Ok(test_stats)
}
