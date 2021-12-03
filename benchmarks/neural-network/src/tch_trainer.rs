use crate::{Library, TrainerDescriptor, TrainerStats};
use autograph::{dataset::mnist::Mnist, result::Result};
use ndarray::s;
use std::time::Instant;
use tch::{
    nn::{ModuleT, OptimizerConfig},
    vision::dataset::Dataset,
    Device, Kind, Tensor,
};

#[derive(Debug)]
struct Lenet5 {
    conv1: tch::nn::Conv2D,
    conv2: tch::nn::Conv2D,
    dense1: tch::nn::Linear,
    dense2: tch::nn::Linear,
    dense3: tch::nn::Linear,
}

impl Lenet5 {
    fn new(vs: &tch::nn::Path) -> Self {
        let mut conv_config = tch::nn::ConvConfig::default();
        conv_config.bias = false;
        let conv1 = tch::nn::conv2d(vs, 1, 6, 5, conv_config);
        let conv2 = tch::nn::conv2d(vs, 6, 16, 5, conv_config);
        let mut linear_config = tch::nn::LinearConfig::default();
        linear_config.bias = false;
        let dense1 = tch::nn::linear(vs, 256, 120, linear_config);
        let dense2 = tch::nn::linear(vs, 120, 84, linear_config);
        linear_config.bias = true;
        let dense3 = tch::nn::linear(vs, 84, 10, linear_config);
        Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        }
    }
}

impl ModuleT for Lenet5 {
    fn forward_t(&self, xs: &tch::Tensor, _train: bool) -> tch::Tensor {
        xs.apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view([-1, 256])
            .apply(&self.dense1)
            .relu()
            .apply(&self.dense2)
            .relu()
            .apply(&self.dense3)
    }
}

pub enum Tch {}

impl Library for Tch {
    fn name() -> &'static str {
        "tch"
    }
    fn benchmark(
        trainer_descriptor: &TrainerDescriptor,
        mut epoch_cb: impl FnMut(usize),
    ) -> Result<TrainerStats> {
        let dataset_descriptor = &trainer_descriptor.dataset;
        let train_batch_size = dataset_descriptor.train_batch_size;
        let test_batch_size = dataset_descriptor.test_batch_size;
        let epochs = trainer_descriptor.epochs;

        let mnist = mnist()?;
        let vs = tch::nn::VarStore::new(Device::Cuda(0));
        let network = Lenet5::new(&vs.root());
        let lr = 0.1;
        let mut optim = tch::nn::Sgd::default().build(&vs, lr).unwrap();

        let mut total_time = vec![0f32; epochs];
        let mut test_accuracy = vec![0f32; epochs];

        let start = Instant::now();
        for epoch in 0..epochs {
            for (x, t) in mnist
                .train_iter(train_batch_size as i64)
                .shuffle()
                .to_device(vs.device())
            {
                let loss = network.forward_t(&x, true).cross_entropy_for_logits(&t);
                optim.backward_step(&loss);
            }
            test_accuracy[epoch] = network.batch_accuracy_for_logits(
                &mnist.test_images,
                &mnist.test_labels,
                vs.device(),
                test_batch_size as i64,
            ) as f32;
            total_time[epoch] = start.elapsed().as_secs_f32();
            epoch_cb(epoch);
        }

        Ok(TrainerStats {
            total_time,
            test_accuracy,
        })
    }
}

fn mnist() -> Result<Dataset> {
    let mnist = Mnist::builder().download(true).build()?;
    // Use the first 60_000 images as the training set.
    let train_images = mnist.images().slice(s![..60_000, .., .., ..]);
    let train_images = Tensor::of_slice(train_images.as_slice().unwrap())
        .view((-1, 1, 28, 28))
        .to_kind(Kind::Float)
        .f_mul_scalar(1. / 255.)?;
    let train_classes = mnist.classes().slice(s![..60_000]);
    let train_labels = Tensor::of_slice(train_classes.as_slice().unwrap());
    // Use the last 10_000 images as the test set.
    let test_images = mnist.images().slice(s![60_000.., .., .., ..]);
    let test_images = Tensor::of_slice(test_images.as_slice().unwrap())
        .view((-1, 1, 28, 28))
        .to_kind(Kind::Float)
        .f_mul_scalar(1. / 255.)?;
    let test_classes = mnist.classes().slice(s![60_000..]);
    let test_labels = Tensor::of_slice(test_classes.as_slice().unwrap()).to_kind(Kind::Int64);
    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    })
}
