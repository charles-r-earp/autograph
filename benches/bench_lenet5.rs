#![allow(warnings)]
#[macro_use]
extern crate autograph;
use autograph::nn::{
    Conv2d, Dense, Forward, Layer,
    autograd::{Graph, ParameterD, Variable, Variable2, Variable4}
};
#[cfg(feature = "cuda")]
use autograph::CudaGpu;
use autograph::{ArcTensor, Cpu, Device, Pool2dArgs, Tensor, Tensor2, Tensor4, TensorView4};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Dimension, Ix2, Ix4};
use num_traits::ToPrimitive;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

#[derive(Layer)]
struct Lenet5 {
    conv1: Conv2d,
    conv2: Conv2d,
    dense1: Dense,
    dense2: Dense,
    dense3: Dense,
}

impl Lenet5 {
    pub fn new(device: &Device) -> Self {
        let conv1 = Conv2d::builder()
            .device(&device)
            .inputs(1)
            .outputs(6)
            .kernel(5)
            .build();
        let conv2 = Conv2d::builder()
            .device(&device)
            .inputs(6)
            .outputs(16)
            .kernel(5)
            .build();
        let dense1 = Dense::builder()
            .device(&device)
            .inputs(256)
            .outputs(120)
            .build();
        let dense2 = Dense::builder()
            .device(&device)
            .inputs(120)
            .outputs(84)
            .build();
        let dense3 = Dense::builder()
            .device(&device)
            .inputs(84)
            .outputs(10)
            .bias()
            .build();
        Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        }
    }
}

impl Forward<Ix4> for Lenet5 {
    type OutputDim = Ix2;
    fn forward(&self, input: &Variable4) -> Variable2 {
        let pool_args = Pool2dArgs::default().kernel(2).strides(2);
        input
            .forward(&self.conv1)
            .relu()
            .max_pool2d(&pool_args)
            .forward(&self.conv2)
            .relu()
            .max_pool2d(&pool_args)
            .flatten()
            .forward(&self.dense1)
            .relu()
            .forward(&self.dense2)
            .relu()
            .forward(&self.dense3)
    }
}

fn bench_autograph_lenet5(c: &mut Criterion, device: &Device, batch_size: usize) {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut model = Lenet5::new(device);
    model.parameters().into_iter().for_each(|w| {
        let dim = w.value().raw_dim();
        if dim.ndim() > 1 {
            w.value()
                .write()
                .unwrap()
                .fill_random(&Normal::new(0., 0.01).unwrap(), &mut rng)
        }
    });

    let lr = 0.001;

    let x = ArcTensor::ones(&device, [batch_size, 1, 28, 28]);
    let t = Tensor::<u8, _>::zeros(&device, batch_size);
    c.bench_function(
        &format!("autograph_lenet5_train_{}_{:?}", batch_size, device),
        |b| {
            b.iter(|| {
                model.set_training(true);
                let graph = Graph::new();
                let x = Variable::new(Some(&graph), x.clone(), false);
                let y = model.forward(&x);
                let t = ArcTensor::from(t.to_one_hot_f32(10));
                let loss = y.cross_entropy_loss(&t);
                loss.backward(graph);
                model.parameters().iter().for_each(|w| {
                    let mut w_value = w.value().write().unwrap();
                    let mut w_grad = w.grad().unwrap().write().unwrap();
                    w_value.scaled_add(-lr, &w_grad);
                });
                device.synchronize();
            })
        },
    );
    let x = ArcTensor::ones(&device, [batch_size, 1, 28, 28]);
    let t = Tensor::<u8, _>::zeros(&device, batch_size);
    c.bench_function(
        &format!("autograph_lenet5_eval_{}_{:?}", batch_size, device),
        |b| {
            b.iter(|| {
                model.set_training(false);
                let x = Variable::new(None, x.clone(), false);
                let y = model.forward(&x);
                let t = ArcTensor::from(t.to_one_hot_f32(10));
                let loss = y.cross_entropy_loss(&t);
                device.synchronize();
            })
        },
    );
}

#[derive(Debug)]
struct TchLenet5 {
    conv1: tch::nn::Conv2D,
    conv2: tch::nn::Conv2D,
    dense1: tch::nn::Linear,
    dense2: tch::nn::Linear,
    dense3: tch::nn::Linear,
}

impl TchLenet5 {
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

impl tch::nn::ModuleT for TchLenet5 {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
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

fn bench_tch_lenet5(c: &mut Criterion, device: tch::Device, batch_size: usize) {
    use tch::nn::ModuleT;
    use tch::nn::OptimizerConfig;
    let vs = tch::nn::VarStore::new(device);
    let model = TchLenet5::new(&vs.root());
    let lr = 0.001;
    let mut optim = tch::nn::Sgd::default().build(&vs, lr).unwrap();

    let device_name = match device {
        tch::Device::Cpu => String::from("cpu"),
        tch::Device::Cuda(i) => format!("cuda:{}", i),
    };
    let x = tch::Tensor::ones(&[batch_size as i64, 1, 28, 28], (tch::Kind::Float, device));
    let t = tch::Tensor::zeros(&[batch_size as i64], (tch::Kind::Int64, device));
    c.bench_function(
        &format!("tch_lenet5_train_{}_{}", batch_size, &device_name),
        |b| {
            b.iter(|| {
                let y = model.forward_t(&x, true);
                let loss = y.cross_entropy_for_logits(&t);
                optim.backward_step(&loss);
            })
        },
    );
    let x = tch::Tensor::ones(&[batch_size as i64, 1, 28, 28], (tch::Kind::Float, device));
    let t = tch::Tensor::zeros(&[batch_size as i64], (tch::Kind::Int64, device));
    c.bench_function(
        &format!("tch_lenet5_eval_{}_{}", batch_size, &device_name),
        |b| {
            b.iter(|| {
                let y = model.forward_t(&x, false);
                let loss = y.cross_entropy_for_logits(&t);
            })
        },
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let cpu = Device::from(Cpu::new());
    bench_autograph_lenet5(c, &cpu, 256);
    #[cfg(feature = "cuda")]
    {
        let gpu = Device::from(CudaGpu::new(0));
        bench_autograph_lenet5(c, &gpu, 256);
    }
    bench_tch_lenet5(c, tch::Device::Cpu, 256);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
