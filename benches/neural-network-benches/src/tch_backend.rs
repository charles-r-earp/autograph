use anyhow::Result;
use tch::{
    kind::Kind,
    nn::{
        Conv2D, ConvConfig, Linear, LinearConfig, Module, Optimizer, OptimizerConfig, Sgd, VarStore,
    },
    Device, Reduction, Tensor,
};

pub struct Lenet5Classifier {
    device: Device,
    kind: Kind,
    model: Lenet5,
    optimizer: Option<Optimizer>,
    var_store: VarStore,
}

impl Lenet5Classifier {
    pub fn new(device: Device, kind: Kind) -> Result<Self> {
        let mut var_store = VarStore::new(device);
        let model = Lenet5::new(&var_store);
        var_store.set_kind(kind);
        Ok(Self {
            device,
            kind,
            model,
            optimizer: None,
            var_store,
        })
    }
    pub fn with_sgd(self, momentum: bool) -> Result<Self> {
        let momentum = if momentum { 0.01 } else { 0.0 };
        let learning_rate = 0.01;
        let optimizer = Sgd {
            momentum,
            ..Sgd::default()
        }
        .build(&self.var_store, learning_rate)?;
        Ok(Self {
            optimizer: Some(optimizer),
            ..self
        })
    }
    pub fn infer(&self, batch_size: usize) -> Result<()> {
        let x = Tensor::zeros([batch_size as i64, 1, 28, 28], (self.kind, self.device));
        let mut y = vec![0f32; batch_size * 10];
        self.model.forward(&x).copy_data(&mut y, batch_size * 10);
        Ok(())
    }
    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        let x = Tensor::zeros([batch_size as i64, 1, 28, 28], (self.kind, self.device));
        let t = Tensor::zeros([batch_size as i64], (Kind::Uint8, self.device));
        let y = self.model.forward(&x);
        let loss = y.cross_entropy_loss::<Tensor>(&t, None, Reduction::Sum, -1, 0.);
        self.optimizer.as_mut().unwrap().backward_step(&loss);
        Ok(())
    }
}

#[derive(Debug)]
struct Lenet5 {
    conv1: Conv2D,
    conv2: Conv2D,
    dense1: Linear,
    dense2: Linear,
    dense3: Linear,
}

impl Lenet5 {
    fn new(var_store: &VarStore) -> Self {
        let conv1 = tch::nn::conv2d(var_store.root(), 1, 6, 5, ConvConfig::default());
        let conv2 = tch::nn::conv2d(var_store.root(), 6, 16, 5, ConvConfig::default());
        let dense1 = tch::nn::linear(var_store.root(), 16 * 4 * 4, 128, LinearConfig::default());
        let dense2 = tch::nn::linear(var_store.root(), 128, 84, LinearConfig::default());
        let dense3 = tch::nn::linear(
            var_store.root(),
            84,
            10,
            LinearConfig {
                bias: true,
                ..LinearConfig::default()
            },
        );
        Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        }
    }
}

impl Module for Lenet5 {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        } = self;
        let x = conv1.forward(xs).relu().max_pool2d_default(2);
        let x = conv2
            .forward(&x)
            .relu()
            .max_pool2d_default(2)
            .flatten(1, -1);
        let x = dense1.forward(&x).relu();
        let x = dense2.forward(&x).relu();
        dense3.forward(&x)
    }
}
