use anyhow::Result;
use tch::{
    kind::Kind,
    nn::{Linear, LinearConfig, ModuleT, Optimizer, OptimizerConfig, Sgd, VarStore},
    Device, Reduction, Tensor,
};

pub struct LinearClassifier {
    device: Device,
    kind: Kind,
    inputs: usize,
    outputs: usize,
    linear: Linear,
    optimizer: Option<Optimizer>,
    var_store: VarStore,
}

impl LinearClassifier {
    pub fn new(device: Device, kind: Kind, inputs: usize, outputs: usize) -> Result<Self> {
        let mut var_store = VarStore::new(device);
        let linear = tch::nn::linear(
            var_store.root(),
            inputs as i64,
            outputs as i64,
            LinearConfig {
                bias: true,
                ..LinearConfig::default()
            },
        );
        var_store.set_kind(kind);
        Ok(Self {
            device,
            kind,
            inputs,
            outputs,
            linear,
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
        let x = Tensor::zeros(
            [batch_size as i64, self.inputs as i64],
            (self.kind, self.device),
        );
        let mut y = vec![0f32; batch_size * self.outputs];
        self.linear
            .forward_t(&x, false)
            .copy_data(&mut y, batch_size * self.outputs);
        Ok(())
    }
    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        let x = Tensor::zeros(
            [batch_size as i64, self.inputs as i64],
            (self.kind, self.device),
        );
        let t = Tensor::zeros([batch_size as i64], (Kind::Uint8, self.device));
        let y = self.linear.forward_t(&x, true);
        let loss = y.cross_entropy_loss::<Tensor>(&t, None, Reduction::Sum, -1, 0.);
        self.optimizer.as_mut().unwrap().backward_step(&loss);
        Ok(())
    }
}
