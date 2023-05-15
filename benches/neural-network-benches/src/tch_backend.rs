use anyhow::Result;
use tch::{
    kind::Kind,
    nn::{Linear, LinearConfig, ModuleT, Optimizer, OptimizerConfig, Sgd, VarStore},
    Device, Reduction, Tensor,
};

pub struct LinearClassifier {
    device: Device,
    kind: Kind,
    var_store: VarStore,
    inputs: usize,
    outputs: usize,
    linear: Linear,
    optimizer: Option<Optimizer>,
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
            var_store,
            inputs,
            outputs,
            linear,
            optimizer: None,
        })
    }
    pub fn with_sgd(self, momentum: bool) -> Result<Self> {
        let momentum = if momentum { 0.01 } else { 0.0 };
        let learning_rate = 0.01;
        let optimizer = Sgd {
            momentum,
            ..Sgd::default()
        }
        .build(&self.var_store, 0.01)?;
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
        let _ = self.linear.forward_t(&x, false);
        Ok(())
    }
    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        let x = Tensor::zeros(
            [batch_size as i64, self.inputs as i64],
            (self.kind, self.device),
        );
        let t = Tensor::zeros(
            [batch_size as i64, self.outputs as i64],
            (self.kind, self.device),
        );
        let y = self.linear.forward_t(&x, true);
        let loss = y.cross_entropy_loss::<Tensor>(&t, None, Reduction::Sum, -1, 0.);
        self.optimizer.as_mut().unwrap().backward_step(&loss);
        Ok(())
    }
}
