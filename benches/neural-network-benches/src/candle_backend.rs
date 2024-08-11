use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{
    conv2d_no_bias, linear, linear_no_bias, loss::cross_entropy, Conv2d, Conv2dConfig, Linear,
    Module, Optimizer, VarBuilder, VarMap, SGD,
};

pub struct LeNet5Classifier {
    device: Device,
    dtype: DType,
    model: LeNet5,
    optimizer: Option<SGD>,
    varmap: VarMap,
    _var_builder: VarBuilder<'static>,
}

impl LeNet5Classifier {
    pub fn new(device: Device, dtype: DType) -> Result<Self> {
        let varmap = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = LeNet5::new(&var_builder)?;
        Ok(Self {
            device,
            dtype,
            model,
            optimizer: None,
            varmap,
            _var_builder: var_builder,
        })
    }
    pub fn with_sgd(self, momentum: bool) -> Result<Self> {
        if momentum {
            anyhow::bail!("Momentum not supported by candle!");
        }
        /*
        let momentum = if momentum { 0.01 } else { 0.0 };
        */
        let learning_rate = 0.01;
        let optimizer = SGD::new(self.varmap.all_vars(), learning_rate)?;
        Ok(Self {
            optimizer: Some(optimizer),
            ..self
        })
    }
    pub fn infer(&self, batch_size: usize) -> Result<()> {
        let x = Tensor::zeros((batch_size, 1, 28, 28), self.dtype, &self.device)?;
        let _y = self.model.forward(&x)?;
        Ok(())
    }
    pub fn train(&mut self, batch_size: usize) -> Result<()> {
        let x = Var::zeros((batch_size, 1, 28, 28), self.dtype, &self.device)?;
        let t = Tensor::zeros(batch_size, DType::U32, &self.device)?;
        let y = self.model.forward(&x)?;
        let loss = cross_entropy(&y, &t)?;
        self.optimizer.as_mut().unwrap().backward_step(&loss)?;
        Ok(())
    }
}

#[derive(Debug)]
struct LeNet5 {
    conv1: Conv2d,
    conv2: Conv2d,
    dense1: Linear,
    dense2: Linear,
    dense3: Linear,
}

impl LeNet5 {
    fn new(var_builder: &VarBuilder) -> Result<Self> {
        let conv1 = conv2d_no_bias(1, 6, 5, Conv2dConfig::default(), var_builder.pp("conv1"))?;
        let conv2 = conv2d_no_bias(6, 16, 5, Conv2dConfig::default(), var_builder.pp("conv2"))?;
        let dense1 = linear_no_bias(16 * 4 * 4, 128, var_builder.pp("dense1"))?;
        let dense2 = linear_no_bias(128, 84, var_builder.pp("dense2"))?;
        let dense3 = linear(84, 10, var_builder.pp("dense3"))?;
        Ok(Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        })
    }
}

impl Module for LeNet5 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::error::Error> {
        let Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        } = self;
        let x = conv1.forward(xs)?.relu()?.max_pool2d(2)?;
        let x = conv2.forward(&x)?.relu()?.max_pool2d(2)?.flatten_from(1)?;
        let x = dense1.forward(&x)?.relu()?;
        let x = dense2.forward(&x)?.relu()?;
        dense3.forward(&x)
    }
}
