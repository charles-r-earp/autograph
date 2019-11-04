use super::{Var, Param, functional, init, init::Initializer, optim::Optimizer}; 
use std::{ops::Deref, marker::PhantomData};
use ndarray as nd;

#[derive(Debug)]
pub struct Parameter<T, D: nd::Dimension, I: Initializer<T>, O: Optimizer<T, D>> {
  param: Param<T, D>,
  init: I,
  optim: O
}

impl<T, D: nd::Dimension, I: Initializer<T>, O: Optimizer<T, D>> Parameter<T, D, I, O> {
  pub fn placeholder(init: I, optim: O) -> Self
    where T: Default {
    Self{param: Param::default(), init, optim}
  }
  pub fn step(&mut self, lr: T)
    where T: Clone {
    self.param.step(&mut self.optim, lr);
  }
  pub fn param(&self) -> &Param<T, D> { &self.param }
  pub fn initialize(&mut self, shape: impl nd::ShapeBuilder<Dim=D>)
    where T: Copy {
    self.param = Param::new(unsafe { nd::ArcArray::uninitialized(shape) });
    self.param.initialize(&self.init);
  } 
  pub fn train(&mut self)
    where T: num_traits::Zero + Clone {
    self.param.zero_grad();
  }
  pub fn eval(&mut self) {
    self.param.none_grad();
  }
} 

pub type Parameter1<T, I, O> = Parameter<T, nd::Ix1, I, O>;
pub type Parameter2<T, I, O> = Parameter<T, nd::Ix2, I, O>;

pub trait Forward<X> {
  fn forward(&self, input: &X) -> X;
}

pub trait Layer<T>: Forward<nd::ArrayD<T>> + Forward<Var<T>> {
  fn build(&mut self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    self.forward(input)
  }
  fn train(&mut self) {}
  fn eval(&mut self) {}
  fn step(&mut self, lr: T) {}
}

pub struct DenseBuilder<T, Ki: Initializer<T>, Ko: Optimizer<T, nd::Ix2>, Bi: Initializer<T>, Bo: Optimizer<T, nd::Ix1>> {
  units: Option<usize>,
  use_bias: bool,
  kernel_initializer: Ki,
  kernel_optimizer: Ko,
  bias_initializer: Bi,
  bias_optimizer: Bo,
  _m: PhantomData<T>
} 
pub fn dense_builder<T>() -> DenseBuilder<T, init::HeNormal, (), init::Zeros, ()>
  where init::HeNormal: Initializer<T>,
        init::Zeros: Initializer<T>,
        (): Optimizer<T, nd::Ix2> + Optimizer<T, nd::Ix1> {
  DenseBuilder{
    units: None,
    use_bias: false,
    kernel_initializer: init::HeNormal,
    kernel_optimizer: (),
    bias_initializer: init::Zeros,
    bias_optimizer: (),
    _m: PhantomData::default()
  }
}
  

impl<T, Ki: Initializer<T>, Ko: Optimizer<T, nd::Ix2>, Bi: Initializer<T>, Bo: Optimizer<T, nd::Ix1>> DenseBuilder<T, Ki, Ko, Bi, Bo> {
  pub fn units(mut self, units: usize) -> Self {
    self.units.replace(units);
    self
  }
  pub fn use_bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn kernel_initializer<Ki2: Initializer<T>>(self, kernel_initializer: Ki2) -> DenseBuilder<T, Ki2, Ko, Bi, Bo> {
    DenseBuilder{
      units: self.units,
      use_bias: self.use_bias,
      kernel_initializer,
      kernel_optimizer: self.kernel_optimizer,
      bias_initializer: self.bias_initializer,
      bias_optimizer: self.bias_optimizer,
      _m: self._m
    }
  }   
  pub fn kernel_optimizer<Ko2: Optimizer<T, nd::Ix2>>(self, kernel_optimizer: Ko2) -> DenseBuilder<T, Ki, Ko2, Bi, Bo> {
    DenseBuilder{
      units: self.units,
      use_bias: self.use_bias,
      kernel_initializer: self.kernel_initializer,
      kernel_optimizer,
      bias_initializer: self.bias_initializer,
      bias_optimizer: self.bias_optimizer,
      _m: self._m
    }
  }   
  pub fn bias_initializer<Bi2: Initializer<T>>(self, bias_initializer: Bi2) -> DenseBuilder<T, Ki, Ko, Bi2, Bo> {
    DenseBuilder{
      units: self.units,
      use_bias: self.use_bias,
      kernel_initializer: self.kernel_initializer,
      kernel_optimizer: self.kernel_optimizer,
      bias_initializer,
      bias_optimizer: self.bias_optimizer,
      _m: self._m
    }
  }   
  pub fn bias_optimizer<Bo2: Optimizer<T, nd::Ix1>>(self, bias_optimizer: Bo2) -> DenseBuilder<T, Ki, Ko, Bi, Bo2> {
    DenseBuilder{
      units: self.units,
      use_bias: self.use_bias,
      kernel_initializer: self.kernel_initializer,
      kernel_optimizer: self.kernel_optimizer,
      bias_initializer: self.bias_initializer,
      bias_optimizer,
      _m: self._m
    }
  }   
  pub fn optimizer<O2: Clone + Optimizer<T, nd::Ix2> + Optimizer<T, nd::Ix1>>(self, optimizer: O2) -> DenseBuilder<T, Ki, O2, Bi, O2> {
    self.kernel_optimizer(optimizer.clone())
      .bias_optimizer(optimizer) 
  }
  pub fn build(self) -> Dense<T, Ki, Ko, Bi, Bo>
    where T: Default,
          Ki: Initializer<T>, 
          Ko: Optimizer<T, nd::Ix2>,
          Bi: Initializer<T>,
          Bo: Optimizer<T, nd::Ix1> {
    Dense{
      units: self.units.unwrap(),
      kernel: Parameter::placeholder(self.kernel_initializer, self.kernel_optimizer),
      bias: if self.use_bias { Some(Parameter::placeholder(self.bias_initializer, self.bias_optimizer)) } else { None }
    }
  }
}

#[derive(Debug)]
pub struct Dense<T, Ki: Initializer<T>, Ko: Optimizer<T, nd::Ix2>, Bi: Initializer<T>, Bo: Optimizer<T, nd::Ix1>> {
  units: usize,
  kernel: Parameter2<T, Ki, Ko>,
  bias: Option<Parameter1<T, Bi, Bo>>
} 

impl<T, Ki: Initializer<T>, Ko: Optimizer<T, nd::Ix2>, Bi: Initializer<T>, Bo: Optimizer<T, nd::Ix1>, X: functional::Dense<T>>
  Forward<X> for Dense<T, Ki, Ko, Bi, Bo> {
  fn forward(&self, input: &X) -> X {
    input.dense(self.kernel.param(), self.bias.as_ref().map(|bias| bias.param()))
  }
}

impl<T: Copy + num_traits::Zero, Ki: Initializer<T>, Ko: Optimizer<T, nd::Ix2>, Bi: Initializer<T>, Bo: Optimizer<T, nd::Ix1>> Layer<T> for Dense<T, Ki, Ko, Bi, Bo>
  where Self: Forward<nd::ArrayD<T>> + Forward<Var<T>> {
  fn build(&mut self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    use nd::ShapeBuilder;
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1..].iter().product();
    if self.kernel.param().value().shape() != &[self.units, in_channels] {
      self.kernel.initialize([self.units, in_channels].f());
    }
    if let Some(ref mut bias) = self.bias {
      if bias.param().value().shape() != &[self.units] {
        bias.initialize(self.units);
      }
    }
    self.forward(input)
  }
  fn train(&mut self) {
    self.kernel.train();
    self.bias.as_mut()
      .map(|bias| bias.train()); 
  }
  fn eval(&mut self) {
    self.kernel.eval();
    self.bias.as_mut()
      .map(|bias| bias.eval());
  }
  fn step(&mut self, lr: T) {
    self.kernel.step(lr);
    self.bias.as_mut()
      .map(|bias| bias.step(lr)); 
  }
}



