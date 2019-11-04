use super::{Var, Param, functional, init, init::Initializer, optim, optim::Optimizer}; 
use std::{ops::Deref, marker::PhantomData};
use ndarray as nd;
/*
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
pub type Parameter2<T, I, O> = Parameter<T, nd::Ix2, I, O>;*/

pub trait Forward<X> {
  fn forward(&self, input: &X) -> X;
}

pub trait Layer<T>: Forward<nd::ArrayD<T>> + Forward<Var<T>> {
  fn build(&mut self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    self.forward(input)
  }
  fn param_iter(&self) -> std::vec::IntoIter<&Param<T>> { Vec::new().into_iter() }
  fn param_iter_mut(&mut self) -> std::vec::IntoIter<&mut Param<T>> { Vec::new().into_iter() }
}

#[derive(Default)]
pub struct DenseBuilder<T> {
  units: Option<usize>,
  use_bias: bool,
  kernel_initializer: Option<Box<dyn Initializer<T>>>,
  bias_initializer: Option<Box<dyn Initializer<T>>>
} 
  

impl<T> DenseBuilder<T> {
  pub fn units(mut self, units: usize) -> Self {
    self.units.replace(units);
    self
  }
  pub fn use_bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn kernel_initializer(mut self, kernel_initializer: impl Initializer<T> + 'static) -> Self {
    self.kernel_initializer.replace(Box::new(kernel_initializer));
    self
  }   
  pub fn bias_initializer(mut self, bias_initializer: impl Initializer<T> + 'static) -> Self {
    self.bias_initializer.replace(Box::new(bias_initializer));
    self
  }   
  pub fn build(self) -> Dense<T>
    where T: Default,
          init::HeNormal: Initializer<T> {
    let mut kernel = Param::default();
    kernel.initializer.replace(self.kernel_initializer.unwrap_or(Box::new(init::HeNormal)));
    let bias = if self.use_bias {
      let mut bias = Param::default();
      bias.initializer = self.bias_initializer;
      Some(bias)
    }
    else { None };
    Dense{
      units: self.units.unwrap(),
      kernel,
      bias
    }
  }
}

#[derive(Debug)]
pub struct Dense<T> {
  units: usize,
  kernel: Param<T>,
  bias: Option<Param<T>>
} 

impl<T: Default> Dense<T> {
  pub fn builder() -> DenseBuilder<T> { Default::default() }
} 

impl<T, X: functional::Dense<T>>
  Forward<X> for Dense<T> {
  fn forward(&self, input: &X) -> X {
    input.dense(&self.kernel, self.bias.as_ref())
  }
}

impl<T: 'static + num_traits::Float> Layer<T> for Dense<T>
  where Self: Forward<nd::ArrayD<T>> + Forward<Var<T>> {
  fn build(&mut self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    use nd::{IntoDimension, ShapeBuilder};
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1..].iter().product();
    if self.kernel.value().shape() != &[self.units, in_channels] {
      self.kernel.initialize([self.units, in_channels].as_ref().into_dimension().f());
    }
    if let Some(ref mut bias) = self.bias {
      if bias.value().shape() != &[self.units] {
        bias.initialize([self.units].as_ref().into_dimension());
      }
    }
    self.forward(input)
  }
  fn param_iter(&self) -> std::vec::IntoIter<&Param<T>> { 
    let params = if let Some(ref bias) = self.bias {
      vec![&self.kernel, bias]
    }
    else {
      vec![&self.kernel]
    };
    params.into_iter()
  }
  fn param_iter_mut(&mut self) -> std::vec::IntoIter<&mut Param<T>> { 
    let params = if let Some(ref mut bias) = self.bias {
      vec![&mut self.kernel, bias]
    }
    else {
      vec![&mut self.kernel]
    };
    params.into_iter()
  }
}



