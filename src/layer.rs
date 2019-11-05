use super::{autograd::{Var, Param}, functional, init, init::Initializer}; 
use std::{fmt::Debug, ops::{Index, IndexMut}};
use ndarray as nd;

pub trait Forward<X> {
  fn forward(&self, input: &X) -> X;
}

pub trait Layer<T>: Forward<nd::ArrayD<T>> + Forward<Var<T>> + Debug {
  fn build(&mut self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    self.forward(input)
  }
  fn param_iter(&self) -> std::vec::IntoIter<&Param<T>> { Vec::new().into_iter() }
  fn param_iter_mut(&mut self) -> std::vec::IntoIter<&mut Param<T>> { Vec::new().into_iter() }
}

#[derive(Default, Debug)]
pub struct Sequential<T> {
  layers: Vec<Box<dyn Layer<T>>>
}

impl<T> Sequential<T> {
  pub fn len(&self) -> usize { self.layers.len() }
  pub fn push(&mut self, layer: impl Layer<T> + 'static) { self.layers.push(Box::new(layer)); }
  pub fn insert(&mut self, index: usize, layer: impl Layer<T> + 'static) {
    self.layers.insert(index, Box::new(layer));
  }
  pub fn remove(&mut self, index: usize) -> Box<dyn Layer<T>> { self.layers.remove(index) }
}

impl<T> Index<usize> for Sequential<T> {
  type Output = dyn Layer<T>;
  fn index(&self, index: usize) -> &(dyn Layer<T> + 'static) { &*self.layers[index] }
}

impl<T> IndexMut<usize> for Sequential<T> {
  fn index_mut(&mut self, index: usize) -> &mut (dyn Layer<T> + 'static) { &mut *self.layers[index] }
}

impl<T: Clone> Forward<nd::ArrayD<T>> for Sequential<T> {
  fn forward(&self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    let mut layer_iter = self.layers.iter();
    if let Some(first) = layer_iter.next() {
      layer_iter.fold(first.forward(input), |x, layer| layer.forward(&x))
    }
    else {
      input.clone()
    }  
  }
}

impl<T: Clone> Forward<Var<T>> for Sequential<T> {
  fn forward(&self, input: &Var<T>) -> Var<T> {
    let mut layer_iter = self.layers.iter();
    if let Some(first) = layer_iter.next() {
      layer_iter.fold(first.forward(input), |x, layer| layer.forward(&x))
    }
    else {
      input.clone()
    }  
  }
}

impl<T: Clone + Debug> Layer<T> for Sequential<T> {
  fn build(&mut self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    let mut layer_iter = self.layers.iter_mut();
    if let Some(first) = layer_iter.next() {
      layer_iter.fold(first.build(input), |x, layer| layer.build(&x))
    }
    else {
      input.clone()
    }  
  }
  fn param_iter(&self) -> std::vec::IntoIter<&Param<T>> {
    self.layers.iter()
      .flat_map(|layer| layer.param_iter())
      .collect::<Vec<_>>()
      .into_iter()
  }
  fn param_iter_mut(&mut self) -> std::vec::IntoIter<&mut Param<T>> {
    self.layers.iter_mut()
      .flat_map(|layer| layer.param_iter_mut())
      .collect::<Vec<_>>()
      .into_iter()
  }
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
          init::HeNormal: Initializer<T>,
          init::Zeros: Initializer<T> {
    let mut kernel = Param::default();
    kernel.initializer.replace(self.kernel_initializer.unwrap_or(Box::new(init::HeNormal)));
    let bias = if self.use_bias {
      let mut bias = Param::default();
      bias.initializer.replace(self.bias_initializer.unwrap_or(Box::new(init::Zeros)));
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

impl<T: 'static + num_traits::Float + Debug> Layer<T> for Dense<T>
  where Self: Forward<nd::ArrayD<T>> + Forward<Var<T>> {
  fn build(&mut self, input: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    use nd::{IntoDimension, ShapeBuilder};
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

#[derive(Default, Debug, Clone, Copy)]
pub struct Relu; 

impl<X: functional::Relu<Output=X>> Forward<X> for Relu {
  fn forward(&self, x: &X) -> X { x.relu() }
}

impl<T: Debug> Layer<T> for Relu where Self: Forward<nd::ArrayD<T>> + Forward<Var<T>> {}



