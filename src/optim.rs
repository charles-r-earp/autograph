use std::{fmt::Debug, marker::{Send, Sync}};
use ndarray as nd;

pub trait Optimizer<T>: Debug + Send + Sync {
  fn step(&mut self, value: &mut nd::ArrayViewMutD<T>, grad: &nd::ArrayViewD<T>, lr: T);
}

#[derive(Default, Clone, Copy)]
pub struct SGDBuilder<T> {
  momentum: Option<T>,
  dampening: Option<T>,
  weight_decay: Option<T>,
  nesterov: bool
}

impl<T> SGDBuilder<T> {
  pub fn momentum(mut self, momentum: T) -> Self {
    self.momentum.replace(momentum);
    self
  }
  pub fn dampening(mut self, dampening: T) -> Self {
    self.dampening.replace(dampening);
    self
  }
  pub fn weight_decay(mut self, weight_decay: T) -> Self {
    self.weight_decay.replace(weight_decay);
    self
  }
  pub fn nesterov(mut self) -> Self {
    self.nesterov = true;
    self
  }
  pub fn build(self) -> SGD<T>
    where T: Default + num_traits::Zero {
    SGD{
      velocity: Default::default(),
      momentum: self.momentum.unwrap_or(T::zero()),
      dampening: self.dampening.unwrap_or(T::zero()),
      weight_decay: self.weight_decay.unwrap_or(T::zero()),
      nesterov: self.nesterov
    }
  }
}

#[derive(Default, Clone, Debug)]
pub struct SGD<T> {
  velocity: nd::ArrayD<T>,
  momentum: T,
  dampening: T,
  weight_decay: T,
  nesterov: bool
}

impl<T: Default> SGD<T> {
  pub fn builder() -> SGDBuilder<T> { Default::default() }
}

impl<T: nd::LinalgScalar + num_traits::Float + num_traits::NumAssign + Debug + Send + Sync> Optimizer<T> for SGD<T> {
  fn step(&mut self, value: &mut nd::ArrayViewMutD<T>, grad: &nd::ArrayViewD<T>, lr: T) {
   if self.velocity.raw_dim() != value.raw_dim() {
      self.velocity = nd::Array::zeros(value.raw_dim());
    }
    let momentum = self.momentum;
    let dampening = self.dampening; 
    let weight_decay = self.weight_decay;
    self.velocity.zip_mut_with(&grad, |v, &dx| {
      *v = momentum * *v + lr * (T::one() - dampening) * (dx - weight_decay);
    });
    if self.nesterov {
      value.scaled_add(-momentum, &self.velocity);
    }
    else {
      value.scaled_add(-T::one(), &self.velocity);
    }
  }
}
