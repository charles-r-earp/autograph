use std::fmt::Debug;
use ndarray as nd;

pub trait Optimizer<T, D> {
  fn step(&mut self, value: &mut nd::ArrayViewMut<T, D>, grad: &nd::Array<T, D>, lr: T);
}

impl<T: nd::LinalgScalar + num_traits::Float, D: nd::Dimension> Optimizer<T, D> for () {
  fn step(&mut self, value: &mut nd::ArrayViewMut<T, D>, grad: &nd::Array<T, D>, lr: T) {
    value.scaled_add(-lr, grad);
  }
}
