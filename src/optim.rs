use std::{any::Any, fmt::Debug};
use ndarray as nd;

pub trait Optimizer<T> {
  fn step<'a>(&mut self, value: nd::ArrayViewMutD<'a, T>, grad: Option<nd::ArrayViewD<T>>, payload: &mut Option<Box<dyn OptimizerPayload<T>>>);
} 

pub trait OptimizerPayload<T>: Any + Debug {}

pub struct LearningRate<T>(pub T);

impl<T: num_traits::Float + num_traits::NumAssign> Optimizer<T> for LearningRate<T> {
  fn step<'a>(&mut self, mut value: nd::ArrayViewMutD<'a, T>, grad: Option<nd::ArrayViewD<T>>, payload: &mut Option<Box<dyn OptimizerPayload<T>>>) {
    *payload = None;
    if let Some(ref grad) = grad {
      value.iter_mut()
        .zip(grad.iter())
        .for_each(|(x, &dx)| *x -= self.0 * dx);
    }
  }
} 

