use std::{ops::Deref, fmt::Debug};
use ndarray as nd;

pub mod init;
use init::{Initializer, Zeros, HeNormal};
pub mod optim;
use optim::{Optimizer, OptimizerPayload};
pub mod iter;
pub mod datasets;


#[derive(Debug)]
pub struct Param<T, D: nd::Dimension> {
  value: nd::Array<T, D>,
  grad: Option<nd::Array<T, D>>,
  initializer: Box<dyn Initializer<T, D>>,
  payload: Option<Box<dyn OptimizerPayload<T>>>,
}

impl<T, D: nd::Dimension> Param<T, D> {
  pub fn placeholder(shape: impl nd::ShapeBuilder<Dim=D>, initializer: Box<dyn Initializer<T, D>>) -> Self
    where T: Default {
    Self{
      value: nd::Array::default(shape),
      grad: None,
      initializer,
      payload: None
    }
  }
  pub fn initialize(&mut self, shape: impl nd::ShapeBuilder<Dim=D>)
    where T: Copy {
    self.value = unsafe { nd::Array::uninitialized(shape) };
    self.initializer.initialize(self.value.raw_dim(), self.value.as_slice_memory_order_mut().unwrap());
    self.grad = None;
    self.payload = None;
  }
  pub fn add_grad(&mut self, grad: nd::Array<T, D>) {
    self.grad.replace(grad);
  }
  pub fn view_mut(&mut self) -> ParamViewMut<T> {
    ParamViewMut{
      value: self.value.view_mut().into_dyn(), 
      grad: self.grad.take().map(|grad| grad.into_dyn()), 
      payload: &mut self.payload
    }
  }
}

impl<T, D: nd::Dimension> Deref for Param<T, D> {
  type Target = nd::Array<T, D>;
  fn deref(&self) -> &Self::Target {
    &self.value
  }
}

pub struct ParamViewMut<'a, T> {
  value: nd::ArrayViewMutD<'a, T>,
  grad: Option<nd::ArrayD<T>>,
  payload: &'a mut Option<Box<dyn OptimizerPayload<T>>>
}

impl<'a, T> ParamViewMut<'a, T> {
  pub fn step(&mut self, optimizer: &mut impl Optimizer<T>) {
    optimizer.step(self.value.view_mut(), self.grad.as_ref().map(|grad| grad.view()), &mut self.payload);
  }
}

pub trait Layer<T>: Debug {
  fn forward(&mut self, input: nd::ArrayD<T>) -> nd::ArrayD<T>;
  fn backward(&mut self, grad: &nd::ArrayD<T>) -> nd::ArrayD<T>;
  fn params_mut(&mut self) -> Vec<ParamViewMut<T>> { Vec::new() }
}

#[derive(Debug)]
pub struct Dense<T> {
  kernel: Param<T, nd::Ix2>, 
  bias: Option<Param<T, nd::Ix1>>,
  saved_input: Option<nd::Array2<T>>,
}

impl<T> Dense<T> {
  pub fn builder() -> DenseBuilder<T>
    where HeNormal: Initializer<T, nd::Ix2>,
          Zeros: Initializer<T, nd::Ix1> {
    DenseBuilder{
      units: 0,
      kernel_initializer: Box::new(HeNormal),
      use_bias: false,
      bias_initializer: Box::new(Zeros)
    }
  }
}

#[derive(Debug)]
pub struct DenseBuilder<T> {
  units: usize,
  use_bias: bool,
  kernel_initializer: Box<dyn Initializer<T, nd::Ix2>>,
  bias_initializer: Box<dyn Initializer<T, nd::Ix1>>,
}

impl<T> DenseBuilder<T> {
  pub fn units(mut self, units: usize) -> Self {
    self.units = units;
    self
  }
  pub fn use_bias(mut self) -> Self {
    self.use_bias = true;
    self
  }
  pub fn kernel_initializer(mut self, kernel_initializer: impl Initializer<T, nd::Ix2> + 'static) -> Self {
    self.kernel_initializer = Box::new(kernel_initializer);
    self
  }
  pub fn bias_initializer(mut self, bias_initializer: impl Initializer<T, nd::Ix1> + 'static) -> Self {
    self.bias_initializer = Box::new(bias_initializer);
    self
  }
  pub fn build(self) -> Dense<T>
    where T: Default {
    use nd::ShapeBuilder;
    assert!(self.units != 0);
    Dense{
      kernel: Param::placeholder([0, self.units].f(), self.kernel_initializer),
      bias: if self.use_bias { Some(Param::placeholder([0], self.bias_initializer)) } else { None },
      saved_input: None
    }
  }
}

impl<T: nd::LinalgScalar + num_traits::NumAssign + num_traits::NumCast + Debug + Default> Layer<T> for Dense<T> {
  fn forward(&mut self, input: nd::ArrayD<T>) -> nd::ArrayD<T> {
    use nd::{ShapeBuilder};
    let input = input.into_dimensionality::<nd::Ix2>()
      .unwrap();
    if self.kernel.shape()[0] != input.shape()[1] {
      self.kernel.initialize([input.shape()[1], self.kernel.shape()[1]].f());
    }
    let mut y = input.dot(&*self.kernel);
    if let Some(ref mut bias) = self.bias {
      if bias.shape()[0] != y.shape()[1] {
        bias.initialize([y.shape()[1]]);
      }
      y.as_slice_mut()
        .unwrap()
        .chunks_exact_mut(bias.shape()[0])
        .for_each(|y| 
          y.iter_mut()
            .zip(bias.iter())
            .for_each(|(y, &b)| *y += b));
    }
    self.saved_input.replace(input);
    y.into_dyn()
  }
  fn backward(&mut self, grad: &nd::ArrayD<T>) -> nd::ArrayD<T> {
    let grad = grad.view()
      .into_dimensionality::<nd::Ix2>()
      .unwrap();
    let input = self.saved_input.take()
      .unwrap();
    self.kernel.add_grad(input.t().dot(&grad));
    if let Some(ref mut bias) = self.bias {
      bias.add_grad(nd::Array1::ones(grad.shape()[0]).dot(&grad));
    }
    grad.dot(&self.kernel.t())
      .into_dyn()
  }
  fn params_mut(&mut self) -> Vec<ParamViewMut<T>> {
    if let Some(ref mut bias) = self.bias {
      vec![self.kernel.view_mut(), bias.view_mut()]
    }
    else {
      vec![self.kernel.view_mut()]
    }
  } 
}

pub fn cross_entropy_loss<T, U>(pred: &nd::Array2<T>, target: &nd::Array1<U>) -> (T, nd::Array2<T>)
  where T: num_traits::Float + num_traits::NumAssign + std::iter::Sum + iter::Mean,
        U: num_traits::Unsigned + num_traits::ToPrimitive + Copy {
  use iter::MeanExt;
  let classes = pred.shape()[1];
  let mut grad = unsafe { nd::Array2::uninitialized(pred.raw_dim()) };
  let loss = grad.as_slice_mut()
    .unwrap()
    .chunks_exact_mut(classes)
    .zip(pred.as_slice().unwrap().chunks_exact(classes))
    .zip(target.iter())
    .map(|((dy, y), &t)| {
      let t = t.to_usize().unwrap();
      let sum: T = y.iter()
        .map(|&y| y.exp())
        .sum();
      dy.iter_mut()
        .zip(y.iter()
          .map(|y| y.exp().div(sum))
          .enumerate()
          .map(|(u, y)| if u == t { y - T::one() } else { y }))
          .for_each(|(dy, y)| *dy = y);
      y[t].exp()
        .div(sum)
        .ln()
        .neg()
    }).mean();
  (loss, grad)
} 

pub fn correct<T, U>(pred: &nd::Array2<T>, target: &nd::Array1<U>) -> usize
  where T: num_traits::Float,
        U: num_traits::Unsigned + num_traits::NumCast + Copy {
  let classes = pred.shape()[1];
  pred.as_slice()
    .unwrap()
    .chunks_exact(classes)
    .zip(target.iter())
    .filter(|(p, &t)| !p.iter().any(|&x| x > p[t.to_usize().unwrap()]))
    .count()
}

#[cfg(test)]
mod tests {
  use ndarray as nd;
  #[test]
  fn test_correct() {
    let pred = nd::arr2(&[[1., 2.]]);
    let target = nd::arr1(&[1u8]);
    assert_eq!(super::correct(&pred, &target), 1);
  }
}
