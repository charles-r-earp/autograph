use std::{any::Any, fmt::Debug, ops::{Deref, Index}};
pub use ndarray as nd;

pub mod iter;
pub mod datasets;

pub trait Initializer<T, D>: Debug {
  fn initialize(&mut self, dim: D, data: &mut [T]);
}

#[derive(Debug)]
pub struct Zeros;

impl<T: num_traits::Zero, D: nd::Dimension> Initializer<T, D> for Zeros {
  fn initialize(&mut self, _: D, data: &mut [T]) {
    data.iter_mut()
      .for_each(|x| *x = T::zero());
  }
}

#[derive(Debug)]
pub struct RandomNormal<T: rand_distr::Float + Debug> {
  pub mean: T,
  pub std_dev: T
}

impl<T: rand_distr::Float + Debug, D> Initializer<T, D> for RandomNormal<T>
  where rand_distr::Normal<T>: rand_distr::Distribution<T>,
        rand_distr::StandardNormal: rand_distr::Distribution<T> {
  fn initialize(&mut self, _: D, data: &mut [T]) {
    use rand_distr::Distribution;
    let normal = rand_distr::Normal::new(self.mean, self.std_dev).unwrap();
    data.iter_mut()
      .zip(normal.sample_iter(&mut rand::thread_rng()))
      .for_each(|(x, n)| *x = n);
  }
}

#[derive(Debug)]
pub struct HeNormal;

impl<T: num_traits::Float + Debug, D: Index<usize, Output=usize>> Initializer<T, D> for HeNormal
  where T: rand_distr::Float, 
        RandomNormal<T>: Initializer<T, D> {
  fn initialize(&mut self, dim: D, data: &mut [T]) {
    use num_traits::{Float, NumCast};
    let units: T = <T as NumCast>::from(dim[1]).unwrap();
    let two: T = <T as NumCast>::from(2.).unwrap();
    let std_dev = Float::sqrt(two / units);
    RandomNormal{mean: T::zero(), std_dev}.initialize(dim, data);
  }
}

pub trait Optimizer<T> {
  fn step<'a>(&mut self, value: nd::ArrayViewMutD<'a, T>, grad: Option<nd::ArrayViewD<T>>, payload: &mut Option<Box<OptimizerPayload<T>>>);
} 

pub trait OptimizerPayload<T>: Any + Debug {}

pub struct LearningRate<T>(pub T);

impl<T: num_traits::Float + num_traits::NumAssign> Optimizer<T> for LearningRate<T> {
  fn step<'a>(&mut self, mut value: nd::ArrayViewMutD<'a, T>, grad: Option<nd::ArrayViewD<T>>, payload: &mut Option<Box<OptimizerPayload<T>>>) {
    *payload = None;
    if let Some(ref grad) = grad {
      value.iter_mut()
        .zip(grad.iter())
        .for_each(|(x, &dx)| *x -= self.0 * dx);
    }
  }
} 

#[derive(Debug)]
pub struct Param<T, D: nd::Dimension> {
  value: nd::Array<T, D>,
  grad: Option<nd::Array<T, D>>,
  initializer: Box<Initializer<T, D>>,
  payload: Option<Box<OptimizerPayload<T>>>,
}

impl<T, D: nd::Dimension> Param<T, D> {
  pub fn placeholder(shape: impl nd::ShapeBuilder<Dim=D>, initializer: Box<Initializer<T, D>>) -> Self
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
  payload: &'a mut Option<Box<OptimizerPayload<T>>>
}

impl<'a, T> ParamViewMut<'a, T> {
  pub fn step(&mut self, optimizer: &mut impl Optimizer<T>) {
    optimizer.step(self.value.view_mut(), self.grad.as_ref().map(|grad| grad.view()), &mut self.payload);
  }
}

pub trait Layer<T> {
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
  kernel_initializer: Box<Initializer<T, nd::Ix2>>,
  bias_initializer: Box<Initializer<T, nd::Ix1>>,
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

pub fn cross_entropy_loss<T, U, D>(pred: &nd::Array<T, D>, target: &nd::Array1<U>) -> (T, nd::Array2<T>)
  where T: num_traits::Float + num_traits::NumAssign + std::iter::Sum + iter::Mean,
        U: num_traits::Unsigned + num_traits::ToPrimitive + Copy,
        D: nd::Dimension {
  use iter::MeanExt;
  let pred = pred.view()
    .into_dimensionality::<nd::Ix2>()
    .unwrap();
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

pub fn correct<T, U, D>(pred: &nd::Array<T, D>, target: &nd::Array1<U>) -> usize
  where T: num_traits::Float,
        U: num_traits::Unsigned + num_traits::NumCast + Copy,
        D: nd::Dimension {
  let pred = pred.view()
    .into_dimensionality::<nd::Ix2>()
    .unwrap();
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
