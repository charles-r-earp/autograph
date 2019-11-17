use std::{fmt::Debug, marker::{Send, Sync}};
use ndarray as nd;
use rand_distr::{Distribution, StandardNormal, Normal, Uniform};

pub trait Initializer<T>: Debug + Send + Sync {
  fn fill(&self, array: &mut nd::ArrayViewMutD<T>);
}

#[derive(Debug)]
pub struct Zeros;

impl<T: num_traits::Zero + Clone + Send + Sync> Initializer<T> for Zeros {
  fn fill(&self, array: &mut nd::ArrayViewMutD<T>) {
    array.fill(T::zero());
  }
}

#[derive(Debug)]
pub struct Random<R> {
  distr: R
}

impl<R> Random<R> {
  pub fn new(distr: R) -> Self {
    Self{distr}
  }
} 

impl<T: Send + Sync, R: Distribution<T> + Copy + Debug + Send + Sync> Initializer<T> for Random<R> {
  fn fill(&self, array: &mut nd::ArrayViewMutD<T>) {
    array.as_slice_memory_order_mut()
      .unwrap()
      .iter_mut()
      .zip(self.distr.sample_iter(&mut rand::thread_rng()))
      .for_each(|(x, r)| *x = r);
  }
}

#[derive(Debug)]
pub struct HeNormal;

impl<T: num_traits::Float + Debug + Send + Sync> Initializer<T> for HeNormal
  where T: rand_distr::Float,
        StandardNormal: Distribution<T>,
        Normal<T>: Distribution<T> {
  fn fill(&self, array: &mut nd::ArrayViewMutD<T>) {
    let units = <T as num_traits::NumCast>::from(array.shape()[1]).unwrap();
    let two = <T as num_traits::NumCast>::from(2.).unwrap();
    let std_dev = num_traits::Float::sqrt(two / units);
    Random::new(Normal::new(T::zero(), std_dev).unwrap())
      .fill(array)
  }
}

#[derive(Debug)]
pub struct XavierUniform;

impl<T: num_traits::Float + Debug + Send + Sync> Initializer<T> for XavierUniform
  where T: rand_distr::Float + rand_distr::uniform::SampleUniform,
        Uniform<T>: Distribution<T> + Copy + Debug + Send + Sync {
  fn fill(&self, array: &mut nd::ArrayViewMutD<T>) {
    let units = <T as num_traits::NumCast>::from(array.shape()[0] * array.shape()[1]).unwrap();
    let six = <T as num_traits::NumCast>::from(6.).unwrap();
    let limit = num_traits::Float::sqrt(six / units);
    Random::new(Uniform::new(-limit, limit))
      .fill(array)
  }
}
