use std::{fmt::Debug, ops::Index}; 

pub trait Initializer<T, D>: Debug {
  fn initialize(&mut self, dim: D, data: &mut [T]);
}

#[derive(Debug)]
pub struct Zeros;

impl<T: num_traits::Zero, D> Initializer<T, D> for Zeros {
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

