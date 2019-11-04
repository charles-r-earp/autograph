use ndarray as nd;
use rand_distr::{Distribution, StandardNormal, Normal};

pub trait Initializer<T> {
  fn fill<D: nd::Dimension>(&self, array: &mut nd::ArrayViewMut<T, D>);
}

#[derive(Debug)]
pub struct Zeros;

impl<T: num_traits::Zero> Initializer<T> for Zeros {
  fn fill<D: nd::Dimension>(&self, array: &mut nd::ArrayViewMut<T, D>) {
    array.as_slice_memory_order_mut()
      .unwrap()
      .iter_mut()
      .for_each(|x| *x = T::zero());
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

impl<T, R: Distribution<T> + Copy> Initializer<T> for Random<R> {
  fn fill<D: nd::Dimension>(&self, array: &mut nd::ArrayViewMut<T, D>) {
    array.as_slice_memory_order_mut()
      .unwrap()
      .iter_mut()
      .zip(self.distr.sample_iter(&mut rand::thread_rng()))
      .for_each(|(x, r)| *x = r);
  }
}

#[derive(Debug)]
pub struct HeNormal;

impl<T: num_traits::Float> Initializer<T> for HeNormal
  where T: rand_distr::Float,
        StandardNormal: Distribution<T>,
        Normal<T>: Distribution<T> {
  fn fill<D: nd::Dimension>(&self, array: &mut nd::ArrayViewMut<T, D>) {
    let units = <T as num_traits::NumCast>::from(array.shape()[1]).unwrap();
    let two = <T as num_traits::NumCast>::from(2.).unwrap();
    let std_dev = num_traits::Float::sqrt(two / units);
    Random::new(Normal::new(T::zero(), std_dev).unwrap())
      .fill(array)
  }
}
