use super::{AutographResult, Element, Tensor, Activation};
use rand::{Rng, rngs::SmallRng, SeedableRng}; 
use rand_distr::{Distribution, Normal};
use std::{thread_local, cell::UnsafeCell};

thread_local! {
  static THREAD_RNG: UnsafeCell<Option<SmallRng>> = UnsafeCell::new(None);
}

pub fn thread_rng() -> &'static mut SmallRng { 
  THREAD_RNG.with(|t| {
    let mut rng = unsafe { &mut *t.get() };
    if rng.is_none() {
      rng.replace(SmallRng::seed_from_u64(0));
    }
    rng.as_mut().unwrap()
  })
} 

pub fn random<T: Element, D: Distribution<T>, R: Rng>(tensor: &Tensor<T>, dist: D, rng: &mut R) -> AutographResult<()> {
  let vec: Vec<T> = dist.sample_iter(rng)
    .take(tensor.len())
    .collect();
  tensor.write(&vec)?;
  Ok(())
}

pub fn normal<R: Rng>(tensor: &Tensor<f32>, mean: f32, std_dev: f32, rng: &mut R) -> AutographResult<()> {
  random(tensor, Normal::new(mean, std_dev).unwrap(), rng)?;
  Ok(())
}

pub fn calculate_gain(act: Activation) -> f32 {
  match act {
    Activation::Relu => std::f32::consts::SQRT_2
  }
}
