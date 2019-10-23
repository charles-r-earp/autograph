use std::{iter::Iterator};

pub trait Mean<T=Self>: Sized {
  fn mean<I: Iterator<Item=T>>(iter: I) -> Self;
}

macro_rules! impl_float_mean {
  ($($t:ty)*) => ($(
    impl Mean for $t {
      fn mean<I: Iterator<Item=Self>>(iter: I) -> Self {
        let (acc, n) = iter.fold((0., 0), |(acc, n), x| { 
          (acc + x, n + 1) 
        }); 
        acc / <$t as num_traits::NumCast>::from(n).unwrap()
      }
    }
  )*)
}

impl_float_mean!(f32 f64);

pub trait MeanExt: Iterator + Sized {
  fn mean<T>(self) -> T
    where T: Mean<Self::Item> {
    T::mean(self)
  }
}

impl<I: Iterator> MeanExt for I {}



#[cfg(test)]
mod tests {
  use super::*;
  
  #[test]
  fn test_mean() {
    assert_eq!(vec![1., 2., 3.].into_iter().mean::<f32>(), 2.);
  }
}
