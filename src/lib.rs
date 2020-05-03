#![allow(warnings)]
#![recursion_limit="256"]
use std::{result::Result};

pub mod error;
use error::AutographError;
type AutographResult<T> = Result<T, AutographError>;

pub mod backend;

mod private {
  use num_traits::Zero;
  
  #[cfg(feature="cuda")]
  use rustacuda::memory::DeviceCopy;
  
  #[cfg(not(feature="cuda"))]
  pub trait DeviceCopy {}
  
  #[cfg(not(feature="cuda"))]
  impl<T> DeviceCopy for T {}
  
  pub trait PrivateElement: 'static + Copy + Zero + DeviceCopy {}
  
  impl PrivateElement for u8 {}
  impl PrivateElement for f32 {}
}

pub trait Element: private::PrivateElement {}

impl<T> Element for T where T: private::PrivateElement {}







