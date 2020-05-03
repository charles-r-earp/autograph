use std::{error::Error, fmt::{self, Debug, Display}};
#[cfg(feature="cuda")]
use rustacuda::error::CudaError;

#[derive(Debug)]
pub enum AutographError {
  Uninitialized,
  CudaUnavailable,
  #[cfg(feature="cuda")]
  CudaError(CudaError),
}


impl Display for AutographError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{:?}", self)
  }
}

impl Error for AutographError {
  fn description(&self) -> &str {
    "AutographError"
  }
}



