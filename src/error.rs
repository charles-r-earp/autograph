use super::Graph;
use std::{result::Result, error::Error, fmt::{self, Debug, Display}, sync::{Arc}};

#[derive(Debug)]
pub enum AutographError {
  
}

impl Display for AutographError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "AutographError")
  }
}

impl Error for AutographError {
  fn description(&self) -> &str {
    "AutographError"
  }
}

