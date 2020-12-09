pub mod backend;
pub mod error;
pub mod tensor;

pub use ndarray;

pub type Result<T, E = error::Error> = std::result::Result<T, E>;
