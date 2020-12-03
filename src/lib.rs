pub type Result<T, E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

pub use ndarray;
pub mod backend;
pub mod tensor;
