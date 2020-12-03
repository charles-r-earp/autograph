pub type Result<T, E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

pub mod backend;
