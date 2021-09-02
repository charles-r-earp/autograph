use crate::result::Result;

/// A criterion for training.
pub trait Criterion<Y, T> {
    /// The output type.
    type Output;
    /// Computes the loss.
    fn loss(&self, prediction: Y, target: T) -> Result<Self::Output>;
}

/// Cross entropy loss.
#[derive(Default, Debug)]
pub struct CrossEntropyLoss {}
