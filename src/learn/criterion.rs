use crate::result::Result;
use serde::{Deserialize, Serialize};

/// A criterion for training.
pub trait Criterion<Y, T> {
    /// The output type.
    type Output;
    /// Evaluates the criterion for the `prediction` and `target`.
    ///
    /// **Errors**
    ///
    /// Returns an error if the arguments are invalid or the operation cannot be performed.
    fn eval(&self, prediction: Y, target: T) -> Result<Self::Output>;
}

/// Cross entropy loss.
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct CrossEntropyLoss {}
