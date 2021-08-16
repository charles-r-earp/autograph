use super::autograd::ParameterD;
use crate::result::Result;

/// A trait for optimizers.
///
/// The optimizer updates the parameters of the model based on their gradients.
pub trait Optimizer {
    /// Updates a set of parameters.
    ///
    /// The optimizer may assume that `parameters` is the same set in the same order on each call to [`.update()`](Self::update()). Typically this is from [`Layer::parameters_mut()`](super::layer::Layer::parameters_mut()).
    ///
    /// # Note
    /// When training a layer, don't call [`.update()`](Self::update()) directly, use [`Layer::update()`](super::layer::Layer::update()) instead.
    ///
    /// **Errors**
    ///
    /// Returns an error if the operation could not be performed. Some parameters may be modified even when returning an error.
    fn update(&mut self, parameters: &mut [&mut ParameterD]) -> Result<()>;
}
