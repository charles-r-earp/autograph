use super::autograd::ParameterD;
use crate::{ops::ScaledAdd, result::Result, tensor::float::FloatTensorD};
use anyhow::bail;

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

/// Sgd optimizer.
///
/// Defaults:
/// - learning_rate: 0.001
/// - momentum: 0.
#[derive(Debug)]
pub struct Sgd {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<FloatTensorD>,
}

impl Default for Sgd {
    fn default() -> Self {
        Self::new(0.001).unwrap()
    }
}

impl Sgd {
    /// Creates a new [`Sgd`] with `learning_rate`.
    ///
    /// Defaults:
    /// - momentum: 0.
    ///
    /// **Errors**
    ///
    /// If `learning_rate` is not between 0 and 1.
    pub fn new(learning_rate: f32) -> Result<Self> {
        if !(0f32..1f32).contains(&learning_rate) {
            bail!("Learning rate must be between 0 and 1!");
        }
        Ok(Self {
            learning_rate,
            momentum: 0.,
            velocities: Vec::new(),
        })
    }
}

impl Optimizer for Sgd {
    fn update(&mut self, parameters: &mut [&mut ParameterD]) -> Result<()> {
        for parameter in parameters {
            if let Some(grad) = parameter.take_grad() {
                /*{
                    let _value = smol::block_on(parameter.value().cast_to::<f32>()?.read())?;
                    dbg!(_value.as_array());
                    let _grad = smol::block_on(grad.cast_to::<f32>()?.read())?;
                    dbg!(_grad.as_array());
                }*/
                parameter
                    .value_mut()
                    .make_mut()?
                    .scaled_add(-self.learning_rate, &grad)?;
            }
        }
        Ok(())
    }
}
