use crate::{Buffer, Tensor, TensorD};
use super::autograd::{Parameter, ParameterD, OptimizerDataEntry};
use std::{
    hash::{Hash, Hasher},
    collections::HashMap,
    sync::{Arc, Weak, RwLock}
};
use ndarray::Dimension;
use serde::{Serialize, Deserialize};
use std::fmt::Debug;

pub mod builders;
use builders::SgdBuilder;

/// Optimizers update the parameters based on their gradients\
/// 
/// They may also store per parameter state like velocity (Sgd) or learning_rates (Adam) 
///```
///optim.step(model.parameters());
pub trait Optimizer {
    fn step(&mut self, parameters: impl IntoIterator<Item=ParameterD>);
}

/// Stochastic Gradient Descent 
///
///```
///let optim = Sgd::builder()
///    .learning_rate(0.001)
///    .momentum(0.1)
///    .build();
///```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Sgd {
    learning_rate: f32,
    momentum: f32
}

impl Sgd {
    pub fn builder() -> SgdBuilder {
        SgdBuilder::default()
    }
}

impl From<SgdBuilder> for Sgd {
    fn from(builder: SgdBuilder) -> Self {
        Self {
            learning_rate: builder.learning_rate,
            momentum: builder.momentum
        }
    }
}

impl Default for Sgd {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, parameters: impl IntoIterator<Item=ParameterD>) {        
        parameters.into_iter()
            .for_each(|weight| {
                if let Some(weight_grad) = weight.grad() {
                    if let Some(weight_grad) = weight_grad.read() {
                        let mut weight_value = weight.value()
                            .write()
                            .unwrap(); // Panics if RwLock is poisoned
                        let weight_grad = weight_grad.unwrap(); // Panics if RwLock is poisoned
                        if self.momentum > 0. {
                            let mut optimizer_data = weight.meta()
                                .optimizer_data()
                                .write()
                                .unwrap();
                            if let [OptimizerDataEntry::VelocityTensor(ref mut velocity)] = optimizer_data.as_mut_slice() {
                                crate::sgd_with_momentum(&mut weight_value, &weight_grad, self.learning_rate, self.momentum, velocity);  
                            }
                            else {
                                let mut velocity = Tensor::zeros(weight_value.device(), weight_value.raw_dim());
                                crate::sgd_with_momentum(&mut weight_value, &weight_grad, self.learning_rate, self.momentum, &mut velocity);
                                *optimizer_data = vec![OptimizerDataEntry::VelocityTensor(velocity)]
                            }
                        }
                        else {
                            weight_value.scaled_add(-self.learning_rate, &weight_grad.view());
                        }
                    }
                }
            });
    }
}


