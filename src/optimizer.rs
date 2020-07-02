use crate::{Buffer, Tensor, TensorD};
use crate::autograd::{Parameter, ParameterD};
use std::{
    hash::{Hash, Hasher},
    collections::HashMap,
    sync::{Arc, Weak, RwLock}
};
use ndarray::Dimension;

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

/// Unique Identifier For Tracking Parameters
#[derive(Eq, PartialEq, Hash, Clone)]
struct ParameterId(usize);

/// Wrapper around a Weak<RwLockBuffer<f32>>>\
/// Used to track when parameters are dropped. If a parameter's value buffer has no strong references, the optimizer can drop any meta data associated with the parameter.
struct ParameterHandle(Weak<RwLock<Buffer<f32>>>); 

impl ParameterHandle {
    fn strong_count(&self) -> usize {
        Weak::strong_count(&self.0)
    }
}

impl<D: Dimension> Parameter<D> {
    fn id(&self) -> ParameterId {
        ParameterId(unsafe { 
            &*self.value()
                .data
                .buffer as *const RwLock<Buffer<f32>> as usize 
        })
    }
    fn handle(&self) -> ParameterHandle { 
        ParameterHandle(Arc::downgrade(&self.value().data.buffer))
    }
}

/// Stochastic Gradient Descent 
///
///```
///let optim = Sgd::builder()
///    .learning_rate(0.001)
///    .momentum(0.1)
///    .build();
///```
pub struct Sgd {
    learning_rate: f32,
    momentum: f32,
    velocities: HashMap<ParameterId, (ParameterHandle, TensorD<f32>)>
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
            momentum: builder.momentum,
            velocities: HashMap::new()
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
        // remove entries that ref parameters that have been dropped
        let to_remove: Vec<ParameterId> = self.velocities.iter()
            .filter_map(|(id, (handle, _))| {
                if handle.strong_count() == 0 {
                    Some(id.clone())
                } else { None }
            })
            .collect();
        to_remove.iter()
            .for_each(|id| {
                self.velocities.remove(&id);
            });
        
        // update 
        parameters.into_iter()
            .for_each(|weight| {
                if let Some(weight_grad) = weight.grad() {
                    if let Some(weight_grad) = weight_grad.read() {
                        let mut weight_value = weight.value()
                            .write()
                            .unwrap(); // Panics if RwLock is poisoned
                        let weight_grad = weight_grad.unwrap(); // Panics if RwLock is poisoned
                        if self.momentum > 0. {
                            // The id retains tha arc, so 
                            let id = weight.id();
                            if let Some(handle_velocity) = self.velocities.get_mut(&id) {
                                let mut velocity = &mut handle_velocity.1;
                                crate::sgd_with_momentum(&mut weight_value, &weight_grad, self.learning_rate, self.momentum, &mut velocity);  
                            }
                            else {
                                let mut velocity = Tensor::zeros(weight_value.device(), weight_value.raw_dim());
                                crate::sgd_with_momentum(&mut weight_value, &weight_grad, self.learning_rate, self.momentum, &mut velocity);
                                self.velocities.insert(id, (weight.handle(), velocity));    
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


