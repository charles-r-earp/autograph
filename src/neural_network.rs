use crate::{tensor::float_tensor::FloatTensorViewD, Result};
//use std::collections::HashMap;

pub mod autograd;
use autograd::{ParameterD, VariableD};

//pub mod builders;

pub trait Optimizer {
    fn step(&mut self, parameter: &mut ParameterD, gradient: FloatTensorViewD) -> Result<()>;
}
/*
pub struct Sgd {
    learning_rate: f32,
    momentum: f32,
    velocities: HashMap<Vertex, FloatTensorD>,
}
*/
pub trait Forward {
    fn forward(&self, input: VariableD) -> Result<VariableD>;
}

pub trait Network: Forward {
    /// Implementation method for parameters_mut\
    ///
    /// Mutable references to the parameters of the network (or layer) should be pushed into the\
    /// provided vec.
    #[allow(unused_variables)]
    fn collect_paramters_mut(&mut self, parameters: &mut Vec<&mut ParameterD>) {}
    /// Returns a Vec containing mutable references to all the parameters in the network.
    ///
    /// Generally this does should not be implemented, as the default implementation calls
    /// collect_paramters_mut.
    fn parameters_mut(&mut self) -> Vec<&mut ParameterD> {
        let mut parameters = Vec::new();
        self.collect_paramters_mut(&mut parameters);
        parameters
    }
    /*
    fn to_device_mut(&mut self, device: &Device) -> Result<()> {
        for parameter in self.parameters_mut() {
            todo!() // parameter.to_device_mut(device)?;
        }
    }
    */
}
/*
#[derive(Default)]
pub struct Identity;

pub struct Dense<A: Network + Default> {
    weight: ParameterD,
    bias: Option<ParameterD>,
    activation: A,
}

impl<A: Network + Default> Dense<A> {
    fn builder() -> DenseBuilder<A> {

    }
}
*/
