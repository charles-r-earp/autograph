use super::autograd::ParameterViewMutD;
use crate::{
    scalar::{Scalar, ScalarElem, ScalarType},
    tensor::{ScalarTensorD, ScalarTensorViewD, ScalarTensorViewMutD},
};
use anyhow::{bail, Result};
use std::sync::Arc;

pub mod builder {
    use super::*;

    #[derive(Default)]
    pub struct SGDBuilder {}

    impl SGDBuilder {
        pub fn build(self) -> SGD {
            SGD {}
        }
    }
}
use builder::SGDBuilder;

#[derive(Debug)]
pub enum Value {
    Tensor(ScalarTensorD),
}

#[derive(Debug)]
pub enum ValueMut<'a> {
    Tensor(ScalarTensorViewMutD<'a>),
}

#[derive(Debug)]
pub struct ValuesMut<'a> {
    inner: Vec<ValueMut<'a>>,
}

#[derive(Debug)]
struct StateInner {
    values: Vec<Value>,
}

#[derive(Default, Debug)]
pub(crate) struct State {
    inner: Option<Arc<StateInner>>,
}

impl State {
    pub(crate) fn cast_mut(self, scalar_type: ScalarType) -> Result<State> {
        todo!()
    }
    pub(crate) fn to_device_mut(&mut self) -> Result<()> {
        todo!()
    }
}

pub struct StateMut<'a> {
    state: &'a mut State,
}

impl<'a> StateMut<'a> {
    pub fn make_mut_or_init(
        &mut self,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<ValuesMut> {
        todo!()
    }
    pub fn clear(&mut self) {
        todo!()
    }
}

pub trait Optimizer {
    fn update(&self, learning_rate: f32, parameter: ParameterViewMutD) -> Result<()>;
}

#[derive(Default, Debug)]
pub struct SGD {}

impl SGD {
    pub fn builder() -> SGDBuilder {
        SGDBuilder::default()
    }
}

impl Optimizer for SGD {
    fn update(&self, learning_rate: f32, mut parameter: ParameterViewMutD) -> Result<()> {
        if let Some(grad) = parameter.grad() {
            parameter
                .value_view_mut()
                .scaled_add((-learning_rate).into(), &grad)?;
        }
        Ok(())
    }
}
