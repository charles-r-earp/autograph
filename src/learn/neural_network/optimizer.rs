use super::autograd::ParameterViewMutD;
#[cfg(feature = "device")]
use crate::tensor::{ScalarTensorView, ScalarTensorViewMut};
use crate::{
    device::Device,
    scalar::{Scalar, ScalarElem, ScalarType},
    tensor::{ScalarTensor, ScalarTensorD, ScalarTensorViewMutD, TensorViewD, TensorViewMutD},
};
use anyhow::{bail, Result};
#[cfg(feature = "device")]
use krnl::macros::module;
use ndarray::Zip;
use serde::{Deserialize, Serialize};
use std::any::TypeId;

pub mod builder {
    use super::*;

    pub struct TensorValueBuilder {
        tensor: ScalarTensorD,
        parameter_device: bool,
        parameter_type: bool,
    }

    impl TensorValueBuilder {
        pub(super) fn new(tensor: ScalarTensorD) -> Self {
            Self {
                tensor,
                parameter_device: false,
                parameter_type: false,
            }
        }
        pub fn parameter_device(self, parameter_device: bool) -> Self {
            Self {
                parameter_device,
                ..self
            }
        }
        pub fn parameter_type(self, parameter_type: bool) -> Self {
            Self {
                parameter_type,
                ..self
            }
        }
        pub fn build(self) -> TensorValue {
            let Self {
                tensor,
                parameter_device,
                parameter_type,
            } = self;
            TensorValue {
                tensor,
                parameter_device,
                parameter_type,
            }
        }
    }

    pub struct SGDBuilder {
        momentum: Option<f32>,
    }

    impl SGDBuilder {
        pub(super) fn new() -> Self {
            Self { momentum: None }
        }
        pub fn momentum(self, momentum: f32) -> Self {
            Self {
                momentum: Some(momentum),
            }
        }
        pub fn build(self) -> SGD {
            let Self { momentum } = self;
            SGD { momentum }
        }
    }
}
use builder::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorValue {
    tensor: ScalarTensorD,
    parameter_device: bool,
    parameter_type: bool,
}

impl TensorValue {
    pub fn builder(tensor: ScalarTensorD) -> TensorValueBuilder {
        TensorValueBuilder::new(tensor)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Value {
    Tensor(TensorValue),
    Elem(ScalarElem),
}

impl Value {
    fn as_mut(&mut self) -> ValueMut {
        match self {
            Self::Tensor(x) => ValueMut::Tensor(x.tensor.view_mut()),
            Self::Elem(x) => ValueMut::Elem(x),
        }
    }
}

#[derive(Debug, derive_more::Unwrap)]
pub enum ValueMut<'a> {
    Tensor(ScalarTensorViewMutD<'a>),
    Elem(&'a mut ScalarElem),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct State {
    name: String,
    #[serde(skip, default = "default_type_id")]
    id: TypeId,
    key_values: Vec<(String, Value)>,
}

fn default_type_id() -> TypeId {
    TypeId::of::<()>()
}

impl State {
    pub(crate) fn new(
        device: Device,
        scalar_type: ScalarType,
        name: String,
        id: TypeId,
        key_values: Vec<(String, Value)>,
    ) -> Result<Self> {
        for (_key, value) in key_values.iter() {
            if let Value::Tensor(tensor_value) = value {
                if tensor_value.parameter_device && tensor_value.tensor.device() != device {
                    bail!("Wrong device!");
                }
                if tensor_value.parameter_type && tensor_value.tensor.scalar_type() != scalar_type {
                    bail!("Wrong type!");
                }
            }
        }
        Ok(Self {
            name,
            id,
            key_values,
        })
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn id(&self) -> TypeId {
        self.id
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.key_values
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, ValueMut)> {
        self.key_values
            .iter_mut()
            .map(|(key, value)| (key.as_str(), value.as_mut()))
    }
    pub(crate) fn to_owned(&self) -> Result<Self> {
        todo!()
    }
    /*
    pub(crate) fn to_device(&self, device: Device) -> Result<Self> {
        todo!()
    }
    pub(crate) fn to_device_mut(&mut self, device: Device) -> Result<()> {
        todo!()
    }*/
}

pub trait Optimizer {
    fn update(&self, learning_rate: f32, parameter: ParameterViewMutD) -> Result<()>;
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct SGD {
    momentum: Option<f32>,
}

impl SGD {
    pub fn builder() -> SGDBuilder {
        SGDBuilder::new()
    }
    fn init_state(&self, parameter: &mut ParameterViewMutD) -> Result<()> {
        if let Some(state) = parameter.optimizer_state() {
            if state.id() == TypeId::of::<Self>() {
                if self.momentum.is_some() == state.iter().next().is_some() {
                    return Ok(());
                }
            }
        }
        let mut key_values = Vec::new();
        if self.momentum.is_some() {
            let velocity = ScalarTensor::zeros(
                parameter.device(),
                parameter.raw_dim(),
                parameter.scalar_type(),
            )?;
            key_values.push((
                "velocity".to_string(),
                Value::Tensor(
                    TensorValue::builder(velocity)
                        .parameter_device(true)
                        .parameter_type(true)
                        .build(),
                ),
            ));
        }
        parameter.init_optimizer_state("SGD", TypeId::of::<Self>(), key_values)
    }
}

impl Optimizer for SGD {
    fn update(&self, learning_rate: f32, mut parameter: ParameterViewMutD) -> Result<()> {
        let scalar_type = parameter.scalar_type();
        if !matches!(scalar_type, ScalarType::BF16 | ScalarType::F32) {
            bail!("SGD {scalar_type:?} unimplemented!");
        }
        self.init_state(&mut parameter)?;
        if let Some(grad) = parameter.grad() {
            let (value, state) = parameter.value_view_optimizer_state_mut();
            let state = state.unwrap();
            let grad = grad.view();
            if let Some(momentum) = self.momentum {
                let (_, velocity) = state.iter_mut().next().unwrap();
                let velocity = velocity.unwrap_tensor();
                match scalar_type {
                    ScalarType::BF16 => todo!(),
                    ScalarType::F32 => sgd_update_with_momentum::<f32>(
                        value.try_into().unwrap(),
                        learning_rate,
                        grad.try_into().unwrap(),
                        momentum,
                        velocity.try_into().unwrap(),
                    )?,
                    _ => unreachable!(),
                }
            } else {
                parameter.value_view_mut().scaled_add(
                    ScalarElem::F32(-learning_rate).scalar_cast(scalar_type),
                    &grad,
                )?;
            }
        }
        Ok(())
    }
}

fn sgd_update_with_momentum<T: Scalar>(
    mut value: TensorViewMutD<T>,
    learning_rate: f32,
    grad: TensorViewD<T>,
    momentum: f32,
    mut velocity: TensorViewMutD<T>,
) -> Result<()> {
    if let Some(((value, grad), velocity)) = value
        .as_array_mut()
        .zip(grad.as_array())
        .zip(velocity.as_array_mut())
    {
        Zip::from(value)
            .and(grad)
            .and(velocity)
            .for_each(|value, grad, velocity| {
                let mut value_f32 = value.cast::<f32>();
                let grad_f32 = grad.cast::<f32>();
                let mut velocity_f32 = velocity.cast::<f32>();
                kernels::sgd_update_with_momentum(
                    &mut value_f32,
                    grad_f32,
                    learning_rate,
                    momentum,
                    &mut velocity_f32,
                );
                *velocity = velocity_f32.cast();
                *value = value_f32.cast();
            });
        return Ok(());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let mut value = ScalarTensorViewMut::from(value)
            .try_into_tensor_view_mut::<f32>()
            .unwrap();
        let grad = ScalarTensorView::from(grad)
            .try_into_tensor_view::<f32>()
            .unwrap();
        let mut velocity = ScalarTensorViewMut::from(velocity)
            .try_into_tensor_view_mut::<f32>()
            .unwrap();
        kernels::sgd_update_with_momentum_f32::builder()?
            .build(value.device())?
            .dispatch(
                value.as_slice_mut().unwrap(),
                grad.as_slice().unwrap(),
                learning_rate,
                momentum,
                velocity.as_slice_mut().unwrap(),
            )?;
        Ok(())
    }
}

#[cfg_attr(feature = "device", module)]
mod kernels {
    #[cfg(all(feature = "device", not(target_arch = "spirv")))]
    use krnl::krnl_core;
    #[cfg(any(feature = "device", target_arch = "spirv"))]
    use krnl_core::macros::kernel;

    pub fn sgd_update_with_momentum(w: &mut f32, dw: f32, lr: f32, m: f32, v: &mut f32) {
        *v = m * *v + dw;
        *w -= lr * *v;
    }

    #[cfg(any(feature = "device", target_arch = "spirv"))]
    pub mod device {
        use super::*;
        #[kernel(threads(256))]
        pub fn sgd_update_with_momentum_f32(
            #[item] w: &mut f32,
            #[item] dw: f32,
            lr: f32,
            m: f32,
            #[item] v: &mut f32,
        ) {
            sgd_update_with_momentum(w, dw, lr, m, v);
        }
    }
    #[cfg(any(feature = "device", target_arch = "spirv"))]
    pub use device::*;
}
