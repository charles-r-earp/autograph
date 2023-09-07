#[cfg(doc)]
use super::autograd::Parameter;
use super::autograd::ParameterViewMutD;
#[cfg(feature = "device")]
use crate::tensor::{ScalarTensorView, ScalarTensorViewMut};
use crate::{
    device::Device,
    scalar::{Scalar, ScalarElem, ScalarType},
    tensor::{ScalarTensor, ScalarTensorD, ScalarTensorViewMutD, TensorViewD, TensorViewMutD},
};
use anyhow::{bail, Result};
use dry::macro_for;
use half::bf16;
#[cfg(feature = "device")]
use krnl::macros::module;
use ndarray::Zip;
use paste::paste;
use serde::{Deserialize, Serialize};
use std::any::TypeId;

/// Optimizer builders.
pub mod builder {
    use super::*;

    /// Builder for creating a [`TensorValue`].
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
        /// Whether the value device should match the parameter device.
        ///
        /// If true, transfering the parameter to a device will copy this value
        /// to that device. Otherwise, it will be kept on the host.
        pub fn parameter_device(self, parameter_device: bool) -> Self {
            Self {
                parameter_device,
                ..self
            }
        }
        /// Whether the value scalar_type should match the parameter scalar_type.
        ///
        /// If true, casting the parameter will cast this value to that type.
        pub fn parameter_type(self, parameter_type: bool) -> Self {
            Self {
                parameter_type,
                ..self
            }
        }
        /// Builds the value.
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

    /// Builder for creating a [`SGD`].
    pub struct SGDBuilder {
        momentum: Option<f32>,
    }

    impl SGDBuilder {
        pub(super) fn new() -> Self {
            Self { momentum: None }
        }
        /// Momentum. Default is 0.
        ///
        /// If `momentum` is greater than 0, a "velocity" tensor will
        /// be added to the [`State`] of each [`Parameter`].
        pub fn momentum(self, momentum: f32) -> Self {
            Self {
                momentum: Some(momentum),
            }
        }
        /// Builds the optimizer.
        pub fn build(self) -> SGD {
            let Self { momentum } = self;
            SGD { momentum }
        }
    }
}
use builder::*;

/// Tensor value.
#[derive(Debug, Serialize, Deserialize)]
pub struct TensorValue {
    tensor: ScalarTensorD,
    parameter_device: bool,
    parameter_type: bool,
}

impl TensorValue {
    /// A builder for creating a [`TensorValue`].
    pub fn builder(tensor: ScalarTensorD) -> TensorValueBuilder {
        TensorValueBuilder::new(tensor)
    }
}

/// [`State`] value.
#[derive(Debug, Serialize, Deserialize)]
pub enum Value {
    /// A tensor.
    Tensor(TensorValue),
    /// An elem.
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

/// Mutable [`Value`].
#[derive(Debug)]
pub enum ValueMut<'a> {
    /// A tensor.
    Tensor(ScalarTensorViewMutD<'a>),
    /// An elem.
    Elem(&'a mut ScalarElem),
}

impl<'a> ValueMut<'a> {
    fn unwrap_tensor(self) -> ScalarTensorViewMutD<'a> {
        if let Self::Tensor(tensor) = self {
            tensor
        } else {
            panic!("Expected tensor!")
        }
    }
    fn unwrap_elem(self) -> ScalarElem {
        if let Self::Elem(elem) = self {
            *elem
        } else {
            panic!("Expected elem!")
        }
    }
}

/// Optimizer State.
///
/// Created with [`ParameterBase::init_optimizer_state()`](super::autograd::ParameterBase::init_optimizer_state).
/// Stores per parameter training progress.
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
        for (key, value) in key_values.iter() {
            if let Value::Tensor(tensor_value) = value {
                let value_device = tensor_value.tensor.device();
                if tensor_value.parameter_device && value_device != device {
                    bail!("Expected {name:?}.{key:?} device {value_device:?} to match parameter {device:?}!");
                }
                let value_scalar_type = tensor_value.tensor.scalar_type();
                if tensor_value.parameter_type && value_scalar_type != scalar_type {
                    bail!("Expected {name:?}.{key:?} scalar_type {value_scalar_type:?} to match parameter {scalar_type:?}!");
                }
            }
        }
        Ok(Self {
            name,
            id,
            key_values,
        })
    }
    /// Name of the [`Optimizer`].
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Type id of the [`Optimizer`].
    pub fn id(&self) -> TypeId {
        self.id
    }
    /// Iterator over keys and values.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.key_values
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }
    /// Iterator over keys and mutable values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, ValueMut)> {
        self.key_values
            .iter_mut()
            .map(|(key, value)| (key.as_str(), value.as_mut()))
    }
    pub(crate) fn to_owned(&self) -> Result<Self> {
        let mut key_values = Vec::with_capacity(self.key_values.len());
        for (key, value) in self.key_values.iter() {
            let value = match value {
                Value::Tensor(tensor_value) => Value::Tensor(TensorValue {
                    tensor: tensor_value.tensor.to_owned()?,
                    parameter_type: tensor_value.parameter_type,
                    parameter_device: tensor_value.parameter_device,
                }),
                Value::Elem(elem) => Value::Elem(*elem),
            };
            key_values.push((key.clone(), value));
        }
        Ok(Self {
            name: self.name.clone(),
            id: self.id,
            key_values,
        })
    }
    pub(crate) fn to_device(&self, device: Device) -> Result<Self> {
        let mut key_values = Vec::with_capacity(self.key_values.len());
        for (key, value) in self.key_values.iter() {
            let value = match value {
                Value::Tensor(tensor_value) => {
                    let tensor = if tensor_value.parameter_device {
                        tensor_value.tensor.to_device(device.clone())?
                    } else {
                        tensor_value.tensor.to_owned()?
                    };
                    Value::Tensor(TensorValue {
                        tensor,
                        parameter_type: tensor_value.parameter_type,
                        parameter_device: tensor_value.parameter_device,
                    })
                }
                Value::Elem(elem) => Value::Elem(*elem),
            };
            key_values.push((key.clone(), value));
        }
        Ok(Self {
            name: self.name.clone(),
            id: self.id,
            key_values,
        })
    }
    /*pub(crate) fn to_device_mut(&mut self, device: Device) -> Result<()> {
        for (_, value) in self.key_values.iter_mut() {
            if let Value::Tensor(tensor_value) = value {
                if tensor_value.parameter_device {
                    tensor_value.tensor.to_device_mut(device.clone())?;
                }
            }
        }
        Ok(())
    }*/
}

/// Optimizer.
pub trait Optimizer {
    /// Performs the optimization, updating the parameter with `learning_rate`.
    fn update(&self, learning_rate: f32, parameter: ParameterViewMutD) -> Result<()>;
}

/// Stochastic Gradient Descent.
///
/// Implemented for bf16 and f32.
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct SGD {
    momentum: Option<f32>,
}

impl SGD {
    /// An SGD builder.
    pub fn builder() -> SGDBuilder {
        SGDBuilder::new()
    }
    fn init_state(&self, parameter: &mut ParameterViewMutD) -> Result<()> {
        if let Some(state) = parameter.optimizer_state() {
            if state.id() == TypeId::of::<Self>()
                && self.momentum.is_some() == state.iter().next().is_some()
            {
                return Ok(());
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

/// Implemented for bf16 and f32.
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
                    ScalarType::BF16 => sgd_update_with_momentum::<bf16>(
                        value.try_into().unwrap(),
                        learning_rate,
                        grad.try_into().unwrap(),
                        momentum,
                        velocity.try_into().unwrap(),
                    )?,
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
        macro_for!($T in [bf16, f32] {
            if value.scalar_type() == $T::scalar_type() {
                let mut value = ScalarTensorViewMut::from(value)
                    .try_into_tensor_view_mut::<$T>()
                    .unwrap();
                let grad = ScalarTensorView::from(grad)
                    .try_into_tensor_view::<$T>()
                    .unwrap();
                let mut velocity = ScalarTensorViewMut::from(velocity)
                    .try_into_tensor_view_mut::<$T>()
                    .unwrap();
                let kernel = paste! {
                    kernels::[<sgd_update_with_momentum_ $T>]::builder()?
                    .build(value.device())?
                };
                return kernel
                    .dispatch(
                        value.as_slice_mut().unwrap(),
                        grad.as_slice().unwrap(),
                        learning_rate,
                        momentum,
                        velocity.as_slice_mut().unwrap(),
                    );
            }
        });
        unreachable!()
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
        #[cfg(target_arch = "spirv")]
        use krnl_core::half::bf16;

        #[kernel]
        pub fn sgd_update_with_momentum_bf16(
            #[item] w: &mut bf16,
            #[item] dw: bf16,
            lr: f32,
            m: f32,
            #[item] v: &mut bf16,
        ) {
            let mut w_f32 = w.to_f32();
            let mut v_f32 = v.to_f32();
            sgd_update_with_momentum(&mut w_f32, dw.to_f32(), lr, m, &mut v_f32);
            *w = bf16::from_f32(w_f32);
            *v = bf16::from_f32(v_f32);
        }

        #[kernel]
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
