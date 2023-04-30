use crate::{
    device::Device,
    ops::AddAssign,
    scalar::ScalarType,
    tensor::{ScalarArcTensor, ScalarArcTensorD},
};
use anyhow::{bail, format_err, Error, Result};
use atomicbox::AtomicOptionBox;
use crossbeam_channel::{Receiver, RecvError, Sender};
use ndarray::{linalg::Dot, Dimension, Ix0, Ix2, IxDyn, ShapeError};
use parking_lot::{RwLock, RwLockWriteGuard};
use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Weak,
    },
};

#[derive(Clone)]
pub struct Variable<D: Dimension> {
    value: ScalarArcTensor<D>,
    grad: Option<GradientLock<D>>,
}

pub type Variable0 = Variable<Ix0>;
pub type Variable2 = Variable<Ix2>;
pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> Variable<D> {
    pub fn value(&self) -> &ScalarArcTensor<D> {
        &self.value
    }
    pub fn grad(&self) -> Option<&GradientLock<D>> {
        self.grad.as_ref()
    }
    pub fn device(&self) -> Device {
        self.value.device()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.value.scalar_type()
    }
    pub fn with_backward(
        self,
        input_grads: impl IntoIterator<Item = Option<GradientLock<D>>>,
        backward: impl FnOnce(Gradient<D>, &mut [Option<GradientMut<D>>]) -> Result<()>
            + Send
            + Sync
            + 'static,
    ) -> Self
    where
        D: 'static,
    {
        let input_grads: Vec<_> = input_grads.into_iter().collect();
        if input_grads.iter().all(Option::is_none) {
            return self;
        }
        for input_grad in input_grads.iter().flatten() {
            input_grad.inner.ref_count.fetch_add(1, Ordering::SeqCst);
        }
        let mut backward = Some(backward);
        let backward = BackwardOp(Box::new(
            move |output_grad: GradientD, sender: Weak<Sender<Error>>| {
                if let Some(backward) = backward.take() {
                    let output_grad: Gradient<D> = output_grad.into_dimensionality().unwrap();
                    let mut input_grads: Vec<_> = input_grads
                        .iter()
                        .map(|grad| {
                            if let Some(grad) = grad {
                                Some(grad.write(sender.clone()))
                            } else {
                                None
                            }
                        })
                        .collect();
                    (backward)(output_grad, input_grads.as_mut_slice());
                }
            },
        ));
        let inner = Arc::new(GradientInner {
            gradient: RwLock::default(),
            ref_count: AtomicUsize::default(),
            backward: AtomicOptionBox::new(Some(Box::new(backward))),
        });
        let grad = Some(GradientLock {
            inner,
            _m: PhantomData::default(),
        });
        Self { grad, ..self }
    }
}

impl Variable0 {
    pub fn backward(&self) -> Result<()> {
        let grad = if let Some(grad) = self.grad.as_ref() {
            grad
        } else {
            bail!("Variable does not have a gradient!")
        };
        let inner = &grad.inner;
        let guard = inner.gradient.write();
        if guard.is_some() {
            bail!("Variable gradient already computed!");
        }
        let gradient = Gradient::Dense(ScalarArcTensor::ones(
            self.device(),
            IxDyn::default(),
            self.scalar_type(),
        )?);
        let mut backward = if let Some(backward) = inner.backward.take(Ordering::SeqCst) {
            backward
        } else {
            bail!("Variable does not have a backward!")
        };
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let sender = Arc::new(sender);
        let sender2 = Arc::downgrade(&sender);
        rayon::spawn(move || (backward)(gradient, sender2));
        if let Ok(error) = receiver.recv() {
            return Err(error);
        }
        Ok(())
    }
}

impl<D: Dimension> From<ScalarArcTensor<D>> for Variable<D> {
    fn from(tensor: ScalarArcTensor<D>) -> Self {
        Self {
            value: tensor,
            grad: None,
        }
    }
}

impl Dot<Self> for Variable2 {
    type Output = Result<Self>;
    fn dot(&self, rhs: &Self) -> Result<Self> {
        todo!()
        /*
        let lhs = self;
        let output = Self::from(lhs.value().dot(rhs.value())?);
        let lhs_grad = lhs.grad().clone();
        let rhs_grad = rhs.grad().clone();
        let lhs = lhs.values().clone();
        let rhs = rhs.value().clone();
        Ok(Self::from(self.value().dot(rhs)?).with_backward(
            [lhs_grad, rhs_grad],
            move |output_grad, input_grads| {
                let [lhs_grad, rhs_grad] =
                    <[Option<GradientMut2>; 2]>::try_from(input_grads).unwrap();
                let output_grad = output_grad.unwrap_dense();
                if let Some(mut lhs_grad) = lhs_grad {
                    lhs_grad.add_assign(output_grad.dot(lhs.t()))?;
                }
                if let Some(mut rhs_grad) = rhs_grad {
                    rhs_grad.add_assign(lhs.t().dot(&output_grad))?;
                }
            },
        ))*/
    }
}

#[derive(Clone)]
pub struct GradientLock<D: Dimension> {
    inner: Arc<GradientInner>,
    _m: PhantomData<D>,
}

impl<D: Dimension> GradientLock<D> {
    pub fn load(&self) -> Option<Gradient<D>> {
        todo!()
    }
    fn write(&self, sender: Weak<Sender<Error>>) -> GradientMut<D> {
        let inner = &self.inner;
        let mut guard = inner.gradient.write();
        let gradient = guard.take().map(|x| x.into_dimensionality().unwrap());
        GradientMut {
            inner,
            guard,
            gradient,
            sender,
        }
    }
}

pub struct GradientMut<'a, D: Dimension> {
    inner: &'a GradientInner,
    guard: RwLockWriteGuard<'a, Option<GradientD>>,
    gradient: Option<Gradient<D>>,
    sender: Weak<Sender<Error>>,
}

pub type GradientMut2<'a> = GradientMut<'a, Ix2>;

impl<D: Dimension> GradientMut<'_, D> {
    pub fn as_mut(&mut self) -> Option<&mut Gradient<D>> {
        self.gradient.as_mut()
    }
}

impl<D: Dimension> AddAssign<ScalarArcTensor<D>> for GradientMut<'_, D> {
    type Error = Error;
    fn add_assign(&mut self, rhs: ScalarArcTensor<D>) -> Result<()> {
        if let Some(gradient) = self.gradient.as_mut() {
            todo!()
        } else {
            self.gradient.replace(Gradient::Dense(rhs));
        }
        Ok(())
    }
}

impl<'a, D: Dimension> Drop for GradientMut<'a, D> {
    fn drop(&mut self) {
        let gradient = self.gradient.take().map(Gradient::into_dyn);
        *self.guard = gradient.clone();
        if self.inner.ref_count.fetch_sub(1, Ordering::SeqCst) == 0 {
            if let Some((gradient, mut backward)) =
                gradient.zip(self.inner.backward.take(Ordering::SeqCst))
            {
                if Weak::strong_count(&self.sender) > 0 {
                    let sender = self.sender.clone();
                    rayon::spawn(move || (backward)(gradient, sender));
                }
            }
        }
    }
}

#[derive(derive_more::Deref, derive_more::DerefMut)]
struct BackwardOp(Box<dyn FnMut(GradientD, Weak<Sender<Error>>) + Send + Sync>);

struct GradientInner {
    gradient: RwLock<Option<Gradient<IxDyn>>>,
    ref_count: AtomicUsize,
    backward: AtomicOptionBox<BackwardOp>,
}

#[derive(Clone, derive_more::IsVariant, derive_more::Unwrap)]
pub enum Gradient<D: Dimension> {
    Dense(ScalarArcTensor<D>),
}

pub type GradientD = Gradient<IxDyn>;

impl<D: Dimension> Gradient<D> {
    /// Converts the gradient into dimension `D2`.
    ///
    /// Typically this is used to downcast from [`IxDyn`](type@ndarray::IxDyn) to a static dimensionality. For conversions to [`IxDyn`](type@ndarray::IxDyn), use [`.into_dyn()`](TensorBase::into_dyn()).
    ///
    /// **Errors**
    /// The number of axes of `D2` must be the same as `D`.
    pub fn into_dimensionality<D2>(self) -> Result<Gradient<D2>, ShapeError>
    where
        D2: Dimension,
    {
        match self {
            Self::Dense(x) => x.into_dimensionality().map(Gradient::Dense),
        }
    }
    /// Converts the dimensionality of the gradient to [`IxDyn`](type@ndarray::IxDyn).
    pub fn into_dyn(self) -> GradientD {
        match self {
            Self::Dense(x) => Gradient::Dense(x.into_dyn()),
        }
    }
}

#[derive(Clone)]
pub struct Parameter<D: Dimension> {
    variable: Variable<D>,
}

pub type ParameterD = Parameter<IxDyn>;

impl<D: Dimension> Parameter<D> {
    pub fn variable(&self) -> &Variable<D> {
        &self.variable
    }
    pub fn to_device_mut(&mut self, device: Device) -> Result<()> {
        todo!()
    }
    pub fn grad(&self) -> Option<&GradientLock<D>> {
        self.variable.grad()
    }
    /// Adds a gradient computed on backward pass.
    ///
    /// If `training` is true, [`.grad()`] will be Some.
    /// else it will be None.
    ///
    /// This should be called with `training` = true prior to training.
    /// Setting `training` to false prevents gradients from being computed
    /// and can be used for inference.
    pub fn set_training(&mut self, training: bool) {
        if training {
            if self.grad().is_some() {
                return;
            }
            let inner = Arc::new(GradientInner {
                gradient: RwLock::default(),
                ref_count: AtomicUsize::default(),
                backward: AtomicOptionBox::default(),
            });
            self.variable.grad = Some(GradientLock {
                inner,
                _m: PhantomData::default(),
            });
        } else {
            self.variable.grad = None;
        }
    }
}
