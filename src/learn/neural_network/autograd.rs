use super::{
    layer::Forward,
    optimizer::{Optimizer, State, StateMut},
};
use crate::{
    buffer::{
        ArcBuffer, ArcBufferRepr, BufferBase, Data, DataMut, ScalarArcBuffer, ScalarArcBufferRepr,
        ScalarBufferBase, ScalarData, ScalarDataMut, ScalarSliceMutRepr, SliceMutRepr,
    },
    device::Device,
    ops::AddAssign,
    scalar::{Scalar, ScalarElem, ScalarType},
    tensor::{
        ArcTensor, ScalarArcTensor, ScalarArcTensorD, ScalarTensor, ScalarTensor4,
        ScalarTensorBase, ScalarTensorViewMut, Tensor, TensorView, TensorViewMut,
    },
};
use anyhow::{bail, Error, Result};
use ndarray::{linalg::Dot, Axis, Dimension, IntoDimension, Ix1, Ix2, Ix4, IxDyn, ShapeError};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::VecDeque,
    fmt::Debug,
    marker::PhantomData,
    sync::{atomic::Ordering, Arc, Weak},
};

pub mod builder {
    use serde::de::VariantAccess;

    use super::*;

    pub struct VariableBuilder<D: Dimension> {
        grad: Option<Arc<RwLock<Option<ScalarArcTensorD>>>>,
        edges: Vec<EdgeInner>,
        _m: PhantomData<D>,
    }

    impl<D: Dimension> VariableBuilder<D> {
        pub(super) fn new() -> Self {
            Self {
                grad: None,
                edges: Vec::new(),
                _m: PhantomData::default(),
            }
        }
        pub fn edge<D2, F>(&mut self, node: &Node<D2>, f: F)
        where
            D2: Dimension,
            F: FnOnce(ScalarArcTensor<D>) -> Result<ScalarArcTensor<D2>> + Send + Sync + 'static,
        {
            if self.grad.is_none() {
                self.grad.replace(Arc::new(RwLock::default()));
            }
            let output_grad_lock = self.grad.clone().unwrap();
            let node = node.inner.clone();
            let mut input_grad_lock = Arc::downgrade(&node.grad);
            let device = node.device.clone();
            let dim = node.dim.clone();
            let scalar_type = node.scalar_type;
            let name = std::any::type_name::<F>();
            let mut f = Some(f);
            let op = Box::new(move || {
                let input_grad_lock = Weak::upgrade(&std::mem::take(&mut input_grad_lock));
                if let Some((f, input_grad_lock)) = f.take().zip(input_grad_lock) {
                    let grad = output_grad_lock
                        .read()
                        .clone()
                        .unwrap()
                        .into_dimensionality()
                        .unwrap();
                    let grad = (f)(grad)?;
                    assert_eq!(grad.device(), device);
                    assert_eq!(grad.shape(), dim.slice());
                    assert_eq!(grad.scalar_type(), scalar_type);
                    let mut guard = input_grad_lock.write();
                    if let Some(input_grad) = guard.as_mut() {
                        input_grad.make_view_mut()?.add_assign(&grad)?;
                    } else {
                        guard.replace(grad.into_dyn());
                    }
                }
                Ok(())
            });
            self.edges.push(EdgeInner { name, op, node })
        }
        pub fn build(self, value: ScalarArcTensor<D>) -> Variable<D> {
            let node = if let Some(grad) = self.grad {
                Some(Node::new(
                    value.device(),
                    value.raw_dim().into_dyn(),
                    value.scalar_type(),
                    grad,
                    self.edges,
                ))
            } else {
                None
            };
            Variable { value, node }
        }
    }
}
use builder::*;

struct EdgeInner {
    name: &'static str,
    op: Box<dyn FnMut() -> Result<()> + Send + Sync + 'static>,
    node: Arc<NodeInner>,
}

impl Debug for EdgeInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeInner")
            .field("node", &self.node)
            .finish()
    }
}

#[derive(Debug)]
struct NodeInner {
    device: Device,
    dim: IxDyn,
    scalar_type: ScalarType,
    grad: Arc<RwLock<Option<ScalarArcTensorD>>>,
    edges: Mutex<Vec<EdgeInner>>,
}

impl NodeInner {
    fn ready(&self) -> bool {
        Arc::weak_count(&self.grad) == 0
    }
}

#[derive(Clone, Debug)]
pub struct Node<D: Dimension> {
    inner: Arc<NodeInner>,
    _m: PhantomData<D>,
}

impl<D: Dimension> Node<D> {
    fn new(
        device: Device,
        dim: IxDyn,
        scalar_type: ScalarType,
        grad: Arc<RwLock<Option<ScalarArcTensorD>>>,
        edges: Vec<EdgeInner>,
    ) -> Self {
        Self {
            inner: Arc::new(NodeInner {
                device,
                dim,
                scalar_type,
                grad,
                edges: Mutex::new(edges),
            }),
            _m: PhantomData::default(),
        }
    }
    pub fn grad(&self) -> Option<ScalarArcTensor<D>> {
        Some(
            self.inner
                .grad
                .read()
                .clone()?
                .into_dimensionality()
                .unwrap(),
        )
    }
    pub fn backward(&self) -> Result<()> {
        let NodeInner {
            device,
            dim,
            scalar_type,
            grad,
            edges,
        } = self.inner.as_ref();
        {
            let mut guard = grad.write();
            if guard.is_some() {
                return Ok(());
            }
            let batch_size = dim.slice().first().copied().unwrap_or(1);
            let scale = ScalarElem::F32(1f32 / batch_size as f32).scalar_cast(*scalar_type);
            guard.replace(ScalarArcTensor::from_elem(
                device.clone(),
                dim.clone(),
                scale,
            )?);
        }
        let mut queue = VecDeque::new();
        queue.push_back(self.inner.clone());
        while let Some(node) = queue.pop_front() {
            for mut edge in node.edges.lock().drain(..) {
                (edge.op)()?;
                let node = edge.node;
                if node.ready() {
                    queue.push_back(node.clone())
                }
            }
        }
        Ok(())
    }
    fn into_dyn(self) -> Node<IxDyn> {
        Node {
            inner: self.inner,
            _m: PhantomData::default(),
        }
    }
    fn into_dimensionality<D2: Dimension>(self) -> Node<D2> {
        Node {
            inner: self.inner,
            _m: PhantomData::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Variable<D: Dimension> {
    value: ScalarArcTensor<D>,
    node: Option<Node<D>>,
}

pub type Variable2 = Variable<Ix2>;
pub type Variable4 = Variable<Ix4>;
pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> Variable<D> {
    pub fn builder() -> VariableBuilder<D> {
        VariableBuilder::new()
    }
    pub fn value(&self) -> &ScalarArcTensor<D> {
        &self.value
    }
    pub fn into_value(self) -> ScalarArcTensor<D> {
        self.value
    }
    pub fn node(&self) -> Option<&Node<D>> {
        self.node.as_ref()
    }
    pub fn forward<F: Forward<Self>>(self, f: &F) -> Result<F::Output> {
        f.forward(self)
    }
    pub fn backward(&self) -> Result<()> {
        if let Some(node) = self.node.as_ref() {
            node.backward()?;
        }
        Ok(())
    }
    pub fn device(&self) -> Device {
        self.value.device()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.value.scalar_type()
    }
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }
    pub fn raw_dim(&self) -> D {
        self.value.raw_dim()
    }
    pub fn into_dimensionality<D2>(self) -> Result<Variable<D2>, ShapeError>
    where
        D2: Dimension,
    {
        let value = self.value.into_dimensionality()?;
        Ok(Variable {
            value,
            node: self.node.map(Node::into_dimensionality),
        })
    }
    pub fn into_dyn(self) -> VariableD {
        Variable {
            value: self.value.into_dyn(),
            node: self.node.map(Node::into_dyn),
        }
    }
}

impl<D: Dimension + 'static> Variable<D> {
    pub fn into_shape<E>(self, shape: E) -> Result<Variable<E::Dim>, ShapeError>
    where
        E: IntoDimension,
    {
        let dim = self.raw_dim();
        let mut builder = Variable::builder();
        if let Some(node) = self.node() {
            builder.edge(node, |output_grad| {
                output_grad
                    .into_shape(dim)
                    .map_err(Error::msg)
                    .map(Into::into)
            })
        }
        Ok(builder.build(self.value.into_shape(shape)?))
    }
    pub fn flatten(self) -> Result<Variable2, ShapeError> {
        let dim = crate::tensor::flatten(self.shape());
        self.into_shape(dim)
    }
    /// Reverses (transposes) the axes of the tensor.
    pub fn reversed_axes(self) -> Self {
        let mut builder = Self::builder();
        if let Some(node) = self.node() {
            builder.edge(node, |output_grad| Ok(output_grad.reversed_axes()));
        }
        builder.build(self.value.reversed_axes())
    }
    pub fn t(&self) -> Self {
        self.clone().reversed_axes()
    }
    pub fn broadcast<E>(&self, dim: E) -> Option<Variable<E::Dim>>
    where
        E: IntoDimension,
    {
        let mut builder = Variable::builder();
        if let Some(node) = self.node() {
            builder.edge(node, |output_grad| {
                let input_grad = output_grad
                    .cast_into_tensor::<f32>()?
                    .into_array()?
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .sum_axis(Axis(0))
                    .into_dimensionality()
                    .unwrap();
                Tensor::from(input_grad).into_scalar_tensor().into_shared()
            })
        }
        Some(builder.build(self.value.broadcast_shared(dim)?))
    }
}

impl<T: Scalar, D: Dimension> From<Tensor<T, D>> for Variable<D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        Self::from(ScalarArcTensor::from(tensor))
    }
}

impl<T: Scalar, D: Dimension> From<ArcTensor<T, D>> for Variable<D> {
    fn from(tensor: ArcTensor<T, D>) -> Self {
        Self::from(ScalarArcTensor::from(tensor))
    }
}

impl<D: Dimension> From<ScalarTensor<D>> for Variable<D> {
    fn from(tensor: ScalarTensor<D>) -> Self {
        Self::from(ScalarArcTensor::from(tensor))
    }
}

impl<D: Dimension> From<ScalarArcTensor<D>> for Variable<D> {
    fn from(tensor: ScalarArcTensor<D>) -> Self {
        Self {
            value: tensor,
            node: None,
        }
    }
}

impl<D1: Dimension + 'static, D2: Dimension + 'static> AddAssign<Variable<D2>> for Variable<D1> {
    fn add_assign(&mut self, rhs: Variable<D2>) -> Result<()> {
        if self.node.is_none() && rhs.node.is_none() {
            return self.value.make_view_mut()?.add_assign(&rhs.value);
        }
        let rhs = if self.shape() != rhs.shape() {
            if let Some(rhs) = rhs.broadcast(self.raw_dim()) {
                rhs
            } else {
                bail!("Can not broadcast {:?} -> {:?}!", self, rhs);
            }
        } else {
            rhs.into_dimensionality().unwrap()
        };
        self.value.make_view_mut()?.add_assign(rhs.value())?;
        let mut builder = Self::builder();
        if let Some(node) = self.node() {
            builder.edge(node, Ok)
        }
        if let Some(node) = rhs.node() {
            let dim = rhs.raw_dim();
            builder.edge(node, |output_grad| {
                output_grad.into_shape(dim).map_err(Error::msg)
            })
        }
        *self = builder.build(self.value.clone());
        Ok(())
    }
}

impl<D1: Dimension + 'static, D2: Dimension + 'static> AddAssign<&Variable<D2>> for Variable<D1> {
    fn add_assign(&mut self, rhs: &Variable<D2>) -> Result<()> {
        self.add_assign(rhs.clone())
    }
}

impl Dot<Self> for Variable2 {
    type Output = Result<Self>;
    fn dot(&self, rhs: &Self) -> Result<Self> {
        let lhs = self;
        let mut builder = Self::builder();
        if let Some(node) = lhs.node() {
            let rhs = rhs.value().clone();
            builder.edge(node, move |output_grad| {
                output_grad.dot(&rhs.t()).map(Into::into)
            });
        }
        if let Some(node) = rhs.node() {
            let lhs = lhs.value().clone();
            builder.edge(node, move |output_grad| {
                lhs.t().dot(&output_grad).map(Into::into)
            });
        }
        let value = lhs.value().dot(rhs.value())?.into();
        Ok(builder.build(value))
    }
}

pub struct ParameterBase<S: ScalarData, D: Dimension> {
    value: ScalarTensorBase<S, D>,
    grad: Option<Arc<RwLock<Option<ScalarArcTensorD>>>>,
    state: State,
}

pub type Parameter<D> = ParameterBase<ScalarArcBufferRepr, D>;
pub type Parameter1 = Parameter<Ix1>;
pub type Parameter2 = Parameter<Ix2>;

pub type ParameterViewMut<'a, D> = ParameterBase<ScalarSliceMutRepr<'a>, D>;
pub type ParameterViewMut1<'a> = ParameterViewMut<'a, Ix1>;
pub type ParameterViewMut2<'a> = ParameterViewMut<'a, Ix2>;
pub type ParameterViewMutD<'a> = ParameterViewMut<'a, IxDyn>;

impl<S: ScalarData, D: Dimension> ParameterBase<S, D> {
    pub fn value(&self) -> &ScalarTensorBase<S, D> {
        &self.value
    }
    pub fn value_view_mut(&mut self) -> ScalarTensorViewMut<D>
    where
        S: ScalarDataMut,
    {
        self.value.view_mut()
    }
    pub fn grad(&self) -> Option<ScalarArcTensor<D>> {
        Some(
            self.grad
                .as_ref()?
                .read()
                .clone()?
                .into_dimensionality()
                .unwrap(),
        )
    }
    pub fn set_training(&mut self, training: bool) {
        if training && self.grad.is_none() {
            self.grad.replace(Arc::new(RwLock::default()));
        } else if !training {
            self.grad = None;
        }
    }
    pub fn state_mut(&mut self) -> StateMut {
        todo!()
    }
    pub fn into_dimensionality<D2>(self) -> Result<ParameterBase<S, D2>, ShapeError>
    where
        D2: Dimension,
    {
        Ok(ParameterBase {
            value: self.value.into_dimensionality()?,
            grad: self.grad.clone(),
            state: State::default(),
        })
    }
    pub fn into_dyn(self) -> ParameterBase<S, IxDyn> {
        ParameterBase {
            value: self.value.into_dyn(),
            grad: self.grad.clone(),
            state: State::default(),
        }
    }
}

impl<D: Dimension> Parameter<D> {
    pub fn to_variable(&self) -> Variable<D> {
        let value = self.value.clone();
        let node = self.grad.as_ref().map(|grad| {
            Node::new(
                value.device(),
                value.raw_dim().into_dyn(),
                value.scalar_type(),
                grad.clone(),
                Vec::new(),
            )
        });
        Variable { value, node }
    }
    pub fn make_view_mut(&mut self) -> Result<ParameterViewMut<D>> {
        Ok(ParameterViewMut {
            value: self.value.make_view_mut()?,
            grad: self.grad.clone(),
            state: State::default(),
        })
    }
    pub fn to_device_mut(&mut self, device: Device) -> Result<()> {
        if device == self.value.device() {
            return Ok(());
        }
        self.value.to_device_mut(device)?;
        todo!()
    }
}

impl<T: Scalar, D: Dimension> From<Tensor<T, D>> for Parameter<D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        Self::from(ScalarArcTensor::from(tensor))
    }
}

impl<T: Scalar, D: Dimension> From<ArcTensor<T, D>> for Parameter<D> {
    fn from(tensor: ArcTensor<T, D>) -> Self {
        Self::from(ScalarArcTensor::from(tensor))
    }
}

impl<D: Dimension> From<ScalarTensor<D>> for Parameter<D> {
    fn from(tensor: ScalarTensor<D>) -> Self {
        Self::from(ScalarArcTensor::from(tensor))
    }
}

impl<D: Dimension> From<ScalarArcTensor<D>> for Parameter<D> {
    fn from(tensor: ScalarArcTensor<D>) -> Self {
        Self {
            value: tensor,
            grad: None,
            state: State::default(),
        }
    }
}
