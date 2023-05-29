use super::{
    layer::Forward,
    optimizer::{OptimizerId, State as OptimizerState, Value as OptimizerValue},
};
use crate::{
    buffer::{ScalarArcBufferRepr, ScalarData, ScalarDataMut, ScalarDataOwned, ScalarSliceMutRepr},
    device::Device,
    ops::AddAssign,
    scalar::{Scalar, ScalarType},
    tensor::{
        ArcTensor, ScalarArcTensor, ScalarArcTensorD, ScalarTensor, ScalarTensorBase,
        ScalarTensorView, ScalarTensorViewMut, Tensor, TensorView,
    },
};
use anyhow::{bail, Error, Result};
use dry::macro_wrap;
use half::{bf16, f16};
#[cfg(feature = "device")]
use krnl::macros::module;
use ndarray::{
    linalg::Dot, Array, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix4, IxDyn, ShapeError,
};
use num_traits::ToPrimitive;
use parking_lot::{Mutex, RwLock};
use paste::paste;
//use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::VecDeque,
    fmt::{self, Debug},
    marker::PhantomData,
    sync::{Arc, Weak},
};

pub mod builder {
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
            let mut output_grad_lock = Some(self.grad.clone().unwrap());
            let node = node.inner.clone();
            let mut input_grad_lock = Arc::downgrade(&node.grad);
            let device = node.device.clone();
            let dim = node.dim.clone();
            let scalar_type = node.scalar_type;
            let name = std::any::type_name::<F>();
            let mut f = Some(f);
            let op = Box::new(move || {
                let input_grad_lock = Weak::upgrade(&std::mem::take(&mut input_grad_lock));
                if let Some((f, (input_grad_lock, output_grad_lock))) =
                    f.take().zip(input_grad_lock.zip(output_grad_lock.take()))
                {
                    let grad = output_grad_lock
                        .read()
                        .clone()
                        .unwrap()
                        .into_dimensionality()
                        .unwrap();
                    std::mem::drop(output_grad_lock);
                    let grad = (f)(grad)?;
                    assert_eq!(grad.device(), device, "{name}");
                    assert_eq!(grad.shape(), dim.slice(), "{name}");
                    assert_eq!(grad.scalar_type(), scalar_type, "{name}");
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
            .field("name", &self.name)
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
            edges: _,
        } = self.inner.as_ref();
        {
            let mut guard = grad.write();
            if guard.is_some() {
                return Ok(());
            }
            guard.replace(ScalarArcTensor::ones(
                device.clone(),
                dim.clone(),
                *scalar_type,
            )?);
        }
        let mut queue = VecDeque::new();
        queue.push_back(self.inner.clone());
        while let Some(node) = queue.pop_front() {
            let edges = std::mem::take(&mut *node.edges.lock());
            std::mem::drop(node);
            for mut edge in edges {
                (edge.op)()?;
                let node = edge.node;
                if node.ready() {
                    queue.push_back(node.clone())
                }
            }
        }
        Ok(())
    }
    /*pub fn backward_par(&self) -> Result<()> {
        let NodeInner {
            device,
            dim,
            scalar_type,
            grad,
            edges: _,
        } = self.inner.as_ref();
        {
            let mut guard = grad.write();
            if guard.is_some() {
                return Ok(());
            }
            guard.replace(ScalarArcTensor::ones(
                device.clone(),
                dim.clone(),
                *scalar_type,
            )?);
        }
        let start = Instant::now();
        let result = Arc::new(Mutex::new(Ok(())));
        let (sender, receiver) = crossbeam_channel::bounded(16);
        let edges = std::mem::take(&mut *self.inner.edges.lock());
        let counter = Arc::new(AtomicUsize::new(edges.len()));
        for edge in edges {
            sender.send(edge).unwrap();
        }
        let result2 = result.clone();
        rayon::spawn_broadcast(move |_| {
            let mut local_edge = Option::<EdgeInner>::None;
            while counter.load(Ordering::SeqCst) > 0 {
                let mut edge = if let Some(edge) = local_edge.take() {
                    edge
                } else if let Ok(edge) = receiver.try_recv() {
                    edge
                } else {
                    continue;
                };
                if let Err(e) = (edge.op)() {
                    let mut result = result2.lock();
                    if result.is_ok() {
                        *result = Err(e);
                    }
                    return;
                }
                let node = edge.node;
                if node.ready() {
                    let edges = std::mem::take(&mut *node.edges.lock());
                    counter.fetch_add(edges.len(), Ordering::SeqCst);
                    std::mem::drop(node);
                    let mut edges = edges.into_iter();
                    local_edge = edges.next();
                    for edge in edges {
                        sender.send(edge);
                    }
                }
                counter.fetch_sub(1, Ordering::SeqCst);
            }
        });
        while Arc::strong_count(&result) > 1 {
            std::thread::yield_now();
        }
        dbg!(start.elapsed());
        Arc::try_unwrap(result).ok().unwrap().into_inner()
    }*/
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

pub type Variable0 = Variable<Ix0>;
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
    pub fn device(&self) -> Device {
        self.value.device()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.value.scalar_type()
    }
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }
    pub fn dim(&self) -> D::Pattern {
        self.value.dim()
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

impl Variable0 {
    pub fn backward(&self) -> Result<()> {
        if let Some(node) = self.node.as_ref() {
            node.backward()?;
        }
        Ok(())
    }
    /*pub fn backward_par(&self) -> Result<()> {
        if let Some(node) = self.node.as_ref() {
            node.backward_par()?;
        }
        Ok(())
    }*/
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
            let input_dim = self.raw_dim();
            builder.edge(node, move |output_grad| {
                macro_wrap!(paste! { match output_grad.scalar_type() {
                    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                        ScalarType::[<$T:upper>] => Ok(broadcast_backward::<$T, E::Dim, D>(output_grad.view().try_into().unwrap(), input_dim)?.into_scalar_tensor().into()),
                    })
                    _ => bail!("Broadcast backward {:?} unimplemented!", output_grad.scalar_type()),
                }})
            })
        }
        Some(builder.build(self.value.broadcast_shared(dim)?))
    }
}

fn broadcast_backward<T: Scalar, D1: Dimension, D2: Dimension>(
    output_grad: TensorView<T, D1>,
    input_dim: D2,
) -> Result<Tensor<T, D2>> {
    if let Some(output_grad) = output_grad.as_array() {
        let mut output_grad_iter = output_grad.iter().copied();
        let mut input_grad: Vec<T> = output_grad_iter.by_ref().take(input_dim.size()).collect();
        let mut output_grad_iter = output_grad_iter.peekable();
        while output_grad_iter.peek().is_some() {
            for (dy, dx) in output_grad_iter.by_ref().zip(input_grad.iter_mut()) {
                *dx += dy;
            }
        }
        return Ok(Tensor::from(
            Array::from(input_grad).into_shape(input_dim).unwrap(),
        ));
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let dy = ScalarTensorView::from(output_grad)
            .try_into_tensor_view::<f32>()
            .unwrap();
        let n = dy.len().to_u32().unwrap();
        assert!(n != 0);
        let batch_size = input_dim.size().to_u32().unwrap() / n;
        let mut dx = unsafe { Tensor::uninit(dy.device(), input_dim)? };
        kernels::broadcast_backward_f32::builder()?
            .build(dx.device())?
            .with_global_threads(batch_size)
            .dispatch(n, dy.as_slice().unwrap(), dx.as_slice_mut().unwrap())?;
        Ok(dx.cast_into().unwrap())
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

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "D: Serialize",
    deserialize = "S: ScalarDataOwned, D: Deserialize<'de>"
))]
pub struct ParameterBase<S: ScalarData, D: Dimension> {
    value: ScalarTensorBase<S, D>,
    #[serde(skip)]
    grad: Option<Arc<RwLock<Option<ScalarArcTensorD>>>>,
    #[serde(skip_serializing_if = "OptimState::is_none", default)]
    optim_state: OptimState<'static>,
}

pub type Parameter<D> = ParameterBase<ScalarArcBufferRepr, D>;
pub type Parameter1 = Parameter<Ix1>;
pub type Parameter2 = Parameter<Ix2>;
pub type Parameter4 = Parameter<Ix4>;

pub type ParameterViewMut<'a, D> = ParameterBase<ScalarSliceMutRepr<'a>, D>;
pub type ParameterViewMut1<'a> = ParameterViewMut<'a, Ix1>;
pub type ParameterViewMut2<'a> = ParameterViewMut<'a, Ix2>;
pub type ParameterViewMut4<'a> = ParameterViewMut<'a, Ix4>;
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
    pub fn device(&self) -> Device {
        self.value.device()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.value.scalar_type()
    }
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }
    pub fn dim(&self) -> D::Pattern {
        self.value.dim()
    }
    pub fn raw_dim(&self) -> D {
        self.value.raw_dim()
    }
    pub fn set_training(&mut self, training: bool) {
        if training && self.grad.is_none() {
            self.grad.replace(Arc::new(RwLock::default()));
        } else if !training {
            self.grad = None;
        }
    }
    pub fn optimizer_state(&self) -> Option<&OptimizerState> {
        self.optim_state.get()
    }
    pub fn optimzer_state_mut(&mut self) -> Option<&mut OptimizerState> {
        self.optim_state.get_mut()
    }
    pub fn value_view_optimizer_state_mut(
        &mut self,
    ) -> (ScalarTensorViewMut<D>, Option<&mut OptimizerState>)
    where
        S: ScalarDataMut,
    {
        (self.value.view_mut(), self.optim_state.get_mut())
    }
    pub fn init_optimizer_state(
        &mut self,
        name: impl Into<String>,
        id: OptimizerId,
        key_values: impl IntoIterator<Item = (String, OptimizerValue)>,
    ) -> Result<()> {
        let state = OptimizerState::new(
            self.device(),
            self.scalar_type(),
            name.into(),
            id,
            key_values.into_iter().collect(),
        )?;
        self.optim_state.make_mut()?.replace(Arc::new(state));
        Ok(())
    }
    pub fn into_dimensionality<D2>(self) -> Result<ParameterBase<S, D2>, ShapeError>
    where
        D2: Dimension,
    {
        Ok(ParameterBase {
            value: self.value.into_dimensionality()?,
            grad: self.grad.clone(),
            optim_state: self.optim_state,
        })
    }
    pub fn into_dyn(self) -> ParameterBase<S, IxDyn> {
        ParameterBase {
            value: self.value.into_dyn(),
            grad: self.grad.clone(),
            optim_state: self.optim_state,
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
        let value = self.value.make_view_mut()?;
        let grad = self.grad.clone();
        let optim_state = self.optim_state.make_mut()?;
        let optim_state = OptimState::StateMut(unsafe {
            let optim_state_ptr = optim_state as *mut Option<Arc<OptimizerState>>;
            &mut *optim_state_ptr as &mut Option<Arc<OptimizerState>>
        });
        Ok(ParameterViewMut {
            value,
            grad,
            optim_state,
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
            optim_state: OptimState::default(),
        }
    }
}

impl<S: ScalarData, D: Dimension> Debug for ParameterBase<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ParameterBase")
            .field("value:", &self.value)
            .field("grad", &self.grad)
            .field("optim_state", &self.optim_state)
            .finish()
    }
}

#[derive(Debug)]
enum OptimState<'a> {
    State(Option<Arc<OptimizerState>>),
    StateMut(&'a mut Option<Arc<OptimizerState>>),
}

impl OptimState<'_> {
    fn is_none(&self) -> bool {
        match self {
            Self::State(x) => x.is_none(),
            Self::StateMut(x) => x.is_none(),
        }
    }
    fn as_ref(&self) -> &Option<Arc<OptimizerState>> {
        match self {
            Self::State(x) => x,
            Self::StateMut(x) => &*x,
        }
    }
    fn as_mut(&mut self) -> &mut Option<Arc<OptimizerState>> {
        match self {
            Self::State(x) => x,
            Self::StateMut(x) => &mut *x,
        }
    }
    fn get(&self) -> Option<&OptimizerState> {
        self.as_ref().as_deref()
    }
    fn get_mut(&mut self) -> Option<&mut OptimizerState> {
        if let Some(state) = self.as_mut() {
            Arc::get_mut(state)
        } else {
            None
        }
    }
    fn make_mut(&mut self) -> Result<&mut Option<Arc<OptimizerState>>> {
        let inner = self.as_mut();
        if let Some(state) = inner.as_mut() {
            if Arc::get_mut(state).is_none() {
                *state = Arc::new(state.as_ref().to_owned()?);
            }
        }
        Ok(self.as_mut())
    }
    /*fn to_device_mut(&mut self, device: Device) -> Result<()> {
        todo!()
    }*/
}

impl Default for OptimState<'_> {
    fn default() -> Self {
        Self::State(None)
    }
}

impl Clone for OptimState<'_> {
    fn clone(&self) -> Self {
        match self {
            Self::State(x) => Self::State(x.clone()),
            Self::StateMut(_) => unreachable!(),
        }
    }
}

impl Serialize for OptimState<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(state) = self.get() {
            state.serialize(serializer)
        } else {
            serializer.serialize_none()
        }
    }
}

impl<'de> Deserialize<'de> for OptimState<'_> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::State(Some(Arc::new(OptimizerState::deserialize(
            deserializer,
        )?))))
    }
}

#[cfg(feature = "device")]
#[module]
pub mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    #[kernel(threads(256))]
    pub fn broadcast_backward_f32(n: u32, #[global] dy: Slice<f32>, #[item] dx: &mut f32) {
        let idx = kernel.item_id();
        *dx = 0f32;
        for i in 0..n {
            *dx += dy[(idx * n + i) as usize];
        }
    }
}
