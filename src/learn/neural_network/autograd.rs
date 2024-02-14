use super::{
    layer::Forward,
    optimizer::{State as OptimizerState, Value as OptimizerValue},
};
#[cfg(doc)]
use crate::{learn::neural_network::optimizer::Optimizer, tensor::TensorBase};
use crate::{
    ops::AddAssign,
    tensor::{
        ArcTensor, CowTensor, ScalarArcTensor, ScalarArcTensorD, ScalarTensor, ScalarTensorBase,
        ScalarTensorViewMut, Tensor, TensorView,
    },
};
use anyhow::{bail, Error, Result};
use dry::macro_wrap;
use half::{bf16, f16};
use krnl::{
    buffer::{ScalarArcBufferRepr, ScalarData, ScalarDataMut, ScalarDataOwned, ScalarSliceMutRepr},
    device::Device,
    scalar::{Scalar, ScalarType},
};
use ndarray::{
    linalg::Dot, Axis, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
    ShapeError,
};
use parking_lot::{Mutex, RwLock};
use paste::paste;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    any::TypeId,
    collections::VecDeque,
    fmt::{self, Debug},
    marker::PhantomData,
    sync::{Arc, Weak},
};

/// Builders.
pub mod builder {
    use super::*;

    /// VariableBuilder.
    ///
    ///```no_run
    /// # use anyhow::Result;
    /// # use autograph::{tensor::ScalarArcTensor2, learn::neural_network::autograd::{Variable, Variable2}};
    /// # let input: Variable2 = todo!();
    /// let mut builder = Variable::builder();
    /// if let Some(node) = input.node() {
    ///     // Add an edge computing the input gradient from the output gradient.
    ///     builder.edge(node, |output_grad: ScalarArcTensor2| -> Result<ScalarArcTensor2> { todo!() });
    /// }
    /// let output_value: ScalarArcTensor2 = todo!();
    /// # let _ = {
    /// builder.build(output_value)
    /// # };
    ///```
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
                _m: PhantomData,
            }
        }
        /// Adds a node.
        ///
        /// Ensures a node is created even if edges are not added. May be useful for testing or for
        /// connecting backward passes together.
        pub fn node(mut self) -> Self {
            if self.grad.is_none() {
                self.grad.replace(Arc::new(RwLock::default()));
            }
            self
        }
        /// Adds an edge.
        ///
        /// During the backward pass, for each edge to `node`, `f` computes the gradient of `node`
        /// given the gradient of `self`.
        /// When multiple edges compute the same gradient, they are added together.
        /// Once there are no more edges needed to compute a gradient for a node, its edges can
        /// be computed.
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
        /// Builds the variable with `value`.
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

/// Node.
///
/// Nodes store gradients and can be connected via [`VariableBuilder::edge()`] to
/// form a graph that is traversed in [`.backward()`].
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
            _m: PhantomData,
        }
    }
    /// The gradient.
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
    /// Executes the backward pass.
    pub fn backward(&self) -> Result<()> {
        self.backward_grad(
            ScalarArcTensor::ones(
                self.inner.device.clone(),
                self.inner.dim.slice(),
                self.inner.scalar_type,
            )?
            .into_dimensionality::<D>()
            .map_err(Error::msg)?,
        )
    }
    /// Executes the backward pass with `grad`.
    pub fn backward_grad(&self, grad: ScalarArcTensor<D>) -> Result<()> {
        {
            let mut guard = self.inner.grad.write();
            if guard.is_some() {
                return Ok(());
            }
            guard.replace(grad.into_dyn());
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
    fn into_dyn(self) -> Node<IxDyn> {
        Node {
            inner: self.inner,
            _m: PhantomData,
        }
    }
    fn into_dimensionality<D2: Dimension>(self) -> Node<D2> {
        Node {
            inner: self.inner,
            _m: PhantomData,
        }
    }
}

/// Variable.
///
/// Variables are tensors with an optional [`Node`] that stores a gradient. Numerical operations
/// on variables with a node create a graph of edges that is traversed during the backward pass
/// to compute the gradients.
///
/// Variables can be created from tensors via [`From`].
/// Use [`builder()`](Variable::builder) to create a Variable as a function of another variable.
#[derive(Clone, Debug)]
pub struct Variable<D: Dimension> {
    value: ScalarArcTensor<D>,
    node: Option<Node<D>>,
}

/// Variable with 1 element
pub type Variable0 = Variable<Ix0>;
/// Variable with 1 dimension
pub type Variable1 = Variable<Ix1>;
/// Variable with 2 dimensions
pub type Variable2 = Variable<Ix2>;
/// Variable with 3 dimensions
pub type Variable3 = Variable<Ix3>;
/// Variable with 4 dimensions
pub type Variable4 = Variable<Ix4>;
/// Variable with 5 dimensions
pub type Variable5 = Variable<Ix5>;
/// Variable with 6 dimensions
pub type Variable6 = Variable<Ix6>;
/// Variable with dynamic dimensions
pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> Variable<D> {
    /// A `VariableBuilder` for creating nodes and edges.
    pub fn builder() -> VariableBuilder<D> {
        VariableBuilder::new()
    }
    /// The value of the variable.
    pub fn value(&self) -> &ScalarArcTensor<D> {
        &self.value
    }
    /// Converts the variable into a tensor.
    pub fn into_value(self) -> ScalarArcTensor<D> {
        self.value
    }
    /// The node.
    pub fn node(&self) -> Option<&Node<D>> {
        self.node.as_ref()
    }
    /// Maps the variable with `F`.
    ///
    /// Shortcut for `f.forward(self)`. This allows chaining methods together.
    pub fn forward<F: Forward<Self>>(self, f: &F) -> Result<F::Output> {
        f.forward(self)
    }
    /// The device.
    pub fn device(&self) -> Device {
        self.value.device()
    }
    /// The scalar_type.
    pub fn scalar_type(&self) -> ScalarType {
        self.value.scalar_type()
    }
    /// The shape.
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }
    /// The dim in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.value.dim()
    }
    /// The dim.
    pub fn raw_dim(&self) -> D {
        self.value.raw_dim()
    }
    /// Converts into dimensionality `D2`.
    ///
    /// See [`TensorBase::into_dimensionality`].
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
    /// Converts into a dynamic dimensional variable.
    pub fn into_dyn(self) -> VariableD {
        Variable {
            value: self.value.into_dyn(),
            node: self.node.map(Node::into_dyn),
        }
    }
}

impl Variable0 {
    /// Executes the backward pass.
    ///
    /// See [`Node::backward`].
    pub fn backward(&self) -> Result<()> {
        if let Some(node) = self.node.as_ref() {
            node.backward()?;
        }
        Ok(())
    }
}

impl<D: Dimension + 'static> Variable<D> {
    /// Converts into `shape`.
    ///
    /// See [`TensorBase::into_shape`].
    pub fn into_shape<E>(self, shape: E) -> Result<Variable<E::Dim>, ShapeError>
    where
        E: IntoDimension,
        E::Dim: 'static,
    {
        let dim = self.raw_dim();
        let mut builder = Variable::builder();
        if let Some(node) = self.node() {
            builder.edge(node, |output_grad| {
                if let Ok(input_grad) = output_grad.clone().into_shape(dim.clone()) {
                    Ok(input_grad)
                } else {
                    Ok(output_grad
                        .to_standard_layout_shared()?
                        .into_shape(dim)
                        .unwrap())
                }
            })
        }
        Ok(builder.build(self.value.into_shape(shape)?))
    }
    /// Flattens the variable into 2 dimensions.
    ///
    /// See [`TensorBase::flatten()`].
    pub fn flatten(self) -> Result<Variable2, ShapeError> {
        let dim = crate::tensor::flatten(self.shape());
        self.into_shape(dim)
    }
    /// Reverses (transposes) the axes of the variable.
    ///
    /// See [`TensorBase::reversed_axes()`].
    pub fn reversed_axes(self) -> Self {
        let mut builder = Self::builder();
        if let Some(node) = self.node() {
            builder.edge(node, |output_grad| Ok(output_grad.reversed_axes()));
        }
        builder.build(self.value.reversed_axes())
    }
    /// Permute the axes of the tensor.
    ///
    /// See [`TensorBase::permuted_axes()`].
    pub fn permuted_axes<A>(self, axes: A) -> Self
    where
        A: IntoDimension<Dim = D>,
    {
        let mut builder = Self::builder();
        let axes = axes.into_dimension();
        let mut input_axes = D::zeros(axes.ndim());
        for (i, a) in axes.slice().iter().copied().enumerate() {
            input_axes[a] = i;
        }
        if let Some(node) = self.node() {
            builder.edge(node, move |output_grad| {
                Ok(output_grad.permuted_axes(input_axes))
            })
        }
        builder.build(self.into_value().permuted_axes(axes))
    }
    #[doc(hidden)]
    /// Converts to standard layout.
    ///
    /// **Errors**
    ///
    /// See [`ArcTensor::to_standard_layout_shared()`].
    pub fn to_standard_layout(&self) -> Result<Self> {
        Ok(Self {
            value: self.value.to_standard_layout_shared()?,
            ..self.clone()
        })
    }
    /// Transposes the variable.
    pub fn t(&self) -> Self {
        self.clone().reversed_axes()
    }
    /// Attempts to broadcast the variable into `dim`.
    ///
    /// See [`TensorBase::broadcast()`].
    pub fn broadcast<E>(&self, dim: E) -> Option<Variable<E::Dim>>
    where
        E: IntoDimension,
    {
        if self.node.is_none() {
            return self.value.broadcast_shared(dim).map(Variable::from);
        }
        let dim = dim.into_dimension();
        if self.shape() == dim.slice() {
            return Some(self.clone().into_dimensionality().unwrap());
        }
        let output = self.value.broadcast_shared(dim)?;
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
        Some(builder.build(output))
    }
}

fn broadcast_backward<T: Scalar, D1: Dimension, D2: Dimension>(
    input: TensorView<T, D1>,
    output_dim: D2,
) -> Result<Tensor<T, D2>> {
    // idea
    // strip leading 1's from output_dim
    // pack leading dims of input into batch dim
    // sum along batch dim if necessary
    // sum along each interior dimension if necessary
    //
    // reshape into output dim

    let output_dim_stripped = {
        let strip_dims = output_dim.slice().iter().take_while(|x| **x == 1).count();
        let mut output_dim_stripped = IxDyn::zeros(output_dim.ndim() - strip_dims);
        output_dim_stripped
            .slice_mut()
            .copy_from_slice(&output_dim.slice()[strip_dims..]);
        output_dim_stripped
    };
    let batch_dims = input.ndim().saturating_sub(output_dim_stripped.ndim());
    let input_dim_packed = if batch_dims > 0 {
        let batch_size = input.shape()[0..batch_dims].iter().product();
        let non_batch_dims = &input.shape()[batch_dims..];
        let mut input_dim_packed = IxDyn::zeros(1 + non_batch_dims.len());
        input_dim_packed[0] = batch_size;
        input_dim_packed.slice_mut()[1..].copy_from_slice(non_batch_dims);
        input_dim_packed
    } else {
        input.shape().into_dimension()
    };
    let mut output = CowTensor::from(input.into_shape(input_dim_packed.clone()).unwrap());
    let output_batch_dim = if batch_dims > 0 { Some(1) } else { None };
    for (axis, (x, y)) in std::iter::zip(
        input_dim_packed.slice().iter().copied(),
        output_batch_dim
            .into_iter()
            .chain(output_dim_stripped.slice().iter().copied()),
    )
    .enumerate()
    {
        if x != y {
            let mut tmp_dim = output.raw_dim();
            tmp_dim[axis] = 1;
            output = output
                .sum_axis(Axis(axis))?
                .into_shape(tmp_dim)
                .unwrap()
                .into();
        }
    }
    Ok(output.into_owned()?.into_shape(output_dim).unwrap())
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

/// Parameter.
///
/// Parameter values are updated during training by the [`Optimizer`]. A Parameter
/// can be converted to a [`Variable`] via [`.to_variable()`](Parameter::to_variable),
/// which allows it to be used in operations.
/// During training, [`.set_training(true)`](Parameter::set_training) ensures that
/// the variable created from this parameter has a [`Node`].
/// A parameter stores the [`OptimizerState`] which can be updated during training
/// in [`Optimizer::update`]. Training progress may be saved by serializing with [`serde`].
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "D: Serialize",
        deserialize = "S: ScalarDataOwned, D: Deserialize<'de>"
    ))
)]
pub struct ParameterBase<S: ScalarData, D: Dimension> {
    value: ScalarTensorBase<S, D>,
    #[cfg_attr(feature = "serde", serde(skip))]
    grad: Option<Arc<RwLock<Option<ScalarArcTensorD>>>>,
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "OptimState::is_none", default)
    )]
    optim_state: OptimState<'static>,
}

/// Parameter with a [`ScalarArcTensor`] value.
///
/// See [`ParameterBase`].
pub type Parameter<D> = ParameterBase<ScalarArcBufferRepr, D>;
/// Parameter with 1 element.
pub type Parameter0 = Parameter<Ix0>;
/// Parameter with 1 dimension.
pub type Parameter1 = Parameter<Ix1>;
/// Parameter with 2 dimensions.
pub type Parameter2 = Parameter<Ix2>;
/// Parameter with 3 dimensions.
pub type Parameter3 = Parameter<Ix3>;
/// Parameter with 4 dimensions.
pub type Parameter4 = Parameter<Ix4>;
/// Parameter with 5 dimensions.
pub type Parameter5 = Parameter<Ix5>;
/// Parameter with 6 dimensions.
pub type Parameter6 = Parameter<Ix6>;
/// Parameter with dynamic dimensions.
pub type ParameterD = Parameter<IxDyn>;

/// Mutable parameter view.
///
/// See [`ParameterBase`].
pub type ParameterViewMut<'a, D> = ParameterBase<ScalarSliceMutRepr<'a>, D>;
/// Mutable parameter view with 1 element.
pub type ParameterViewMut0<'a> = ParameterViewMut<'a, Ix0>;
/// Mutable parameter view with 1 dimension.
pub type ParameterViewMut1<'a> = ParameterViewMut<'a, Ix1>;
/// Mutable parameter view with 2 dimensions.
pub type ParameterViewMut2<'a> = ParameterViewMut<'a, Ix2>;
/// Mutable parameter view with 3 dimensions.
pub type ParameterViewMut3<'a> = ParameterViewMut<'a, Ix3>;
/// Mutable parameter view with 4 dimensions.
pub type ParameterViewMut4<'a> = ParameterViewMut<'a, Ix4>;
/// Mutable parameter view with 5 dimensions.
pub type ParameterViewMut5<'a> = ParameterViewMut<'a, Ix5>;
/// Mutable parameter view with 6 dimensions.
pub type ParameterViewMut6<'a> = ParameterViewMut<'a, Ix6>;
/// Mutable parameter view with dynamic dimensions.
pub type ParameterViewMutD<'a> = ParameterViewMut<'a, IxDyn>;

impl<S: ScalarData, D: Dimension> ParameterBase<S, D> {
    /// The value of the parameter.
    pub fn value(&self) -> &ScalarTensorBase<S, D> {
        &self.value
    }
    /// Borrows the value of the parameter as a mutable tensor view.
    pub fn value_view_mut(&mut self) -> ScalarTensorViewMut<D>
    where
        S: ScalarDataMut,
    {
        self.value.view_mut()
    }
    /// The gradient of the parameter.
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
    /// The device.
    pub fn device(&self) -> Device {
        self.value.device()
    }
    /// The scalar_type.
    pub fn scalar_type(&self) -> ScalarType {
        self.value.scalar_type()
    }
    /// The shape.
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }
    /// The dim in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.value.dim()
    }
    /// The dim.
    pub fn raw_dim(&self) -> D {
        self.value.raw_dim()
    }
    /// Enables / disables training.
    ///
    /// If `training`, ensures that when the parameter is converted to a [`Variable`],
    /// it will have a [`Node`] for computing a gradient.
    /// If `training` is false, discards any gradient that has been computed.
    pub fn set_training(&mut self, training: bool) {
        if training && self.grad.is_none() {
            self.grad.replace(Arc::new(RwLock::default()));
        } else if !training {
            self.grad = None;
        }
    }
    /// Borrows the optimizer state.
    pub fn optimizer_state(&self) -> Option<&OptimizerState> {
        self.optim_state.get()
    }
    /// Borrows the optimizer state mutably.
    pub fn optimzer_state_mut(&mut self) -> Option<&mut OptimizerState> {
        self.optim_state.get_mut()
    }
    /// Borrows the value and optimizer state mutably.
    pub fn value_view_optimizer_state_mut(
        &mut self,
    ) -> (ScalarTensorViewMut<D>, Option<&mut OptimizerState>)
    where
        S: ScalarDataMut,
    {
        (self.value.view_mut(), self.optim_state.get_mut())
    }
    /// Initializes the optimizer state.
    ///
    /// The `name` should be the name of the optimizer, for example "SGD".
    /// The `id` is the [`TypeId`] of the optimizer.
    pub fn init_optimizer_state(
        &mut self,
        name: impl Into<String>,
        id: TypeId,
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
    /// Converts into dimensionality `D2`.
    ///
    /// See [`TensorBase::into_dimensionality`].
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
    /// Converts into a dynamic dimensional parameter.
    pub fn into_dyn(self) -> ParameterBase<S, IxDyn> {
        ParameterBase {
            value: self.value.into_dyn(),
            grad: self.grad.clone(),
            optim_state: self.optim_state,
        }
    }
}

impl<D: Dimension> Parameter<D> {
    /// Converts to a `Variable`.
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
    /// Makes a mutable parameter view.
    ///
    /// Copies the value and optimizer state if they are not exclusive.
    ///
    /// See [`TensorBase::make_view_mut`].
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
    /// Moves the parameter into `device`.
    ///
    /// See [`.to_device_mut()`](Self::to_device_mut).
    pub fn into_device(self, device: Device) -> Result<Self> {
        let mut parameter = self.clone();
        parameter.to_device_mut(device)?;
        Ok(parameter)
    }
    /// Transfers the parameter to `device` if necessary.
    pub fn to_device_mut(&mut self, device: Device) -> Result<()> {
        self.value.to_device_mut(device.clone())?;
        if let Some(grad) = self.grad.as_mut() {
            let value = if let Some(value) = grad.read().clone() {
                Some(value.clone().into_device_shared(device.clone())?)
            } else {
                None
            };
            *grad = Arc::new(RwLock::new(value));
        }
        self.optim_state.to_device_mut(device)?;
        Ok(())
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
    #[cfg(feature = "serde")]
    fn is_none(&self) -> bool {
        match self {
            Self::State(x) => x.is_none(),
            Self::StateMut(x) => x.is_none(),
        }
    }
    fn as_ref(&self) -> &Option<Arc<OptimizerState>> {
        match self {
            Self::State(x) => x,
            Self::StateMut(x) => x,
        }
    }
    fn as_mut(&mut self) -> &mut Option<Arc<OptimizerState>> {
        match self {
            Self::State(x) => x,
            Self::StateMut(x) => x,
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
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        let inner = self.as_mut();
        if let Some(state) = inner.as_mut() {
            if Arc::get_mut(state).is_none() {
                *state = Arc::new(state.as_ref().to_device(device)?);
            }
        }
        Ok(())
    }
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

#[cfg(feature = "serde")]
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

#[cfg(feature = "serde")]
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
