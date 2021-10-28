use crate::{
    buffer::float::FloatBuffer,
    device::Device,
    glsl_shaders,
    linalg::{Dot, DotAcc, DotBias},
    ops::AddAssign,
    ops::{Col2Im, Im2Col, KernelArgs, KernelKind},
    result::Result,
    scalar::FloatType,
    tensor::float::{
        FloatArcTensor, FloatArcTensor2, FloatData, FloatTensor, FloatTensor2, FloatTensorBase,
        FloatTensorD, FloatTensorView0, FloatTensorView2, FloatTensorViewMut, FloatTensorViewMut1,
        FloatTensorViewMut2,
    },
    util::type_eq,
};
use anyhow::{anyhow, bail};
#[doc(hidden)]
pub use autograph_derive::*;
use ndarray::{Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, IxDyn, ShapeBuilder};
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::VecDeque,
    fmt::{self, Debug},
    sync::Arc,
};

mod sealed {
    /// Sealed base trait for [`Node`](super::Node).
    pub trait NodeBase {}
}
use sealed::NodeBase;

/// A reflection trait for backward ops.
///
/// [`Autograd`] can an should be [derived](autograph_derive). See [`Backward`].
pub trait Autograd: Send + Sync + 'static {
    /// A name for the the operation. The derive implementation uses the raw type name without path prefix or generics.
    fn name(&self) -> Cow<'static, str>;
    /// Returns the input gradients.
    fn grads(&self) -> Vec<GradientD>;
}

/// A trait for backward ops.
///
/// Backward is not called directly, instead it is provided via [`Variable::with_backward()`](VertexBase::backward()), and executed in [`Variable::backward()`](VertexBase::backward()), which runs the backward pass. Each variable gradient is computed before [`.backward()`](Backward::backward()) is called with that gradient to propagate the gradients through the graph. [`.backward()`](Backward::backward()) is called only once per [`Variable`] during the execution of the backward pass.
pub trait Backward: Autograd {
    /// Computes input gradients given the output gradient.
    ///
    /// **Errors**
    ///
    /// Returns an error if the operation cannot be performed. May return an error even if the input gradients are modified.
    fn backward(&self, output_grad: FloatTensorD) -> Result<()>;
}

struct BackwardOp {
    backward: Box<dyn Backward>,
    dim: IxDyn,
    strides: IxDyn,
}

impl Autograd for BackwardOp {
    fn name(&self) -> Cow<'static, str> {
        self.backward.name()
    }
    fn grads(&self) -> Vec<GradientD> {
        self.backward.grads()
    }
}

impl Backward for BackwardOp {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        let output_grad = unsafe {
            output_grad
                .with_raw_dim(self.dim.clone())
                .with_raw_strides(self.strides.clone())
        };
        self.backward.backward(output_grad)
    }
}

/// Marker trait for [`VertexBase`] nodes.
pub trait Node: Default + Clone + Send + Sync + 'static + NodeBase {
    #[doc(hidden)]
    fn is_variable(&self) -> bool {
        false
    }
    #[doc(hidden)]
    fn is_parameter(&self) -> bool {
        !self.is_variable()
    }
    #[doc(hidden)]
    fn into_vertex(self) -> VertexNode;
    #[doc(hidden)]
    fn try_into_variable(self) -> Result<VariableNode, Self> {
        Err(self)
    }
    #[doc(hidden)]
    fn as_variable(&self) -> Option<&VariableNode> {
        None
    }
    #[doc(hidden)]
    fn as_variable_mut(&mut self) -> Option<&mut VariableNode> {
        None
    }
}

/// A [`Variable`] node.
#[derive(Default, Clone)]
pub struct VariableNode {
    backward: Option<Arc<dyn Backward>>,
    training: bool,
}

impl NodeBase for VariableNode {}

impl Node for VariableNode {
    fn is_variable(&self) -> bool {
        true
    }
    fn into_vertex(self) -> VertexNode {
        self.into()
    }
    fn try_into_variable(self) -> Result<Self, Self> {
        Ok(self)
    }
    fn as_variable(&self) -> Option<&Self> {
        Some(self)
    }
    fn as_variable_mut(&mut self) -> Option<&mut Self> {
        Some(self)
    }
}

/// A [`Parameter`] node.
#[derive(Default, Clone)]
pub struct ParameterNode {}

impl NodeBase for ParameterNode {}

impl Node for ParameterNode {
    fn into_vertex(self) -> VertexNode {
        self.into()
    }
}

/// A variable or parameter node.
#[derive(Clone)]
pub enum VertexNode {
    /// VariableNode
    Variable(VariableNode),
    /// ParameterNode
    Parameter(ParameterNode),
}

impl Default for VertexNode {
    fn default() -> Self {
        Self::Variable(VariableNode::default())
    }
}

impl NodeBase for VertexNode {}

impl Node for VertexNode {
    fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }
    fn into_vertex(self) -> Self {
        self
    }
    fn try_into_variable(self) -> Result<VariableNode, Self> {
        match self {
            Self::Variable(node) => Ok(node),
            this => Err(this),
        }
    }
    fn as_variable(&self) -> Option<&VariableNode> {
        match self {
            Self::Variable(node) => Some(node),
            _ => None,
        }
    }
    fn as_variable_mut(&mut self) -> Option<&mut VariableNode> {
        match self {
            Self::Variable(node) => Some(node),
            _ => None,
        }
    }
}

impl From<VariableNode> for VertexNode {
    fn from(node: VariableNode) -> Self {
        Self::Variable(node)
    }
}

impl From<ParameterNode> for VertexNode {
    fn from(node: ParameterNode) -> Self {
        Self::Parameter(node)
    }
}

#[derive(Clone, Debug)]
struct FloatTensorDesc<D> {
    float_type: FloatType,
    device: Device,
    dim: D,
    strides: D,
}

impl<D: Dimension> FloatTensorDesc<D> {
    fn into_dyn(self) -> FloatTensorDesc<IxDyn> {
        FloatTensorDesc {
            float_type: self.float_type,
            device: self.device,
            dim: self.dim.into_dyn(),
            strides: self.strides.into_dyn(),
        }
    }
}

impl<S: FloatData, D: Dimension> From<&'_ FloatTensorBase<S, D>> for FloatTensorDesc<D> {
    fn from(tensor: &FloatTensorBase<S, D>) -> Self {
        Self {
            float_type: tensor.float_type(),
            device: tensor.device(),
            dim: tensor.raw_dim(),
            strides: tensor.raw_strides(),
        }
    }
}

/// A gradient.
#[derive(Clone)]
pub struct GradientBase<D: Dimension, N: Node> {
    desc: FloatTensorDesc<D>,
    grad: Arc<Mutex<Option<FloatBuffer>>>,
    node: N,
}

/// A gradient.
pub type Gradient<D> = GradientBase<D, VertexNode>;
/// A gradient with 2 dimensions.
pub type Gradient2 = Gradient<Ix2>;
/// A dynamically dimensional gradient.
pub type GradientD = Gradient<IxDyn>;

/// A Variable gradient.
pub type VariableGradient<D> = GradientBase<D, VariableNode>;
/// A dynamically dimensional variable gradient.
pub type VariableGradientD = VariableGradient<IxDyn>;

impl<D: Dimension, N: Node> GradientBase<D, N> {
    fn new(
        desc: impl Into<FloatTensorDesc<D>>,
        grad: Arc<Mutex<Option<FloatBuffer>>>,
        node: N,
    ) -> Self {
        Self {
            desc: desc.into(),
            grad,
            node,
        }
    }
    /// Converts into a dynamically dimensioned gradient.
    pub fn into_dyn(self) -> GradientBase<IxDyn, N> {
        GradientBase {
            desc: self.desc.into_dyn(),
            grad: self.grad,
            node: self.node,
        }
    }
    /// Converts into a [`Gradient`].
    pub fn into_gradient(self) -> Gradient<D> {
        Gradient {
            desc: self.desc,
            grad: self.grad,
            node: self.node.into_vertex(),
        }
    }
    /// Locks the gradient, returning a guard.
    ///
    /// The gradient is protected by a [`Mutex`], and the guard ensures exclusive access. Use this method in [`Backward::backward()`] to compute the gradients of the inputs / parameters of a function.
    pub fn lock(&self) -> GradientGuard<D> {
        GradientGuard {
            desc: self.desc.clone(),
            guard: self.grad.lock(),
        }
    }
    fn try_into_variable(self) -> Result<VariableGradient<D>, Self> {
        match self.node.try_into_variable() {
            Ok(node) => Ok(GradientBase {
                desc: self.desc,
                grad: self.grad,
                node,
            }),
            Err(node) => Err(GradientBase {
                desc: self.desc,
                grad: self.grad,
                node,
            }),
        }
    }
    /// The device of the gradient.
    pub fn device(&self) -> Device {
        self.desc.device.clone()
    }
    /// The dimensions of the gradient in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.desc.dim.clone().into_pattern()
    }
    /// The dimensions of the gradient.
    pub fn raw_dim(&self) -> D {
        self.desc.dim.clone()
    }
    /// The dimensions of the vertex as a slice.
    pub fn shape(&self) -> &[usize] {
        self.desc.dim.slice()
    }
    /// The strides of the vertex as a slice.
    #[allow(unused)]
    pub(crate) fn strides(&self) -> &[isize] {
        bytemuck::cast_slice(self.desc.strides.slice())
    }
    /// The length of the gradient.
    pub fn len(&self) -> usize {
        self.desc.dim.size()
    }
    /// Whether the gradient is empty.
    pub fn is_empty(&self) -> bool {
        self.desc.dim.slice().iter().any(|x| *x == 0)
    }
    /// The dimensionality of the gradient.
    pub fn ndim(&self) -> usize {
        self.desc.dim.ndim()
    }
    /// The [`FloatType`] of the gradient.
    pub fn float_type(&self) -> FloatType {
        self.desc.float_type
    }
}

impl VariableGradientD {
    fn into_grad_node(self) -> Option<(FloatTensorD, VariableNode)> {
        // TODO: with strides?
        let grad = FloatTensor::from(Arc::try_unwrap(self.grad).ok()?.into_inner()?)
            .into_shape(self.desc.dim)
            .unwrap();
        let grad = unsafe { grad.with_raw_strides(self.desc.strides) };
        Some((grad, self.node))
    }
}

/// Gradient guard.
///
/// See [`GradientBase::lock()`].
pub struct GradientGuard<'a, D: Dimension> {
    desc: FloatTensorDesc<D>,
    guard: MutexGuard<'a, Option<FloatBuffer>>,
}

impl<D: Dimension> GradientGuard<'_, D> {
    pub(crate) fn view_mut(&mut self) -> Option<FloatTensorViewMut<D>> {
        if let Some(buffer) = self.guard.as_mut() {
            let view = FloatTensorViewMut::from(buffer.as_slice_mut())
                .into_shape(self.desc.dim.clone())
                .unwrap();
            let view = unsafe { view.with_raw_strides(self.desc.strides.clone()) };
            Some(view)
        } else {
            None
        }
    }
    /// Returns a mutable view, zeroing if necessary.
    ///
    /// If the gradient has not been computed, it is initialized with 0's.
    ///
    /// **Errors**
    ///
    /// See [`FloatTensor::zeros`](FloatTensorBase::zeros).
    pub fn zeroed_mut(&mut self) -> Result<FloatTensorViewMut<D>> {
        if self.guard.is_none() {
            self.guard.replace(FloatBuffer::zeros(
                self.desc.float_type,
                self.desc.device.clone(),
                self.desc.dim.size(),
            )?);
        }
        Ok(self.view_mut().unwrap())
    }
    fn oned_mut(&mut self) -> Result<()> {
        if self.guard.is_none() {
            self.guard.replace(FloatBuffer::ones(
                self.desc.float_type,
                self.desc.device.clone(),
                self.desc.dim.size(),
            )?);
        }
        Ok(())
    }
    /// Accumulates the gradient with `tensor`.
    ///
    /// If the gradient has not been computed, stores `tensor`. Otherwise accumulates the gradient.
    pub(crate) fn add_assign<S2: FloatData>(
        &mut self,
        tensor: FloatTensorBase<S2, D>,
    ) -> Result<()> {
        if self.desc.dim.slice() != tensor.shape() {
            bail!(
                "Shape does not match gradient! {:?} != {:?}",
                self.desc.dim.slice(),
                tensor.shape()
            );
        }
        if self.desc.device != tensor.device() {
            bail!(
                "Device does not match gradient! {:?} != {:?}",
                self.desc.device,
                tensor.device()
            );
        }
        if self.guard.is_some() || self.desc.strides != tensor.raw_strides() {
            self.zeroed_mut()?.add_assign(&tensor)?;
        } else {
            self.guard.replace(tensor.into_raw_buffer()?);
        }
        Ok(())
    }
    /// The device of the gradient.
    pub fn device(&self) -> Device {
        self.desc.device.clone()
    }
    /// The dimensions of the gradient in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.desc.dim.clone().into_pattern()
    }
    /// The dimensions of the gradient.
    pub fn raw_dim(&self) -> D {
        self.desc.dim.clone()
    }
    /// The dimensions of the vertex as a slice.
    pub fn shape(&self) -> &[usize] {
        self.desc.dim.slice()
    }
    /// The strides of the vertex as a slice.
    #[allow(unused)]
    pub(crate) fn strides(&self) -> &[isize] {
        bytemuck::cast_slice(self.desc.strides.slice())
    }
    /// The length of the gradient.
    pub fn len(&self) -> usize {
        self.desc.dim.size()
    }
    /// Whether the gradient is empty.
    pub fn is_empty(&self) -> bool {
        self.desc.dim.slice().iter().any(|x| *x == 0)
    }
    /// The dimensionality of the gradient.
    pub fn ndim(&self) -> usize {
        self.desc.dim.ndim()
    }
    /// The [`FloatType`] of the gradient.
    pub fn float_type(&self) -> FloatType {
        self.desc.float_type
    }
}

/// A vertex for autograd ops.
///
/// [`Variable`] and [`Parameter`] are vertices, convertible into [`Vertex`]. In the forward pass, a graph is built connecting outputs to inputs, storing backward ops. In the backward pass, the graph is traversed from the loss to the inputs, computing the gradients of variables and parameters in order.
#[derive(Clone, Serialize, Deserialize)]
pub struct VertexBase<D: Dimension, N: Node> {
    value: FloatArcTensor<D>,
    #[serde(skip)]
    grad: Option<Arc<Mutex<Option<FloatBuffer>>>>,
    #[serde(skip)]
    node: N,
}

/// A variable of a network.
///
/// Inputs to the network are wrapped in variables, which potentially construct the graph of backward ops as the forward pass is computed. Use the [`.backward()`](VertexBase::backward()) method to execute the backward pass, computing the gradients.
pub type Variable<D> = VertexBase<D, VariableNode>;
/// A variable with 1 element
pub type Variable0 = Variable<Ix0>;
/// A variable with 1 dimensions
pub type Variable1 = Variable<Ix1>;
/// A variable with 2 dimensions
pub type Variable2 = Variable<Ix2>;
/// A variable with 3 dimensions
pub type Variable3 = Variable<Ix3>;
/// A variable with 4 dimensions
pub type Variable4 = Variable<Ix4>;
/// A dynamic dimensional variable
pub type VariableD = Variable<IxDyn>;

/// A parameter of a network.
///
/// The parameters are [updated](super::layer::Layer::update()) after one or more forward + backward passes, training the network.
pub type Parameter<D> = VertexBase<D, ParameterNode>;
/// A parameter with 1 dimension
pub type Parameter1 = Parameter<Ix1>;
/// A parameter with 2 dimensions
pub type Parameter2 = Parameter<Ix2>;
/// A parameter with 3 dimensions
pub type Parameter3 = Parameter<Ix3>;
/// A parameter with 4 dimensions
pub type Parameter4 = Parameter<Ix4>;
/// A dynamic dimensional parameter
pub type ParameterD = Parameter<IxDyn>;

/// A vertex of a network.
///
/// [`Vertex`] can be either a [`Variable`] or [`Parameter`].
pub type Vertex<D> = VertexBase<D, VertexNode>;
/// A vertex with 1 element
pub type Vertex0 = Vertex<Ix0>;
/// A vertex with 1 dimension
pub type Vertex1 = Vertex<Ix1>;
/// A vertex with 2 dimensions
pub type Vertex2 = Vertex<Ix2>;
/// A vertex with 3 dimensions
pub type Vertex3 = Vertex<Ix3>;
/// A vertex with 4 dimensions
pub type Vertex4 = Vertex<Ix4>;
/// A dynamic dimensional vertex
pub type VertexD = Vertex<IxDyn>;

impl<D: Dimension, N: Node> VertexBase<D, N> {
    /// Returns a reference to the value of the vertex.
    pub fn value(&self) -> &FloatArcTensor<D> {
        &self.value
    }
    /// Returns the value of the vertex.
    pub fn into_value(self) -> FloatArcTensor<D> {
        self.value
    }
    /// Returns a mutable reference to the value of the vertex.
    pub fn value_mut(&mut self) -> &mut FloatArcTensor<D> {
        &mut self.value
    }
    /// Creates a vertex of type `float_type` on `device` with `shape` filled with 0's.
    ///
    /// See [`FloatTensor::zeros()`](FloatTensorBase::zeros()).
    pub fn zeros<Sh>(float_type: FloatType, device: Device, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Ok(Self {
            value: FloatArcTensor::zeros(float_type, device, shape)?,
            grad: None,
            node: N::default(),
        })
    }
    /// The device of the vertex.
    pub fn device(&self) -> Device {
        self.value.device()
    }
    /// The dimensions of the vertex in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.value.dim()
    }
    /// The dimensions of the vertex.
    pub fn raw_dim(&self) -> D {
        self.value.raw_dim()
    }
    /// The dimensions of the vertex as a slice.
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }
    /// The strides of the vertex as a slice.
    pub fn strides(&self) -> &[isize] {
        self.value.strides()
    }
    fn raw_strides(&self) -> D {
        self.value.raw_strides()
    }
    /// The length of the vertex.
    pub fn len(&self) -> usize {
        self.value.len()
    }
    /// Whether the vertex is empty.
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }
    /// The dimensionality of the vertex.
    pub fn ndim(&self) -> usize {
        self.value.ndim()
    }
    /// The [`FloatType`] of the vertex.
    pub fn float_type(&self) -> FloatType {
        self.value.float_type()
    }
    pub(super) fn is_variable(&self) -> bool {
        self.node.is_variable()
    }
    pub(super) fn is_parameter(&self) -> bool {
        self.node.is_parameter()
    }
    /// Whether to train the network's parameters.
    ///
    /// For parameters, returns false. For variables, returns true if [`.with_training()`](VertexBase::with_training()) was called with training = true.
    ///
    /// For a given function, the output will require grad (and thus propagate gradients in the backward pass) if:
    /// - Any input variable requires grad.
    /// - Any input variable is training and any parameter requires grad.
    ///
    /// If the output requires grad, then any variables or parameters with a gradient should have that gradient computed in [`Backward::backward()`].
    pub fn training(&self) -> bool {
        self.node.as_variable().map_or(false, |node| node.training)
    }
    /// Converts the vertex into dimension `D2`.
    ///
    /// See [`FloatTensor::into_dimensionality()`](FloatTensorBase::into_dimensionality()).
    pub fn into_dimensionality<D2>(self) -> Result<VertexBase<D2, N>>
    where
        D2: Dimension,
    {
        Ok(VertexBase {
            value: self.value.into_dimensionality()?,
            grad: self.grad,
            node: self.node,
        })
    }
    /// Returns the vertex with dim `shape`.
    ///
    /// See [`FloatTensor::into_shape()`](FloatTensorBase::into_dimensionality()).
    pub fn into_shape<E>(self, shape: E) -> Result<VertexBase<E::Dim, N>>
    where
        E: IntoDimension,
    {
        Ok(VertexBase {
            value: self.value.into_shape(shape)?,
            grad: self.grad,
            node: self.node,
        })
    }
    /// Permute the axes of the vertex.
    ///
    /// See [`FloatTensor::permuted_axes()`].
    pub fn permuted_axes<A>(mut self, axes: A) -> Self
    where
        A: IntoDimension<Dim = D>,
    {
        self.value = self.value.permuted_axes(axes);
        self
    }
    /// Reverses (transposes) the axes of the vertex.
    pub fn reversed_axes(mut self) -> Self {
        self.value = self.value.reversed_axes();
        self
    }
    /// Clones self with reversed (transposed) axes.
    pub fn t(&self) -> Self {
        self.clone().reversed_axes()
    }
    /// Whether the vertex is standard layout.
    ///
    /// See [`TensorBase::is_standard_layout()`](crate::tensor::TensorBase::is_standard_layout()).
    pub fn is_standard_layout(&self) -> bool {
        self.value.is_standard_layout()
    }
    #[allow(unused)]
    pub(crate) fn flatten(self) -> Result<VertexBase<Ix2, N>> {
        Ok(VertexBase {
            value: self.value.flatten()?,
            grad: self.grad,
            node: self.node,
        })
    }
    /// Converts the dimensionality of the vertex to [`IxDyn`](type@ndarray::IxDyn).
    pub fn into_dyn(self) -> VertexBase<IxDyn, N> {
        VertexBase {
            value: self.value.into_dyn(),
            grad: self.grad,
            node: self.node,
        }
    }
    /// Converts into a [`Vertex`].
    pub fn into_vertex(self) -> Vertex<D> {
        Vertex {
            value: self.value,
            grad: self.grad,
            node: self.node.into_vertex(),
        }
    }
    /*fn try_into_variable(self) -> Result<Variable<D>, Self> {
        match self.node.try_into_variable() {
            Ok(node) => Ok(Variable {
                value: self.value,
                grad: self.grad,
                node,
            }),
            Err(node) => Err(Self {
                value: self.value,
                grad: self.grad,
                node,
            }),
        }
    }*/
    /// Potentially adds a gradient to the vertex.
    ///
    /// # Note
    /// - If `requires_grad`, gradients **can** be computed for this vertex. The gradient is lazily allocated during the backward pass.
    ///   - For [`Variable`]'s, gradients will be computed for all inputs which require a gradient.
    ///   - For [`Parameter`]'s, gradients will be computed if any input variables are training. See [`.with_training()`](VertexBase::with_training()).
    /// - If `requires_grad` is false, drops the gradient. Gradients will not be computed for this vertex.
    /// - This method must be called on parameters (with `requires_grad` = true) before the first backward pass.
    /// - It *does not* need to be called for each pass.
    /// - Use [`Variable::with_training()`](VertexBase::with_training()) to control which model / parameters are trained (ie parameter gradients computed).
    /// - Input variables typically do not require a gradient.
    ///     - Input variables can have a gradient, this can be used to connect several backward passes together for distributed training.
    pub fn require_grad_mut(&mut self, requires_grad: bool) {
        if requires_grad {
            if self.grad.is_none() {
                self.grad.replace(Arc::default());
            }
        } else {
            self.grad.take();
        }
    }
    /// Potentially adds a gradient to the vertex.
    ///
    /// See [.require_grad_mut()](Self::require_grad_mut()).
    pub fn require_grad(mut self, require_grad: bool) -> Self {
        self.require_grad_mut(require_grad);
        self
    }
    /// Whether the vertex requires grad.
    pub fn requires_grad(&self) -> bool {
        self.grad.is_some()
    }
    /// Returns the gradient of the vertex.
    ///
    /// If None, the vertex does not require grad.
    pub fn grad(&self) -> Option<GradientBase<D, N>> {
        Some(GradientBase::new(
            &self.value,
            self.grad.clone()?,
            self.node.clone(),
        ))
    }
    /// Takes the gradient from the vertex.
    ///
    /// Returns None if:
    /// - The vertex does not require grad.
    /// - The vertex is not exclusively held.
    /// - The gradient was not computed.
    ///
    /// Typically this method should be called in [`Optimizer::update()`](super::optimizer::Optimizer::update()), after one or more backwards passes.
    pub fn take_grad(&mut self) -> Option<FloatTensor<D>> {
        if let Some(grad) = self.grad.as_mut().map(Arc::get_mut).flatten() {
            assert_eq!(
                self.value.strides(),
                bytemuck::cast_slice(self.value.raw_dim().default_strides().slice())
            );
            Some(
                FloatTensor::from(grad.get_mut().take()?)
                    .into_shape(self.value.dim())
                    .unwrap(),
            )
        } else {
            None
        }
    }
    /// Transfers the vertex into `device`.
    ///
    /// NOOP when the vertex is on `device`.
    ///
    /// # Autograd
    /// For a variable with a gradient, the output will have a backward op to compute the input gradient.
    ///
    /// **Panics**
    /// - Not yet implemented when the input has multiple dependent gradients.
    ///
    /// See [`FloatArcTensor::into_device_shared()`](FloatTensorBase::into_device_shared()).
    pub async fn into_device(self, device: Device) -> Result<Self> {
        if self.device() == device {
            Ok(self)
        } else {
            // TODO: users can't implement this pattern, where the return type may be a variable or parameter.
            // TODO: For Vertex From returns a variable, invalid if parameter.
            if type_eq::<N, VertexNode>() && self.is_parameter() {
                todo!()
            }
            let mut output = Self::from(smol::block_on(
                self.value().clone().into_device_shared(device),
            )?);
            let input_grad = self
                .grad()
                .map(|g| g.try_into_variable().ok())
                .flatten()
                .map(VariableGradient::into_dyn);
            if let Some(node) = output.node.as_variable_mut() {
                node.training = self.node.as_variable().unwrap().training;
                if let Some(input_grad) = input_grad {
                    node.backward
                        .replace(Arc::new(IntoDeviceBackward { input_grad }));
                }
            }
            if self.requires_grad() {
                output.grad.replace(Arc::default());
            }
            Ok(output)
        }
    }
}

#[derive(Autograd)]
#[autograph(crate)]
struct IntoDeviceBackward {
    #[autograph(gradient)]
    input_grad: VariableGradientD,
}

impl Backward for IntoDeviceBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        let device = self.input_grad.device();
        let output_grad = smol::block_on(output_grad.into_device(device))?;
        self.input_grad.lock().add_assign(output_grad)
    }
}

impl<D: Dimension, N: Node> From<FloatArcTensor<D>> for VertexBase<D, N> {
    fn from(tensor: FloatArcTensor<D>) -> Self {
        Self {
            value: tensor,
            grad: None,
            node: N::default(),
        }
    }
}

impl<D: Dimension, N: Node> From<FloatTensor<D>> for VertexBase<D, N> {
    fn from(tensor: FloatTensor<D>) -> Self {
        Self {
            value: tensor.into(),
            grad: None,
            node: N::default(),
        }
    }
}

impl<D: Dimension, N: Node> Debug for VertexBase<D, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ty = if self.node.is_variable() {
            "Variable"
        } else {
            "Parameter"
        };
        let mut builder = f.debug_struct(ty);
        builder
            .field("float_type", &self.float_type())
            .field("device", &self.device())
            .field("shape", &self.shape());
        let strides = self.strides();
        if strides != bytemuck::cast_slice(self.raw_dim().default_strides().slice()) {
            builder.field("strides", &strides);
        }
        builder.finish()
    }
}

impl<D: Dimension> Variable<D> {
    /// Whether to train the parameters.
    ///
    /// Controls whether to compute the gradients of parameters of functions involving this and dependent variables. If any input is training and any parameter requires grad (which is set for trainers), then the implementation should provide a backward op via [`.backward()`](Variable::backward()) which computes the gradients of any parameters (and input variables) which have a gradient.
    ///
    /// Implementations should inherit [`.training()`](Variable::training()) such that if any input is training, all output variables are training.
    ///
    /// This can be toggled on and off, for example to train a generator but not a discriminator (and vice versa) in a Generative Adversarial Network.
    ///
    /// Note that disabling training does not prevent gradients from being computed for variables, only for parameters.
    pub fn with_training(mut self, training: bool) -> Self {
        self.node.training = training;
        self
    }
    /// Adds a `backward` op.
    ///
    /// If the input variables to a function [`.requires_grad()`](Variable::require_grad()), use this method to provide a [`Backward`] op. [`Backward::backward()`] will be called on 'backward` during the backward pass to compute the input gradients. This variable will also require grad.
    pub fn with_backward(mut self, backward: impl Backward) -> Self {
        if self.grad.is_none() {
            self.grad.replace(Arc::default());
        }
        self.node.backward.replace(Arc::new(BackwardOp {
            backward: Box::new(backward),
            dim: self.raw_dim().into_dyn(),
            strides: self.raw_strides().into_dyn(),
        }));
        self
    }
    /// Runs the backward pass.
    ///
    /// Recursively calls all backward ops, which compute the gradients of variables and parameters.
    ///
    /// **Errors**
    ///
    /// The variable must require grad.
    ///
    /// # Important
    /// The backward algorithm relies on the exclusivity of gradients in order to visit each node of the graph exactly once, even in a potentially multithreaded context. If a variable or gradient still exists outside of a [`Backward`] op (which is dropped after [`.backward()`](Backward::backward())), all upstream gradients will not be computed.
    /// If the value of an intermediate variable needs to be retained past [`.backward()`](Variable::backward), use [`.into_value()`](Variable::into_value()) to extract the value while dropping the gradient.
    pub fn backward(self) -> Result<()> {
        let grad = self
            .grad()
            .ok_or_else(|| anyhow!("Variable does not require grad!"))?;
        std::mem::drop(self);
        grad.lock().oned_mut()?;
        let mut queue = VecDeque::new();
        queue.push_back(grad.into_dyn());
        while let Some(gradient) = queue.pop_front() {
            if let Some((output_grad, node)) = gradient.into_grad_node() {
                if let Some(backward) = node.backward {
                    backward.backward(output_grad)?;
                    queue.extend(
                        backward
                            .grads()
                            .into_iter()
                            .filter_map(|grad| grad.try_into_variable().ok()),
                    )
                }
            }
        }
        Ok(())
    }
}

/*
fn get_permuted_axes(input_strides: &[isize], output_strides: &[isize]) -> Result<IxDyn> {
    let mut axes = IxDyn::zeros(input_strides.len());
    for (a, os) in axes.slice_mut().iter_mut().zip(output_strides) {
        *a = input_strides.iter()
            .position(|s| s == os)
            .ok_or_else(|| anyhow!("Unsupported input strides: {:?} output strides: {:?}!", input_strides, output_strides))?;
    }
    Ok(axes)
}
*/
#[derive(Autograd)]
#[autograph(crate)]
struct NHWCIntoNCHWBackward {
    #[autograph(gradient)]
    input_grad: VariableGradient<Ix4>,
}

impl Backward for NHWCIntoNCHWBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        let output_grad = output_grad
            .into_dimensionality()?
            .permuted_axes([0, 2, 3, 1])
            .into_standard_layout()?;
        self.input_grad.lock().add_assign(output_grad)
    }
}

impl Variable4 {
    pub(super) fn nhwc_into_nchw(self) -> Result<Self> {
        if !self.is_standard_layout() {
            bail!("Must be standard_layout!");
        }
        let mut output = Self::from(
            self.value()
                .clone()
                .permuted_axes([0, 3, 1, 2])
                .to_standard_layout_shared()?,
        )
        .with_training(self.training());
        if let Some(input_grad) = self.grad() {
            output = output.with_backward(NHWCIntoNCHWBackward { input_grad });
        }
        Ok(output)
    }
}

/*
#[derive(Autograd)]
#[autograph(crate)]
struct IntoStandardLayoutBackward {
    input_grad: VariableGradientD,
}

impl Backward for IntoStandardLayoutBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        assert!(output_grad.is_standard_layout());
        let axes = get_permuted_axes(self.input_grad.strides(), output_grad.strides())?;
        let output_grad = output_grad.permuted_axes(axes).into_standard_layout()?;
        self.input_grad.lock().add_assign(output_grad)
    }
}

impl<D: Dimension> Variable<D> {
    /// Converts into standard layout.
    ///
    /// See [`FloatArcTensor::to_standard_layout_shared()`](FloatTensorBase::to_standard_layout_shared()).
    ///
    /// # Note
    /// Autograd only supports standard layout and its permutations, that is the dimensions have been reordered.
    pub(crate) fn into_standard_layout(self) -> Result<Self> {
        if self.is_standard_layout() {
            return Ok(self);
        }
        let mut output = Self::from(self.value.to_standard_layout_shared()?)
            .with_training(self.training());
        if let Some(grad) = self.grad() {
            if cfg!(debug_assertions) {
                // TODO: Validate only permutation?
                dbg!(self.shape());
                dbg!(self.strides());
                dbg!(output.shape());
                dbg!(output.strides());
                get_permuted_axes(self.strides(), output.strides())?;
            }
            output = output.with_backward(IntoStandardLayoutBackward {
                input_grad: grad.into_dyn(),
            });
        }
        Ok(output)
    }
}*/

#[derive(Autograd)]
#[autograph(crate)]
struct DotBackward {
    #[autograph(vertex)]
    input: Vertex2,
    #[autograph(vertex)]
    weight: Vertex2,
    #[autograph(optional_vertex)]
    bias: Option<Vertex1>,
}

impl Backward for DotBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        let dy = output_grad.into_dimensionality()?;
        let x = &self.input;
        let w = &self.weight;
        if let Some(dx) = x.grad() {
            let mut dx = dx.lock();
            dy.dot_acc(1f32, &w.value().t(), &mut dx.zeroed_mut()?)?;
        }
        if let Some(dw) = w.grad() {
            let mut dw = dw.lock();
            x.value()
                .t()
                .dot_acc(1. / dy.dim().0 as f32, &dy.view(), &mut dw.zeroed_mut()?)?;
        }
        if let Some(db) = self.bias.as_ref().map(Vertex::grad).flatten() {
            let mut db = db.lock();
            bias_backward(&mut db.zeroed_mut()?, &dy.view())?;
        }
        Ok(())
    }
}

// linalg
impl<N1: Node, N2: Node> Dot<VertexBase<Ix2, N2>> for VertexBase<Ix2, N1> {
    type Output = Variable2;
    fn dot(&self, rhs: &VertexBase<Ix2, N2>) -> Result<Variable2> {
        self.dot_bias(rhs, Option::<&VertexBase<Ix1, N2>>::None)
    }
}

fn bias_backward(
    bias_grad: &mut FloatTensorViewMut1,
    output_grad: &FloatTensorView2,
) -> Result<()> {
    let (n, c) = output_grad.dim();
    if bias_grad.dim() != c {
        bail!(
            "bias_grad dim {:?} (expected {:?}) not valid for output_grad dim {:?}!",
            bias_grad.shape(),
            [c],
            output_grad.shape(),
        );
    }
    if bias_grad.float_type() != output_grad.float_type() {
        bail!(
            "bias_grad float_type {:?} != output_grad float_type {:?}!",
            bias_grad.float_type(),
            output_grad.float_type(),
        );
    }
    let output_grad_slice = output_grad.to_slice()?;
    let builder = glsl_shaders::module(&format!(
        "bias_backward_{}",
        bias_grad.float_type().as_str()
    ))?
    .compute_pass("main")?
    .float_slice_mut(bias_grad.as_raw_slice_mut())?
    .float_slice(output_grad_slice.as_slice())?
    .push([n as u32, c as u32])?;
    unsafe { builder.submit([c as u32, 1, 1]) }
}

impl<N1: Node, N2: Node, N3: Node> DotBias<VertexBase<Ix2, N2>, VertexBase<Ix1, N3>>
    for VertexBase<Ix2, N1>
{
    fn dot_bias(
        &self,
        rhs: &VertexBase<Ix2, N2>,
        bias: Option<&VertexBase<Ix1, N3>>,
    ) -> Result<Variable2> {
        let input = self;
        let training =
            input.training() || rhs.training() || bias.map_or(false, VertexBase::training);
        let output = Variable::from(input.value().dot(rhs.value())?).with_training(training);
        let train = training
            && (input.is_parameter()
                || rhs.is_parameter()
                || bias.map_or(false, VertexBase::is_parameter));
        let requires_grad = train
            || (input.is_variable() && input.requires_grad())
            || (rhs.is_variable() && rhs.requires_grad())
            || bias.map_or(false, |bias| bias.is_variable() && bias.requires_grad());
        if requires_grad {
            let mut input = input.clone().into_vertex();
            let mut weight = rhs.clone().into_vertex();
            let mut bias = bias.map(|bias| bias.clone().into_vertex());
            if !train {
                if input.is_parameter() {
                    input.require_grad_mut(false);
                }
                if weight.is_parameter() {
                    weight.require_grad_mut(false);
                }
                if let Some(bias) = bias.as_mut() {
                    if bias.is_parameter() {
                        bias.require_grad_mut(false);
                    }
                }
            }
            Ok(output.with_backward(DotBackward {
                input,
                weight,
                bias,
            }))
        } else {
            Ok(output)
        }
    }
}

#[derive(Autograd)]
#[autograph(crate)]
struct Im2ColBackward {
    #[autograph(vertex)]
    input: Variable4,
    kernel: Ix2,
    kind: KernelKind,
    args: KernelArgs<Ix2>,
}

impl Backward for Im2ColBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        use crate::tensor::Tensor;
        smol::block_on(async {
            let input = &self.input;
            if let Some(input_grad) = input.grad() {
                let shape = [input.dim().2, input.dim().3].into_dimension();
                let grad = output_grad
                    .into_dimensionality()?
                    .cast_into::<f32>()?
                    .read()
                    .await?
                    .as_array()
                    .col2im(&shape, &self.kernel, self.kind, &self.args)?;
                let grad = Tensor::from(grad)
                    .into_device(input.device())
                    .await?
                    .into_shared()?
                    .into_float();
                input_grad.lock().add_assign(grad)?;
            }
            Ok(())
        })
    }
}

impl Im2Col<Ix2> for Variable4 {
    type Output = Variable2;
    fn im2col(
        &self,
        kernel: &Ix2,
        kind: KernelKind,
        args: &KernelArgs<Ix2>,
    ) -> Result<Self::Output> {
        let mut output = Variable2::from(self.value().im2col(kernel, kind, args)?)
            .with_training(self.training());
        if self.requires_grad() {
            output = output.with_backward(Im2ColBackward {
                input: self.clone(),
                kernel: *kernel,
                kind,
                args: args.clone(),
            });
        }
        Ok(output)
    }
}

#[allow(unused_variables)]
fn cross_entropy_loss(input: &FloatTensorView2, target: &FloatTensorView2) -> Result<FloatTensor2> {
    if input.shape() != target.shape() {
        bail!(
            "input shape {:?} != target shape {:?}!",
            input.shape(),
            target.shape(),
        );
    }
    if input.float_type() != target.float_type() {
        bail!(
            "input float_type {:?} != target float_type {:?}!",
            input.float_type(),
            target.float_type(),
        )
    }
    let device = input.device();
    let (n, nclasses) = input.dim();
    let float_type = input.float_type();
    let mut output = match float_type {
        FloatType::BF16 => FloatTensor::zeros(float_type, device, [n, nclasses])?,
        FloatType::F32 => unsafe { FloatTensor::alloc(float_type, device, [n, nclasses])? },
    };
    let nclasses_str = if nclasses <= 64 {
        "64"
    } else if nclasses <= 256 {
        "256"
    } else if nclasses <= 1024 {
        "1024"
    } else {
        bail!("nclasses > 1024 unimplemented!");
    };
    let input_slice = input.to_slice()?;
    let target_slice = target.to_slice()?;
    let name = format!(
        "cross_entropy_loss_{}_{}",
        float_type.as_str(),
        nclasses_str
    );
    let builder = glsl_shaders::module(&name)?
        .compute_pass("main")?
        .float_slice(input_slice.as_slice())?
        .float_slice(target_slice.as_slice())?
        .float_slice_mut(output.as_raw_slice_mut())?
        .push([n as u32, nclasses as u32])?;
    unsafe {
        builder.submit([n as u32, 1, 1])?;
    }
    Ok(output)
}

fn cross_entropy_loss_backward(
    input: &FloatTensorView2,
    input_grad: &mut FloatTensorViewMut2,
    target: &FloatTensorView2,
    output_grad: &FloatTensorView0,
) -> Result<()> {
    let n = input.dim().0 as u32;
    let float_type = input.float_type();
    let input_slice = input.to_slice()?;
    let target_slice = target.to_slice()?;
    let output_grad_slice = output_grad.to_slice()?;
    let builder = glsl_shaders::module(&format!(
        "cross_entropy_loss_backward_{}",
        float_type.as_str()
    ))?
    .compute_pass("main")?
    .float_slice(input_slice.as_slice())?
    .float_slice_mut(input_grad.as_raw_slice_mut())?
    .float_slice(target_slice.as_slice())?
    .float_slice(output_grad_slice.as_slice())?
    .push(n)?;
    unsafe { builder.submit([n, 1, 1]) }
}

#[derive(Autograd)]
#[autograph(crate)]
struct CrosEntropyLossBackward {
    #[autograph(vertex)]
    input: Variable2,
    target: FloatArcTensor2,
}

impl Backward for CrosEntropyLossBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        let x = &self.input;
        let t = &self.target;
        let dy = output_grad.into_dimensionality()?;
        if let Some(dx) = x.grad() {
            let mut dx = dx.lock();
            cross_entropy_loss_backward(
                &x.value().view(),
                &mut dx.zeroed_mut()?,
                &t.view(),
                &dy.view(),
            )?;
        }
        Ok(())
    }
}

impl<D: Dimension> Variable<D> {
    pub(super) fn cross_entropy_loss(&self, target: FloatArcTensor<D>) -> Result<Variable0> {
        let input = self.clone().into_dimensionality()?;
        let target = target.into_dimensionality()?;
        let mut output =
            Variable::from(cross_entropy_loss(&input.value().view(), &target.view())?.sum()?)
                .with_training(input.training());
        if input.requires_grad() {
            output = output.with_backward(CrosEntropyLossBackward { input, target });
        }
        Ok(output)
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::{scalar::Float, tensor::CowTensor, util::type_eq};
    use approx::assert_relative_eq;
    use half::bf16;
    use ndarray::{Array, Array2, ArrayView2, ArrayViewMut1, Axis};
    use num_traits::FromPrimitive;

    fn array_bias_backward(db: &mut ArrayViewMut1<f32>, dy: &ArrayView2<f32>) {
        let scale = 1. / dy.dim().0 as f32;
        for (db, dy) in db.iter_mut().zip(dy.axis_iter(Axis(1))) {
            *db += dy.iter().map(|dy| scale * *dy).sum::<f32>();
        }
    }

    async fn test_bias_backward<T: Float>() -> Result<()> {
        let batch_size = 3;
        let units = 7;
        let data = (0..batch_size * units)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let dy_array = Array::from_shape_vec([batch_size, units], data)?;
        let mut db_true = Array::from_elem(units, 1f32);
        array_bias_backward(&mut db_true.view_mut(), &dy_array.view());
        let device = Device::new()?;
        let _s = device.acquire().await;
        let dy = CowTensor::from(dy_array.map(|x| T::from_f32(*x).unwrap()))
            .into_device(device.clone())
            .await?
            .into_float();
        let mut db = FloatTensor::ones(T::float_type(), device, units)?;
        bias_backward(&mut db.view_mut(), &dy.view())?;
        let db_array = db.cast_into::<f32>()?.read().await?;
        let (epsilon, max_relative) = match T::float_type() {
            FloatType::BF16 => (0.01, 0.01),
            FloatType::F32 => (f32::EPSILON, 0.000_1),
        };
        assert_relative_eq!(
            db_array.as_array(),
            db_true.view(),
            epsilon = epsilon,
            max_relative = max_relative
        );
        Ok(())
    }

    #[tokio::test]
    async fn bias_backward_bf16() -> Result<()> {
        test_bias_backward::<bf16>().await
    }

    #[tokio::test]
    async fn bias_backward_f32() -> Result<()> {
        test_bias_backward::<f32>().await
    }

    fn array_cross_entropy_loss(x: &ArrayView2<f32>, t: &ArrayView2<f32>) -> Array2<f32> {
        let mut y = Array::zeros(x.raw_dim());
        for (mut y, (x, t)) in y.outer_iter_mut().zip(x.outer_iter().zip(t.outer_iter())) {
            let m = x
                .iter()
                .copied()
                .fold(x[0], |m, x| if x > m { x } else { m });
            let x = x.map(|x| x - m);
            let s: f32 = x.iter().map(|x| x.exp()).sum();
            for (y, (x, t)) in y.iter_mut().zip(x.iter().copied().zip(t.iter().copied())) {
                *y = (s.ln() - x) * t;
            }
        }
        y
    }

    async fn test_cross_entropy_loss<T: Float + From<u8> + FromPrimitive>() -> Result<()> {
        let n = 67;
        let c = 9;
        let x_data: Vec<T> = (0..n * c).into_iter().map(|x| (x as u8).into()).collect();
        let t_data: Vec<T> = x_data.iter().copied().rev().collect();
        let x_array = Array::from_shape_vec([n, c], x_data)?;
        let t_array = Array::from_shape_vec([n, c], t_data)?;
        let y_true = {
            let x_array = x_array.map(|x| x.to_f32().unwrap());
            let t_array = t_array.map(|t| t.to_f32().unwrap());
            array_cross_entropy_loss(&x_array.view(), &t_array.view())
        };
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = CowTensor::from(x_array.view())
            .into_device(device.clone())
            .await?
            .into_float();
        let t = CowTensor::from(t_array.view())
            .into_device(device.clone())
            .await?
            .into_float();
        let y = cross_entropy_loss(&x.view(), &t.view())?;
        let y_array = y.cast_into::<T>()?.read().await?;
        let y_array = y_array.as_array().map(|x| x.to_f32().unwrap());
        if type_eq::<T, bf16>() {
            assert_relative_eq!(y_array, y_true, epsilon = 0.01, max_relative = 0.01);
        } else {
            assert_relative_eq!(y_array, y_true, max_relative = 0.000_1);
        }
        Ok(())
    }

    #[tokio::test]
    async fn cross_entropy_loss_bf16() -> Result<()> {
        test_cross_entropy_loss::<bf16>().await
    }

    #[tokio::test]
    async fn cross_entropy_loss_f32() -> Result<()> {
        test_cross_entropy_loss::<f32>().await
    }
}
