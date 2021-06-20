use crate::{
    backend::Device,
    tensor::{
        float::{
            FloatArcRepr, FloatArcTensor, FloatData, FloatDataMut, FloatOwnedRepr, FloatTensor,
            FloatTensorBase, FloatTensorView, FloatTensorViewMut, FloatType, FloatViewMutRepr,
            FloatViewRepr,
        },
        Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
    },
    util::type_eq,
    Result,
};
use anyhow::anyhow;

use futures_util::future::ready;
use serde::{Deserialize, Deserializer, Serialize};
use smol::lock::{Mutex, MutexGuard};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    convert::TryInto,
    fmt::{self, Debug},
    future::Future,
    hash::{Hash, Hasher},
    iter::{once, FromIterator},
    marker::PhantomData,
    mem::transmute,
    ops::DerefMut as _,
    pin::Pin,
    sync::Arc,
};

#[doc(hidden)]
#[derive(Clone)]
pub struct VertexBase<D: Dimension> {
    device: Device,
    float_type: FloatType,
    dim: D,
}

impl<D: Dimension> VertexBase<D> {
    fn into_dimensionality<D2>(self) -> Result<VertexBase<D2>>
    where
        D2: Dimension,
    {
        Ok(VertexBase {
            device: self.device,
            float_type: self.float_type,
            dim: into_dimensionality(self.dim)?,
        })
    }
    fn into_dyn(self) -> VertexBase<IxDyn> {
        VertexBase {
            device: self.device,
            float_type: self.float_type,
            dim: self.dim.into_dyn(),
        }
    }
}

impl<D: Dimension> Debug for VertexBase<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Vertex")
            .field("device", &self.device)
            .field("float_type", &self.float_type)
            .field("dim", &self.dim.slice())
            .finish()
    }
}

#[derive(Clone)]
pub struct Vertex {
    base: Arc<VertexBase<IxDyn>>,
}

impl Debug for Vertex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.base.fmt(f)
    }
}

impl Vertex {
    fn from_float_tensor<S: FloatData, D: Dimension>(tensor: &FloatTensorBase<S, D>) -> Self {
        let base = VertexBase {
            device: tensor.device().clone(),
            dim: tensor.raw_dim().into_dyn(),
            float_type: tensor.float_type(),
        };
        Self {
            base: Arc::new(base),
        }
    }
    fn as_key(&self) -> usize {
        Arc::as_ptr(&self.base) as usize
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.base, &other.base)
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.as_key().hash(hasher);
    }
}

#[non_exhaustive]
pub enum GradientBase<S: FloatData, D: Dimension> {
    Dense(FloatTensorBase<S, D>),
    // Sparse
}

macro_rules! gradient_impl {
    ($this:ident, $i:ident => $e:expr) => {{
        match $this {
            GradientBase::Dense($i) => $e,
        }
    }};
    ($this:ident, map $i:ident => $e:expr) => {{
        match $this {
            GradientBase::Dense($i) => GradientBase::Dense($e),
        }
    }};
}

pub type Gradient<D> = GradientBase<FloatOwnedRepr, D>;
pub type GradientD = Gradient<IxDyn>;

pub type GradientView<'a, D> = GradientBase<FloatViewRepr<'a>, D>;
pub type GradientViewD<'a> = GradientView<'a, IxDyn>;

pub type GradientViewMut<'a, D> = GradientBase<FloatViewMutRepr<'a>, D>;
pub type GradientViewMutD<'a> = GradientViewMut<'a, IxDyn>;

impl<S: FloatData, D: Dimension> GradientBase<S, D> {
    pub fn into_dimensionality<D2>(self) -> Result<GradientBase<S, D2>>
    where
        D2: Dimension,
    {
        Ok(gradient_impl!(self, map x => x.into_dimensionality()?))
    }
    pub fn into_dyn(self) -> GradientBase<S, IxDyn> {
        gradient_impl!(self, map x => x.into_dyn())
    }
}

impl<S: FloatDataMut, D: Dimension> GradientBase<S, D> {
    pub fn view_mut(&mut self) -> GradientViewMut<D> {
        gradient_impl!(self, map x => x.view_mut())
    }
}

impl<D: Dimension> Gradient<D> {
    /// converts to dense and returns a mutable view
    pub fn dense_view_mut(&mut self) -> FloatTensorViewMut<D> {
        match self {
            Self::Dense(x) => x.view_mut(),
        }
    }
    pub fn to_dense_mut(&mut self) -> Result<()> {
        match self {
            Self::Dense(_) => Ok(()),
        }
    }
    pub fn into_dense(mut self) -> Result<FloatTensor<D>> {
        self.to_dense_mut()?;
        match self {
            Self::Dense(x) => Ok(x),
        }
    }
}

fn into_dimensionality<D1: Dimension, D2: Dimension>(dim: D1) -> Result<D2> {
    D2::from_dimension(&dim)
        .ok_or_else(|| anyhow!("Incompatible Shapes! {:?} => {:?}", dim, D2::NDIM))
}

pub struct OccupiedEntry<'a, D: Dimension> {
    vertex: VertexBase<D>,
    gradient: &'a mut GradientD,
}

impl<'a, D: Dimension> OccupiedEntry<'a, D> {
    pub fn gradient_view_mut(&mut self) -> GradientViewMut<D> {
        self.gradient.view_mut().into_dimensionality().unwrap()
    }
    pub fn dense_view_mut(&mut self) -> FloatTensorViewMut<D> {
        self.gradient
            .dense_view_mut()
            .into_dimensionality()
            .unwrap()
    }
    pub fn into_dense_view_mut(self) -> Result<FloatTensorViewMut<'a, D>> {
        todo!()
    }
    pub fn into_dimensionality<D2>(self) -> Result<OccupiedEntry<'a, D2>>
    where
        D2: Dimension,
    {
        Ok(OccupiedEntry {
            vertex: self.vertex.into_dimensionality()?,
            gradient: self.gradient,
        })
    }
    pub fn into_dyn(self) -> OccupiedEntry<'a, IxDyn> {
        OccupiedEntry {
            vertex: self.vertex.into_dyn(),
            gradient: self.gradient,
        }
    }
}

pub struct VacantEntry<'a, D: Dimension> {
    vertex: VertexBase<D>,
    gradient: &'a mut Option<GradientD>,
}

impl<'a, D: Dimension> VacantEntry<'a, D> {
    pub fn into_dense_zeroed(self) -> Result<FloatTensorViewMut<'a, D>> {
        let vertex = self.vertex;
        self.gradient.replace(Gradient::Dense(FloatTensor::zeros(
            &vertex.device,
            vertex.float_type,
            vertex.dim.into_dyn(),
        )?));
        match self.gradient {
            Some(Gradient::Dense(x)) => Ok(x.view_mut().into_dimensionality()?),
            None => unreachable!(),
        }
    }
    /// # Safety
    /// The tensor may be uninitialized.
    pub unsafe fn into_dense_uninitialized(self) -> Result<FloatTensorViewMut<'a, D>> {
        let vertex = self.vertex;
        self.gradient
            .replace(Gradient::Dense(FloatTensor::uninitialized(
                &vertex.device,
                vertex.float_type,
                vertex.dim.into_dyn(),
            )?));
        match self.gradient {
            Some(Gradient::Dense(x)) => Ok(x.view_mut().into_dimensionality()?),
            None => unreachable!(),
        }
    }
    pub fn into_dimensionality<D2>(self) -> Result<VacantEntry<'a, D2>>
    where
        D2: Dimension,
    {
        Ok(VacantEntry {
            vertex: self.vertex.into_dimensionality()?,
            gradient: self.gradient,
        })
    }
    pub fn into_dyn(self) -> VacantEntry<'a, IxDyn> {
        VacantEntry {
            vertex: self.vertex.into_dyn(),
            gradient: self.gradient,
        }
    }
}

pub enum GradientEntry<'a, D: Dimension> {
    Occupied(OccupiedEntry<'a, D>),
    Vacant(VacantEntry<'a, D>),
}

pub type GradientEntryD<'a> = GradientEntry<'a, IxDyn>;

impl<'a, D: Dimension> GradientEntry<'a, D> {
    fn new(vertex: VertexBase<D>, gradient: &'a mut Option<GradientD>) -> Self {
        match gradient {
            Some(gradient) => Self::Occupied(OccupiedEntry { vertex, gradient }),
            None => Self::Vacant(VacantEntry { vertex, gradient }),
        }
    }
    pub fn or_dense_zeroed(self) -> Result<FloatTensorViewMut<'a, D>> {
        match self {
            Self::Occupied(x) => x.into_dense_view_mut()?.into_dimensionality(),
            Self::Vacant(x) => x.into_dense_zeroed()?.into_dimensionality(),
        }
    }
    pub fn into_dimensionality<D2>(self) -> Result<GradientEntry<'a, D2>>
    where
        D2: Dimension,
    {
        match self {
            Self::Occupied(x) => Ok(GradientEntry::Occupied(x.into_dimensionality()?)),
            Self::Vacant(x) => Ok(GradientEntry::Vacant(x.into_dimensionality()?)),
        }
    }
}

pub struct GradientVec<'a> {
    gradients: Vec<Option<GradientEntryD<'a>>>,
}

impl<'a, const N: usize> TryInto<[Option<GradientEntryD<'a>>; N]> for GradientVec<'a> {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<[Option<GradientEntryD<'a>>; N]> {
        self.gradients
            .try_into()
            .map_err(|vec: Vec<_>| anyhow!("Expected {}, found {} gradients!", N, vec.len()))
    }
}

impl<'a> GradientVec<'a> {
    pub fn try_into_array<const N: usize>(self) -> Result<[Option<GradientEntryD<'a>>; N]> {
        self.try_into()
    }
}

pub trait Backward: Send + Sync + 'static {
    fn backward(&mut self, input_grads: GradientVec, output_grad: GradientD) -> Result<()>;
    fn backward_parameters(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + Sync + 'static>> {
        Box::pin(ready(Ok(())))
    }
}

pub struct Graph {
    training: bool,
    node: Option<Arc<Node>>,
}

struct Node {
    nodes: Vec<Option<Arc<Node>>>,
    backward: Box<dyn Backward>,
    parameterized: bool,
    vertex: Vertex,
}

#[derive(Clone)]
pub struct Variable<D: Dimension> {
    value: FloatArcTensor<D>,
    training: bool,
    node: Option<Arc<Node>>,
}

pub type Variable0 = Variable<Ix0>;
pub type Variable1 = Variable<Ix1>;
pub type Variable2 = Variable<Ix2>;
pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> From<FloatTensor<D>> for Variable<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        FloatArcTensor::from(tensor).into()
    }
}

impl<D: Dimension> From<FloatArcTensor<D>> for Variable<D> {
    fn from(tensor: FloatArcTensor<D>) -> Self {
        Self {
            value: tensor,
            training: false,
            node: None,
        }
    }
}

impl<D: Dimension> Variable<D> {
    /// Returns whether the variable is training.
    ///
    /// If training, parameter gradients will be computed.
    pub fn training(&self) -> bool {
        self.training
    }
    /// Returns self with training set to the provided value.
    ///
    /// Use this method to enable or disable training, by default variables have training = false.\
    /// Variables inherit both the graph and training via the forward_op method.
    pub fn with_training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }
    /// Whether the variable has a gradient.
    ///
    /// Inputs to the network do not have gradients. If training and a backward parameter op is\
    /// added using builder().backward_parameter_op().build(), that variable will have a\
    /// gradient. Any dependent variables created using builder().backward_op().build() will also\
    /// have a gradient.
    pub fn requires_grad(&self) -> bool {
        self.node.is_some()
    }
    pub fn graph(&self) -> Graph {
        Graph {
            training: self.training,
            node: self.node.clone(),
        }
    }
    pub fn into_dyn(self) -> VariableD {
        VariableD {
            value: self.value.into_dyn(),
            training: self.training,
            node: self.node,
        }
    }
    pub fn value(&self) -> &FloatArcTensor<D> {
        &self.value
    }
    pub fn into_value(self) -> FloatArcTensor<D> {
        self.value
    }
    /*
    /// Convenience method for Forward::forward
    pub fn forward<F: Forward>(self, f: &F) -> Result<VariableD> {
        f.forward(self.into_dyn())
    }
    /// Convenience method for Forward::forward_mut
    pub fn forward_mut<F: Forward>(self, f: &mut F) -> Result<VariableD> {
        f.forward_mut(self.into_dyn())
    }*/
    pub fn into_dimensionality<D2>(self) -> Result<Variable<D2>>
    where
        D2: Dimension,
    {
        Ok(Variable {
            value: self.value.into_dimensionality()?,
            training: self.training,
            node: self.node,
        })
    }
    /*
    // TODO: may want to implement this in graph
    pub fn into_device(self, device: &Device) -> Result<impl Future<Output = Result<Self>>> {
        fn into_device_impl<T: Float, D: Dimension>(
            this: Variable<D>,
            value: Tensor<T, D>,
        ) -> Result<Variable<D>> {
            let mut builder = this.forward_op(move |_| Ok(value))?;
            builder.backward_op(|mut dx, dy| {
                let device = dx.device().clone();
                dx.add_assign(&smol::block_on(dy.into_device(&device)?)?)
            });
            Ok(builder.build())
        }
        let device = device.clone();
        Ok(async move {
            if self.value().device() == &device {
                Ok(self)
            } else {
                let value = self.value.float_view().float_into_device(&device)?.await?;
                // TODO: macro for this pattern?
                match value.float_type() {
                    FloatType::BF16 => into_device_impl::<bf16, _>(self, value.float_cast_into()?),
                    FloatType::F32 => into_device_impl::<f32, _>(self, value.float_cast_into()?),
                }
            }
        })
    }
    */
    /// Returns a builder used to attach a backward op\
    ///
    ///```
    /// # use autograph::{
    /// #    Result,
    /// #    backend::Device,
    /// #    tensor::float::{FloatType, FloatArcTensor, FloatArcTensor1},
    /// #    neural_network::autograd::{Backward, GradientVec, Variable, Variable1, GradientD},
    /// # };
    /// # struct MyBackwardOp {
    /// #    input: FloatArcTensor1,
    /// # }
    /// # impl MyBackwardOp {
    /// #     fn new(input: FloatArcTensor1) -> Self {
    /// #         Self { input }
    /// #     }
    /// # }
    /// # impl Backward for MyBackwardOp {
    /// #     fn backward(&mut self, input_grads: GradientVec, output_grad: GradientD) -> Result<()> {
    /// #         todo!()
    /// #     }
    /// # }
    /// # fn main() -> Result<()> {
    ///     let device = Device::new_cpu();
    ///     let x = Variable::from(FloatArcTensor::zeros(
    ///         &device,
    ///        FloatType::F32,
    ///        1
    ///     )?);
    ///     # let f = |x: FloatArcTensor1| -> Result<FloatArcTensor1> { Ok(x) };
    ///     let y = Variable1::builder([x.graph()]) // the graphs of the inputs
    ///         .parameterized() // when the op has parameters
    ///         .with_backward(|| Box::new(MyBackwardOp::new(x.value().clone())))
    ///         .build(f(x.into_value())?); // Construct a Variable with the value
    /// #   Ok(())
    /// # }
    ///```
    pub fn builder(graphs: impl IntoIterator<Item = Graph>) -> VariableBuilder {
        let graphs = graphs.into_iter();
        let mut training = false;
        let mut requires_grad = false;
        let nodes = graphs
            .into_iter()
            .map(|graph| {
                training |= graph.training;
                requires_grad |= graph.node.is_some();
                graph.node
            })
            .collect();
        VariableBuilder {
            nodes,
            training,
            parameterized: false,
            requires_grad,
            backward: None,
        }
    }
    /// Runs the backward pass\
    ///
    /// Err:
    /// - The variable doesn't require grad.
    /// - The variable isn't exclusively held.
    ///
    /// Additionally, all variables should only be retained in Backward ops (generally only the
    /// values need to be retained). The backward algorithm deconstructs the graph as it goes,
    /// visiting all variables in appropriate order, calling [`Backwrd::backward`] and
    /// [`Backward::backward_parameters`] on each as necessary, dropping the op immediately. If a
    /// [`Variable`] or [`Graph`] is retained elsewhere, it will not not be visited.
    pub async fn backward(self) -> Result<()> {
        let node = self
            .node
            .ok_or_else(|| anyhow!("Variable does not require grad!"))?;
        let node =
            Arc::try_unwrap(node).map_err(|_| anyhow!("Variable must be exclusively held!"))?;
        let output_grad = Gradient::Dense(FloatTensor::ones(
            self.value.device(),
            self.value.float_type(),
            self.value.raw_dim().into_dyn(),
        )?);
        let mut gradients = HashMap::new();
        gradients.insert(node.vertex.clone(), RefCell::new(Some(output_grad)));
        let mut queue = VecDeque::from_iter(once(node));
        while let Some(node) = queue.pop_front() {
            let output_grad = gradients
                .remove(&node.vertex)
                .expect("Gradient was not computed!")
                .into_inner()
                .expect("Gradient was not computed!");
            for node in node.nodes.iter().filter_map(Option::as_ref) {
                gradients.entry(node.vertex.clone()).or_default();
            }
            let mut input_grad_muts = Vec::with_capacity(node.nodes.len());
            for node in node.nodes.iter() {
                if let Some(node) = node.as_ref() {
                    let grad = gradients.get(&node.vertex).unwrap();
                    input_grad_muts.push(grad.try_borrow_mut().ok());
                } else {
                    input_grad_muts.push(None);
                }
            }
            let mut input_grads = Vec::with_capacity(input_grad_muts.len());
            for (node, input_grad_mut) in node.nodes.iter().zip(input_grad_muts.iter_mut()) {
                let grad = if let Some((node, grad)) = node.as_ref().zip(input_grad_mut.as_mut()) {
                    let vertex = VertexBase::clone(&*node.vertex.base);
                    let grad = grad.deref_mut();
                    Some(GradientEntry::new(vertex, grad))
                } else {
                    None
                };
                input_grads.push(grad);
            }
            let input_grads = GradientVec {
                gradients: input_grads,
            };

            let mut backward = node.backward;
            backward.backward(input_grads, output_grad)?;
            // Want to optimize this in parallel training with multiple devices
            // the parameter ops can be performed after the gradients have been passed
            // back to the previous device
            if node.parameterized {
                backward.backward_parameters().await?;
            }
            for node in node.nodes.into_iter().filter_map(Option::into) {
                if let Ok(node) = Arc::try_unwrap(node) {
                    queue.push_back(node);
                }
            }
        }
        Ok(())
    }
}

pub struct VariableBuilder {
    nodes: Vec<Option<Arc<Node>>>,
    training: bool,
    parameterized: bool,
    requires_grad: bool,
    backward: Option<Box<dyn Backward>>,
}

impl VariableBuilder {
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    pub fn parameterized(mut self) -> Self {
        self.parameterized = self.training;
        self.requires_grad |= self.training;
        self
    }
    pub fn with_backward(mut self, f: impl FnOnce() -> Box<dyn Backward>) -> Self {
        if self.requires_grad {
            self.backward.replace(f());
        }
        self
    }
    pub fn build<D: Dimension>(self, value: FloatArcTensor<D>) -> Variable<D> {
        let node = if let Some(backward) = self.backward {
            Some(Arc::new(Node {
                nodes: self.nodes,
                backward,
                parameterized: self.parameterized,
                vertex: Vertex::from_float_tensor(&value),
            }))
        } else {
            None
        };
        Variable {
            value,
            training: self.training,
            node,
        }
    }
}

#[derive(Serialize)]
#[serde(bound = "S: FloatData, D: Dimension + Serialize, FloatTensorBase<S, D>: Serialize")]
pub struct ParameterBase<S: FloatData, D: Dimension> {
    value: FloatTensorBase<S, D>,
    #[serde(skip_serializing)]
    vertex: Vertex,
    #[serde(skip_serializing)]
    grad: Arc<Mutex<Option<GradientD>>>,
}

pub type Parameter<D> = ParameterBase<FloatArcRepr, D>;
pub type Parameter0 = Parameter<Ix0>;
pub type Parameter1 = Parameter<Ix1>;
pub type Parameter2 = Parameter<Ix2>;
pub type Parameter3 = Parameter<Ix3>;
pub type Parameter4 = Parameter<Ix4>;
pub type Parameter5 = Parameter<Ix5>;
pub type Parameter6 = Parameter<Ix6>;
pub type ParameterD = Parameter<IxDyn>;

pub type ParameterView<'a, D> = ParameterBase<FloatViewRepr<'a>, D>;
pub type ParameterView1<'a> = ParameterBase<FloatViewRepr<'a>, Ix1>;
pub type ParameterView2<'a> = ParameterBase<FloatViewRepr<'a>, Ix2>;
pub type ParameterViewD<'a> = ParameterBase<FloatViewRepr<'a>, IxDyn>;

pub type ParameterViewMut<'a, D> = ParameterBase<FloatViewMutRepr<'a>, D>;
pub type ParameterViewMut1<'a> = ParameterBase<FloatViewMutRepr<'a>, Ix1>;
pub type ParameterViewMut2<'a> = ParameterBase<FloatViewMutRepr<'a>, Ix2>;
pub type ParameterViewMutD<'a> = ParameterBase<FloatViewMutRepr<'a>, IxDyn>;

impl<D: Dimension> Clone for Parameter<D> {
    fn clone(&self) -> Self {
        Parameter {
            value: self.value.clone(),
            vertex: self.vertex.clone(),
            grad: self.grad.clone(),
        }
    }
}

impl<S: FloatData, D: Dimension> ParameterBase<S, D> {
    pub fn view(&self) -> ParameterView<D> {
        ParameterBase {
            value: self.value.view(),
            vertex: self.vertex.clone(),
            grad: self.grad.clone(),
        }
    }
    pub fn value(&self) -> &FloatTensorBase<S, D> {
        &self.value
    }
    pub fn value_view(&self) -> FloatTensorView<D> {
        self.value.view()
    }
    pub fn vertex(&self) -> &Vertex {
        &self.vertex
    }
    pub fn into_dimensionality<D2>(self) -> Result<ParameterBase<S, D2>>
    where
        D2: Dimension,
    {
        Ok(ParameterBase {
            value: self.value.into_dimensionality()?,
            vertex: self.vertex,
            grad: self.grad,
        })
    }
    pub fn into_dyn(self) -> ParameterBase<S, IxDyn> {
        ParameterBase {
            value: self.value.into_dyn(),
            vertex: self.vertex,
            grad: self.grad,
        }
    }
    pub fn into_device(
        self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<Parameter<D>>>> {
        let device = device.clone();
        Ok(async move {
            let value = self.value.into_device_arc(&device)?.await?;
            let vertex = Vertex::from_float_tensor(&value);
            let grad = Arc::default();
            Ok(Parameter {
                value,
                vertex,
                grad,
            })
        })
    }
    pub async fn grad_lock(&'_ self) -> ParameterGradientGuard<'_, D> {
        ParameterGradientGuard {
            vertex: VertexBase {
                device: self.value.device().clone(),
                float_type: self.value.float_type(),
                dim: self.value.raw_dim(),
            },
            grad: self.grad.lock().await,
        }
    }
    pub fn take_grad(&mut self) -> Option<Gradient<D>> {
        Arc::get_mut(&mut self.grad)
            .map(|grad| {
                grad.get_mut()
                    .take()
                    .map(|grad| grad.into_dimensionality().unwrap())
            })
            .flatten()
    }
}

impl<D: Dimension> Parameter<D> {
    pub fn make_mut(&mut self) -> Result<ParameterViewMut<D>> {
        Ok(ParameterBase {
            value: self.value.make_shared_mut()?,
            vertex: self.vertex.clone(),
            grad: self.grad.clone(),
        })
    }
    pub fn as_mut(&mut self) -> ParameterMut<D>
    where
        D: 'static,
    {
        ParameterMut::from_parameter(self)
    }
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device_mut<'a>(
        &'a mut self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<()>> + 'a> {
        let device = device.clone();
        Ok(async move {
            self.value.to_device_mut(&device)?.await?;
            self.vertex = Vertex::from_float_tensor(&self.value);
            Ok(())
        })
    }
}

impl<S: FloatDataMut, D: Dimension> ParameterBase<S, D> {
    pub fn value_view_mut(&mut self) -> FloatTensorViewMut<D> {
        self.value.view_mut()
    }
}

impl<D: Dimension> From<FloatTensor<D>> for Parameter<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        let vertex = Vertex::from_float_tensor(&tensor);
        Self {
            value: tensor.into(),
            vertex,
            grad: Arc::default(),
        }
    }
}

impl<'de, D: Dimension + Deserialize<'de>> Deserialize<'de> for Parameter<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let value = FloatArcTensor::deserialize(deserializer)?;
        let vertex = Vertex::from_float_tensor(&value);
        let grad = Arc::default();
        Ok(Self {
            value,
            vertex,
            grad,
        })
    }
}

impl<S: FloatData, D: Dimension> Debug for ParameterBase<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Parameter")
            .field("device", self.value.device())
            .field("dim", &self.value.shape())
            .field("float_type", &self.value.float_type())
            .finish()
    }
}

pub struct ParameterGradientGuard<'a, D: Dimension> {
    vertex: VertexBase<D>,
    grad: MutexGuard<'a, Option<GradientD>>,
}

impl<'a, D: Dimension> ParameterGradientGuard<'a, D> {
    pub fn entry(&mut self) -> GradientEntry<D> {
        GradientEntry::new(self.vertex.clone(), &mut self.grad)
    }
}

pub struct ParameterMut<'a, D: Dimension> {
    inner: ParameterMutInner<'a>,
    _m: PhantomData<D>,
}

pub type ParameterMutD<'a> = ParameterMut<'a, IxDyn>;

impl<'a, D: Dimension> ParameterMut<'a, D> {
    fn from_parameter(parameter: &'a mut Parameter<D>) -> Self
    where
        D: 'static,
    {
        Self {
            inner: ParameterMutInner::from_parameter(parameter),
            _m: PhantomData::default(),
        }
    }
    pub fn view(&self) -> ParameterView<D> {
        self.inner.view()
    }
    pub fn make_mut(&mut self) -> Result<ParameterViewMut<D>> {
        self.inner.make_mut()
    }
    pub fn into_dimensionality<D2>(self) -> Result<ParameterMut<'a, D2>>
    where
        D2: Dimension,
    {
        // check that conversion will succeed
        self.view().into_dimensionality::<D2>()?;
        Ok(ParameterMut {
            inner: self.inner,
            _m: PhantomData::default(),
        })
    }
    pub fn into_dyn(self) -> ParameterMutD<'a> {
        ParameterMut {
            inner: self.inner,
            _m: PhantomData::default(),
        }
    }
    pub fn value_view(&self) -> FloatTensorView<D> {
        self.inner.value_view()
    }
    pub fn vertex(&self) -> &Vertex {
        self.inner.vertex()
    }
    pub fn to_device_mut<'b: 'a>(
        &'b mut self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<()>> + 'b> {
        self.inner.to_device_mut(device)
    }
    pub fn take_grad(&mut self) -> Option<Gradient<D>> {
        self.inner.take_grad()
    }
}

#[allow(unused)]
enum ParameterMutInner<'a> {
    Ix0(&'a mut Parameter0),
    Ix1(&'a mut Parameter1),
    Ix2(&'a mut Parameter2),
    Ix3(&'a mut Parameter3),
    Ix4(&'a mut Parameter4),
    Ix5(&'a mut Parameter5),
    Ix6(&'a mut Parameter6),
    IxDyn(&'a mut ParameterD),
}

macro_rules! parameter_mut_inner_impl {
    ($this:ident, $i:ident => $e:expr) => {{
        match $this {
            Self::Ix0($i) => $e,
            Self::Ix1($i) => $e,
            Self::Ix2($i) => $e,
            Self::Ix3($i) => $e,
            Self::Ix4($i) => $e,
            Self::Ix5($i) => $e,
            Self::Ix6($i) => $e,
            Self::IxDyn($i) => $e,
        }
    }};
}

impl<'a> ParameterMutInner<'a> {
    fn from_parameter<D: Dimension + 'static>(parameter: &'a mut Parameter<D>) -> Self {
        if type_eq::<D, Ix0>() {
            Self::Ix0(unsafe { transmute(parameter) })
        } else if type_eq::<D, Ix1>() {
            Self::Ix1(unsafe { transmute(parameter) })
        } else if type_eq::<D, Ix2>() {
            Self::Ix2(unsafe { transmute(parameter) })
        } else if type_eq::<D, Ix3>() {
            Self::Ix3(unsafe { transmute(parameter) })
        } else if type_eq::<D, Ix4>() {
            Self::Ix4(unsafe { transmute(parameter) })
        } else if type_eq::<D, Ix5>() {
            Self::Ix5(unsafe { transmute(parameter) })
        } else if type_eq::<D, Ix6>() {
            Self::Ix6(unsafe { transmute(parameter) })
        } else if type_eq::<D, IxDyn>() {
            Self::IxDyn(unsafe { transmute(parameter) })
        } else {
            unreachable!()
        }
    }
    fn view<D: Dimension>(&self) -> ParameterView<D> {
        parameter_mut_inner_impl!(self, x => x.view().into_dimensionality().unwrap())
    }
    fn make_mut<D: Dimension>(&mut self) -> Result<ParameterViewMut<D>> {
        parameter_mut_inner_impl!(self, x => Ok(x.make_mut()?.into_dimensionality().unwrap()))
    }
    fn value_view<D: Dimension>(&self) -> FloatTensorView<D> {
        parameter_mut_inner_impl!(self, x => x.value_view().into_dimensionality().unwrap())
    }
    fn take_grad<D: Dimension>(&mut self) -> Option<Gradient<D>> {
        parameter_mut_inner_impl!(self, x => Some(x.take_grad()?.into_dimensionality().unwrap()))
    }
    fn vertex(&self) -> &Vertex {
        parameter_mut_inner_impl!(self, x => x.vertex())
    }
    #[allow(clippy::wrong_self_convention)]
    fn to_device_mut<'b: 'a>(
        &'b mut self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<()>> + 'b> {
        let device = device.clone();
        Ok(async move {
            parameter_mut_inner_impl!(
                self,
                x => x.value.to_device_mut(&device)?.await
            )
        })
    }
}
