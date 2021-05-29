use super::Forward;
use crate::{
    backend::Device,
    tensor::{
        float_tensor::{
            FloatArcRepr, FloatArcTensor, FloatData, FloatDataMut, FloatTensor, FloatTensorD,
            FloatTensorView, FloatTensorViewD, FloatTensorViewMut, FloatTensorViewMutD, FloatType,
            FloatViewMutRepr, FloatViewRepr,
        },
        ArcTensor, Data, Dimension, Float, Ix0, Ix1, Ix2, IxDyn, Tensor, TensorBase, TensorView,
        TensorViewMut,
    },
    Result,
};
use half::bf16;
use serde::{Deserialize, Deserializer, Serialize};
use smol::lock::{Mutex, MutexGuardArc};
use std::{
    collections::HashMap,
    convert::TryInto,
    fmt::{self, Debug},
    future::Future,
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::{Arc, Weak},
};

#[doc(hidden)]
pub struct VertexBase {
    device: Device,
    dim: IxDyn,
    float_type: FloatType,
}

impl Debug for VertexBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Vertex")
            .field("device", &self.device)
            .field("dim", &self.dim.slice())
            .field("float_type", &self.float_type)
            .finish()
    }
}

#[derive(Clone)]
pub struct Vertex {
    base: Arc<VertexBase>,
}

impl Debug for Vertex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.base.fmt(f)
    }
}

impl Vertex {
    fn from_tensor<T: Float, S: Data<Elem = T>, D: Dimension>(tensor: &TensorBase<S, D>) -> Self {
        let base = VertexBase {
            device: tensor.device().clone(),
            dim: tensor.raw_dim().into_dyn(),
            float_type: T::float_type(),
        };
        Self {
            base: Arc::new(base),
        }
    }
    fn from_float_tensor<S: FloatData, D: Dimension>(tensor: &TensorBase<S, D>) -> Self {
        let base = VertexBase {
            device: tensor.device().clone(),
            dim: tensor.raw_dim().into_dyn(),
            float_type: tensor.float_type(),
        };
        Self {
            base: Arc::new(base),
        }
    }
    fn device(&self) -> &Device {
        &self.base.device
    }
    fn raw_dim(&self) -> IxDyn {
        self.base.dim.clone()
    }
    fn float_type(&self) -> FloatType {
        self.base.float_type
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

type BackwardOp = Box<dyn Fn(FloatTensorViewMutD, FloatTensorViewD) -> Result<()>>;

#[doc(hidden)]
pub struct Edge {
    output: usize,
    op: BackwardOp,
}

impl Debug for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Edge")
            .field("output", &self.output)
            .finish()
    }
}

#[doc(hidden)]
#[derive(Default, Debug)]
pub struct GraphBase {
    variable_edges: Vec<(Vertex, Vec<Edge>)>,
    parameter_edges: HashMap<Vertex, Vec<Edge>>,
}

impl GraphBase {
    fn variable_grad(&mut self, vertex: Vertex) -> usize {
        let index = self.variable_edges.len();
        self.variable_edges.push((vertex, Vec::new()));
        index
    }
    fn variable_op(&mut self, input: usize, output: usize, op: BackwardOp) {
        self.variable_edges[input].1.push(Edge { output, op });
    }
    fn parameter_op(&mut self, vertex: Vertex, output: usize, op: BackwardOp) {
        self.parameter_edges
            .entry(vertex)
            .or_default()
            .push(Edge { output, op });
    }
    // TODO: make this async for backward device transfers?
    fn backward(self) -> Result<HashMap<Vertex, FloatTensorD>> {
        if self.variable_edges.is_empty() {
            return Ok(HashMap::new());
        }
        let mut variable_grads: Vec<Option<FloatTensorD>> =
            Vec::with_capacity(self.variable_edges.len());
        for _ in 0..self.variable_edges.len() - 1 {
            variable_grads.push(None);
        }
        {
            let vertex = &self.variable_edges.last().as_ref().unwrap().0;
            variable_grads.push(Some(FloatTensor::float_ones(
                vertex.device(),
                vertex.float_type(),
                vertex.raw_dim(),
            )?));
        }
        let variable_grads = variable_grads.as_mut_slice();
        for (input_index, (vertex, edges)) in self.variable_edges.into_iter().enumerate().rev() {
            let offset = input_index + 1;
            let (input_grads, output_grads) = variable_grads.split_at_mut(offset);
            let input_grad = &mut input_grads[input_index];
            if input_grad.is_none() {
                let grad = FloatTensor::float_zeros(
                    vertex.device(),
                    vertex.float_type(),
                    vertex.raw_dim(),
                )?;
                input_grad.replace(grad);
            }
            let input_grad = input_grad.as_mut().unwrap();
            for edge in edges {
                let output_grad = output_grads[edge.output - offset].as_ref().unwrap();
                (edge.op)(input_grad.float_view_mut(), output_grad.float_view())?;
            }
        }
        let mut parameter_grads = HashMap::with_capacity(self.parameter_edges.len());
        for (vertex, edges) in self.parameter_edges.into_iter() {
            let mut parameter_grad =
                FloatTensor::float_zeros(vertex.device(), vertex.float_type(), vertex.raw_dim())?;
            for edge in edges {
                let output_grad = variable_grads[edge.output].as_ref().unwrap();
                (edge.op)(parameter_grad.float_view_mut(), output_grad.float_view())?;
            }
            parameter_grads.insert(vertex, parameter_grad);
        }
        Ok(parameter_grads)
    }
}

#[derive(Default)]
pub struct Graph {
    base: Arc<Mutex<GraphBase>>,
}

impl Graph {
    pub fn backward(self) -> Result<HashMap<Vertex, FloatTensorD>> {
        let mut base = self.base;
        loop {
            // because Graph is not Clone, the only Arcs will be within VariableBuilder::build
            // so there will not be a deadlock and those methods won't block for long
            match Arc::try_unwrap(base) {
                Ok(base) => {
                    return base.into_inner().backward();
                }
                Err(graph) => {
                    base = graph;
                }
            };
        }
    }
}

#[derive(Default, Clone)]
pub struct WeakGraph {
    base: Weak<Mutex<GraphBase>>,
}

impl WeakGraph {
    fn upgrade(&self) -> Option<Arc<Mutex<GraphBase>>> {
        Weak::upgrade(&self.base)
    }
    async fn lock(&self) -> Option<MutexGuardArc<GraphBase>> {
        if let Some(base) = self.upgrade() {
            Some(base.lock_arc().await)
        } else {
            None
        }
    }
}

impl From<&Graph> for WeakGraph {
    fn from(graph: &Graph) -> Self {
        WeakGraph {
            base: Arc::downgrade(&graph.base),
        }
    }
}

#[derive(Clone)]
pub struct Variable<D: Dimension> {
    graph: WeakGraph,
    value: FloatArcTensor<D>,
    training: bool,
    grad_index: Option<usize>,
}

pub type Variable0 = Variable<Ix0>;
pub type Variable2 = Variable<Ix2>;
pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> From<FloatTensor<D>> for Variable<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        Self {
            graph: WeakGraph::default(),
            value: tensor.into(),
            training: false,
            grad_index: None,
        }
    }
}

impl<D: Dimension> Variable<D> {
    /// Returns the graph associated with the the variable.
    pub fn graph(&self) -> WeakGraph {
        self.graph.clone()
    }
    /// Attaches the variable to a graph\
    ///
    /// A graph records backward operations and is required for training but not for inference.
    pub fn with_graph(mut self, graph: impl Into<WeakGraph>) -> Self {
        self.graph = graph.into();
        self
    }
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
    pub fn into_dyn(self) -> VariableD {
        VariableD {
            graph: self.graph,
            value: self.value.into_dyn(),
            training: self.training,
            grad_index: self.grad_index,
        }
    }
    pub fn value(&self) -> &FloatArcTensor<D> {
        &self.value
    }
    pub fn into_value(self) -> FloatArcTensor<D> {
        self.value
    }
    /// Convenience method for Forward::forward
    pub fn forward<F: Forward>(self, f: &F) -> Result<VariableD> {
        f.forward(self.into_dyn())
    }
    /// Returns a VariableBuilder computed via the provided closure.\
    ///
    /// The builder inherits the graph and training of the input variable can be used to apply\
    /// backward ops.
    pub fn forward_op<T, T2, D2, F>(&self, f: F) -> Result<VariableBuilder<T, D, T2, D2>>
    where
        T: Float,
        T2: Float,
        D2: Dimension,
        F: FnOnce(TensorView<T, D>) -> Result<Tensor<T2, D2>>,
    {
        let output_value = f(self.value.float_view().try_into()?)?;
        Ok(VariableBuilder {
            input: self,
            output_value: output_value.into(),
            variable_op: None,
            parameter_ops: Vec::new(),
            _m: PhantomData::default(),
        })
    }
    pub fn into_dimensionality<D2>(self) -> Result<Variable<D2>>
    where
        D2: Dimension,
    {
        Ok(Variable {
            graph: self.graph,
            value: self.value.into_dimensionality()?,
            training: self.training,
            grad_index: self.grad_index,
        })
    }
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
}

pub struct VariableBuilder<'a, T0: Float, D0: Dimension, T: Float, D: Dimension> {
    input: &'a Variable<D0>,
    output_value: ArcTensor<T, D>,
    variable_op: Option<BackwardOp>,
    parameter_ops: Vec<(Vertex, BackwardOp)>,
    _m: PhantomData<T0>,
}

impl<T0: Float, D0: Dimension, T: Float, D: Dimension> VariableBuilder<'_, T0, D0, T, D> {
    pub fn backward_op<F>(&mut self, op: F)
    where
        F: Fn(TensorViewMut<T0, D0>, TensorView<T, D>) -> Result<()> + 'static,
    {
        if self.input.grad_index.is_some() && self.input.graph.upgrade().is_some() {
            let op = Box::new(move |dx: FloatTensorViewMutD, dy: FloatTensorViewD| {
                let dx = dx.into_dimensionality::<D0>()?.try_into()?;
                let dy = dy.into_dimensionality::<D>()?.try_into()?;
                op(dx, dy)
            });
            self.variable_op.replace(op);
        }
    }
    pub fn backward_parameter_op<T2, D2, F>(&mut self, parameter: &ParameterView<D2>, op: F)
    where
        T2: Float,
        D2: Dimension,
        F: Fn(TensorViewMut<T2, D2>, TensorView<T, D>) -> Result<()> + 'static,
    {
        if self.input.training && self.input.graph.upgrade().is_some() {
            let op = Box::new(move |dx: FloatTensorViewMutD, dy: FloatTensorViewD| {
                let dx = dx.into_dimensionality::<D2>()?.try_into()?;
                let dy = dy.into_dimensionality()?.try_into()?;
                op(dx, dy)
            });
            self.parameter_ops.push((parameter.vertex().clone(), op));
        }
    }
    pub fn build(self) -> Variable<D> {
        let training = self.input.training;
        let mut grad_index = None;
        if self.variable_op.is_some() || !self.parameter_ops.is_empty() {
            if let Some(mut graph) = smol::block_on(self.input.graph.lock()) {
                let output_grad_index =
                    graph.variable_grad(Vertex::from_tensor(&self.output_value));
                grad_index.replace(output_grad_index);
                if let Some((input_grad_index, variable_op)) =
                    self.input.grad_index.zip(self.variable_op)
                {
                    graph.variable_op(input_grad_index, output_grad_index, variable_op);
                }
                for (vertex, parameter_op) in self.parameter_ops {
                    graph.parameter_op(vertex, output_grad_index, parameter_op);
                }
            }
        }
        Variable {
            graph: self.input.graph.clone(),
            value: self.output_value.into(),
            training,
            grad_index,
        }
    }
}

#[derive(Serialize)]
#[serde(bound = "S: FloatData + Serialize, D: Dimension + Serialize")]
pub struct ParameterBase<S: FloatData, D: Dimension> {
    value: TensorBase<S, D>,
    #[serde(skip_serializing)]
    vertex: Vertex,
}

pub type Parameter<D> = ParameterBase<FloatArcRepr, D>;
pub type Parameter1 = Parameter<Ix1>;
pub type Parameter2 = Parameter<Ix2>;
pub type ParameterD = Parameter<IxDyn>;

pub type ParameterView<'a, D> = ParameterBase<FloatViewRepr<'a>, D>;
pub type ParameterView1<'a> = ParameterBase<FloatViewRepr<'a>, Ix1>;
pub type ParameterView2<'a> = ParameterBase<FloatViewRepr<'a>, Ix2>;
pub type ParameterViewD<'a> = ParameterBase<FloatViewRepr<'a>, IxDyn>;

pub type ParameterViewMut<'a, D> = ParameterBase<FloatViewMutRepr<'a>, D>;
pub type ParameterViewMut1<'a> = ParameterBase<FloatViewMutRepr<'a>, Ix1>;
pub type ParameterViewMut2<'a> = ParameterBase<FloatViewMutRepr<'a>, Ix2>;
pub type ParameterViewMutD<'a> = ParameterBase<FloatViewMutRepr<'a>, IxDyn>;

impl<S: FloatData + Clone, D: Dimension> Clone for ParameterBase<S, D> {
    fn clone(&self) -> Self {
        ParameterBase {
            value: self.value.clone(),
            vertex: self.vertex.clone(),
        }
    }
}

impl<S: FloatData, D: Dimension> ParameterBase<S, D> {
    pub fn view(&self) -> ParameterView<D> {
        ParameterBase {
            value: self.value.float_view(),
            vertex: self.vertex.clone(),
        }
    }
    pub fn value(&self) -> &TensorBase<S, D> {
        &self.value
    }
    pub fn value_view(&self) -> FloatTensorView<D> {
        self.value.float_view()
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
        })
    }
    pub fn into_dyn(self) -> ParameterBase<S, IxDyn> {
        ParameterBase {
            value: self.value.into_dyn(),
            vertex: self.vertex,
        }
    }
    pub fn into_device(
        self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<Parameter<D>>>> {
        let device = device.clone();
        Ok(async move {
            let value = self.value.float_into_device_arc(&device)?.await?;
            let vertex = Vertex::from_float_tensor(&value);
            Ok(Parameter { value, vertex })
        })
    }
}

impl<D: Dimension> Parameter<D> {
    pub fn make_mut(&mut self) -> Result<ParameterViewMut<D>> {
        Ok(ParameterBase {
            value: self.value.float_make_mut()?,
            vertex: self.vertex.clone(),
        })
    }
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device_mut<'a>(
        &'a mut self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<()>> + 'a> {
        let device = device.clone();
        Ok(async move {
            self.value.float_to_device_mut(&device)?.await?;
            self.vertex = Vertex::from_float_tensor(&self.value);
            Ok(())
        })
    }
}

impl<S: FloatDataMut, D: Dimension> ParameterBase<S, D> {
    pub fn value_view_mut(&mut self) -> FloatTensorViewMut<D> {
        self.value.float_view_mut()
    }
}

impl<D: Dimension> From<FloatTensor<D>> for Parameter<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        let vertex = Vertex::from_float_tensor(&tensor);
        Self {
            value: tensor.into(),
            vertex,
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
        Ok(Self { value, vertex })
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
