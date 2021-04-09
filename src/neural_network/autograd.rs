use super::{Forward, Optimizer};
use crate::{
    backend::Device,
    tensor::{
        float_tensor::{
            FloatArcTensor, FloatTensor, FloatTensorD, FloatTensorView, FloatTensorViewD,
            FloatTensorViewMutD, FloatType, FloatWeakTensor, FloatWeakTensorD,
        },
        Dimension, IxDyn,
    },
    Result,
};
use smol::lock::{Mutex, MutexGuardArc};
use std::{
    cell::UnsafeCell,
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, Weak},
};

#[derive(Clone)]
pub struct Vertex(FloatWeakTensorD);

impl Vertex {
    fn device(&self) -> &Device {
        &self.0.device()
    }
    fn raw_dim(&self) -> IxDyn {
        self.0.raw_dim()
    }
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
    fn as_key(&self) -> usize {
        self.0.vertex_key()
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.as_key() == other.as_key()
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.as_key().hash(hasher);
    }
}

#[doc(hidden)]
pub struct Edge {
    input: Vertex,
    output: Vertex,
    op: Box<dyn Fn(FloatTensorViewMutD, FloatTensorViewD) -> Result<()>>,
}

#[doc(hidden)]
#[derive(Default)]
pub struct GraphBase {
    variable_grads: HashMap<Vertex, Option<FloatTensorD>>,
    variable_edges: Vec<Edge>,
    parameter_grads: HashMap<Vertex, Option<FloatTensorD>>,
    parameter_edges: Vec<Edge>,
}

#[derive(Default)]
pub struct Graph {
    base: Arc<Mutex<GraphBase>>,
}

impl Graph {
    pub fn backward(&mut self) -> Result<()> {
        // TODO: This implementation does not allow simultaneous forward and backward, which may
        // be useful (ie mini batches for distributed training with multiple devices).
        // TODO: May want to ensure that arc is exclusively held and purge weak references before
        // / after backward / update.
        let mut base = smol::block_on(self.base.lock());
        let base = &mut *base;
        if let Some(edge) = base.variable_edges.last() {
            if let Some(ref mut output_grad) = base.variable_grads.get_mut(&edge.output) {
                let output = &edge.output;
                output_grad.replace(FloatTensorD::float_ones(
                    output.device(),
                    output.float_type(),
                    output.raw_dim(),
                )?);
            }
        }
        // TODO: backwards order ensures correctness, but especially when device transfers are
        // involved there may be a more optimum order.
        for edge in base.variable_edges.iter().rev() {
            let input_grad = base.variable_grads.get(&edge.input);
            let output_grad = base.variable_grads.get(&edge.output);
            if let Some((input_grad, Some(output_grad))) = input_grad.zip(output_grad) {
                // input and output vertices are prevented from being equal because the output arc
                // is created internally
                let input_grad = unsafe {
                    &*(input_grad as *const _ as *const UnsafeCell<Option<FloatTensorD>>)
                };
                let input_grad = unsafe { &mut *input_grad.get() };
                if input_grad.is_none() {
                    let input = &edge.input;
                    input_grad.replace(FloatTensorD::float_zeros(
                        input.device(),
                        input.float_type(),
                        input.raw_dim(),
                    )?);
                }
                let input_grad = input_grad.as_mut().unwrap();
                (edge.op)(input_grad.float_view_mut(), output_grad.float_view())?;
            }
        }
        base.variable_edges.clear();
        for edge in base.parameter_edges.iter().rev() {
            let parameter_grad = base.parameter_grads.get_mut(&edge.input);
            let output_grad = base.variable_grads.get(&edge.output);
            if let Some((parameter_grad, Some(output_grad))) = parameter_grad.zip(output_grad) {
                if parameter_grad.is_none() {
                    let input = &edge.input;
                    parameter_grad.replace(FloatTensorD::float_zeros(
                        input.device(),
                        input.float_type(),
                        input.raw_dim(),
                    )?);
                }
                let parameter_grad = parameter_grad.as_mut().unwrap();
                (edge.op)(parameter_grad.float_view_mut(), output_grad.float_view())?;
            }
        }
        base.variable_grads.clear();
        base.parameter_edges.clear();
        Ok(())
    }
    pub fn update<O: Optimizer>(
        &mut self,
        parameters: Vec<&mut ParameterD>,
        optimizer: &mut O,
    ) -> Result<()> {
        // TODO: May want to ensure that arc is exclusively held and purge weak references before / after backward / update.
        let mut base = smol::block_on(self.base.lock());
        for parameter in parameters {
            if let Some(Some(grad)) = base.parameter_grads.get(&parameter.vertex()) {
                optimizer.step(parameter, grad.float_view())?;
            }
        }
        base.parameter_grads.clear();
        Ok(())
    }
}

#[derive(Default, Clone)]
pub struct WeakGraph {
    base: Weak<Mutex<GraphBase>>,
}

impl WeakGraph {
    fn upgrade(&self) -> Option<GraphGuard> {
        todo!()
    }
}

#[doc(hidden)]
pub struct GraphGuard {
    base: MutexGuardArc<GraphBase>,
}

impl GraphGuard {
    fn downgrade(&self) -> WeakGraph {
        todo!()
    }
}

#[derive(Clone)]
pub struct Variable<D: Dimension> {
    graph: WeakGraph,
    value: FloatArcTensor<D>,
    training: bool,
}

pub type VariableD = Variable<IxDyn>;

impl<D: Dimension> From<FloatTensor<D>> for Variable<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        Self {
            graph: WeakGraph::default(),
            value: tensor.into(),
            training: false,
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
        }
    }
    fn vertex(&self) -> Vertex {
        Vertex(FloatWeakTensor::from(&self.value).into_dyn())
    }
    /// Convenience method for Forward::forward
    pub fn forward<F: Forward>(self, f: F) -> Result<VariableD> {
        f.forward(self.into_dyn())
    }
    /// Returns a VariableBuilder computed via the provided closure.\
    ///
    /// The builder inherits the graph and training of the input variable can be used to apply\
    /// backward ops.
    pub fn forward_op<D2, F>(&self, f: F) -> Result<VariableBuilder<D2>>
    where
        D2: Dimension,
        F: Fn(FloatTensorView<D>) -> Result<FloatTensor<D2>>,
    {
        let output = f(self.value.float_view())?;
        Ok(VariableBuilder {
            graph: self.graph.upgrade(),
            input_vertex: self.vertex(),
            value: output.into(),
            training: self.training,
        })
    }
}

impl<D: Dimension> From<VariableBuilder<D>> for Variable<D> {
    fn from(builder: VariableBuilder<D>) -> Self {
        Self {
            graph: builder
                .graph
                .as_ref()
                .map_or(WeakGraph::default(), |g| g.downgrade()),
            value: builder.value,
            training: builder.training,
        }
    }
}

pub struct VariableBuilder<D: Dimension> {
    graph: Option<GraphGuard>,
    input_vertex: Vertex,
    value: FloatArcTensor<D>,
    training: bool,
}

impl<D: Dimension> VariableBuilder<D> {
    pub fn backward_op<F>(&mut self, op: F)
    where
        F: Fn(FloatTensorViewMutD, FloatTensorViewD) -> Result<()> + 'static,
    {
        if let Some(graph) = self.graph.as_mut() {
            if graph.base.variable_grads.contains_key(&self.input_vertex) {
                let vertex = Vertex(FloatWeakTensor::from(&self.value).into_dyn());
                // if input requires grad and backward_op is called, then the output requires grad
                // note that vertex may already exist in the map
                graph.base.variable_grads.insert(vertex.clone(), None);
                graph.base.variable_edges.push(Edge {
                    input: self.input_vertex.clone(),
                    output: vertex,
                    op: Box::new(op),
                });
            } // no op if input does not have a gradient
        } // no op if no graph
    }
    pub fn backward_parameter_op<D2, F>(&mut self, parameter: &Parameter<D2>, op: F)
    where
        D2: Dimension,
        F: Fn(FloatTensorViewMutD, FloatTensorViewD) -> Result<()> + 'static,
    {
        if self.training {
            if let Some(graph) = self.graph.as_mut() {
                let vertex = Vertex(FloatWeakTensor::from(&self.value).into_dyn());
                // if training and backward_op is called, then the output requires grad
                // note that vertex may already exist in the map
                graph.base.variable_grads.insert(vertex.clone(), None);
                let parameter_vertex = parameter.vertex();
                // note that vertex may already exist in the map
                graph
                    .base
                    .parameter_grads
                    .insert(parameter_vertex.clone(), None);
                graph.base.parameter_edges.push(Edge {
                    input: parameter_vertex,
                    output: vertex,
                    op: Box::new(op),
                });
            } // no op if no graph
        } // no op if not training
    }
}

pub struct Parameter<D: Dimension> {
    value: FloatArcTensor<D>,
}

impl<D: Dimension> Parameter<D> {
    pub fn vertex(&self) -> Vertex {
        Vertex(FloatWeakTensor::from(&self.value).into_dyn())
    }
}

pub type ParameterD = Parameter<IxDyn>;
