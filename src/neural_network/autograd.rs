use crate::Result;
use crate::backend::Device;
use crate::tensor::{Dimension, OwnedRepr, ArcRepr, ArcTensor, TensorViewD, TensorViewMutD};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use smol::lock::Mutex;
use std::fmt::Debug;

pub trait Float: crate::tensor::Num + num_traits::Float {}

impl Float for f32 {}

#[doc(hidden)]
pub enum FloatOwnedRepr {
    F32(OwnedRepr<f32>)
}

#[doc(hidden)]
pub enum FloatArcRepr {
    F32(ArcRepr<f32>)
}

pub struct FloatTensorBase<S, D: Dimension> {
    device: Device,
    dim: D,
    strides: D,
    data: S,    
} 

pub trait BackwardOp {
    type Elem: Float;
    fn backward(&self, grad: TensorViewMutD<Self::Elem>, output_grad: TensorViewD<Self::Elem>) -> Result<()>;
}

enum DynBackwardOp {
    F32(Box<dyn BackwardOp<Elem=f32>>)
}

#[doc(hidden)]
pub struct VariableId {
    gen: usize,
    id: usize
}

#[doc(hidden)]
pub struct ParameterId(usize);

#[doc(hidden)]
pub enum GraphId {
    Variable(VariableId),
    Parameter(ParameterId),
}

#[doc(hidden)]
pub struct GraphBase {
    backward_ops: DashMap<GraphId, (DynBackwardOp, VariableId)>,
    parameter_gradients: DashMap<GraphId, Option<(usize, FloatArcTensorD)>>
}

pub struct Graph {
    base: Arc<GraphBase>
}

pub struct Variable<T, D> {
    graph: Option<Graph>,
    value: ArcTensor<T, D>,
    id: VariableId,
}

pub struct Parameter {
    value: FloatArcTensorD,
    grad: FloatArcTensorD,
}


