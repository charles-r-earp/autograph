#![allow(dead_code, unused)]
use std::{ops, marker, alloc, mem, slice, fmt, cell, iter, convert};
pub mod native;
pub use native::{Element, Native};
#[cfg(feature="opencl")]
pub mod opencl;
#[cfg(feature="opencl")]
pub use opencl::Opencl;

#[derive(Default, Clone, Copy)]
pub struct Shape {
  dims: [usize; 4],
  len: usize
}

impl Shape {
  pub fn new(dims: impl AsRef<[usize]>) -> Self {
    let dims = dims.as_ref();
    debug_assert!(dims.len() <= Self::default().dims.len());
    let mut s = Self::default();
    s.len = dims.len();
    s[..].copy_from_slice(dims);
    s
  }
  pub fn len(&self) -> usize { self.len }
  pub fn size(&self) -> usize { 
    self.iter()
      .product::<usize>()
  }
}

pub fn shape(dims: impl AsRef<[usize]>) -> Shape {
  Shape::new(dims)
}

impl ops::Deref for Shape {
  type Target = [usize];
  fn deref(&self) -> &Self::Target {
    &self.dims[..self.len]
  }
}

impl ops::DerefMut for Shape {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.dims[..self.len]
  }
}

impl fmt::Debug for Shape {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    self[..].fmt(f)
  }
}

#[derive(Clone, Copy)]
pub struct Elem<T: Element> {
  _m: marker::PhantomData<T>
}

impl<T: Element> Default for Elem<T> {
  fn default() -> Self { Self{_m: <_>::default()} }
}

impl<T: Element> fmt::Debug for Elem<T> {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{}", T::rtype())
  }
} 

pub trait ContextBase<'c>: 'c {
  type Buffer: fmt::Debug;
  type Executor;
  type Source;
  fn compile(&'c self, sources: impl IntoIterator<Item=Option<Self::Source>>) -> Self::Executor;
}

#[derive(Debug)]
pub struct TensorBase<'c, C: ContextBase<'c>> {
  context: &'c C,
  shape: Shape,
  buffer: C::Buffer,
}

impl<'c, C: ContextBase<'c>> TensorBase<'c, C> {
  pub fn new(context: &'c C, shape: Shape, buffer: C::Buffer) -> Self {
    Self{context, shape, buffer}
  }
  pub fn buffer(&self) -> &C::Buffer { &self.buffer }
  pub fn buffer_mut(&mut self) -> &mut C::Buffer { &mut self.buffer }
  pub fn into_buffer(self) -> C::Buffer { self.buffer }
}

impl<'c, T: Element, C: Context<'c, T>> From<Tensor<'c, T, C>> for TensorBase<'c, C> {
  fn from(tensor: Tensor<'c, T, C>) -> Self { tensor.base }
}

pub trait Context<'c, T: Element>: ContextBase<'c> {}

pub trait FromNative<'c, T: Element, C: Context<'c, T>> {
  fn from_native(native_tensor: Tensor<'static, T, Native>, executor: &C::Executor) -> Self;
}

pub trait ToNative<'c, T: Element, C: Context<'c, T>> {
  fn to_native(self, executor: &C::Executor) -> Tensor<'static, T, Native>;
}

#[derive(Debug)]
pub struct Tensor<'c, T: Element, C: Context<'c, T>> {
  base: TensorBase<'c, C>,
  elem: Elem<T>
}

impl<'c, T: Element, C: Context<'c, T>> Tensor<'c, T, C> {
  pub fn new(context: &'c C, shape: Shape, buffer: C::Buffer) -> Self { 
    TensorBase::new(context, shape, buffer).into()
  } 
  pub fn context(&self) -> &'c C { self.base.context }
  pub fn shape(&self) -> &Shape { &self.base.shape }  
  pub fn base(&self) -> &TensorBase<'c, C> { &self.base }
  pub fn base_mut(&mut self) -> &mut TensorBase<'c, C> { &mut self.base }
  pub fn into_base(self) -> TensorBase<'c, C> { self.base }
}

impl<'c, T: Element, C: Context<'c, T>> From<TensorBase<'c, C>> for Tensor<'c, T, C> {
  fn from(base: TensorBase<'c, C>) -> Self {
    Self{base, elem: <_>::default()}
  }
}

impl<'c, T: Element, C: Context<'c, T>> From<&TensorBase<'c, C>> for &Tensor<'c, T, C> {
  fn from(base: &TensorBase<'c, C>) -> Self {
    unsafe { mem::transmute(base) }
  }
}


pub trait OpBase {
  fn shape(&self) -> &Shape;
}

pub trait ForwardOp<'c, C: ContextBase<'c>>: OpBase {
  fn source(&self) -> Option<C::Source> { None }
  fn forward(&self, executor: &C::Executor, tensors: &[TensorBase<'c, C>]) -> TensorBase<'c, C>;
} 

pub struct ForwardGraph<'c, C: ContextBase<'c>> {
  ops: cell::RefCell<Vec<Box<ForwardOp<'c, C>>>>
}

impl<'c, C: ContextBase<'c>> ForwardGraph<'c, C> {
  fn new() -> Self { Self{ops: cell::RefCell::new(Vec::new())} }
}

impl<'c, C: ContextBase<'c>> ForwardGraph<'c, C> {
  pub fn len(&self) -> usize { 
    self.ops.borrow()
      .len()
  }
  pub unsafe fn op<'g, T: Element>(&'g self, op: impl ForwardOp<'c, C> + 'static) -> Vertex<'g, T, Self>
    where C: Context<'c, T> {
    let idx = self.ops.borrow()
      .len();
    let vertex = Vertex::new(self, idx, *op.shape());
    self.ops.borrow_mut()
      .push(Box::new(op));
    vertex
  } 
}

pub fn forward<'c, T: Element, C: Context<'c, T>>(context: &'c C, mut f: impl FnMut(&ForwardGraph<'c, C>)->VertexBase<T>) -> Tensor<'static, T, Native>
  where Tensor<'c, T, C>: ToNative<'c, T, C> {
  let graph = ForwardGraph::new();
  let y = f(&graph);
  graph.ops.borrow_mut()
    .truncate(y.idx()+1);
  let executor = context.compile(graph.ops.borrow()
    .iter()
    .map(|op| op.source()));
  let mut tensors = Vec::<TensorBase<'c, C>>::with_capacity(graph.ops.borrow().len()-1);
  unsafe { tensors.set_len(tensors.capacity()); }
  graph.ops.borrow()
    .iter()
    .take(tensors.len())
    .enumerate()
    .for_each(|(i, op)| tensors[i] = op.forward(&executor, &tensors[..i]));
  let out: Tensor::<T, C> = graph.ops.into_inner()
    .last()
    .unwrap()
    .forward(&executor, &tensors[..])
    .into();
  out.to_native(&executor)
}


#[derive(Debug, Clone, Copy)]
pub struct VertexBase<T: Element> {
  idx: usize,
  shape: Shape,
  elem: Elem<T>
}

impl<T: Element> VertexBase<T> {
  fn new(idx: usize, shape: Shape) -> Self {
    Self{idx, shape, elem: <_>::default()}
  }
  pub fn idx(&self) -> usize { self.idx }
  pub fn shape(&self) -> &Shape { &self.shape }
}

impl<'g, T: Element, G> From<Vertex<'g, T, G>> for VertexBase<T> {
  fn from(vertex: Vertex<'g, T, G>) -> Self { vertex.base }
}

pub struct Vertex<'g, T: Element, G> {
  graph: &'g G,
  base: VertexBase<T>
}

impl<'g, T: Element, G> Vertex<'g, T, G> {
  pub fn new(graph: &'g G, idx: usize, shape: Shape) -> Self {
    Self{base: VertexBase::new(idx, shape), graph}
  }
  pub fn graph(&self) -> &'g G { self.graph }
  pub fn idx(&self) -> usize { self.base.idx() }
  pub fn shape(&self) -> &Shape { self.base.shape() }
  pub fn base(&self) -> &VertexBase<T> { &self.base }
} 

pub trait Uninitialized<T> {
  type Output;
  unsafe fn uninitialized(&self, shape: Shape) -> Self::Output;
}

pub trait Zeros<T> {
  type Output;
  fn zeros(&self, shape: Shape) -> Self::Output;
}

impl<'g, 'c, T: Element, C: Context<'c, T>> Zeros<T> for &'g ForwardGraph<'c, C>
  where ZerosOp<T>: ForwardOp<'c, C> {
  type Output = Vertex<'g, T, ForwardGraph<'c, C>>;
  fn zeros(&self, shape: Shape) -> Self::Output {
    unsafe { self.op(ZerosOp::new(VertexBase::new(self.len(), shape))) }
  }
}

pub struct ZerosOp<T: Element> {
  vertex: VertexBase<T>
}

impl<T: Element> ZerosOp<T> {
  fn new(vertex: VertexBase<T>) -> Self {
    Self{vertex}
  }
  pub fn vertex(&self) -> &VertexBase<T> { &self.vertex }
}

impl<T: Element> OpBase for ZerosOp<T> {
  fn shape(&self) -> &Shape { self.vertex().shape() }
}

pub trait Ones<T> {
  type Output;
  fn ones(&self, shape: Shape) -> Self::Output;
}

impl<'g, 'c, T: Element, C: Context<'c, T>> Ones<T> for &'g ForwardGraph<'c, C>
  where OnesOp<T>: ForwardOp<'c, C> {
  type Output = Vertex<'g, T, ForwardGraph<'c, C>>;
  fn ones(&self, shape: Shape) -> Self::Output {
    unsafe { self.op(OnesOp::new(VertexBase::new(self.len(), shape))) }
  }
}

pub struct OnesOp<T: Element> {
  vertex: VertexBase<T>
}

impl<T: Element> OnesOp<T> {
  fn new(vertex: VertexBase<T>) -> Self {
    Self{vertex}
  }
  pub fn vertex(&self) -> &VertexBase<T> { &self.vertex }
}

impl<T: Element> OpBase for OnesOp<T> {
  fn shape(&self) -> &Shape { self.vertex().shape() }
}

//#[derive(Debug)]
//pub enum Layout {
  

//pub trait Gemm


