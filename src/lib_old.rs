#![allow(dead_code, unused)]
use std::{cell, convert, marker, fmt, ops};

pub type Ix = u16;
pub type Sx = i32;

type Indices = smallvec::SmallVec<[Ix; 4]>;
type Strides = smallvec::SmallVec<[Sx; 4]>;

#[derive(Default, Clone, Eq, PartialEq)]
pub struct Shape {  
  dims: Indices,
  strides: Strides,
}

impl Shape {
  pub fn new(shape: impl ShapeBuilder) -> Self {
    shape.shape()
  }
  pub fn build(dims: impl AsRef<[Ix]>, strides: impl AsRef<[Sx]>) -> Self {
    let dims = Indices::from_slice(dims.as_ref());
    let strides = Strides::from_slice(strides.as_ref());
    debug_assert_eq!(dims.len(), strides.len(), "{:?}.len() != {:?}.len()", &dims, &strides);
    Self{dims, strides} 
  }
  pub fn default_strides(dims: &[Ix]) -> Strides {
    let mut strides = Strides::with_capacity(dims.len());
    unsafe { strides.set_len(dims.len()); }
    strides.iter_mut()
      .zip(dims.iter())
      .fold(1, |acc, (s, &i)| {
      *s = acc;
      acc * i as Sx
    });
    strides
  }   
  pub fn size(&self) -> Ix {
    self.dims.iter().product::<Ix>()
  } 
  pub fn dims(&self) -> &[Ix] { self.dims.as_slice() }
  pub fn strides(&self) -> &[Sx] { self.strides.as_slice() }
}

impl fmt::Debug for Shape {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    if self.strides() == Self::default_strides(self.dims()).as_slice() {
      write!(f, "{:?}, strides: {:?}", self.dims(), self.strides())
    }
    else {
      write!(f, "{:?}", self.dims())
    }
  }
}

pub trait ShapeBuilder {
  fn shape(&self) -> Shape;
}

impl ShapeBuilder for Shape {
  fn shape(&self) -> Shape { self.clone() }
}

impl<D: AsRef<[Ix]>> ShapeBuilder for D {
  fn shape(&self) -> Shape {
    let dims = Indices::from_slice(self.as_ref());
    let strides = Shape::default_strides(dims.as_slice());
    debug_assert_eq!(dims.len(), strides.len(), "{:?}.len() != {:?}.len()", &dims, &strides);
    Shape{dims, strides} 
  }
} 

pub trait Element: 'static + Copy + Default + fmt::Debug {
  fn rtype() -> &'static str;
  fn ctype() -> &'static str;
}  

impl Element for f32 {
  fn rtype() -> &'static str { "f32" }
  fn ctype() -> &'static str { "float" }
}

#[derive(Clone)]
pub struct Vertex<T> {
  idx: usize,
  shape: Shape,
  _m: marker::PhantomData<T>
}

impl<T: Element> Vertex<T> {
  pub fn new(idx: usize, shape: Shape) -> Self {
    Self{idx, shape, _m: <_>::default()}
  }
  pub fn idx(&self) -> usize { self.idx }
  pub fn shape(&self) -> &Shape { &self.shape }
}

impl<T: Element> fmt::Debug for Vertex<T> {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Vertex<{}>{{idx: {:?}, dims: {:?}}}", T::rtype(), self.idx, self.shape)    
  }
}

#[derive(Default, Clone)]
pub struct Buffer<T: Element> {
  data: smallvec::SmallVec<[T; 128]>
}

impl<T: Element> ops::Deref for Buffer<T> {
  type Target = [T];
  fn deref(&self) -> &[T] {
    self.data.as_slice()
  }
}

impl<T: Element> ops::DerefMut for Buffer<T> {
  fn deref_mut(&mut self) -> &mut [T] {
    self.data.as_mut_slice()
  }
}

impl<T: Element> Buffer<T> {
  pub fn new(data: impl AsRef<[T]>) -> Self {
    Self{data: smallvec::SmallVec::from_slice(data.as_ref())}
  }
  pub fn as_slice<'a>(&'a self) -> &'a [T] { self.data.as_slice() }
}

impl<T: Element> fmt::Debug for Buffer<T> {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{:?}", self.data.as_ptr())
  }
}

#[derive(Default, Clone)]
pub struct Tensor<T: Element> {
  shape: Shape,
  buffer: Option<Buffer<T>>
}

impl<T: Element> Tensor<T> {
  pub fn new(shape: impl ShapeBuilder, buffer: Buffer<T>) -> Self {
    Self{shape: shape.shape(), buffer: Some(buffer)}
  }
  pub fn shape(&self) -> &Shape { &self.shape }
  pub fn buffer(&self) -> Option<&Buffer<T>> { self.buffer.as_ref() }
  pub fn as_slice(&self) -> Option<&[T]> { self.buffer.as_ref().map(|b| b.as_slice()) }
  fn buffer_mut(&mut self) -> Option<&mut Buffer<T>> { self.buffer.as_mut() }
}

impl<T: Element> From<Shape> for Tensor<T> {
  fn from(shape: Shape) -> Self {
    Self{shape, buffer: None}
  }
}

impl<T: Element> From<&Shape> for Tensor<T> {
  fn from(shape: &Shape) -> Self {
    Self::from(shape.clone())
  }
}

impl<T: Element> fmt::Debug for Tensor<T> {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Tensor<{}>{{{:?}, buffer: {:?}}}", T::rtype(), self.shape, self.buffer)    
  }
}

#[derive(Debug, Clone)]
pub enum Op<T: Element> {
  Zero(Vertex<T>)
}

pub trait Workspace<T: Element> {
  fn vertex(&self, tensor: Tensor<T>) -> Vertex<T>;
  fn tensor(&self, vertex: &Vertex<T>) -> Tensor<T>; 
  fn exec(&self, op: &Op<T>);
} 

#[derive(Default, Clone, Debug)]
pub struct Graph<T: Element, W: Workspace<T>> {
  ws: W,
  ops: cell::RefCell<Vec<Op<T>>>,
}

impl<T: Element, W: Workspace<T>> Graph<T, W> {
  pub fn new<C: Context<T, Workspace=W>>(ctx: &C) -> Self { 
    Self{ws: ctx.workspace(), ops: cell::RefCell::new(Vec::new())}
  }
  pub fn var<'g>(&'g self, tensor: impl Into<Tensor<T>>) -> Var<'g, T, W> {
    Var::new(self, self.ws.vertex(tensor.into()))
  } 
  pub fn exec(&self) {
    self.ops.borrow()
      .iter()
      .for_each(|op| self.ws.exec(op));
  }
  fn push(&self, op: Op<T>) {
    self.ops.borrow_mut()
      .push(op);
  }
  fn tensor(&self, vertex: &Vertex<T>) -> Tensor<T> {
    self.ws.tensor(vertex)
  }
}

pub struct Var<'g, T: Element, W: Workspace<T>> {
  graph: &'g Graph<T, W>,
  vertex: Vertex<T>
}

impl<'g, T: Element, W: Workspace<T>> Var<'g, T, W> {
  fn new(graph: &'g Graph<T, W>, vertex: Vertex<T>) -> Self {
    Self{graph, vertex}
  }
  pub fn shape(&self) -> &Shape { self.vertex.shape() }
  pub fn zero(&self) {
    self.graph.push(Op::Zero(self.vertex.clone()))
  }
  pub fn tensor(&self) -> Tensor<T> {
    let tensor = self.graph.tensor(&self.vertex);
    debug_assert_eq!(tensor.shape(), self.shape());
    debug_assert!(tensor.buffer().is_some(), true);
    tensor
  }
}

pub trait Context<T: Element> {
  type Workspace;
  fn workspace(&self) -> Self::Workspace;
}

pub mod native;
pub use native::Native;

