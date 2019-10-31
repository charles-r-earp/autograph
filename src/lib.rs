use std::{ops::{Deref}, iter::{FromIterator, Sum}, fmt, fmt::Debug, mem, alloc, slice, rc::Rc, cell::RefCell, sync::{Arc, Mutex, MutexGuard}, ops::{AddAssign, SubAssign, Mul}};
use matrixmultiply::{sgemm, dgemm};

pub mod datasets;
pub mod iter;

#[derive(Default, Clone, Copy, Eq, PartialEq)]
pub struct Shape {
  dims: [usize; 4],
  len: usize,
  is_t: bool
}

pub trait IntoShape {
  fn into_shape(self) -> Shape;
} 

impl IntoShape for usize {
  fn into_shape(self) -> Shape {
    Shape {
      dims: [self, 0, 0, 0],
      len: 1,
      is_t: false
    }
  }
}

impl IntoShape for [usize; 1] {
  fn into_shape(self) -> Shape {
    Shape {
      dims: [self[0], 0, 0, 0],
      len: 1,
      is_t: false
    }
  }
}

impl IntoShape for [usize; 2] {
  fn into_shape(self) -> Shape {
    Shape {
      dims: [self[0], self[1], 0, 0],
      len: 2,
      is_t: false
    }
  }
}

impl IntoShape for [usize; 3] {
  fn into_shape(self) -> Shape {
    Shape {
      dims: [self[0], self[1], self[2], 0],
      len: 3,
      is_t: false
    }
  }
}

impl IntoShape for [usize; 4] {
  fn into_shape(self) -> Shape {
    Shape {
      dims: [self[0], self[1], self[2], self[3]],
      len: 4,
      is_t: false
    }
  }
}

impl IntoShape for Shape {
  fn into_shape(self) -> Shape { self }
}

impl Shape {
  pub fn len(&self) -> usize { self.len }
  pub fn size(&self) -> usize {
    self.iter().product()
  }
  pub fn dims<A: Default + AsMut<[usize]>>(&self) -> A {
    let mut dims = A::default();
    dims.as_mut()
      .copy_from_slice(self);
    dims
  }
  pub fn t(mut self) -> Shape {
    if self.len == 1 {
      self.len = 2;
      self.dims[1] = 1;
    }
    else if self.len >= 2 {
      let rows = self.dims[self.len-1];
      self.dims[self.len-1] = self.dims[self.len-2];
      self.dims[self.len-2] = rows;
    }
    self.is_t = !self.is_t;
    self
  }
  pub fn strides<A: Default + AsMut<[isize]>>(&self) -> A {
    let mut strides = A::default();
    strides.as_mut()
      .iter_mut()
      .zip(self.dims.iter().take(self.len).copied())
      .rev()
      .fold(1, |acc, (s, d)| { *s = acc as isize; acc * d });
    if self.is_t {
      strides.as_mut()[1] = strides.as_mut()[0];
      strides.as_mut()[0] = 1;
    }
    strides
  }
  pub fn stack(&self, n: usize) -> Self {
    debug_assert!(!self.is_t);
    debug_assert!(self.len < self.dims.len());
    let mut shape = Self::default();
    shape.len = self.len + 1;
    shape.dims[0] = n;
    shape.dims[1..shape.len].copy_from_slice(&self.dims[..self.len]);
    shape
  }
}

impl Deref for Shape {
  type Target = [usize];
  fn deref(&self) -> &Self::Target {
    &self.dims[..self.len]
  }
}

impl Debug for Shape {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    self.dims[..self.len].fmt(f)
  }
}

pub trait DataRef {
  type Elem;
  fn as_slice(&self) -> &[Self::Elem];
}

pub trait DataMut: DataRef {
  fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

#[derive(Debug)]
pub struct Buffer<T>(Box<[T]>);

impl<T> Buffer<T> {
  unsafe fn uninitialized(len: usize) -> Self {
    let layout = alloc::Layout::from_size_align_unchecked(mem::size_of::<T>()*len, mem::align_of::<T>());
    let p = alloc::alloc(layout) as *mut T;
    Buffer(Box::from_raw(slice::from_raw_parts_mut(p, len) as *mut [T]))
  }
  pub fn zeros(len: usize) -> Self {
    unsafe {
      let layout = alloc::Layout::from_size_align_unchecked(mem::size_of::<T>()*len, mem::align_of::<T>());
      let p = alloc::alloc_zeroed(layout) as *mut T;
      Buffer(Box::from_raw(slice::from_raw_parts_mut(p, len) as *mut [T]))
    }
  }
}

impl<T> Default for Buffer<T> {
  fn default() -> Self { Buffer(Box::new([])) }
}

impl<T> From<Vec<T>> for Buffer<T> {
  fn from(data: Vec<T>) -> Self {
    Buffer(data.into_boxed_slice())
  }
}

impl<T> DataRef for Buffer<T> {
  type Elem = T;
  fn as_slice(&self) -> &[T] { &*self.0 }
}

impl<T> DataMut for Buffer<T> {
  fn as_mut_slice(&mut self) -> &mut [T] { &mut *self.0 }
}

#[derive(Debug)]
pub struct BufferRef<'a, T>(&'a [T]);

impl<'a, T> DataRef for BufferRef<'a, T> {
  type Elem = T;
  fn as_slice(&self) -> &[T] { &*self.0 }
}

#[derive(Debug)]
pub struct BufferMut<'a, T>(&'a mut [T]);

impl<'a, T> DataRef for BufferMut<'a, T> {
  type Elem = T;
  fn as_slice(&self) -> &[T] { &*self.0 }
}

impl<'a, T> DataMut for BufferMut<'a, T> {
  fn as_mut_slice(&mut self) -> &mut [T] { &mut *self.0 }
}
  
#[derive(Debug, Default)]
pub struct TensorBase<D> {
  shape: Shape,
  data: D
}

impl<D> TensorBase<D> {
  pub fn shape(&self) -> Shape { self.shape }
}

pub type Tensor<T> = TensorBase<Buffer<T>>;
pub type TensorRef<'a, T> = TensorBase<BufferRef<'a, T>>;
pub type TensorMut<'a, T> = TensorBase<BufferMut<'a, T>>;

impl<T> Tensor<T> {
  fn from_shape_data(shape: impl IntoShape, data: impl Into<Buffer<T>>) -> Self {
    Self{shape: shape.into_shape(), data: data.into()}
  }
  unsafe fn uninitialized(shape: impl IntoShape) -> Self {
    let shape = shape.into_shape();
    Self::from_shape_data(shape, Buffer::uninitialized(shape.size()))
  }
  pub fn zeros(shape: impl IntoShape) -> Self {
    let shape = shape.into_shape();
    Self::from_shape_data(shape, Buffer::zeros(shape.size()))
  }
  pub fn ones(shape: impl IntoShape) -> Self
    where T: num_traits::One + Copy {
    let shape = shape.into_shape();
    Self::from_shape_data(shape, vec![T::one(); shape.size()])
  }
  pub fn from_shape_iter(shape: impl IntoShape, iter: impl IntoIterator<Item=T>) -> Self {
    let shape = shape.into_shape();
    Self::from_iter(iter.into_iter().take(shape.size()))
      .into_shape(shape)
  }
}

impl<T, D: DataRef<Elem=T>> TensorBase<D> {
  pub fn into_shape(self, shape: impl IntoShape) -> Self {
    let shape = shape.into_shape();
    debug_assert_eq!(shape.size(), self.shape.size());
    Self{shape, data: self.data}
  }
  pub fn _ref(&self) -> TensorRef<T> {
    TensorRef{shape: self.shape, data: BufferRef(self.data.as_slice())}
  }
  pub fn _mut(&mut self) -> TensorMut<T>
    where D: DataMut<Elem=T> {
    TensorMut{shape: self.shape, data: BufferMut(self.data.as_mut_slice())}
  }
  pub fn t(self) -> TensorBase<D> { let shape = self.shape.t(); self.into_shape(shape) }
  pub fn stack(&self, n: usize) -> Tensor<T>
    where T: Copy {
    Tensor::from_iter(self.data.as_slice().iter().copied().cycle().take(n*self.shape().size()))
      .into_shape(self.shape().stack(n))
  }
  fn map(&self, mut f: impl FnMut(T)->T) -> Tensor<T>
    where T: Copy {
    self.data.as_slice()
      .iter()
      .copied()
      .map(f)
      .collect::<Tensor<T>>()
      .into_shape(self.shape())
  }
  fn zip_mut_for_each<D2: DataRef<Elem=T>>(&mut self, rhs: &TensorBase<D2>, mut f: impl FnMut(&mut T, T))
    where T: Copy,
          D: DataMut<Elem=T> {
    debug_assert_eq!(self.shape, rhs.shape);
    self.data.as_mut_slice()
      .iter_mut()
      .zip(rhs.data.as_slice().iter().copied())
      .for_each(|(a, b)| f(a, b));
  }  
  pub fn as_slice(&self) -> &[T] {
    debug_assert!(!self.shape.is_t);
    self.data.as_slice()
  }
  pub fn as_mut_slice(&mut self) -> &mut [T]
    where D: DataMut<Elem=T> {
    debug_assert!(!self.shape.is_t);
    self.data.as_mut_slice()
  }
  fn as_ptr(&self) -> *const T {
    self.data.as_slice().as_ptr()
  }
  fn as_mut_ptr(&mut self) -> *mut T
    where D: DataMut<Elem=T> {
    self.data.as_mut_slice().as_mut_ptr()
  }
}

impl<T> FromIterator<T> for Tensor<T> {
  fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
    let data = Vec::from_iter(iter);
    Self::from_shape_data(data.len(), data)
  }
}

impl<T: Copy + AddAssign<T>, D1: DataMut<Elem=T>, D2: DataRef<Elem=T>> AddAssign<TensorBase<D2>> for TensorBase<D1> {
  fn add_assign(&mut self, rhs: TensorBase<D2>) {
    self.zip_mut_for_each(&rhs, |a, b| *a += b);
  }
}

pub trait Float: num_traits::Float + num_traits::NumAssign + Sum {
  fn gemm(a: &TensorRef<Self>, b: &TensorRef<Self>, c: Option<Tensor<Self>>) -> Tensor<Self>;
}

macro_rules! impl_float {
  ($t:ty, $gemm:ident) => (
    impl Float for $t {
      fn gemm(a: &TensorRef<Self>, b: &TensorRef<Self>, c: Option<Tensor<Self>>) -> Tensor<Self> {
        let [m, k] = a.shape().dims::<[usize; 2]>();
        let alpha = 1.;
        let [rsa, csa] = a.shape().strides::<[isize; 2]>();
        let [_k, n] = b.shape().dims::<[usize; 2]>();
        debug_assert_eq!(k, _k);
        let [rsb, csb] = b.shape().strides::<[isize; 2]>();
        let beta = if c.is_some() { 1. } else { 0. };
        let mut c = c.unwrap_or(unsafe { Tensor::uninitialized([m, n]) });
        debug_assert_eq!([m, n], c.shape().dims::<[usize; 2]>());
        let [rsc, csc] = c.shape().strides::<[isize; 2]>();
        unsafe { 
          $gemm(
            m, k, n,
            alpha,
            a.as_ptr(),
            rsa, csa,
            b.as_ptr(),
            rsb, csb,
            beta,
            c.as_mut_ptr(),
            rsc, csc
          )
        };
        c
      }
    }
  )
}

impl_float!(f32, sgemm);
impl_float!(f64, dgemm);

impl<T: Float, D: DataRef<Elem=T>> TensorBase<D> {
  pub fn gemm<D2: DataRef<Elem=T>>(&self, rhs: &TensorBase<D2>, out: Option<Tensor<T>>) -> Tensor<T> {
    T::gemm(&self._ref(), &rhs._ref(), out)
  }
  pub fn mm<D2: DataRef<Elem=T>>(&self, rhs: &TensorBase<D2>) -> Tensor<T> {
    self.gemm(rhs, None)
  }
}

#[derive(Default)]
pub struct Graph<T> {
  tensors: RefCell<Vec<Rc<Tensor<T>>>>
}

impl<T> Graph<T> {
  fn push(&self, value: Rc<Tensor<T>>) {
    self.tensors.borrow_mut()
      .push(value);
  }
  fn last(&self) -> Option<Rc<Tensor<T>>> {
    self.tensors.borrow()
      .last()
      .map(|x| Rc::clone(x))
  }
  fn pop(&self) -> Option<Rc<Tensor<T>>> {
    self.tensors.borrow_mut()
      .pop()
  }
}

pub struct Var<T> {
  graph: Rc<Graph<T>>,
  value: Rc<Tensor<T>>
}

impl<T> Var<T> {
  pub fn new(graph: &Rc<Graph<T>>, value: Tensor<T>) -> Self {
    let value = Rc::new(value);
    graph.push(Rc::clone(&value));
    Self{graph: Rc::clone(graph), value}
  }
  pub fn graph(&self) -> &Rc<Graph<T>> { &self.graph }
  pub fn value(&self) -> &Rc<Tensor<T>> {
    debug_assert!(Rc::ptr_eq(&self.value, &self.graph.last().unwrap()));
    &self.value 
  }    
}

pub struct Grad<T> {
  graph: Rc<Graph<T>>,
  input: Rc<Tensor<T>>,
  value: Rc<Tensor<T>>
}

impl<T> Grad<T> {
  pub fn new(graph: &Rc<Graph<T>>, value: Tensor<T>) -> Self {
    Self{graph: Rc::clone(graph), input: graph.pop().unwrap(), value: Rc::new(value)}
  }
  pub fn graph(&self) -> &Rc<Graph<T>> { &self.graph }
  pub fn input(&self) -> &Rc<Tensor<T>> { &self.input }
  pub fn value(&self) -> &Rc<Tensor<T>> { &self.value }
}

#[derive(Default)]
pub struct Param<T> {
  value: Tensor<T>,
  grad: Option<Arc<Mutex<Tensor<T>>>>
}

impl<T> Param<T> {
  pub fn new(value: Tensor<T>) -> Self {
    Self{value, grad: None}
  }
  pub fn value(&self) -> &Tensor<T> { &self.value }
  pub fn zero_grad(&mut self) {
    self.grad.replace(Arc::new(Mutex::new(Tensor::zeros(self.value.shape()))));
  }
  pub fn none_grad(&mut self) {
    self.grad = None;
  }
  pub fn grad(&self) -> Option<MutexGuard<Tensor<T>>> {
    self.grad.as_ref()
      .map(|grad| grad.lock().unwrap())
  }
  pub fn step(&mut self, lr: T)
    where T: Copy + SubAssign<T> + Mul<T, Output=T> {
    if let Some(ref grad) = self.grad {
      self.value.zip_mut_for_each(&grad.lock().unwrap(), |w, dw| *w -= lr * dw);
    }
    self.none_grad();
  }
}

pub trait Linear<T> {
  fn linear(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Self;
}

impl<T: Float> Linear<T> for Tensor<T> {
  fn linear(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Self {
    let batch_size = self.shape()[0];
    let channels = self.shape()[1..].iter().product();
    self._ref()
      .into_shape([batch_size, channels])
      .gemm(kernel.value(), bias.map(|bias| bias.value().stack(batch_size)))
  }
}

impl<T: Float> Linear<T> for Var<T> {
  fn linear(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Self {
    Self::new(self.graph(), self.value().linear(kernel, bias))
  }
}

pub trait LinearBackward<T>: Sized {
  fn linear_backward(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Option<Self>;
}

impl<T: Float> LinearBackward<T> for Grad<T> {
  fn linear_backward(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Option<Self> {
    let input = self.input();
    let batch_size = input.shape()[0];
    let channels = input.shape()[1..].iter().product();
    let input = self.input._ref()
      .into_shape([batch_size, channels]);
    if let Some(mut kernel_grad) = kernel.grad() {
      *kernel_grad += input._ref().t().mm(self.value());
    }
    if let Some(bias) = bias {
      if let Some(mut bias_grad) = bias.grad() { 
        *bias_grad += Tensor::ones([1, batch_size])
          .mm(self.value())
          .into_shape(bias.value().shape());
      }
    }
    self.graph.last().map(|_| 
      Self::new(self.graph(), self.value().mm(&kernel.value()._ref().t()))
    )
  }
}

pub trait Softmax {
  fn softmax(&self) -> Self;
}

impl<T: Float> Softmax for Tensor<T> {
  fn softmax(&self) -> Self {
    let [batch_size, nclasses] = self.shape().dims::<[usize; 2]>();
    let mut out = self.map(|x| x.exp());
    out.as_mut_slice()
      .chunks_exact_mut(nclasses)
      .for_each(|p| {
      let sum = p.iter().copied().sum();
      p.iter_mut().for_each(|p| *p /= sum);
    });
    out
  }
} 


pub trait CrossEntropy<T, U> {
  fn cross_entropy_loss(&self, target: &Tensor<U>) -> (T, Grad<T>);
} 

impl<T: Float, U: num_traits::AsPrimitive<usize>> CrossEntropy<T, U> for Var<T> {
  fn cross_entropy_loss(&self, target: &Tensor<U>) -> (T, Grad<T>) {
    let [batch_size, nclasses] = self.value().shape().dims::<[usize; 2]>();
    debug_assert_eq!([batch_size], target.shape().dims::<[usize; 1]>());
    let mut loss = T::zero();
    let mut pred = self.value().softmax();
    pred.as_mut_slice()
      .chunks_exact_mut(nclasses)
      .zip(target.as_slice().iter().copied().map(|t| t.as_()))
      .for_each(|(p, t)| { 
      loss += -p[t].ln(); 
      p[t] += -T::one();
    });
    self.graph().pop();
    let loss_grad = Grad::new(self.graph(), pred);
    (loss / T::from(batch_size).unwrap(), loss_grad)
  }
}
      
  


