use std::{iter, slice, ops};

pub unsafe trait Element: ocl::OclPrm + 'static {
  fn rtype() -> String;
  fn ctype() -> String;
  fn is_real() -> bool;
  fn zero() -> Self;
  fn one() -> Self;
}

unsafe impl Element for f32 {
  fn rtype() -> String { String::from("f32") }
  fn ctype() -> String { String::from("float") }
  fn is_real() -> bool { true }
  fn zero() -> Self { 0. }
  fn one() -> Self { 1. }
}

pub unsafe trait Real: Element {}

unsafe impl Real for f32 {}

pub fn source() -> String {
  _source(std::marker::PhantomData::<f32>::default())
}

fn _source<T: Element>(_m: std::marker::PhantomData<T>) -> String {
  include_str!("autograph.cl")
    .replace("RTYPE", &T::rtype())
    .replace("CTYPE", &T::ctype())
    .replace("IS_REAL", &T::is_real().to_string())
    .to_string()
}

#[derive(Debug)]
pub struct Workspace {
  context: ocl::Context,
  program: ocl::Program,
  queue: ocl::Queue
}

impl Workspace {
  pub fn new(context: ocl::Context, src: String) -> Self {
    debug_assert_eq!(context.devices().len(), 1);
    let program = ocl::Program::builder()
      .src(src)
      .devices(0)
      .build(&context)
      .unwrap();
    let queue = ocl::Queue::new(&context, context.devices()[0], None).unwrap();
    Self{context, program, queue}
  }
  pub fn program(&self) -> &ocl::Program { &self.program }
  pub fn queue(&self) -> &ocl::Queue { &self.queue }
}

pub trait Executor {
  fn enq<I: iter::IntoIterator<Item=ocl::Kernel>>(&mut self, kernels: I)
    where I::IntoIter: iter::DoubleEndedIterator;
}

#[derive(Default)]
pub struct Forward;

pub struct Backward {
  kernels: Vec<ocl::Kernel>
}

impl Default for Backward {
  fn default() -> Self { Self{kernels: Vec::new()} }
}

pub struct Graph<'w, E> {
  workspace: &'w Workspace,
  exec: E
}

impl<'w, E> Graph<'w, E> {
  pub fn new(workspace: &'w Workspace) -> Self
    where E: Default { 
    Self{workspace, exec: E::default()}
  }
  pub fn workspace(&self) -> &'w Workspace { self.workspace }
  pub fn forward(&self) -> Graph<'w, Forward> { Graph::new(self.workspace) }
  /*fn backward(&mut self, ) { 
    kernels.iter()
      .for_each(|k| unsafe { println!("{:?}", k.name()); k.enq().unwrap(); });
    kernels.clear();
  }*/
}

impl<'w> Executor for Graph<'w, Forward> {
  fn enq<I: iter::IntoIterator<Item=ocl::Kernel>>(&mut self, kernels: I)
    where I::IntoIter: iter::DoubleEndedIterator {
    kernels.into_iter().for_each(|k| unsafe { k.enq().unwrap(); });
  }
}

impl<'w> Executor for Graph<'w, Backward> {
  fn enq<I: iter::IntoIterator<Item=ocl::Kernel>>(&mut self, kernels: I)
    where I::IntoIter: iter::DoubleEndedIterator {
    self.exec.kernels.extend(kernels.into_iter().rev());
  }
}

#[derive(Debug)]
pub struct Tensor<T: Element> {
  dims: Vec<usize>,
  data: Option<Vec<T>>,
  buffer: ocl::Buffer<T>
}

impl<T: Element> Tensor<T> {
  pub fn new<'w>(ws: &'w Workspace, dims: Vec<usize>, data: Option<Vec<T>>) -> Self {
    let buffer = ocl::Buffer::builder()
      .queue(ws.queue().clone())
      .len(dims.iter().product::<usize>())
      .build()
      .unwrap();
    if let Some(ref data) = data {
      debug_assert_eq!(buffer.len(), data.len());
      buffer.write(data).enq().unwrap();
    }
    Self{dims, data, buffer}
  }
  pub fn dims(&self) -> &Vec<usize> { &self.dims } 
  pub fn data(&self) -> Option<&Vec<T>> { self.data.as_ref() }
  pub fn len(&self) -> usize { self.buffer.len() }
  pub fn buffer(&self) -> &ocl::Buffer<T> { &self.buffer }
  pub fn read(&mut self) {
    let mut data = vec![T::zero(); self.len()];
    self.buffer.read(&mut data).enq().unwrap();
    self.data = Some(data);
  } 
}

pub fn restrict<'a, 'b, T: Element>(lhs: &'a ocl::Buffer<T>, rhs: &'b ocl::Buffer<T>) -> bool {
  rhs.as_core().as_ptr() == lhs.as_core().as_ptr()
}

pub fn binary_restrict_sfx<'a, 'b, T: Element>(lhs: &'a ocl::Buffer<T>, rhs: &'b ocl::Buffer<T>) -> &'static str {
  if restrict(lhs, rhs) { "_restrict" } else { "" }
}

pub struct Variable<T: Real> {
  value: Tensor<T>,
  grad: Option<Tensor<T>>
}

impl<T: Real> Variable<T> {
  pub fn new(value: Tensor<T>, grad: Option<Tensor<T>>) -> Self {
    Self{value, grad}
  } 
  pub fn value(&self) -> &Tensor<T> { &self.value }
  pub fn value_mut(&mut self) -> &mut Tensor<T> { &mut self.value }
  pub fn grad(&self) -> Option<&Tensor<T>> { self.grad.as_ref() }
  pub fn grad_mut(&mut self) -> Option<&mut Tensor<T>> { self.grad.as_mut() }
}

impl<T: Real> From<Tensor<T>> for Variable<T> {
  fn from(value: Tensor<T>) -> Self {
    Self::new(value, None)
  }
}

pub trait Optimizer<T: Real>: Sized {
  type Payload;
  fn payload<'w, 'd>(&self, ws: &'w Workspace, dims: &'d Vec<usize>) -> Self::Payload;
  fn step<'g, 'w>(&self, graph: &'g mut Graph<'w, Forward>, param: &mut Parameter<T, Self>);
}

pub struct Parameter<T: Real, O: Optimizer<T>> {
  var: Variable<T>,
  payload: O::Payload
}

impl<T: Real, O: Optimizer<T>> Parameter<T, O> {
  pub fn new<'w, 'o>(ws: &'w Workspace, var: Variable<T>, opt: &'o O) -> Self {
    let payload = opt.payload(ws, var.value().dims());
    Self{var, payload}
  }
  pub fn var(&self) -> &Variable<T> { &self.var } 
  pub fn var_mut(&mut self) -> &mut Variable<T> { &mut self.var } 
}  
/*
pub struct Net<T: Real, O: Optimizer<T>> {
  params: Vec<Parameter<T, O>>,
  opt: O
} 

impl<T: Real, O: Optimizer<T>> Net<T, O> {
  pub fn new(params: Vec<Parameter<T, O>>, opt: O) -> Self {
    Self{params, opt}
  }
  pub fn opt(&self) -> &O { &self.opt }
  pub fn push(&mut self, param: Parameter<T, O>) {
    self.params.push(param);
  }
  pub fn iter<'n>(&'n self) -> NetIter<'n, T, O> {
    NetIter{iter: self.params.iter()}
  }
}

pub struct NetIter<'n, T: Real, O: Optimizer<T>> {
  iter: slice::Iter<'n, Parameter<T, O>> 
}

impl<'n, T: Real, O: Optimizer<T>> iter::Iterator for NetIter<'n, T, O> {
  type Item = &'n Tensor<T>;
  fn next(&mut self) -> Option<Self::Item> {
    self.iter.next().map(|p| p.var().value())
  }
}

pub struct NetIterMut<'n, T: Real, O: Optimizer<T>> {
  net: &'n mut Net<T, O>,
  i: usize 
}

impl<'n, T: Real, O: Optimizer<T>> NetIterMut<'n, T, O> {
  pub fn push(&mut self, 
  pub fn next<'a>(&'a mut self) -> Option<&'a mut Parameter<T, O>> {
    let p = self.net.params.get(self.i);
    self.i += 1;
    p
  }
}  */
        
// Unary Elementwise 

pub trait Sigmoid<A> {
  type Output;
  fn sigmoid(&mut self, input: A) -> Self::Output;
}

impl<'w, 'a, T: Real> Sigmoid<&'a Tensor<T>> for Graph<'w, Forward> {
  type Output = Tensor<T>;
  fn sigmoid(&mut self, input: &'a Tensor<T>) -> Self::Output {
    let ws = self.workspace();
    let out = Tensor::new(ws, input.dims.clone(), None);
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("sigmoid_{}", T::rtype()))
      .queue(ws.queue().clone())
      .global_work_size(input.len())
      .arg(out.buffer())
      .arg(input.buffer())
      .build()
      .unwrap();
    self.enq(iter::once(kernel));
    out
  }
}

pub trait SigmoidGrad<A> {
  type Output;
  fn sigmoid_grad(&mut self, input: A) -> Self::Output;
}

impl<'w, 'a, T: Real> SigmoidGrad<&'a Tensor<T>> for Graph<'w, Backward> {
  type Output = Tensor<T>;
  fn sigmoid_grad(&mut self, input: &'a Tensor<T>) -> Self::Output {
    let ws = self.workspace();
    let out = Tensor::new(ws, input.dims.clone(), None);
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("sigmoid_grad_{}", T::rtype()))
      .queue(ws.queue().clone())
      .global_work_size(input.len())
      .arg(out.buffer())
      .arg(input.buffer())
      .build()
      .unwrap();
    self.enq(iter::once(kernel));
    out
  }
}

impl<'w, 'a, T: Real> Sigmoid<&'a Variable<T>> for Graph<'w, Backward> {
  type Output = Variable<T>;
  fn sigmoid(&mut self, input: &'a Variable<T>) -> Self::Output {
    let out = Variable::new(self.forward().sigmoid(input.value()), input.grad().map(|input_grad| {
      let out_grad = Tensor::new(self.workspace, input_grad.dims().clone(), Some(vec![T::zero(); input_grad.len()]));
      let partial_grad = self.sigmoid_grad(input.value());
      let partial_grad = self.mul(&out_grad, &partial_grad);
      self.add_assign(input_grad, &partial_grad);
      out_grad
    }));
    out
  } 
}



// Unary 2d 

pub trait Transpose<A> {
  type Output;
  fn transpose(&mut self, input: A) -> Self::Output;
}

impl<'w, 'a, T: Element, E> Transpose<&'a Tensor<T>> for Graph<'w, E>
  where Graph<'w, E>: Executor {
  type Output = Tensor<T>;
  fn transpose(&mut self, input: &'a Tensor<T>) -> Self::Output {
    debug_assert_eq!(input.dims.len(), 2);
    let d0 = input.dims[0];
    let d1 = input.dims[1];
    let ws = self.workspace();
    let out = Self::Output::new(ws, vec![d1, d0], None);
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("transpose_v1_{}", T::rtype()))
      .queue(ws.queue().clone())
      .global_work_size([d1, d0])
      .arg(out.buffer())
      .arg(input.buffer())
      .build()
      .unwrap();
    self.enq(iter::once(kernel));
    out
  }
}

// Binary Elementwise

pub trait AddAssign<A, B> {
  fn add_assign(&mut self, lhs: A, rhs: B);
}

impl<'w, 'a, 'b, T: Element, E> AddAssign<&'a Tensor<T>, &'b Tensor<T>> for Graph<'w, E>
  where Graph<'w, E>: Executor {
  fn add_assign(&mut self, lhs: &'a Tensor<T>, rhs: &'b Tensor<T>) {
    debug_assert_eq!(lhs.dims(), rhs.dims());
    let ws = self.workspace();
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("add_assign{}", binary_restrict_sfx(lhs.buffer(), rhs.buffer())))
      .queue(ws.queue().clone())
      .global_work_size(lhs.len())
      .arg(lhs.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    self.enq(iter::once(kernel));
  }
}
pub trait Add<A, B> {
  type Output;
  fn add(&mut self, lhs: A, rhs: B) -> Self::Output;
}

impl<'w, 'a, 'b, T: Element, E> Add<&'a Tensor<T>, &'b Tensor<T>> for Graph<'w, E>
  where Graph<'w, E>: Executor {
  type Output = Tensor<T>;
  fn add(&mut self, lhs: &'a Tensor<T>, rhs: &'b Tensor<T>) -> Self::Output {
    debug_assert_eq!(lhs.dims(), rhs.dims());
    let ws = self.workspace();
    let out = Tensor::new(ws, lhs.dims().clone(), None);
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("add{}", binary_restrict_sfx(lhs.buffer(), rhs.buffer())))
      .queue(ws.queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(lhs.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    self.enq(iter::once(kernel));
    out
  }
}

pub trait Mul<A, B> {
  type Output;
  fn mul(&mut self, lhs: A, rhs: B) -> Self::Output;
}

impl<'w, 'a, 'b, T: Element, E> Mul<&'a Tensor<T>, &'b Tensor<T>> for Graph<'w, E>
  where Graph<'w, E>: Executor {
  type Output = Tensor<T>;
  fn mul(&mut self, lhs: &'a Tensor<T>, rhs: &'b Tensor<T>) -> Self::Output {
    debug_assert_eq!(lhs.dims(), rhs.dims());
    let ws = self.workspace();
    let out = Tensor::new(ws, lhs.dims().clone(), None);
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("mul{}", binary_restrict_sfx(lhs.buffer(), rhs.buffer())))
      .queue(ws.queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(lhs.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    self.enq(iter::once(kernel));
    out
  }
}

// Binary 2d

pub trait Matmul<A, B> {
  type Output;
  fn matmul(&mut self, lhs: A, rhs: B) -> Self::Output;
}

impl<'w, 'a, 'b, T: Element, E> Matmul<&'a Tensor<T>, &'b Tensor<T>> for Graph<'w, E>
  where Graph<'w, E>: Executor {
  type Output = Tensor<T>;
  fn matmul(&mut self, lhs: &'a Tensor<T>, rhs: &'b Tensor<T>) -> Self::Output {
    debug_assert_eq!(lhs.dims.len(), 2);
    debug_assert_eq!(rhs.dims.len(), 2);
    debug_assert_eq!(lhs.dims[1], rhs.dims[0]);
    let m = lhs.dims[0];
    let n = rhs.dims[1];
    let k = lhs.dims[1];
    let ws = self.workspace();
    let out = Tensor::new(ws, vec![m, n], None);
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("matmul_v1{}", binary_restrict_sfx(lhs.buffer(), rhs.buffer())))
      .queue(ws.queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(lhs.buffer())
      .arg(rhs.buffer())
      .arg(m)
      .arg(n)
      .arg(k)
      .build()
      .unwrap();
    self.enq(iter::once(kernel));
    out
  }
}

impl<'w, 'a, 'b, T: Real> Matmul<&'a Variable<T>, &'b Variable<T>> for Graph<'w, Backward> {
  type Output = Variable<T>;
  fn matmul(&mut self, lhs: &'a Variable<T>, rhs: &'b Variable<T>) -> Variable<T> {
    let out_value = self.forward().matmul(lhs.value(), rhs.value());
    let ws = self.workspace();
    let out_grad = if lhs.grad().is_some() || rhs.grad().is_some() {
      let out_grad = Tensor::new(ws, out_value.dims().clone(), Some(vec![T::zero(); out_value.len()]));
      lhs.grad().map(|lgrad| {
        let rvalue_t = self.transpose(rhs.value());
        let partial_grad = self.matmul(&out_grad, &rvalue_t);
        self.add_assign(&lgrad, &partial_grad);
      });
      rhs.grad().map(|rgrad| {
        let lvalue_t = self.transpose(lhs.value());
        let partial_grad = self.matmul(&lvalue_t, &out_grad);
        self.add_assign(rgrad, &partial_grad);
      });
      Some(out_grad)
    }
    else { None };
    Variable::new(out_value, out_grad)  
  }
}

/*
pub trait Weight<A, B, F> {
  type Output;
  fn weight(&mut self, lhs: A, rhs: B, n: usize, f: F) -> Self::Output;
} 

impl<'w, 'a, 'b, 'n, T: Real, O: Optimizer<T>, F: FnMut([usize; 2]) -> Vec<T>> Weight<&'a Tensor<T>, &'b mut NetIter<'n, T, O>, F> for Graph<'w, Forward> {
  type Output = Tensor<T>;
  fn weight(&mut self, lhs: &'a Tensor<T>, rhs: &'b mut NetIter<'n, T, O>, n: usize, f: F) -> Self::Output {
    let rhs = rhs.next().unwrap();
    debug_assert_eq!(rhs.dims().len(), 2);
    debug_assert_eq!(rhs.dims()[1], n);
    self.matmul(lhs, rhs)
  }
}

impl<'w, 'a, 'b, 'n, T: Real, O: Optimizer<T>, F: FnMut([usize; 2]) -> Vec<T>> Weight<&'a Variable<T>, &'b mut NetIterMut<'n, T, O>, F> for Graph<'w, Forward> {
  type Output = Tensor<T>;
  fn weight(&mut self, lhs: &'a Variable<T>, rhs: &'b mut NetIter<'n, T, O>, n: usize, f: F) -> Self::Output {
    debug_assert_eq!(lhs.value().dims().len(), 2);
    let m = lhs.value().dims()[0];
    let k = lhs.value().dims()[1];
    let ws = self.workspace();
    let rhs = if let Some(ref mut rhs) = rhs.next() {
      debug_assert_eq!(rhs.var().value().dims().len(), 2);
      if rhs.var().value().dims()[0] == k {
        rhs.var()
      }
      else {
        panic!()
      }
    }
    else {
      let value = Tensor::new(ws, vec![k, n], Some((f)([k, n])));
      let grad = Tensor::new(ws, vec![k, n], Some(vec![T::zero(); k*n]));
      rhs.push(ws, Paramter::new(Variable::new(value, Some(grad)), rhs.opt())
    }
    //self.matmul(lhs, rhs)
    panic!()
  }
}*/


/*
// Optimizers 

pub struct LearningRate<T: Real> {
  lr: T
}

impl<T: Real> LearningRate<T> {
  pub fn new(lr: T) -> Self { Self{lr} }
  pub fn lr(&self) -> T { self.lr }
  pub fn lr_mut(&mut self) -> &mut T { &mut self.lr }
}

impl<T: Real> Optimizer<T> for LearningRate<T> {
  type Payload = ();
  fn payload<'d>(&self, dims: &'d [usize]) -> () {}
  fn step<'w, 'g>(&self, param: &mut Parameter<'w, 'g, T, Self>) {
    param.var().grad().map(|grad| {
      let value = param.var().value();
      let exec = value.exec();
      let kernel = ocl::Kernel::builder()
        .program(exec.program())
        .name(format!("learning_rate_step_{}", T::rtype()))
        .queue(exec.queue().clone())
        .global_work_size(value.len())
        .arg(value.buffer())
        .arg(&self.lr)
        .arg(grad.buffer())
        .build()
        .unwrap();
      exec.enq(iter::once(kernel));
    });
  }
}
*/




