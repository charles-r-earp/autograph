use std::{iter, vec, cell, ops};

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
}

pub trait Executor {
  fn queue(&self) -> &ocl::Queue;
  fn program(&self) -> &ocl::Program;
  fn enq(&self, kernel: ocl::Kernel);
}

impl Executor for Workspace {
  fn queue(&self) -> &ocl::Queue { &self.queue }
  fn program(&self) -> &ocl::Program { &self.program }
  fn enq(&self, kernel: ocl::Kernel) { unsafe { kernel.enq().unwrap(); } }
}

pub struct Graph<'w> {
  workspace: &'w Workspace,
  kernels: cell::RefCell<Vec<ocl::Kernel>>
}

impl<'w> Graph<'w> {
  pub fn new(workspace: &'w Workspace) -> Self { 
    Self{workspace, kernels: cell::RefCell::new(Vec::new())}
  }
  pub fn backward(&self) { 
    let mut kernels = self.kernels.borrow_mut();
    kernels.iter()
      .for_each(|k| unsafe { k.enq().unwrap(); });
    kernels.clear();
  }
}

impl<'w> Executor for Graph<'w> {
  fn queue(&self) -> &ocl::Queue { self.workspace.queue() }
  fn program(&self) -> &ocl::Program { self.workspace.program() }
  fn enq(&self, kernel: ocl::Kernel) { self.kernels.borrow_mut().push(kernel) }
} 

#[derive(Debug)]
pub struct TensorBase<'e, E: Executor, T: Element> {
  exec: &'e E,
  dims: Vec<usize>,
  data: Option<Vec<T>>,
  buffer: ocl::Buffer<T>
}

pub type Tensor<'w, T: Element> = TensorBase<'w, Workspace, T>;
pub type Gradient<'g, 'w, T: Element> = TensorBase<'g, Graph<'w>, T>;

impl<'e, E: Executor, T: Element> TensorBase<'e, E, T> {
  pub fn new(exec: &'e E, dims: Vec<usize>, data: Option<Vec<T>>) -> Self {
    let buffer = ocl::Buffer::builder()
      .queue(exec.queue().clone())
      .len(dims.iter().product::<usize>())
      .build()
      .unwrap();
    if let Some(ref data) = data {
      debug_assert_eq!(buffer.len(), data.len());
      buffer.write(data).enq().unwrap();
    }
    Self{exec, dims, data, buffer}
  }
  pub fn exec(&self) -> &'e E { self.exec }
  pub fn dims(&self) -> &Vec<usize> { &self.dims } 
  pub fn data(&self) -> Option<&Vec<T>> { self.data.as_ref() }
  pub fn len(&self) -> usize { self.buffer.len() }
  pub fn buffer(&self) -> &ocl::Buffer<T> { &self.buffer }
  pub fn read(&mut self) {
    let mut data = vec![T::zero(); self.len()];
    self.buffer.read(&mut data).enq().unwrap();
    self.data = Some(data);
  }
  pub fn zero(&mut self) {
    let data = vec![T::zero(); self.len()];
    self.buffer.write(&data).enq().unwrap();
    self.data = Some(data);
  }
  pub fn one(&mut self) {
    let data = vec![T::one(); self.len()];
    self.buffer.write(&data).enq().unwrap();
    self.data = Some(data);
  }
}

impl<'g, 'w, T: Element> Gradient<'g, 'w, T> {
  fn backward(&mut self) {
    self.one();
    self.exec.backward();
  }
}

pub fn restrict<'a, 'b, T: Element>(lhs: &'a ocl::Buffer<T>, rhs: &'b ocl::Buffer<T>) -> bool {
  rhs.as_core().as_ptr() == lhs.as_core().as_ptr()
}

pub struct VariableBase<'g, 'w, T: Real, P> {
  value: Tensor<'w, T>,
  grad: Option<Gradient<'g, 'w, T>>,
  payload: P
}

pub type Variable<'g, 'w, T: Real> = VariableBase<'g, 'w, T, ()>;
pub type Parameter<'g, 'w, T: Real> = VariableBase<'g, 'w, T, Vec<Tensor<'w, T>>>;

impl<'g, 'w, T: Real, P> VariableBase<'g, 'w, T, P> {
  pub fn value(&self) -> &Tensor<'w, T> { &self.value }
  //pub fn value_mut(&mut self) -> &mut Tensor<'w, T> { &mut self.value }
  pub fn grad(&self) -> Option<&Gradient<'g, 'w, T>> { self.grad.as_ref() }
  fn grad_mut(&mut self) -> Option<&mut Gradient<'g, 'w, T>> { self.grad.as_mut() }
  pub fn read(&mut self) { self.value.read(); }
  pub fn read_grad(&mut self) { 
    if let Some(ref mut grad) = self.grad_mut() {
      grad.read();
    }
  }
  pub fn workpace(&self) -> &'w Workspace { self.value().exec() }
  pub fn graph(&self) -> Option<&'g Graph> { self.grad().map(|g| g.exec()) }
  pub fn len(&self) -> usize { self.value().len() }
  pub fn dims(&self) -> &Vec<usize> { self.value().dims() }
  pub fn zero_grad(&mut self) {
    if let Some(ref mut grad) = self.grad_mut() {
      grad.zero();
    }
  }
  pub fn one_grad(&mut self) {
    if let Some(ref mut grad) = self.grad_mut() {
      grad.zero();
    }
  }
}

impl<'g, 'w, T: Real> Variable<'g, 'w, T> {
  pub fn new(value: Tensor<'w, T>, grad: Option<Gradient<'g, 'w, T>>) -> Self {
    Self{value, grad, payload: ()}
  }
  pub fn backward(&mut self) {
    self.grad_mut().unwrap().backward();
  }
}

impl<'g, 'w, T: Real> Parameter<'g, 'w, T> {
  pub fn new(value: Tensor<'w, T>, grad: Option<Gradient<'g, 'w, T>>, payload: Vec<Tensor<'w, T>>) -> Self {
    Self{value, grad, payload}
  }
}

pub trait Sigmoid {
  type Output;
  fn sigmoid(self) -> Self::Output;
}

impl<'a, 'e, E: Executor, T: Real> Sigmoid for &'a TensorBase<'e, E, T> {
  type Output = TensorBase<'e, E, T>;
  fn sigmoid(self) -> Self::Output {
    let exec = self.exec;
    let out = TensorBase::new(exec, self.dims.clone(), None);
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(format!("sigmoid_{}", T::rtype()))
      .queue(exec.queue().clone())
      .global_work_size(self.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    exec.enq(kernel);
    out
  }
}

pub trait SigmoidGrad: Sigmoid {
  fn sigmoid_grad(self) -> Self::Output;
}

impl<'a, 'e, E: Executor, T: Real> SigmoidGrad for &'a TensorBase<'e, E, T> {
  fn sigmoid_grad<'g>(self) -> Self::Output {
    let exec = self.exec();
    let out = Self::Output::new(exec, self.dims.clone(), None);
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(format!("sigmoid_grad_{}", T::rtype()))
      .queue(exec.queue().clone())
      .global_work_size(self.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    exec.enq(kernel);
    out
  }
}

impl<'a, 'g, 'w, T: Real, P> Sigmoid for &'a mut VariableBase<'g, 'w, T, P> {
  type Output = Variable<'g, 'w, T>;
  fn sigmoid(self) -> Self::Output {
    let out = Variable::new(self.value().sigmoid(), self.grad().map(|input_grad| 
      Gradient::new(input_grad.exec(), input_grad.dims().clone(), Some(vec![T::zero(); input_grad.len()]))
    ));
    let sig_grad = self.value().sigmoid_grad();
    self.grad_mut().map(|input_grad| { *input_grad += &(out.grad().unwrap() * &sig_grad) });
    out
  } 
}

impl<'b, 'e1, 'e2, E1: Executor, E2: Executor, T: Element> ops::AddAssign<&'b TensorBase<'e2, E2, T>> for TensorBase<'e1, E1, T> {
  fn add_assign(&mut self, rhs: &'b TensorBase<'e2, E2, T>) {
    debug_assert_eq!(self.dims(), rhs.dims());
    let exec = self.exec();
    let name = if restrict(self.buffer(), rhs.buffer()) {
      format!("add_assign_{}_restrict", T::rtype())
    }
    else {
      format!("add_assign_{}", T::rtype())
    };
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(name)
      .queue(exec.queue().clone())
      .global_work_size(self.len())
      .arg(self.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    exec.enq(kernel);
  }
}

impl<'a, 'b, 'e1, 'e2, E1: Executor, E2: Executor, T: Element> ops::Mul<&'b TensorBase<'e2, E2, T>> for &'a TensorBase<'e1, E1, T> {
  type Output = TensorBase<'e1, E1, T>;
  fn mul(self, rhs: &'b TensorBase<'e2, E2, T>) -> Self::Output {
    debug_assert_eq!(self.len(), rhs.len());
    let exec = self.exec();
    let out = Self::Output::new(exec, self.dims.clone(), None);
    let name = if restrict(self.buffer(), rhs.buffer()) {
      format!("mul_{}_restrict", T::rtype())
    }
    else {
      format!("mul_{}", T::rtype())
    };
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(name)
      .queue(exec.queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    exec.enq(kernel);
    out
  }
}

pub trait Matmul<R> {

impl<'a, 'b, 'e1, 'e2, E1: Executor, E2: Executor, T: Element> ops::Mul<&'b TensorBase<'e2, E2, T>> for &'a TensorBase<'e1, E1, T> {
  type Output = TensorBase<'e1, E1, T>;
  fn mul(self, rhs: &'b TensorBase<'e2, E2, T>) -> Self::Output {
    debug_assert_eq!(self.len(), rhs.len());
    let exec = self.exec();
    let out = Self::Output::new(exec, self.dims.clone(), None);
    let name = if restrict(self.buffer(), rhs.buffer()) {
      format!("mul_{}_restrict", T::rtype())
    }
    else {
      format!("mul_{}", T::rtype())
    };
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(name)
      .queue(exec.queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    exec.enq(kernel);
    out
  }
}


/*
fn add_assign<'w, 'a, 'b, T: Element>(lhs: &'a Tensor<'w, T>, rhs: &'b Tensor<'w, T>) -> ocl::Kernel {
  l
}



impl<'w, 'b, T: Element> ops::AddAssign<&'b Gradient<'g, 'w, T>> for Gradient<'g, 'w, T> {
  fn add_assign(&mut self, rhs: &'b Gradient<'g, 'w, T>) {
    let kernel = add_assign(self.tensor(), rhs.tensor());
    self.graph.push(kernel);
  }
}

fn sigmoid<'w, 'a, T: Real>(input: &'a Tensor<'w, T>) -> (ocl::Kernel, Tensor<'w, T>) {
  let ws = input.workspace();
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
  (kernel, out)
}

pub trait Sigmoid {
  type Output;
  fn sigmoid(self) -> Self::Output;
}

impl<'w, 'a, T: Real> Sigmoid for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn sigmoid<'g>(&self, graph: &'g mut Graph) -> Tensor<'w, T> {
    let (kernel, out) = sigmoid(self);
    unsafe { kernel.enq().unwrap(); }
    out  
  }
} 

impl<'w, T: Real> Sigmoid for Variable<'w, T> {
  type Output = Self;
  fn sigmoid<'g>(&self, graph: &'g mut Graph) -> Self {
    let ws = self.value().workspace();
    let mut fgraph = Graph::new();
    let value = self.value().sigmoid(&mut fgraph);
    fgraph.forward();
    let grad = if let Some(ref input_grad) = self.grad() {
      let mut bgraph = Graph::new();
      let grad = Tensor::new(ws, self.dims().clone(), Some(vec![T::zero(); self.len()]));
      let dx = value.sigmoid_grad(&mut bgraph);
      input_grad.add_assign(&mut bgraph, &dx);
      graph.extend(bgraph.into_iter().rev());
      Some(grad)
    }
    else { None };
    Self::new(value, None)
  }
}

fn sigmoid_grad<'w, 'a, T: Real>(input: &'a Tensor<'

pub trait SigmoidGrad: Sigmoid {
  fn sigmoid_grad(self) -> Self::Output;
}

impl<'w, T: Real> SigmoidGrad for &'a Tensor<'w, T> {
  fn sigmoid_grad(self) -> Self::Output {
    let ws = self.workspace();
    let out = Tensor::new(ws, self.dims.clone(), None);
    let kernel = ocl::Kernel::builder()
      .program(ws.program())
      .name(format!("sigmoid_grad_{}", T::rtype()))
      .queue(ws.queue().clone())
      .global_work_size(self.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    
    out
  }
}*/
/*
pub trait Sigmoid {
  type Output;
  fn sigmoid(self) -> Self::Output; 
}

impl<'w, 'a, T: Element> Sigmoid for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn sigmoid(self) -> Tensor<'w, T> { sigmoid(self, ()) }
}

pub trait Stack {
  type Output;
  fn stack(self, rows: usize) -> Self::Output;
}

impl<'w, 'a, T: Element> Stack for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn stack(self, rows: usize) -> Tensor<'w, T> {
    let mut dims = self.dims.clone();
    dims.insert(0, rows); 
    let out = self.workspace().tensor(dims, None);
    let name = format!("stack_{}", T::rtype());
    let kernel = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(name)
      .queue(self.workspace().queue().clone())
      .global_work_size(rows)
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(self.len())
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    out
  }
}

pub trait Sum {
  type Output;
  fn sum(self) -> Self::Output;
}

impl<'w, 'a, T: Element> Sum for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn sum(self) -> Tensor<'w, T> {
    debug_assert!(T::is_real());
    let out = self.workspace().tensor(vec![1], None);
    let kernel2 = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(format!("sum_{}", T::rtype()))
      .queue(self.workspace().queue().clone())
      .global_work_size(1)
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(self.buffer().len())
      .build()
      .unwrap();
    unsafe { kernel2.enq().unwrap(); }
    out
  }
}

fn add<'w, 'a, 'b, T: Element, E: Executor>(lhs: &'a Tensor<'w, T>, rhs: &'b Tensor<'w, T>, exec: E) -> Tensor<'w, T> {
  let ws = lhs.workspace();
  let out = ws.tensor(lhs.dims.clone(), None);
  let name = if lhs.restrict(rhs) {
      format!("add_{}_restrict", T::rtype())
    }
    else {
      format!("add_{}", T::rtype())
    };
  let kernel = ocl::Kernel::builder()
    .program(ws.program())
    .name(name)
    .queue(ws.queue().clone())
    .global_work_size(lhs.len())
    .arg(out.buffer())
    .arg(lhs.buffer())
    .arg(rhs.buffer())
    .build()
    .unwrap();
  exec.enq(kernel);
  out
}

impl<'w, 'a, 'b, T: Element> ops::Add<&'b Tensor<'w, T>> for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn add(self, rhs: &'b Tensor<'w, T>) -> Tensor<'w, T> {
    debug_assert_eq!(self.len(), rhs.len());
    let out = self.workspace().tensor(self.dims.clone(), None);
    let name = if self.restrict(rhs) {
      format!("add_{}_restrict", T::rtype())
    }
    else {
      format!("add_{}", T::rtype())
    };
    let kernel = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(name)
      .queue(self.workspace().queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    out
  }
}


fn add_assign<'w, 'a, 'b, T: Element, E: Executor>(lhs: &'a Tensor<'w, T>, rhs: &'b Tensor<'w, T>, exec: E) {
  debug_assert_eq!(lhs.dims(), rhs.dims());
  let name = if lhs.restrict(rhs) {
      format!("add_assign_{}_restrict", T::rtype())
    }
    else {
      format!("add_assign_{}", T::rtype())
    };
  let kernel = ocl::Kernel::builder()
    .program(lhs.workspace().program())
    .name(name)
    .queue(lhs.workspace().queue().clone())
    .global_work_size(lhs.len())
    .arg(lhs.buffer())
    .arg(rhs.buffer())
    .build()
    .unwrap();
  exec.enq(kernel);
}

impl<'w, 'b, T: Element> ops::AddAssign<&'b Tensor<'w, T>> for Tensor<'w, T> {
  fn add_assign(&mut self, rhs: &'b Tensor<'w, T>) { add_assign(self, rhs, ()) }
}

impl<'w, 'a, 'b, T: Element> ops::Sub<&'b Tensor<'w, T>> for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn sub(self, rhs: &'b Tensor<'w, T>) -> Tensor<'w, T> {
    debug_assert_eq!(self.len(), rhs.len());
    let out = self.workspace().tensor(self.dims.clone(), None);
    let name = if self.restrict(rhs) {
      format!("sub_{}_restrict", T::rtype())
    }
    else {
      format!("sub_{}", T::rtype())
    };
    let kernel = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(name)
      .queue(self.workspace().queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(rhs.buffer())
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    out
  }
}

fn mul<'w, 'a, 'b, T: Element, E: Executor>(lhs: &'a Tensor<'w, T>, rhs: &'b Tensor<'w, T>, exec: E) -> Tensor<'w, T> {
  let ws = lhs.workspace();
  let out = ws.tensor(lhs.dims.clone(), None);
  let name = if lhs.restrict(rhs) {
      format!("mul_{}_restrict", T::rtype())
    }
    else {
      format!("mul_{}", T::rtype())
    };
  let kernel = ocl::Kernel::builder()
    .program(ws.program())
    .name(name)
    .queue(ws.queue().clone())
    .global_work_size(lhs.len())
    .arg(out.buffer())
    .arg(lhs.buffer())
    .arg(rhs.buffer())
    .build()
    .unwrap();
  exec.enq(kernel);
  out
}



pub trait Square {
  type Output;
  fn sqr(self) -> Self::Output;
} 

impl<'w, 'a, T: Element> Square for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn sqr(self) -> Tensor<'w, T> {
    let out = self.workspace().tensor(self.dims.clone(), None);
    let name = format!("mul_{}_restrict", T::rtype());
    let kernel = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(name)
      .queue(self.workspace().queue().clone())
      .global_work_size(out.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    out
  }
}

pub trait Transpose {
  type Output;
  fn t(self) -> Self::Output;
}

impl<'w, 'a, T: Element> Transpose for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn t(self) -> Tensor<'w, T> {
    debug_assert_eq!(self.dims.len(), 2);
    let d0 = self.dims[0];
    let d1 = self.dims[1];
    let out = self.workspace().tensor(vec![d1, d0], None);
    let name = format!("transpose_v1_{}", T::rtype());
    let kernel = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(name)
      .queue(self.workspace().queue().clone())
      .global_work_size([d1, d0])
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    out
  }
}

pub fn matmul<'w, 'a, 'b, T: Element, E: Executor>(lhs: &'a Tensor<'w, T>, rhs: &'b Tensor<'w, T>, exec: E) {
  debug_assert_eq!(lhs.dims.len(), 2);
  debug_assert_eq!(rhs.dims.len(), 2);
  debug_assert_eq!(lhs.dims[1], rhs.dims[0]);
  let m = lhs.dims[0];
  let n = rhs.dims[1];
  let k = lhs.dims[1];
  let ws = lhs.workspace();
  let out = ws.tensor(vec![m, n], None);
  let name = if lhs.restrict(rhs) {
    format!("matmul_v1_{}_restrict", T::rtype())
  }
  else {
    format!("matmul_v1_{}", T::rtype())
  };
  let kernel = ocl::Kernel::builder()
    .program(ws.program())
    .name(name)
    .queue(ws.queue().clone())
    .global_work_size([m, n])
    .arg(out.buffer())
    .arg(lhs.buffer())
    .arg(rhs.buffer())
    .arg(m)
    .arg(n)
    .arg(k)
    .build()
    .unwrap();
  exec.enq(kernel);
  out
}

pub trait Matmul<R> {
  type Output;
  fn matmul(self, rhs: R) -> Self::Output;
}

impl<'w, 'a, 'b, T: Element> Matmul<&'b Tensor<'w, T>> for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn matmul(self, rhs: &'b Tensor<'w, T>) -> Tensor<'w, T> {
    matmul(self, rhs, ())
  }
}

// Optimization

pub trait Optimizer<T: Element> {
  fn step<'w, 'p>(&mut self, param: &'p mut Parameter<'w, T>);
}  

#[derive(Debug)]
pub struct Parameter<'w, T: Element> {
  value: Tensor<'w, T>,
  grad: Option<Tensor<'w, T>>,
  opt_params: Vec<Tensor<'w, T>>
}

impl<'w, T: Element> Parameter<'w, T> {
  pub fn new(value: Tensor<'w, T>, grad: Option<Tensor<'w, T>>, opt_params: Vec<Tensor<'w, T>>) -> Self { 
    Self{value, grad, opt_params} 
  }
  pub fn value(&self) -> &Tensor<'w, T> { &self.value }
  pub fn grad_mut(&mut self) -> Option<&mut Tensor<'w, T>> { self.grad.as_mut() }
}

// Backward

#[derive(Debug)]
pub struct Graph<'w> {
  workspace: &'w Workspace,
  kernels: cell::RefCell<Vec<ocl::Kernel>>
}

impl<'w> Graph<'w> {
  pub fn variable<'g, T: Element>(&'g self, value: Tensor<'w, T>, grad: Option<Tensor<'w, T>>) -> Variable<'w, 'g, T> {
    Variable{graph: self, value, grad}
  }
  pub fn extend(&self, kernels: Vec<ocl::Kernel>) {
    self.kernels.borrow_mut().extend(kernels.into_iter().rev());
  }
  fn backward(&self) {
    self.kernels.borrow()
      .iter()
      .rev()
      .for_each(|k| unsafe { k.enq().unwrap(); });
    self.kernels.borrow_mut()
      .clear();
  }
}    
  
#[derive(Debug)]
pub struct Variable<'w, 'g, T: Element> {
  graph: &'g Graph<'w>,
  value: Tensor<'w, T>,
  grad: Option<Tensor<'w, T>>
}

impl<'w, 'g, T: Element> Variable<'w, 'g, T> {
  pub fn graph(&self) -> &'g Graph<'w> { &self.graph }
  pub fn value(&self) -> &Tensor<'w, T> { &self.value }
  pub fn grad(&self) -> Option<&Tensor<'w, T>> { self.grad.as_ref() }
  pub fn read(&mut self) { self.value.read(); }
  pub fn read_grad(&mut self) { 
    if let Some(ref mut grad) = self.grad.as_mut() {
      grad.read();
    }
  }
  pub fn backward(&mut self) {
    debug_assert!(T::is_real());
    self.grad
      .as_mut()
      .unwrap()
      .one();
    self.graph.backward();
  }
}

impl<'w, 'g, 'a, T: Element> Sigmoid for &'a Variable<'w, 'g, T> {
  type Output = Variable<'w, 'g, T>;
  fn sigmoid(self) -> Variable<'w, 'g, T> {
    let ws = self.value().workspace();
    let value = self.value().sigmoid();
    let grad = if let Some(ref grad) = self.grad() {
      let mut kernels = Vec::new();
      let out_grad = ws.tensor(value.dims().clone(), Some(vec![T::zero(); value.len()]));
      add_assign(&grad, &mul(&out_grad, &sigmoid_grad(&value, &mut kernels), &mut kernels), &mut kernels);
      self.graph.extend(kernels); 
      Some(out_grad)
    }
    else {
      None
    };
    self.graph().variable(self.value().sigmoid(), grad)
  }
}

impl<'w, 'g, 'a, 'b, T: Element> Matmul<&'b mut Parameter<'w, T>> for &'a Variable<'w, 'g, T> {
  type Output = Variable<'w, 'g, T>;
  fn matmul(self, rhs: &'b mut Parameter<'w, T>) -> Variable<'w, 'g, T> {
    let ws = self.value().workspace();
    let value = self.value().matmul(rhs.value());
    let grad = if self.grad().is_some() || rhs.grad_mut().is_some() {
      Some(ws.tensor(value.dims().clone(), Some(vec![T::zero(); value.len()]))
    }
    else { None };
    let mut kernels = Vec::new();
    if let Some(ref lgrad) = self.grad() {
      add_assign(&lgrad, &matmut(&lgrad
        
*/     


   







