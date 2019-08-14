use std::ops;
use std::cell;

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

#[derive(Debug)]
pub struct Tensor<'w, T: Element> {
  workspace: &'w Workspace,
  dims: Vec<usize>,
  data: Option<Vec<T>>,
  buffer: ocl::Buffer<T>
}

impl<'w, T: Element> Tensor<'w, T> {
  pub fn read(&mut self) {
    let mut data = vec![T::zero(); self.len()];
    self.buffer.read(&mut data)
      .enq()
      .unwrap();
    self.data = Some(data);
  }
  pub fn zero(&mut self) {
    let data = vec![T::zero(); self.len()];
    self.buffer.write(&data)
      .enq()
      .unwrap();
    self.data = Some(data);
  }
  pub fn one(&mut self) {
    let data = vec![T::one(); self.len()];
    self.buffer.write(&data)
      .enq()
      .unwrap();
    self.data = Some(data);
  }
  pub fn workspace(&self) -> &'w Workspace { self.workspace }
  pub fn dims(&self) -> &Vec<usize> { &self.dims } 
  pub fn data(&self) -> Option<&Vec<T>> { self.data.as_ref() }
  pub fn len(&self) -> usize { self.buffer.len() }
  pub fn buffer(&self) -> &ocl::Buffer<T> { &self.buffer }
  pub fn restrict<'b>(&self, other: &'b Tensor<T>) -> bool {
    self.buffer().as_core().as_ptr() == other.buffer().as_core().as_ptr()
  }
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
  pub fn tensor<T: Element>(&self, dims: Vec<usize>, data: Option<Vec<T>>) -> Tensor<T> {
    let buffer = ocl::Buffer::builder()
      .queue(self.queue.clone())
      .len(dims.iter().product::<usize>())
      .build()
      .unwrap();
    if let Some(ref data) = data {
      debug_assert_eq!(buffer.len(), data.len());
      buffer.write(data).enq().unwrap();
    }
    Tensor{workspace: self, dims, data, buffer}
  }
  pub fn graph<'w>(&'w self) -> Graph<'w> {
    Graph{workspace: self, kernels: cell::RefCell::new(Vec::new())}
  }
  pub fn context(&self) -> &ocl::Context { &self.context }
  pub fn program(&self) -> &ocl::Program { &self.program }
  pub fn queue(&self) -> &ocl::Queue { &self.queue }
  pub fn device(&self) -> ocl::Device { self.context.devices()[0] }
}

pub fn source<'c>(context: &'c ocl::Context) -> String {
  _source(std::marker::PhantomData::<f32>::default())
}

fn _source<T: Element>(_m: std::marker::PhantomData<T>) -> String {
  include_str!("autograph.cl")
    .replace("RTYPE", &T::rtype())
    .replace("CTYPE", &T::ctype())
    .replace("IS_REAL", &T::is_real().to_string())
    .to_string()
}

pub trait Executor {
  fn enq(self, kernel: ocl::Kernel);
}

impl Executor for () {
  fn enq(self, kernel: ocl::Kernel) { unsafe { kernel.enq().unwrap(); } }
}

impl<'a> Executor for &'a mut Vec<ocl::Kernel> {
  fn enq(self, kernel: ocl::Kernel) { self.push(kernel); }
}

fn sigmoid<'w, 'a, T: Element, E: Executor>(input: &'a Tensor<'w, T>, exec: E) -> Tensor<'w, T> {
  debug_assert!(T::is_real());
  let out = input.workspace().tensor(input.dims.clone(), None);
  let kernel = ocl::Kernel::builder()
    .program(input.workspace().program())
    .name(format!("sigmoid_{}", T::rtype()))
    .queue(input.workspace().queue().clone())
    .global_work_size(input.len())
    .arg(out.buffer())
    .arg(input.buffer())
    .build()
    .unwrap();
  exec.enq(kernel);
  out
}

fn sigmoid_grad<'w, 'a, T: Element, E: Executor>(input: &'a Tensor<'w, T>, exec: E) -> Tensor<'w, T> {
  debug_assert!(T::is_real());
  let out = input.workspace().tensor(input.dims.clone(), None);
  let kernel = ocl::Kernel::builder()
    .program(input.workspace().program())
    .name(format!("sigmoid_grad_{}", T::rtype()))
    .queue(input.workspace().queue().clone())
    .global_work_size(input.len())
    .arg(out.buffer())
    .arg(input.buffer())
    .build()
    .unwrap();
  exec.enq(kernel);
  out
}
  
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

impl<'w, 'a, 'b, T: Element> ops::Mul<&'b Tensor<'w, T>> for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn mul(self, rhs: &'b Tensor<'w, T>) -> Tensor<'w, T> {
    debug_assert_eq!(self.len(), rhs.len());
    let out = self.workspace().tensor(self.dims.clone(), None);
    let name = if self.restrict(rhs) {
      format!("mul_{}_restrict", T::rtype())
    }
    else {
      format!("mul_{}", T::rtype())
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

pub trait Matmul<R> {
  type Output;
  fn matmul(self, rhs: R) -> Self::Output;
}

impl<'w, 'a, 'b, T: Element> Matmul<&'b Tensor<'w, T>> for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn matmul(self, rhs: &'b Tensor<'w, T>) -> Tensor<'w, T> {
    debug_assert_eq!(self.dims.len(), 2);
    debug_assert_eq!(rhs.dims.len(), 2);
    debug_assert_eq!(self.dims[1], rhs.dims[0]);
    let m = self.dims[0];
    let n = rhs.dims[1];
    let k = self.dims[1];
    let out = self.workspace().tensor(vec![m, n], None);
    let name = if self.restrict(rhs) {
      format!("matmul_v1_{}_restrict", T::rtype())
    }
    else {
      format!("matmul_v1_{}", T::rtype())
    };
    let kernel = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(name)
      .queue(self.workspace().queue().clone())
      .global_work_size([m, n])
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(rhs.buffer())
      .arg(m)
      .arg(n)
      .arg(k)
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    out
  }
}

// Optimization

pub trait Optimizer<'w, T: Element> {

}

pub struct BasiceLearningRate {
  

#[derive(Debug)]
pub struct Parameter<'w, T: Element, O: Optimizer<'w, T>> {
  value: Tensor<'w, T>,
  optim: O
}

impl<'w, T: Element, O: Optimizer<'w, T>> Parameter<'w, T, O> {
  pub fn new(value: Tensor<'w, T>, optim: O) -> Self { 
    Self{value, optim} 
  }
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
    let value = self.value().sigmoid();
    let grad = if let Some(ref grad) = self.grad() {
      let mut kernels = Vec::new();
      let out_grad = value.workspace().tensor(value.dims().clone(), Some(vec![T::zero(); value.len()]));
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


   







