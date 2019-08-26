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

pub struct Graph<'w> {
  workspace: &'w Workspace,
  kernels: cell::RefCell<Vec<ocl::Kernel>>
}

impl<'w> Graph<'w> {
  pub fn new(workspace: &'w Workspace) -> Self { 
    Self{workspace, kernels: cell::RefCell::new(Vec::new())}
  }
}

pub trait Executor {
  fn queue(&self) -> &ocl::Queue;
  fn program(&self) -> &ocl::Program;
  fn enq<I: iter::IntoIterator<Item=ocl::Kernel>>(&self, kernels: I)
    where I::IntoIter: iter::DoubleEndedIterator;
}

impl<'w> Executor for &'w Workspace {
  fn queue(&self) -> &ocl::Queue { &self.queue }
  fn program(&self) -> &ocl::Program { &self.program }
  fn enq<I: IntoIterator<Item=ocl::Kernel>>(&self, kernels: I)
    where I::IntoIter: iter::DoubleEndedIterator {
    kernels.into_iter().for_each(|k| unsafe { k.enq().unwrap(); });
  }
}

impl<'w, 'g> Executor for &'g Graph<'w> {
  fn queue(&self) -> &ocl::Queue { self.workspace.queue() }
  fn program(&self) -> &ocl::Program { self.workspace.program() }
  fn enq<I: IntoIterator<Item=ocl::Kernel>>(&self, kernels: I)
    where I::IntoIter: iter::DoubleEndedIterator {
    self.kernels.borrow_mut().extend(kernels.into_iter().rev());
  }
}

pub trait BinaryExec<R> {
  type Output: Executor;
  fn binary_exec(&self, rhs: R) -> Self::Output;
}

impl<'w, E: Executor> BinaryExec<E> for &'w Workspace {
  type Output = E;
  fn binary_exec(&self, rhs: E) -> Self::Output { rhs }
}

impl<'w, 'g, E: Executor> BinaryExec<E> for &'g Graph<'w> {
  type Output = Self;
  fn binary_exec(&self, rhs: E) -> Self { self }
}


pub trait BackwardExecutor: Executor {
  fn backward(&self);
}

impl<'w, 'g> BackwardExecutor for &'g Graph<'w> {
  fn backward(&self) { 
    let mut kernels = self.kernels.borrow_mut();
    kernels.iter()
      .for_each(|k| unsafe { println!("{:?}", k.name()); k.enq().unwrap(); });
    kernels.clear();
  }
}

#[derive(Debug)]
pub struct Tensor<E, T: Element> {
  exec: E,
  dims: Vec<usize>,
  data: Option<Vec<T>>,
  buffer: ocl::Buffer<T>
}

impl<E: Executor, T: Element> Tensor<E, T> {
  pub fn new(exec: E, dims: Vec<usize>, data: Option<Vec<T>>) -> Self {
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
  pub fn from_elem(exec: E, dims: Vec<usize>, elem: T) -> Self {
    let mut t = Self::new(exec, dims, None);
    t.fill_elem(elem);
    t
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
  pub fn fill_elem(&mut self, elem: T) {
    let data = vec![elem; self.len()];
    self.buffer.write(&data).enq().unwrap();
    self.data = Some(data);
  }
}

pub trait ExecutorRef {
  type Output: Executor;
  fn exec(&self) -> Self::Output;
}

impl<'e, E, T: Element> ExecutorRef for Tensor<&'e E, T> 
  where &'e E: Executor {
  type Output = &'e E;
  fn exec(&self) -> Self::Output { self.exec }
}

impl<'b, E1, E2, T: Element> BinaryExec<&'b Tensor<E2, T>> for Tensor<E1, T>
  where Tensor<E1, T>: ExecutorRef,
        Tensor<E2, T>: ExecutorRef,
        <Tensor<E1, T> as ExecutorRef>::Output: BinaryExec<<Tensor<E2, T> as ExecutorRef>::Output> {
  type Output = <<Tensor<E1, T> as ExecutorRef>::Output as BinaryExec<<Tensor<E2, T> as ExecutorRef>::Output>>::Output;
  fn binary_exec(&self, rhs: &'b Tensor<E2, T>) -> Self::Output { self.exec().binary_exec(rhs.exec()) }
}

impl<'w, E: BackwardExecutor, T: Element> Tensor<E, T> {
  fn backward(&mut self) {
    self.fill_elem(T::one());
    self.exec.backward();
  }
}
 
pub fn restrict<'a, 'b, T: Element>(lhs: &'a ocl::Buffer<T>, rhs: &'b ocl::Buffer<T>) -> bool {
  rhs.as_core().as_ptr() == lhs.as_core().as_ptr()
}

impl<'a, 'b, E1: Executor, E2: Executor, T: Element> ops::Add<&'b Tensor<E2, T>> for &'a Tensor<E1, T>
  where Tensor<E1, T>: BinaryExec<&'b Tensor<E2, T>> {
  type Output = Tensor<<Tensor<E1, T> as BinaryExec<&'b Tensor<E2, T>>>::Output, T>;
  fn add(self, rhs: &'b Tensor<E2, T>) -> Self::Output {
    debug_assert_eq!(self.len(), rhs.len());
    let exec = self.binary_exec(rhs);
    let out = Self::Output::new(exec, self.dims.clone(), None);
    let exec = &out.exec;
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
    out.exec.enq(iter::once(kernel));
    out
  }
}

/*pub struct VariableBase<'g, 'w, T: Real, P> {
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

pub trait Transpose {
  type Output;
  fn t(self) -> Self::Output;
}

impl<'a, 'e, E: Executor, T: Element> Transpose for &'a TensorBase<'e, E, T> {
  type Output = TensorBase<'e, E, T>;
  fn t(self) -> Self::Output {
    debug_assert_eq!(self.dims.len(), 2);
    let exec = self.exec();
    let d0 = self.dims[0];
    let d1 = self.dims[1];
    let out = Self::Output::new(exec, vec![d1, d0], None);
    let name = format!("transpose_v1_{}", T::rtype());
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(name)
      .queue(exec.queue().clone())
      .global_work_size([d1, d0])
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    exec.enq(kernel);
    out
  }
}

pub trait Matmul<R> {
  type Output;
  fn matmul(self, rhs: R) -> Self::Output;
}

impl<'a, 'b, 'e1, 'e2, E1: Executor, E2: Executor, T: Element> Matmul<&'b TensorBase<'e2, E2, T>> for &'a TensorBase<'e1, E1, T>
  where <&'a TensorBase<'e1, E1, T>>: BinaryExpr<&'b TensorBase<'e2, E2, T>> {
  type Output = <TensorBase<'e1, E1, T> as BinaryExpr<&'b TensorBase<'e2, E2, T>>>::Output;
  fn matmul(self, rhs: &'b TensorBase<'e2, E2, T>) -> Self::Output {
    debug_assert_eq!(self.dims.len(), 2);
    debug_assert_eq!(rhs.dims.len(), 2);
    debug_assert_eq!(self.dims[1], rhs.dims[0]);
    let exec = self.binary_exec(rhs);
    let m = self.dims[0];
    let n = rhs.dims[1];
    let k = self.dims[1];
    let out = Self::Output::new(exec, vec![m, n], None);
    let name = if restrict(self.buffer(), rhs.buffer()) {
      format!("matmul_v1_{}_restrict", T::rtype())
    }
    else {
      format!("matmul_v1_{}", T::rtype())
    };
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(name)
      .queue(exec.queue().clone())
      .global_work_size([m, n])
      .arg(out.buffer())
      .arg(self.buffer())
      .arg(rhs.buffer())
      .arg(m)
      .arg(n)
      .arg(k)
      .build()
      .unwrap();
    exec.enq(kernel);
    out
  }
}

impl<'a, 'b, 'g, 'w, T: Real, P1, P2> Matmul<&'b mut VariableBase<'g, 'w, T, P2>> for &'b mut VariableBase<'g, 'w, T, P1> {
  type Output = Variable<'g, 'w, T>;
  fn matmul(self, rhs: &'b mut VariableBase<'g, 'w, T, P2>) -> Variable<'g, 'w, T> {
    let lvalue = self.value();
    let rvalue = rhs.value();
    let out_value = lvalue.matmul(rvalue);
    let m = lvalue.dims()[0];
    let k = lvalue.dims()[1];
    let n = rvalue.dims()[1];
    let ws = lvalue.exec();
    let graph = self.grad().or(rhs.grad()).map(|grad| grad.exec());
    let out_grad = graph.map(|graph| Gradient::new(&graph, out_value.dims().clone(), Some(vec![T::zero(); out_value.len()])));
    self.grad_mut().map(|lgrad| {
      *lgrad += &out_grad.as_ref().unwrap().matmul(&rvalue.t());
    });
    rhs.grad_mut().map(|rgrad| {
      *rgrad += &self.value().t().matmul(out_grad.as_ref().unwrap());
    });
    Variable::new(out_value, out_grad)  
  }
}
*/




