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
  pub fn workspace(&self) -> &'w Workspace { self.workspace }
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
  pub fn fill_fn(&mut self, f: impl FnMut() -> T) {
    
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

pub type BinaryExecOutput<A, B> = <A as BinaryExec<B>>::Output;

impl<'w, E: BackwardExecutor, T: Element> Tensor<E, T> {
  fn backward(&mut self) {
    self.fill_elem(T::one());
    self.exec.backward();
  }
}
 
pub fn restrict<'a, 'b, T: Element>(lhs: &'a ocl::Buffer<T>, rhs: &'b ocl::Buffer<T>) -> bool {
  rhs.as_core().as_ptr() == lhs.as_core().as_ptr()
}

pub struct Variable<'w, 'g, T: Real> {
  value: Tensor<&'w Workspace, T>,
  grad: Option<Tensor<&'g Graph<'w>, T>>
}

impl<'w, 'g, T: Real> Variable<'w, 'g, T> {
  pub fn new(value: Tensor<&'w Workspace, T>, grad: Option<Tensor<&'g Graph<'w>, T>>) -> Self {
    Self{value, grad}
  } 
  pub fn value(&self) -> &Tensor<&'w Workspace, T> { &self.value }
  pub fn value_mut(&mut self) -> &mut Tensor<&'w Workspace, T> { &mut self.value }
  pub fn grad(&self) -> Option<&Tensor<&'g Graph<'w>, T>> { self.grad.as_ref() }
  pub fn grad_mut(&mut self) -> Option<&mut Tensor<&'g Graph<'w>, T>> { self.grad.as_mut() }
  pub fn backward(&mut self) { self.grad_mut().unwrap().backward(); }
}

impl<'w, 'g, T: Real> From<Tensor<&'w Workspace, T>> for Variable<'w, 'g, T> {
  fn from(value: Tensor<&'w Workspace, T>) -> Self {
    Self::new(value, None)
  }
}

pub trait Optimizer<T: Real>: Sized {
  type Payload;
  fn payload<'d>(&self, dims: &'d [usize]) -> Self::Payload;
  fn step<'w, 'g>(&self, param: &mut Parameter<'w, 'g, T, Self>);
}

pub struct Parameter<'w, 'g, T: Real, O: Optimizer<T>> {
  var: Variable<'w, 'g, T>,
  payload: O::Payload
}

impl<'w, 'g, T: Real, O: Optimizer<T>> Parameter<'w, 'g, T, O> {
  pub fn new<'o>(value: Tensor<&'w Workspace, T>, opt: &'o O) -> Self {
    let payload = opt.payload(value.dims().as_slice());
    let var = Variable::from(value);
    Self{var, payload}
  }
  pub fn var(&self) -> &Variable<'w, 'g, T> { &self.var } 
  pub fn var_mut(&mut self) -> &mut Variable<'w, 'g, T> { &mut self.var } 
} 

// Unary Elementwise 

pub trait Sigmoid {
  type Output;
  fn sigmoid(self) -> Self::Output;
}

impl<'a, E: Executor, T: Real> Sigmoid for &'a Tensor<E, T>
  where Tensor<E, T>: ExecutorRef<Output=E> {
  type Output = Tensor<E, T>;
  fn sigmoid(self) -> Self::Output {
    let out = Tensor::new(self.exec(), self.dims.clone(), None);
    let exec = out.exec();
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(format!("sigmoid_{}", T::rtype()))
      .queue(exec.queue().clone())
      .global_work_size(self.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    exec.enq(iter::once(kernel));
    out
  }
}

pub trait SigmoidGrad: Sigmoid {
  fn sigmoid_grad(self) -> Self::Output;
}

impl<'a, E: Executor, T: Real> SigmoidGrad for &'a Tensor<E, T>
  where Tensor<E, T>: ExecutorRef<Output=E> {
  fn sigmoid_grad<'g>(self) -> Self::Output {
    let out = Self::Output::new(self.exec(), self.dims.clone(), None);
    let exec = out.exec();
    let kernel = ocl::Kernel::builder()
      .program(exec.program())
      .name(format!("sigmoid_grad_{}", T::rtype()))
      .queue(exec.queue().clone())
      .global_work_size(self.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    exec.enq(iter::once(kernel));
    out
  }
}

impl<'a, 'g, 'w, T: Real> Sigmoid for &'a mut Variable<'g, 'w, T> {
  type Output = Variable<'g, 'w, T>;
  fn sigmoid(self) -> Self::Output {
    let out = Variable::new(self.value().sigmoid(), self.grad().map(|input_grad| 
      Tensor::from_elem(input_grad.exec(), input_grad.dims().clone(), T::zero())
    ));
    let sig_grad = self.value().sigmoid_grad();
    self.grad_mut().map(|input_grad| { *input_grad += &(out.grad().unwrap() * &sig_grad) });
    out
  } 
}

// Unary 2d 

pub trait Transpose {
  type Output;
  fn t(self) -> Self::Output;
}

impl<'a, E: Executor, T: Element> Transpose for &'a Tensor<E, T>
  where Tensor<E, T>: ExecutorRef<Output=E> {
  type Output = Tensor<E, T>;
  fn t(self) -> Self::Output {
    debug_assert_eq!(self.dims.len(), 2);
    let exec = self.exec();
    let d0 = self.dims[0];
    let d1 = self.dims[1];
    let out = Self::Output::new(exec, vec![d1, d0], None);
    let exec = &out.exec();
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
    exec.enq(iter::once(kernel));
    out
  }
}

// Binary Elementwise

impl<'b, E1: Executor, E2: Executor, T: Element> ops::AddAssign<&'b Tensor<E2, T>> for Tensor<E1, T>
  where Self: ExecutorRef {
  fn add_assign(&mut self, rhs: &'b Tensor<E2, T>) {
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
    exec.enq(iter::once(kernel));
  }
}

impl<'a, 'b, E1: Executor, E2: Executor, T: Element> ops::Add<&'b Tensor<E2, T>> for &'a Tensor<E1, T>
  where Tensor<E1, T>: BinaryExec<&'b Tensor<E2, T>> {
  type Output = Tensor<BinaryExecOutput<Tensor<E1, T>, &'b Tensor<E2, T>>, T>;
  fn add(self, rhs: &'b Tensor<E2, T>) -> Self::Output {
    debug_assert_eq!(self.len(), rhs.len());
    let exec = self.binary_exec(rhs);
    let out = Self::Output::new(exec, self.dims.clone(), None);
    let exec = &out.exec;
    let name = if restrict(self.buffer(), rhs.buffer()) {
      format!("add_{}_restrict", T::rtype())
    }
    else {
      format!("add_{}", T::rtype())
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

impl<'a, 'b, E1: Executor, E2: Executor, T: Element> ops::Mul<&'b Tensor<E2, T>> for &'a Tensor<E1, T>
  where Tensor<E1, T>: BinaryExec<&'b Tensor<E2, T>> {
  type Output = Tensor<BinaryExecOutput<Tensor<E1, T>, &'b Tensor<E2, T>>, T>;
  fn mul(self, rhs: &'b Tensor<E2, T>) -> Self::Output {
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

// Binary 2d

pub trait Matmul<R> {
  type Output;
  fn matmul(self, rhs: R) -> Self::Output;
}

impl<'a, 'b, E1: Executor, E2: Executor, T: Element> Matmul<&'b Tensor<E2, T>> for &'a Tensor<E1, T>
  where Tensor<E1, T>: BinaryExec<&'b Tensor<E2, T>> {
  type Output = Tensor<BinaryExecOutput<Tensor<E1, T>, &'b Tensor<E2, T>>, T>;
  fn matmul(self, rhs: &'b Tensor<E2, T>) -> Self::Output {
    debug_assert_eq!(self.dims.len(), 2);
    debug_assert_eq!(rhs.dims.len(), 2);
    debug_assert_eq!(self.dims[1], rhs.dims[0]);
    let exec = self.binary_exec(rhs);
    let m = self.dims[0];
    let n = rhs.dims[1];
    let k = self.dims[1];
    let out = Self::Output::new(exec, vec![m, n], None);
    let exec = &out.exec;
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
    exec.enq(iter::once(kernel));
    out
  }
}

impl<'a, 'b, 'w, 'g, T: Real> Matmul<&'b mut Variable<'w, 'g, T>> for &'b mut Variable<'w, 'g, T> {
  type Output = Variable<'w, 'g, T>;
  fn matmul(self, rhs: &'b mut Variable<'w, 'g, T>) -> Variable<'w, 'g, T> {
    let lvalue = self.value();
    let rvalue = rhs.value();
    let out_value = lvalue.matmul(rvalue);
    let ws = lvalue.exec();
    let graph = self.grad().or(rhs.grad()).map(|grad| grad.exec());
    let out_grad = graph.map(|graph| Tensor::from_elem(graph, out_value.dims().clone(), T::zero()));
    self.grad_mut().map(|lgrad| {
      *lgrad += &out_grad.as_ref().unwrap().matmul(&rvalue.t());
    });
    rhs.grad_mut().map(|rgrad| {
      *rgrad += &self.value().t().matmul(out_grad.as_ref().unwrap());
    });
    Variable::new(out_value, out_grad)  
  }
}

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





