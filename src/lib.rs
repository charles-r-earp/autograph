use std::{marker, cell, ops, iter, fmt, io, fs, path, env};

pub unsafe trait Element: ocl::OclPrm + 'static {
  fn rtype() -> &'static str;
  fn ctype() -> &'static str;
  fn is_real() -> bool { false }
}

unsafe impl Element for f32 {
  fn rtype() -> &'static str { "f32" }
  fn ctype() -> &'static str { "float" }
  fn is_real() -> bool { true }
}

unsafe impl Element for u8 {
  fn rtype() -> &'static str { "u8" }
  fn ctype() -> &'static str { "uchar" }
}

pub unsafe trait Real: Element {}

unsafe impl Real for f32 {}

pub fn source() -> String {
  _source(marker::PhantomData::<f32>::default())
  + &_source(marker::PhantomData::<u8>::default())
}

fn _source<T: Element>(_m: marker::PhantomData<T>) -> String {
  include_str!("autograph.cl")
    .replace("RTYPE", &T::rtype())
    .replace("CTYPE", &T::ctype())
    .replace("IS_REAL", &T::is_real().to_string())
    .to_string()
}

pub fn is_restrict<T: Element>(lhs: &ocl::Buffer<T>, rhs: &ocl::Buffer<T>) -> bool {
  rhs.as_ptr() == lhs.as_ptr()
}

pub fn restrict_str<T: Element>(lhs: &ocl::Buffer<T>, rhs: &ocl::Buffer<T>) -> &'static str {
  if is_restrict(lhs, rhs) { "_restrict" } else { "" }
}


#[derive(Default, Clone, Copy, Eq, PartialEq)]
pub struct Shape {
  dims: [usize; 4],
  len: usize
}

macro_rules! impl_shape_from {
  ($arr:ty, $n:literal) => {
    impl From<$arr> for Shape {
      fn from(arr: $arr) -> Self {
        let mut s = Self{dims: <_>::default(), len: $n};
        s.copy_from_slice(&arr);
        s
      }
    } 
  }
}

impl_shape_from!([usize; 1], 1);
impl_shape_from!([usize; 2], 2);
impl_shape_from!([usize; 3], 3);
impl_shape_from!([usize; 4], 4);

impl From<&Shape> for Shape {
  fn from(shape: &Shape) -> Self {
    *shape
  }
}

impl Shape {
  pub fn len(&self) -> usize { self.len }
  pub fn rows(&self) -> usize {
    match self.len {
      0 => 0,
      1 => 1,
      _ => self[self.len-2]
    }
  }
  pub fn cols(&self) -> usize {
    match self.len {
      0 => 0,
      _ => self[self.len-1]
    }
  }
  pub fn size(&self) -> usize {
    self.iter().product()
  }
}

impl ops::Deref for Shape {
  type Target = [usize];
  fn deref(&self) -> &[usize] {
    &self.dims[..self.len]
  }
}

impl ops::DerefMut for Shape {
  fn deref_mut(&mut self) -> &mut [usize] {
    &mut self.dims[..self.len]
  }
}

impl fmt::Debug for Shape {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ops::Deref::deref(self).fmt(f)
  }
}

#[derive(Clone, Debug)]
pub struct TensorRef<'a, T: Element> {
  shape: Shape,
  data: &'a [T]
}

#[derive(Clone)]
pub struct TensorRefChunks<'a, T: Element, B: iter::Iterator<Item=usize>> {
  shape: Shape,
  data: &'a [T],
  batch_sizes: iter::Peekable<B>
}

impl<'a, T: Element, B: iter::Iterator<Item=usize>> iter::Iterator for TensorRefChunks<'a, T, B> {
  type Item = TensorRef<'a, T>;
  fn next(&mut self) -> Option<Self::Item> {
    self.batch_sizes.next()
      .and_then(|b| {
      if self.batch_sizes.peek().is_none() {
        Some(TensorRef::new(self.shape.clone(), self.data))
      }
      else {
        self.shape[0] -= b;
        let mut shape = self.shape.clone();
        shape[0] = b;
        let data = &self.data[..b];
        self.data = &self.data[b..];
        Some(TensorRef::new(shape, data))
      }
    })
  }
}

impl<'a, T: Element> TensorRef<'a, T> {
  pub fn new(shape: Shape, data: &'a [T]) -> Self {
    Self{shape, data}
  }
  pub fn shape(&self) -> &Shape { &self.shape }
  pub fn data(&self) -> &[T] { self.data }
  fn batches(&self, batch_sizes: impl IntoIterator<Item=usize>) -> impl iter::Iterator<Item=TensorRef<'a, T>> {
    TensorRefChunks{shape: self.shape.clone(), data: self.data, batch_sizes: batch_sizes.into_iter().peekable()} 
  }
}

#[derive(Default, Debug)]
pub struct TensorBase<T: Element, A: smallvec::Array<Item=T>> {
  shape: Shape,
  data: smallvec::SmallVec<A>
}

impl<T: Element, A: smallvec::Array<Item=T>> TensorBase<T, A> {
  pub fn new(shape: impl Into<Shape>, data: impl AsRef<[T]>) -> Self
    where A::Item: Copy {
    let shape = shape.into();
    let data = smallvec::SmallVec::from_slice(data.as_ref());
    debug_assert_eq!(shape.size(), data.len());
    Self{shape, data}
  }
  unsafe fn uninitialized(shape: impl Into<Shape>) -> Self {
    let shape = shape.into();
    let n = shape.size();
    let mut data = smallvec::SmallVec::with_capacity(n);
    data.set_len(n);
    Self{shape, data}
  }
  pub fn shape(&self) -> &Shape { &self.shape }
  pub fn data(&self) -> &[T] { self.data.as_slice() }
  pub fn data_mut(&mut self) -> &mut [T] { self.data.as_mut_slice() }
}

impl<'a, T: Element, A: smallvec::Array<Item=T>> From<&'a TensorBase<T, A>> for TensorRef<'a, T> {
  fn from(tensor: &'a TensorBase<T, A>) -> Self {
    Self::new(tensor.shape.clone(), tensor.data.as_slice())
  }
}

pub type Tensor<T> = TensorBase<T, [T; 32]>;

#[derive(Debug, Clone)]
pub struct Queue {
  queue: ocl::Queue,
  program: ocl::Program,
  power: usize
}

impl Queue {
  fn new(context: &ocl::Context, device: ocl::Device, program: &ocl::Program, power: usize) -> Self {
    let queue = ocl::Queue::new(context, device, Some(ocl::CommandQueueProperties::new().out_of_order()))
      .unwrap();
    Self{queue, program: program.clone(), power}
  }
  fn power(&self) -> usize { self.power }
  pub fn buffer_builder<T: Element>(&self) -> ocl::builders::BufferBuilder<T> {
    ocl::Buffer::builder()
      .queue(self.queue.clone())
  }
  pub fn kernel_builder(&self) -> ocl::builders::KernelBuilder { 
    let mut builder = ocl::Kernel::builder();
    builder.program(&self.program)
      .queue(self.queue.clone());
    builder
  }
} 

#[derive(Debug)]
pub struct Graph {
  queues: Vec<Queue>
}

impl Graph {
  pub fn new(platforms: impl iter::IntoIterator<Item=(ocl::Platform, ocl::enums::DeviceSpecifier)>, source: impl Into<String> + Clone) -> Self {
    let queues = platforms.into_iter()
      .flat_map(|(platform, device_specifier)| {
      let context = ocl::Context::builder()
        .platform(platform)
        .build()
        .unwrap();
      let program = ocl::Program::builder()
        .devices(device_specifier)
        .source(source.clone())
        .build(&context)
        .unwrap();
      let power = 1; // needs impl
      ocl::Device::list_from_core(program.devices().unwrap())
        .into_iter()
        .map(move |device| Queue::new(&context, device, &program, power))
    }).collect();
    Self{queues}
  }
  pub fn batch_sizes(&self, batch_size: usize) -> impl iter::Iterator<Item=usize> + '_ {
    let mut powers = self.queues.iter()
      .map(|q| q.power());
    let power = powers.clone()
      .sum::<usize>();
    let mut batch_sizes = powers.map(move |p| p*batch_size/power);
    let rem = batch_size - batch_sizes.clone()
      .sum::<usize>();
    batch_sizes.zip(iter::once(rem).chain(iter::repeat(0)))
      .map(|(b, r)| b + r)
  }
  pub fn variable<'a, T: Element>(&self, tensor: impl Into<TensorRef<'a, T>>, req_grad: bool) -> Variable<T> {
    let tensor = tensor.into();
    let batch_size = tensor.shape()[0];
    self.queues.iter()
      .zip(tensor.batches(self.batch_sizes(batch_size)))
      .map(|(q, t)| DualVertex::input(t, q, req_grad))
      .collect()
  }
}

#[derive(Debug)]
pub struct Vertex<T: Element> {
  queue: Queue,
  shape: Shape,
  buffer: ocl::Buffer<T>
}

impl<T: Element> Vertex<T> {
  pub fn input<'a>(tensor: impl Into<TensorRef<'a, T>>, queue: &Queue) -> Self {
    let queue = queue.clone();
    let tensor = tensor.into();
    let mut buffer = queue.buffer_builder()
      .len(tensor.shape().size())
      .copy_host_slice(tensor.data())
      .build()
      .unwrap();
    Self{queue, shape: tensor.shape().clone(), buffer}
  }
  pub fn output(queue: &Queue, shape: &Shape) -> Self {
    let queue = queue.clone();
    let shape = shape.clone();
    let mut buffer = queue.buffer_builder()
      .len(shape.size())
      .build()
      .unwrap();
    Self{queue, shape, buffer}
  }
  pub fn queue(&self) -> &Queue { &self.queue }
  pub fn shape(&self) -> &Shape { &self.shape }
  pub fn buffer(&self) -> &ocl::Buffer<T> { &self.buffer }
  fn read(&self, mut data: impl AsMut<[T]>) {
    self.buffer.read(data.as_mut()).enq().unwrap();
  }
}

#[derive(Debug)]
pub struct DualVertex<T: Element> {
  value: Vertex<T>,
  grad: Option<Vertex<T>>,
  req_grad: bool
}

impl<T: Element> DualVertex<T> {
  pub fn input<'a>(tensor: impl Into<TensorRef<'a, T>>, queue: &Queue, req_grad: bool) -> Self {
    let value = Vertex::input(tensor, queue);
    Self{value, grad: None, req_grad}
  }
  pub fn output(value: Vertex<T>, req_grad: bool) -> Self {
    let grad = if req_grad {
      Some(Vertex::output(value.queue(), value.shape()))
    }
    else { None }; 
    Self{value, grad, req_grad}
  }
  pub fn value(&self) -> &Vertex<T> { &self.value }
  pub fn shape(&self) -> &Shape { self.value.shape() }
  pub fn req_grad(&self) -> bool { self.req_grad }
}

#[derive(Debug)]
pub struct Variable<T: Element> {
  vertices: smallvec::SmallVec<[DualVertex<T>; 8]>
}

impl<T: Element> Variable<T> {
  pub fn shape(&self) -> Shape {
    let mut shape = self.vertices[0].value().shape().clone();
    self.vertices.iter()
      .map(|v| v.shape())
      .skip(1)
      .for_each(|s| {
      debug_assert!((shape.len() <= 1 && s.len() <= 1) || (shape[1..] == s[1..]));
      shape[0] += s[0];
    });
    shape
  }
}

impl<T: Element> iter::FromIterator<DualVertex<T>> for Variable<T> {
  fn from_iter<I: iter::IntoIterator<Item=DualVertex<T>>>(vertices: I) -> Self {
    Self{vertices: vertices.into_iter().collect()}
  }
}

impl<T: Element> ops::Deref for Variable<T> {
  type Target = [DualVertex<T>];
  fn deref(&self) -> &Self::Target {
    self.vertices.as_slice()
  }
}

impl<T: Element, A: smallvec::Array<Item=T> + Default> From<&Variable<T>> for TensorBase<T, A> {
  fn from(variable: &Variable<T>) -> Self {
    let mut tensor = unsafe { Self::uninitialized(variable.shape()) };
    variable.vertices.iter()
      .fold(tensor.data_mut(), |data, v| {
      let n = v.shape().size();
      v.value().read(&mut data[..n]);
      &mut data[n..]
    });
    tensor
  }
}

impl<T: Element, A: smallvec::Array<Item=T> + Default> From<Variable<T>> for TensorBase<T, A> {
  fn from(variable: Variable<T>) -> Self { Self::from(&variable) }
}


macro_rules! impl_binary_op {
  ($op_trait:ident, $func:ident, $kernel:literal) => {
    use ops::*;
    impl<T: Element> $op_trait<&Vertex<T>> for &Vertex<T> {
      type Output = Vertex<T>;
      fn $func(self, rhs: &Vertex<T>) -> Self::Output {
        let lhs = self;
        debug_assert_eq!(lhs.shape(), rhs.shape());
        let out = Vertex::output(self.queue(), lhs.shape());
        let kernel = self.queue().kernel_builder()
          .name(format!("{}{}_{}", $kernel, restrict_str(lhs.buffer(), rhs.buffer()), T::rtype()))
          .global_work_size(lhs.shape().size())
          .arg(out.buffer())
          .arg(lhs.buffer())
          .arg(rhs.buffer())
          .build()
          .unwrap();
        unsafe { kernel.enq().unwrap(); }
        out
      }
    }
    impl<T: Element> $op_trait<&DualVertex<T>> for &DualVertex<T> {
      type Output = DualVertex<T>;
      fn $func(self, rhs: &DualVertex<T>) -> Self::Output {
        let lhs = self;
        debug_assert_eq!(lhs.shape(), rhs.shape());
        let out = DualVertex::output(lhs.value() + rhs.value(), lhs.req_grad() || rhs.req_grad());
        // grad ops
        out
      }
    }
    impl<T: Element> $op_trait<&Variable<T>> for &Variable<T> {
      type Output = Variable<T>;
      fn $func(self, rhs: &Variable<T>) -> Self::Output {
        use std::thread;
        let lhs = self;
        debug_assert_eq!(lhs.shape(), rhs.shape());
        lhs.iter()
          .zip(rhs.iter())
          .map(|(a, b)| a.$func(b))
          .collect()
      }
    }
  }
}

impl_binary_op!(Add, add, "add");
impl_binary_op!(Sub, sub, "sub");
impl_binary_op!(Mul, mul, "mul");
impl_binary_op!(Div, div, "div");

/*
pub struct Op {
  kernels: Vec<ocl::Kernel>,
  verbose: bool
}

impl Default for Op {
  fn default() -> Self {
    Self{kernels: Vec::new(), verbose: true}
  }
}

impl Op {
  fn enq(&self) { 
    self.kernels.iter()
      .for_each(|k| {
      if self.verbose { println!("{:?}", k.name()); }
      unsafe { k.enq().unwrap(); }
   });
  }
  pub fn extend(&mut self, kernels: impl iter::IntoIterator<Item=ocl::Kernel>) {
    self.kernels.extend(kernels)
  }
}

impl<I: IntoIterator<Item=ocl::Kernel>> From<I> for Op {
  fn from(kernels: I) -> Self {
    Self{kernels: kernels.into_iter().collect(), verbose: true}
  }
}*/
/*
pub struct Graph {
  program: ocl::Program,
  queue: ocl::Queue,
  backward: cell::RefCell<Vec<Op>>
}

impl Graph {
  pub fn new(source: impl Into<String>, context: &ocl::Context, device: ocl::Device) -> Self {
    let program = ocl::Program::builder()
      .source(source)
      .devices(device)
      .build(context)
      .unwrap();
    let queue = ocl::Queue::new(context, device, None)
      .unwrap();
    Self{program, queue, backward: cell::RefCell::new(Vec::new())}
  }
  
  pub fn begin_op(&self) {
    self.backward.borrow_mut()
      .push(Op::default());
  }
  pub fn extend_op(&self, kernels: impl iter::IntoIterator<Item=ocl::Kernel>) {
    self.backward.borrow_mut()
      .last_mut()
      .unwrap()
      .extend(kernels);
  } 
  pub fn backward(&self) {
    self.backward.borrow()
      .iter()
      .rev()
      .for_each(|op| { op.enq(); });
    self.backward.borrow_mut()
      .clear();
  }
}

pub trait Execution: Default + Copy {
  type Exec: Execution;
  fn exec<I: iter::IntoIterator<Item=ocl::Kernel>>(graph: &Graph, kernels: I) { Self::Exec::exec(&graph, kernels); }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct Forward;

impl Execution for Forward {
  type Exec = Self;
  fn exec<I: iter::IntoIterator<Item=ocl::Kernel>>(graph: &Graph, kernels: I) { Op::from(kernels).enq() }
}

impl<E: Execution> Execution for (Forward, E) {
  type Exec = E;
}

#[derive(Default, Debug, Copy, Clone)]
pub struct Backward;

impl Execution for Backward {
  type Exec = Self;
  fn exec<I: iter::IntoIterator<Item=ocl::Kernel>>(graph: &Graph, kernels: I) {
    graph.extend_op(kernels);
  }
}

impl<E: Execution> Execution for (Backward, E) {
  type Exec = Backward;
}



#[derive(Clone)]
pub struct Vertex<'g, T: Element, E: Execution> {
  graph: &'g Graph,
  shape: Shape,
  buffer: ocl::Buffer<T>,
  exec: E
}

impl<'g, T: Element, E: Execution> Vertex<'g, T, E> {
  pub fn new(graph: &'g Graph, shape: impl Into<Shape>) -> Self {
    let shape = shape.into();
    let buffer = graph.buffer_builder()
      .len(shape.size())
      .build()
      .unwrap();
    Self{graph, shape, buffer, exec: E::default()}
  }
  pub fn graph(&self) -> &'g Graph { self.graph }
  pub fn shape(&self) -> &Shape { &self.shape }
  pub fn buffer(&self) -> &ocl::Buffer<T> { &self.buffer }
  pub fn zero(&self) {
    let kernel = self.graph.kernel_builder()
      .name(format!("zero_{}", T::rtype()))
      .global_work_size(self.shape.size())
      .arg(self.buffer())
      .build()
      .unwrap();
    Forward::exec(&self.graph, vec![kernel]);
  }
  pub fn one(&self) {
    let kernel = self.graph.kernel_builder()
      .name(format!("one_{}", T::rtype()))
      .global_work_size(self.shape.size())
      .arg(self.buffer())
      .build()
      .unwrap();
    Forward::exec(&self.graph, vec![kernel]);
  }
}

impl<'g, T: Element> Vertex<'g, T, Forward> {
  pub fn from_tensor(graph: &'g Graph, tensor: &Tensor<T>) -> Self {
     let buffer = graph.buffer_builder()
      .len(tensor.shape().size())
      .copy_host_slice(&tensor.data())
      .build()
      .unwrap();
    Self{graph, shape: tensor.shape().clone(), buffer, exec: Forward}
  }
}

impl<'g, T: Element, E: Execution> From<Vertex<'g, T, E>> for Tensor<T> {
  fn from(vertex: Vertex<'g, T, E>) -> Tensor<T> {
    Self::from(&vertex)
  }
} 

impl<'g, T: Element, E: Execution> From<&Vertex<'g, T, E>> for Tensor<T> {
  fn from(vertex: &Vertex<'g, T, E>) -> Tensor<T> {
    let mut data = Vec::with_capacity(vertex.shape().size());
    unsafe { data.set_len(data.capacity()); }
    vertex.buffer().read(&mut data).enq().unwrap();
    Tensor::new(vertex.shape(), data)
  }
}



macro_rules! impl_assign_op {
  ($op_trait:ident, $func:ident, $kernel:literal) => {
    use ops::*;
    impl<'g, T: Element, E1: Execution, E2: Execution> $op_trait<&Vertex<'g, T, E2>> for Vertex<'g, T, E1>
      where (E1, E2): Execution<Exec=E1> {
      fn $func(&mut self, rhs: &Vertex<'g, T, E2>) {
        let lhs = &self;
        let graph = self.graph();
        debug_assert_eq!(lhs.shape(), rhs.shape());
        let kernel = graph.kernel_builder()
          .name(format!("{}{}_{}", $kernel, restrict_str(lhs.buffer(), rhs.buffer()), T::rtype()))
          .global_work_size(lhs.shape().size())
          .arg(lhs.buffer())
          .arg(rhs.buffer())
          .build()
          .unwrap();
        <(E1, E2)>::exec(graph, vec![kernel]);
      }
    }
  }
}

impl_assign_op!(AddAssign, add_assign, "add_assign"); 

pub trait Transpose {
  type Output;
  fn t(self) -> Self::Output;
}

impl<'g, T: Element, E: Execution> Transpose for &Vertex<'g, T, E> {
  type Output = Vertex<'g, T, E>;
  fn t(self) -> Self::Output {
    debug_assert!(self.shape().len() <= 2);
    let out = Vertex::new(self.graph(), [self.shape.cols(), self.shape.rows()]);
    let graph = self.graph();
    let kernel = graph.kernel_builder()
      .name(format!("t_{}", T::rtype()))
      .global_work_size([out.shape().rows(), out.shape().cols()])
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    E::exec(graph, vec![kernel]);
    out
  }
}
    
pub trait Dot<R> {
  type Output;
  fn dot(self, rhs: R) -> Self::Output;
}

impl<'g, T: Element, E1: Execution, E2: Execution> Dot<&Vertex<'g, T, E2>> for &Vertex<'g, T, E1>
  where (E1, E2): Execution {
  type Output = Vertex<'g, T, <(E1, E2) as Execution>::Exec>;
  fn dot(self, rhs: &Vertex<'g, T, E2>) -> Self::Output {
    let lhs = self;
    debug_assert_eq!(lhs.shape().cols(), rhs.shape().rows(), "Invalid shapes: {:?} dot {:?}", lhs.shape(), rhs.shape());
    let graph = self.graph();
    let out = Vertex::new(graph, [lhs.shape().rows(), rhs.shape().cols()]);
    let kernel = graph.kernel_builder()
      .name(format!("dot{}_{}", restrict_str(lhs.buffer(), rhs.buffer()), T::rtype()))
      .global_work_size([out.shape().rows(), out.shape().cols()])
      .arg(out.buffer())
      .arg(lhs.buffer())
      .arg(rhs.buffer())
      .arg(out.shape().rows())
      .arg(out.shape().cols())
      .arg(lhs.shape().cols())
      .build()
      .unwrap();
    <(E1, E2)>::exec(graph, vec![kernel]);
    out
  }
}

pub struct Variable<'g, T: Real> {
  value: Vertex<'g, T, Forward>,
  grad: Option<cell::RefCell<Vertex<'g, T, Backward>>>
}

impl<'g, T: Real> Variable<'g, T> {
  pub fn new(value: Vertex<'g, T, Forward>, req_grad: bool) -> Self {
    let grad = if req_grad {
      let grad = Vertex::new(value.graph(), value.shape().clone());
      grad.zero();
      Some(cell::RefCell::new(grad))
    }
    else { None };
    Self{value, grad}
  }
  pub fn from_tensor(graph: &'g Graph, tensor: &Tensor<T>) -> Self {
    Self{value: Vertex::from_tensor(graph, tensor), grad: None}
  }
  pub fn value(&self) -> &Vertex<'g, T, Forward> { &self.value }
  pub fn value_mut(&mut self) -> &Vertex<'g, T, Forward> { &mut self.value }
  pub fn grad(&self) -> Option<cell::Ref<Vertex<'g, T, Backward>>> {
    self.grad.as_ref()
      .map(|c| c.borrow())
  }
  pub fn grad_mut(&self) -> Option<cell::RefMut<Vertex<'g, T, Backward>>> {
    self.grad.as_ref()
      .map(|c| c.borrow_mut())
  }
  pub fn req_grad(&self) -> bool { self.grad.is_some() }
  pub fn zero_grad(&mut self) {
    if !self.req_grad() {
      let grad = Vertex::new(self.value.graph(), self.value.shape().clone());
      self.grad = Some(cell::RefCell::new(grad));
    }
    self.grad_mut()
      .unwrap()
      .zero();
  }   
  pub fn one_grad(&self) {
    self.grad_mut()
      .unwrap()
      .one();
  }   
  pub fn backward(&self) {
    self.one_grad();
    self.value()
      .graph()
      .backward();
  }
}

impl<'g, T: Real> Dot<&Variable<'g, T>> for &Variable<'g, T> {
  type Output = Variable<'g, T>;
  fn dot(self, rhs: &Variable<'g, T>) -> Self::Output {
    let lhs = self;
    let out = Variable::new(lhs.value().dot(rhs.value()), lhs.req_grad() || rhs.req_grad());
    self.value().graph().begin_op();
    out.grad().map(|ograd| {
      lhs.grad_mut().map(|mut lgrad| *lgrad += &ograd.dot(&rhs.value().t()));
      rhs.grad_mut().map(|mut rgrad| *rgrad += &lhs.value().t().dot(&ograd));
    });
    out
  }
}*/

/*
// Dataset

pub fn download<'f>(url: impl AsRef<str>, f: &'f mut fs::File) -> reqwest::Result<()> {
  let mut req = reqwest::get(url.as_ref())?;
  io::copy(&mut req, f);
  Ok(())
}

#[derive(Clone)]
pub struct Mnist {
  train_images: Vec<u8>,
  train_labels: Vec<u8>,
  test_images: Vec<u8>,
  test_labels: Vec<u8>
}

impl Mnist {
  pub fn new() -> io::Result<Self> {
    let mnist_path = Self::mnist_path()?;
    if !mnist_path.exists() { 
      fs::create_dir_all(&mnist_path)?; 
    }
    let train_images_file = Self::load(mnist_path.join("train-images-idx3-ubyte.gz"), "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")?;
    let train_images = Self::load_images(train_images_file, 60_000)?;
    let train_labels_file = Self::load(mnist_path.join("train-labels-idx1-ubyte.gz"), "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")?;
    let train_labels = Self::load_labels(train_labels_file, 60_000)?;
    let test_images_file = Self::load(mnist_path.join("t10k-images-idx3-ubyte.gz"), "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")?;
    let test_images = Self::load_images(test_images_file, 10_000)?;
    let test_labels_file = Self::load(mnist_path.join("t10k-labels-idx1-ubyte.gz"), "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")?;
    let test_labels = Self::load_labels(test_labels_file, 10_000)?;
    Ok(Self{train_images, train_labels, test_images, test_labels}) 
  }
  /*pub fn train(&self, batch_size: usize) -> impl Iterator<Item=(&[T], &[u8])> + '_ {
    self.train_images
      .chunks(batch_size*28*28)
      .zip(self.train_labels
        .chunks(batch_size))
  }*/
  pub fn test<'g, T: Real>(&self, batch_size: usize) -> ([usize; 4], impl Iterator<Item=> + '_) {
    self.test_images.chunks_exact(batch_size*28*28)
      .map(
      .zip(self.test_labels.chunks_exact(batch_size))
  }
  fn mnist_path() -> io::Result<path::PathBuf> { Ok(env::current_dir()?.join("datasets/mnist")) }
  fn load(fpath: impl AsRef<path::Path>, url: impl AsRef<str>) -> io::Result<fs::File> {
    let mut f: fs::File;
    if fpath.as_ref().exists() { 
      f = fs::File::open(fpath)?;
    }
    else {
      f = fs::File::create(fpath)?;
      download(&url, &mut f).expect(&format!("Mnist unable to download: {:?}", url.as_ref()));
    }
    Ok(f)
  }
  fn load_images(f: fs::File, n: usize) -> io::Result<Vec<u8>> {
    let mut gz = flate2::read::GzDecoder::new(&f);
    use io::Read;
    let mut magic = [0; 4];
    gz.read(&mut magic)?;
    assert_eq!(magic[..], (2051 as u32).to_be_bytes());
    let mut nimages = [0; 4];
    gz.read(&mut nimages);
    assert_eq!(nimages[..], (n as u32).to_be_bytes());
    let mut nrows = [0; 4];
    gz.read(&mut nrows);
    assert_eq!(nrows[..], (28 as u32).to_be_bytes());
    let mut ncols = [0; 4];
    gz.read(&mut ncols);
    assert_eq!(ncols[..], (28 as u32).to_be_bytes());
    let mut images = Vec::new();
    gz.read_to_end(&mut images);
    Ok(images)
  } 
  fn load_labels(f: fs::File, n: usize) -> io::Result<Vec<u8>> {
    let mut gz = flate2::read::GzDecoder::new(&f);
    use io::Read;
    let mut magic = [0; 4];
    gz.read(&mut magic)?;
    assert_eq!(magic[..], (2049 as u32).to_be_bytes());
    let mut nlabels = [0; 4];
    gz.read(&mut nlabels);
    assert_eq!(nlabels[..], (n as u32).to_be_bytes());
    let mut buf = [0; 1024];
    let mut labels = Vec::new();
    gz.read_to_end(&mut labels)?;
    Ok(labels)
  }
}*/
/*
pub struct MnistIter<'w, 'm, T: Real> {
  ws: &'w Workspace,
  images: &'m [T],
  labels: &'m [u8],
  dims: Vec<usize>
}

impl<'w, 'm, T: Real> MnistIter<'w, 'm, T> {
  fn new(ws: &'w Workspace, images: &'m [T], labels: &'m [u8], dims: Vec<usize>) -> Self {
    Self{ws, images, labels dims}
  }
}

impl<'w, 'm, T: Real> iter::Iterator for MnistIter<'w, 'm, T> {
  type Item = (Tensor<T>, Tensor<u16>);
  fn next(&mut self) -> Option<Self::Item> {
    if labels.len() == 
    
}*/

