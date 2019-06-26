use std::ops;

pub trait Dimension: std::fmt::Debug + ops::Deref<Target=[usize]> + Copy + PartialEq {
  fn rmaj_strides(&self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dim<D> {
  dims: D
}

impl<D> From<D> for Dim<D> 
  where Self: Dimension {
  fn from(dims: D) -> Self {
    Self{dims}
  }
}

pub type Ix = Dim<[usize; 1]>;
pub type Ix1 = Ix;

impl ops::Deref for Ix1 {
  type Target = [usize];
  fn deref(&self) -> &[usize] {
    &self.dims
  }
}

impl Dimension for Ix1 {
  fn rmaj_strides(&self) -> Self {
    Self{dims: [1]}
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape<D> {
  dims: D,
  strides: D
}

impl<D: Dimension> Shape<D> {
  pub fn rmaj<Dims: Into<D>>(dims: Dims) -> Self {
    let dims = dims.into();
    let strides = dims.rmaj_strides();
    Self{dims, strides}
  }
  pub fn size(&self) -> usize {
    self.dims.iter().product::<usize>()
  }
}

#[derive(Debug)]
pub struct OpenCL {
  context: ocl::Context,
  program: ocl::Program
}

impl OpenCL {
  pub fn new(context: ocl::Context, program: ocl::Program) -> Self {
    Self{context, program}
  }
  pub fn context(&self) -> &ocl::Context { &self.context }
  pub fn program(&self) -> &ocl::Program { &self.program }
  pub fn queue(&self) -> ocl::Queue {
    ocl::Queue::new(&self.context,
                    self.context.get_device_by_wrapping_index(0),
                    None)
      .unwrap()
  }
}

#[derive(Debug)]
pub enum Buffer {
  F32(ocl::Buffer<f32>),
  I32(ocl::Buffer<i32>)
}

pub trait Element: ocl::OclPrm {
  fn ocl_buffer_ref<'b>(buffer: &'b Buffer) -> &'b ocl::Buffer<Self>;
  fn buffer(ocl_buffer: ocl::Buffer<Self>) -> Buffer;
  fn ctype_name() -> &'static str;
}

impl Element for f32 {
  fn ocl_buffer_ref<'b>(buffer: &'b Buffer) -> &'b ocl::Buffer<Self> {
    match buffer {
      Buffer::F32(ref buffer) => buffer,
      Buffer::I32(_) => panic!()
    }
  }
  fn buffer(ocl_buffer: ocl::Buffer<Self>) -> Buffer {
    Buffer::F32(ocl_buffer)
  }
  fn ctype_name() -> &'static str { "float" }
}

impl Element for i32 {
  fn ocl_buffer_ref<'b>(buffer: &'b Buffer) -> &'b ocl::Buffer<Self> {
    match buffer {
      Buffer::F32(_) => panic!(),
      Buffer::I32(ref buffer) => buffer
    }
  }
  fn buffer(ocl_buffer: ocl::Buffer<Self>) -> Buffer {
    Buffer::I32(ocl_buffer)
  }
  fn ctype_name() -> &'static str { "int" } 
}

#[derive(Debug)]
pub struct Graph<'o> {
  backend: &'o OpenCL,
  vertices: Vec<Buffer>,
  pub ops: Vec<ocl::Kernel>
}

impl<'o> Graph<'o> {
  pub fn new(backend: &'o OpenCL) -> Self {
    Self{backend, vertices: Vec::new(), ops: Vec::new()}
  }
  pub fn vertex<'b, T: Element>(&mut self, data: Option<&'b Vec<T>>, len: usize) -> usize {
    let buffer = ocl::Buffer::<T>::builder()
      .context(&self.backend.context())
      .len(len)
      .build()
      .unwrap();
    if let Some(data) = data {
      buffer.write(data)
        .queue(&self.backend.queue())
        .enq()
        .unwrap();
    }
    let idx = self.vertices.len();
    self.vertices.push(T::buffer(buffer));
    idx
  }
  
  pub fn backend(&self) -> &'o OpenCL { self.backend }
  pub fn op(&mut self, kernel: ocl::Kernel) {
    self.ops.push(kernel);
  }
  pub fn exec<'q>(&self) {
    let queue = self.backend.queue();
    self.ops.iter()
      .for_each(|k| { 
        unsafe {
          k.cmd()
            .queue(&queue)
            .enq()
            .unwrap();
        }
      });
    queue.finish()
      .unwrap();
  }
}

impl<'o, 'b, T: Element, D: Dimension> ops::Index<&'b Tensor<T, D>> for Graph<'o> {
  type Output = ocl::Buffer<T>;
  fn index(&self, tensor: &'b Tensor<T, D>) -> &Self::Output {
    T::ocl_buffer_ref(&self.vertices[tensor.idx])
  }
}

#[derive(Debug)]
pub struct Tensor<T, D> {
  shape: Shape<D>,
  data: Option<Vec<T>>,
  idx: usize
}

impl<T: Element, D: Dimension> Tensor<T, D> {
  pub fn new<'g, 'o>(graph: &'g mut Graph<'o>, shape: Shape<D>, data: Option<Vec<T>>) -> Self {
    let idx = graph.vertex(data.as_ref(), shape.size());
    Self{shape, data, idx}
  }
  pub fn shape(&self) -> &Shape<D> { &self.shape }
  pub fn read<'g, 'o>(&mut self, graph: &'g Graph<'o>)
    where T: Default + Copy {
   if self.data.is_none() {
      let mut data = Vec::with_capacity(self.shape.size());
      unsafe { data.set_len(data.capacity()); }
      self.data = Some(data);
    }
    graph[self].read(self.data.as_mut().unwrap())
      .queue(&graph.backend().queue())
      .enq()
      .unwrap();
  }
}

pub struct Add<T: Element> {
  _m: std::marker::PhantomData<T>
}

impl<T: Element> Add<T> {
  pub fn src() -> String {
    r#"
      kernel void add_T(global T* out, global T* lhs, global T* rhs) {
        size_t gid = get_global_id(0);
        out[gid] = lhs[gid] + rhs[gid];
      }
    "#.replace("T", T::ctype_name())
      .to_string()
  }
  pub fn op<'g, 'o, 'l, 'r, D: Dimension>(graph: &'g mut Graph<'o>, lhs: &'l Tensor<T, D>, rhs: &'r Tensor<T, D>) -> Tensor<T, D> {
    debug_assert_eq!(lhs.shape(), rhs.shape());
    let out = Tensor::new(graph, *lhs.shape(), lhs.data.clone());
    let kernel = ocl::Kernel::builder()
      .program(graph.backend().program())
      .name("add_T".replace("T", T::ctype_name()))
      .global_work_size([lhs.shape().size()])
      .arg(&graph[&out])
      .arg(&graph[&lhs])
      .arg(&graph[&rhs])
      .build()
      .unwrap();
    graph.op(kernel);
    out
  }
}

pub struct Ones<T: Element> {
  _m : std::marker::PhantomData<T>
}

impl<T: Element> Ones<T> {
  pub fn src() -> String {
    r#"
      kernel void ones_T(global T* out) {
        out[get_global_id(0)] = 1;
      }
    "#.replace("T", T::ctype_name())
      .to_string()
  }
  pub fn op<'g, 'o, D: Dimension>(graph: &'g mut Graph<'o>, shape: Shape<D>) -> Tensor<T, D> {
    let out = Tensor::new(graph, shape, None);
    let kernel = ocl::Kernel::builder()
      .program(&graph.backend().program())
      .name("ones_T".replace("T", T::ctype_name()))
      .global_work_size([out.shape().size()])
      .arg(&graph[&out])
      .build()
      .unwrap();
    graph.op(kernel);
    out
  }
}
    
    
    
    




