use super::*;

#[derive(Debug)] 
pub enum Buffer {
  U8(ocl::Buffer<u8>),
  U16(ocl::Buffer<u16>),
  U32(ocl::Buffer<u32>),
  U64(ocl::Buffer<u64>),
  I8(ocl::Buffer<i8>),
  I16(ocl::Buffer<i16>),
  I32(ocl::Buffer<i32>),
  I64(ocl::Buffer<i64>),
  F32(ocl::Buffer<f32>),
  F64(ocl::Buffer<f64>)
}

use Buffer::*;

pub trait Element: native::Element + ocl::OclPrm {
  fn ctype() -> &'static str;
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer;
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self>;
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self>;
}

impl Element for u8 {
  fn ctype() -> &'static str { "uchar" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { U8(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let U8(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let U8(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for u16 {
  fn ctype() -> &'static str { "ushort" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { U16(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let U16(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let U16(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for u32 {
  fn ctype() -> &'static str { "uint" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { U32(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let U32(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let U32(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for u64 {
  fn ctype() -> &'static str { "ulong" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { U64(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let U64(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let U64(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for i8 {
  fn ctype() -> &'static str { "char" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { I8(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let I8(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let I8(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for i16 {
  fn ctype() -> &'static str { "short" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { I16(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let I16(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let I16(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for i32 {
  fn ctype() -> &'static str { "int" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { I32(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let I32(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let I32(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for i64 {
  fn ctype() -> &'static str { "long" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { I64(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let I64(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let I64(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for f32 {
  fn ctype() -> &'static str { "float" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { F32(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let F32(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let F32(ref mut b) = buffer { b } else { panic!() } }
}

impl Element for f64 {
  fn ctype() -> &'static str { "double" }
  fn opencl_buffer(b: ocl::Buffer<Self>) -> Buffer { F64(b) }
  fn opencl_ref(buffer: &Buffer) -> &ocl::Buffer<Self> { if let F64(ref b) = buffer { b } else { panic!() } }
  fn opencl_mut(buffer: &mut Buffer) -> &mut ocl::Buffer<Self> { if let F64(ref mut b) = buffer { b } else { panic!() } }
}


#[derive(Debug)]
pub struct Opencl {
  ocl_context: ocl::Context,
}

impl Opencl {
  pub fn new(ocl_context: ocl::Context) -> Self {
    Self{ocl_context}
  }
  pub fn ocl_context(&self) -> &ocl::Context { &self.ocl_context }
}

fn namespace(ns: impl fmt::Display, src: impl Into<String>) -> String {
  src.into()
    .replace("NAMESPACE", &ns.to_string())
}

impl<'c> ContextBase<'c> for Opencl
  where Self: 'c {
  type Buffer = Buffer;
  type Executor = Executor<'c>;
  type Source = String;
  fn compile(&'c self, sources: impl iter::IntoIterator<Item=Option<Self::Source>>) -> Self::Executor {
    let source = itertools::join(sources.into_iter()
      .enumerate()
      .map(|(i, s)| s.map_or(String::new(), |s| namespace(i, s))), "");
    println!("source: {}", &source);
    let device = self.ocl_context().get_device_by_wrapping_index(0);
    let program = ocl::Program::builder()
      .source(source)
      .devices(device)
      .build(self.ocl_context())
      .unwrap();
    let queue = ocl::Queue::new(self.ocl_context(), device, None)
      .unwrap();
    Executor{context: self, program, queue}
  }
}

impl<'c, T: Element> Context<'c, T> for Opencl 
  where Self: ContextBase<'c> {}
  
impl<'c, T: Element> Tensor<'c, T, Opencl> {
  pub fn buffer(&self) -> &ocl::Buffer<T> {
    T::opencl_ref(self.base().buffer())
  }
  pub fn buffer_mut(&mut self) -> &mut ocl::Buffer<T> {
    T::opencl_mut(self.base_mut().buffer_mut())
  }
  pub fn len(&self) -> usize {
    debug_assert_eq!(self.buffer().len(), self.shape().size());
    self.buffer().len()
  }
}


#[derive(Debug)]
pub struct Executor<'c> {
  context: &'c Opencl,
  program: ocl::Program,
  queue: ocl::Queue
}

impl<'c> Executor<'c> {
  pub fn context(&self) -> &'c Opencl { self.context }
  pub fn ocl_context(&self) -> &'c ocl::Context { self.context.ocl_context() }
  pub fn queue(&self) -> &ocl::Queue { &self.queue }
  pub fn buffer_builder<T: Element>(&self) -> ocl::builders::BufferBuilder<T> {
    ocl::Buffer::builder()
      .context(&self.ocl_context())
  }
  pub fn kernel_builder(&self) -> ocl::builders::KernelBuilder {
    let mut builder = ocl::Kernel::builder();
    builder.program(&self.program);
    builder
  }
}

impl<'c, T: Element> FromNative<'c, T, Opencl> for Tensor<'c, T, Opencl> {
  fn from_native(native_tensor: Tensor<'static, T, Native>, executor: &Executor<'c>) -> Self {
    let data = &native_tensor[..];
    let buffer = unsafe { executor.buffer_builder()
      .len(data.len())
      .use_host_slice(data)
      .build()
      .unwrap() };
    Tensor::new(executor.context(), *native_tensor.shape(), T::opencl_buffer(buffer)) 
  }
}


impl<'c, T: Element> ToNative<'c, T, Opencl> for Tensor<'c, T, Opencl> {
  fn to_native(self, executor: &Executor<'c>) -> Tensor<'static, T, Native> {
    let mut out = unsafe { native::uninitialized(*self.shape()) };
    self.buffer()
      .read(&mut out[..])
      .queue(&executor.queue)
      .enq()
      .unwrap();
    out
  }
}

impl<'c, T: Element> Uninitialized<T> for Executor<'c> {
  type Output = Tensor<'c, T, Opencl>;
  unsafe fn uninitialized(&self, shape: Shape) -> Self::Output {
    let buffer = self.buffer_builder()
      .len(shape.size())
      .build()
      .unwrap();
    Tensor::new(self.context(), shape, T::opencl_buffer(buffer))
  }
}


impl<'c, T: Element> ForwardOp<'c, Opencl> for ZerosOp<T> {
  fn source(&self) -> Option<String> {
Some(r#"kernel void _NAMESPACE_zeros_RTYPE_(global CTYPE* out) {
  out[get_global_id(0)] = 0;
}
"#.replace("RTYPE", T::rtype())
    .replace("CTYPE", T::ctype()))
  }   
  fn forward(&self, executor: &Executor<'c>, tensors: &[TensorBase<'c, Opencl>]) -> TensorBase<'c, Opencl> {
    let out: Tensor<T, _> = unsafe { executor.uninitialized(*self.vertex().shape()) };
    let kernel = executor.kernel_builder()
      .name(format!("_{}_zeros_{}_", self.vertex().idx(), T::rtype()))
      .global_work_size(out.len())
      .arg(out.buffer())
      .build()
      .unwrap();
    unsafe { kernel.cmd()
      .queue(executor.queue())
      .enq()
      .unwrap() };
    out.into_base()
  }
}

impl<'c, T: Element> ForwardOp<'c, Opencl> for OnesOp<T> {
  fn source(&self) -> Option<String> {
Some(r#"kernel void _NAMESPACE_ones_RTYPE_(global CTYPE* out) {
  out[get_global_id(0)] = 1;
}
"#.replace("RTYPE", T::rtype())
    .replace("CTYPE", T::ctype()))
  }   
  fn forward(&self, executor: &Executor<'c>, tensors: &[TensorBase<'c, Opencl>]) -> TensorBase<'c, Opencl> {
    let out: Tensor<T, _> = unsafe { executor.uninitialized(*self.vertex().shape()) };
    let out_buffer = out.buffer();
    let kernel = executor.kernel_builder()
      .name(format!("_{}_ones_{}_", self.vertex().idx(), T::rtype()))
      .global_work_size(out.len())
      .arg(out.buffer())
      .build()
      .unwrap();
    unsafe { kernel.cmd()
      .queue(executor.queue())
      .enq()
      .unwrap() };
    out.into_base()
  }
}
      
