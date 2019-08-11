use std::ops;

pub unsafe trait Element: ocl::OclPrm + 'static {
  fn rtype() -> String;
  fn ctype() -> String;
  fn is_real() -> bool;
  fn cwidth(device: ocl::Device) -> usize;
  fn cvector(device: ocl::Device) -> String {
    Self::ctype() + &Self::cwidth(device).to_string()
  } 
}

unsafe impl Element for f32 {
  fn rtype() -> String { String::from("f32") }
  fn ctype() -> String { String::from("float") }
  fn is_real() -> bool { true }
  fn cwidth(device: ocl::Device) -> usize {
    use ocl::enums::{DeviceInfoResult, DeviceInfo};
    if let DeviceInfoResult::NativeVectorWidthFloat(w) = device.info(DeviceInfo::NativeVectorWidthFloat).unwrap() {
      w as usize
    }
    else {
      panic!()
    }
  }
  
}

unsafe impl Element for i32 {
  fn rtype() -> String { String::from("i32") }
  fn ctype() -> String { String::from("int") }
  fn is_real() -> bool { false }
  fn cwidth(device: ocl::Device) -> usize {
    use ocl::enums::{DeviceInfoResult, DeviceInfo};
    if let DeviceInfoResult::NativeVectorWidthInt(w) = device.info(DeviceInfo::NativeVectorWidthInt).unwrap() {
      w as usize
    }
    else {
      panic!()
    }
  }
}

#[derive(Debug)]
pub struct Tensor<'w, T: Element> {
  workspace: &'w Workspace,
  dims: Vec<usize>,
  data: Option<Vec<T>>,
  buffer: ocl::Buffer<T>
}

impl<'w, T: Element> Tensor<'w, T> {
  pub fn read<'q>(&mut self)
    where T: Default {
    let mut data = vec![T::default(); self.len()];
    self.buffer.read(&mut data)
      .enq()
      .unwrap();
    self.data = Some(data);
  }
  pub fn workspace(&self) -> &'w Workspace { self.workspace }
  pub fn dims(&self) -> &Vec<usize> { &self.dims } 
  pub fn data(&self) -> Option<&Vec<T>> { self.data.as_ref() }
  pub fn len(&self) -> usize { self.buffer.len() }
  pub fn vlen(&self) -> usize { self.buffer.len() / self.cwidth() + self.buffer.len() % self.cwidth() }
  pub fn buffer(&self) -> &ocl::Buffer<T> { &self.buffer }
  pub fn restrict<'b>(&self, other: &'b Tensor<T>) -> bool {
    self.buffer().as_core().as_ptr() == other.buffer().as_core().as_ptr()
  }
  pub fn cwidth(&self) -> usize { T::cwidth(self.workspace.device()) }
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
  pub fn context(&self) -> &ocl::Context { &self.context }
  pub fn program(&self) -> &ocl::Program { &self.program }
  pub fn queue(&self) -> &ocl::Queue { &self.queue }
  pub fn device(&self) -> ocl::Device { self.context.devices()[0] }
}

pub fn source<'c>(context: &'c ocl::Context) -> String {
  debug_assert_eq!(context.devices().len(), 1);
  let device = context.devices()[0];
  _source(std::marker::PhantomData::<f32>::default(), device)
  + &_source(std::marker::PhantomData::<i32>::default(), device)
}

fn _source<T: Element>(_m: std::marker::PhantomData<T>, device: ocl::Device) -> String {
  include_str!("autograph.cl")
    .replace("RTYPE", &T::rtype())
    .replace("CTYPE", &T::ctype())
    .replace("IS_REAL", &T::is_real().to_string())
    .replace("CVECTOR", &T::cvector(device))
    .to_string()
}
  
pub trait Sigmoid {
  type Output;
  fn sigmoid(self) -> Self::Output;
}

impl<'w, 'a, T: Element> Sigmoid for &'a Tensor<'w, T> {
  type Output = Tensor<'w, T>;
  fn sigmoid(self) -> Tensor<'w, T> {
    debug_assert!(T::is_real());
    let out = self.workspace().tensor(self.dims.clone(), None);
    let name = format!("sigmoid_{}", T::rtype());
    let kernel = ocl::Kernel::builder()
      .program(self.workspace().program())
      .name(name)
      .queue(self.workspace().queue().clone())
      .global_work_size(self.len())
      .arg(out.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    out
  }
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








