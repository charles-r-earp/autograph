use super::*;

#[derive(Debug)] 
pub enum Buffer {
  U8(Vec<u8>),
  U16(Vec<u16>),
  U32(Vec<u32>),
  U64(Vec<u64>),
  I8(Vec<i8>),
  I16(Vec<i16>),
  I32(Vec<i32>),
  I64(Vec<i64>),
  F32(Vec<f32>),
  F64(Vec<f64>)
}

use Buffer::*;

pub trait Element: 'static + Default + Copy + num_traits::Zero + num_traits::One + fmt::Debug {
  fn rtype() -> &'static str;
  fn native_buffer(v: Vec<Self>) -> Buffer;
  fn native_ref(buffer: &Buffer) -> &Vec<Self>;
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self>;
}

impl Element for u8 {
  fn rtype() -> &'static str { "u8" }
  fn native_buffer(v: Vec<Self>) -> Buffer { U8(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let U8(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let U8(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for u16 {
  fn rtype() -> &'static str { "u16" }
  fn native_buffer(v: Vec<Self>) -> Buffer { U16(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let U16(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let U16(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for u32 {
  fn rtype() -> &'static str { "u32" }
  fn native_buffer(v: Vec<Self>) -> Buffer { U32(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let U32(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let U32(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for u64 {
  fn rtype() -> &'static str { "u64" }
  fn native_buffer(v: Vec<Self>) -> Buffer { U64(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let U64(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let U64(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for i8 {
  fn rtype() -> &'static str { "i8" }
  fn native_buffer(v: Vec<Self>) -> Buffer { I8(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let I8(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let I8(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for i16 {
  fn rtype() -> &'static str { "i16" }
  fn native_buffer(v: Vec<Self>) -> Buffer { I16(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let I16(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let I16(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for i32 {
  fn rtype() -> &'static str { "i32" }
  fn native_buffer(v: Vec<Self>) -> Buffer { I32(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let I32(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let I32(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for i64 {
  fn rtype() -> &'static str { "i64" }
  fn native_buffer(v: Vec<Self>) -> Buffer { I64(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let I64(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let I64(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for f32 {
  fn rtype() -> &'static str { "f32" }
  fn native_buffer(v: Vec<Self>) -> Buffer { F32(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let F32(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let F32(ref mut v) = buffer { v } else { panic!() } }
}

impl Element for f64 {
  fn rtype() -> &'static str { "f64" }
  fn native_buffer(v: Vec<Self>) -> Buffer { F64(v) }
  fn native_ref(buffer: &Buffer) -> &Vec<Self> { if let F64(ref v) = buffer { v } else { panic!() } }
  fn native_mut(buffer: &mut Buffer) -> &mut Vec<Self> { if let F64(ref mut v) = buffer { v } else { panic!() } }
}

#[derive(Default, Debug)]
pub struct Native;

impl<'c> ContextBase<'c> for Native
  where Self: 'c {
  type Buffer = Buffer;
  type Executor = Executor<'c>;
  type Source = ();
  fn compile(&'c self, sources: impl iter::IntoIterator<Item=Option<Self::Source>>) -> Self::Executor {
    Self::Executor::new(self)
  }
}

impl<'c, T: Element> Context<'c, T> for Native
  where Self: ContextBase<'c> {}
  
impl<'c, T: Element> Tensor<'c, T, Native> {
  pub fn buffer(&self) -> &Vec<T> {
    T::native_ref(self.base().buffer())
  }
  pub fn buffer_mut(&mut self) -> &mut Vec<T> {
    T::native_mut(self.base_mut().buffer_mut())
  }
  pub fn len(&self) -> usize {
    debug_assert_eq!(self.buffer().len(), self.shape().size());
    self.buffer().len()
  }
}

impl<'c, T: Element> ops::Deref for Tensor<'c, T, Native> {
  type Target = [T];
  fn deref(&self) -> &Self::Target { 
    self.buffer().as_slice()
  }
}  

impl<'c, T: Element> ops::DerefMut for Tensor<'c, T, Native> {
  fn deref_mut(&mut self) -> &mut Self::Target { 
    self.buffer_mut().as_mut_slice()
  }
} 
        
#[derive(Debug)]
pub struct Executor<'c> {
  context: &'c Native
}

impl<'c> Executor<'c> {
  fn new(context: &'c Native) -> Self {
    Self{context}
  }
} 

impl<'c, T: Element> FromNative<'c, T, Native> for Tensor<'c, T, Native> {
  fn from_native(native_tensor: Tensor<'static, T, Native>, executor: &Executor<'c>) -> Self {
    Tensor::new(executor.context, *native_tensor.shape(), native_tensor.into_base().into_buffer())
  }
}

impl<'c, T: Element> ToNative<'c, T, Native> for Tensor<'c, T, Native> {
  fn to_native(self, executor: &Executor<'c>) -> Tensor<'static, T, Native> {
    Tensor::new(&Native, *self.shape(), self.into_base().into_buffer())
  }
}

pub unsafe fn uninitialized<T: Element>(shape: Shape) -> Tensor<'static, T, Native> {
  let mut v = Vec::with_capacity(shape.size());
  unsafe { v.set_len(v.capacity()); }
  Tensor::new(&Native, shape, T::native_buffer(v))
}

impl<'c, T: Element> Uninitialized<T> for Executor<'c> {
  type Output = Tensor<'c, T, Native>;
  unsafe fn uninitialized(&self, shape: Shape) -> Self::Output {
    Tensor::from_native(uninitialized(shape), self)
  }
}

pub fn zeros<T: Element>(shape: Shape) -> Tensor<'static, T, Native> {
  Tensor::new(&Native, shape, T::native_buffer(vec![T::zero(); shape.size()]))
}

impl<'c, T: Element> ForwardOp<'c, Native> for ZerosOp<T> {
  fn forward(&self, executor: &Executor<'c>, tensors: &[TensorBase<'c, Native>]) -> TensorBase<'c, Native> {
    Tensor::<T, _>::from_native(zeros::<T>(*self.vertex().shape()), executor).into_base()
  }
} 

pub fn ones<T: Element>(shape: Shape) -> Tensor<'static, T, Native> {
  Tensor::new(&Native, shape, T::native_buffer(vec![T::one(); shape.size()]))
}
  
impl<'c, T: Element> ForwardOp<'c, Native> for OnesOp<T> {
  fn forward(&self, executor: &Executor<'c>, tensors: &[TensorBase<'c, Native>]) -> TensorBase<'c, Native> {
    Tensor::<T, _>::from_native(ones::<T>(*self.vertex().shape()), executor).into_base()
  }
} 
