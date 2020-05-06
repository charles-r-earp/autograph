use super::{AutographResult, Element, Activation, Autograd, ForwardMode, BackwardMode};
use std::{rc::Rc, fmt::Debug, cell::RefCell};
use rc_cell::RcCell;
use cpp::*;

cpp!({
  #include <dnnl.hpp>
  
  struct dense_op_impl {
    dnnl::matmul inference_prim;
    std::unordered_map<int, dnnl::memory> inference_args;
  };
  
});

#[derive(Clone)]
pub struct Buffer<T> {
  data: RcCell<Vec<T>>
}

impl<T: Element> Buffer<T> {
  pub fn zeros(cpu: &Cpu, len: usize) -> Self {
    Self{data: RcCell::new(vec![T::zero(); len])}
  }
  pub fn len(&self) -> usize { 
    self.data.borrow().len()
  }
  pub fn write(&self, cpu: &Cpu, slice: &[T]) {
    cpu.sync();
    self.data.borrow_mut()
      .copy_from_slice(slice)
  }
  pub fn read(&self, cpu: &Cpu, slice: &mut [T]) {
    cpu.sync();
    slice.copy_from_slice(self.data.borrow().as_slice());
  }
  pub fn as_mut_ptr(&self) -> *mut T { 
    self.data.borrow_mut()
      .as_mut_ptr()
  }
}

pub struct DualBuffer {
  value: Buffer<f32>,
  grad: Option<Buffer<f32>>
}

impl DualBuffer {
  pub fn new(value: Buffer<f32>, grad: Option<Buffer<f32>>) -> Self {
    Self{value, grad}
  }
  pub fn dual_mut_ptr(&self) -> (*mut f32, *mut f32) {
    use std::ptr::null_mut;
    (self.value.as_mut_ptr(), 
      self.grad.as_ref().map_or(null_mut(), |g| g.as_mut_ptr()))
  } 
}

cpp_class!(pub unsafe struct Engine as "dnnl::engine");

cpp_class!(pub unsafe struct Stream as "dnnl::stream");

pub struct Cpu {
  engine: Engine,
  stream: RefCell<Stream>
}

impl Cpu {
  pub(super) fn new() -> Self {
    let engine = cpp!(unsafe [] -> Engine as "dnnl::engine" {
      return dnnl::engine(dnnl::engine::kind::cpu, 0); 
    });
    let stream = {
      let engine = unsafe { &engine as *const Engine };
      let stream = cpp!(unsafe [engine as "const dnnl::engine*"] -> Stream as "dnnl::stream" {
        return dnnl::stream(*engine, dnnl::stream::flags::in_order);
      });
      RefCell::new(stream)
    };
    Self{engine, stream}
  }
  pub fn engine(&self) -> &Engine { &self.engine }
  pub fn stream(&self) -> &RefCell<Stream> { &self.stream }
  pub fn sync(&self) {
    let mut stream = self.stream.borrow_mut();
    let stream = unsafe { &mut *stream as *mut Stream };
    cpp!(unsafe [stream as "dnnl::stream*"] {
      stream->wait();
    });
  }
}

cpp_class!(pub unsafe struct DenseOpImpl as "dense_op_impl");

pub struct DenseOp {
  cpu: Rc<Cpu>,
  input: DualBuffer,
  weight: DualBuffer,
  bias: Option<DualBuffer>,
  output: DualBuffer,
  dense_op_impl: DenseOpImpl
}

impl DenseOp {
  pub(super) fn new(cpu: &Rc<Cpu>, 
                    mkn: [usize; 3], 
                    input: DualBuffer,
                    weight: DualBuffer,
                    bias: Option<DualBuffer>,
                    act: Option<Activation>,
                    output: DualBuffer) -> Self {
    let cpu = cpu.clone();
    let dense_op_impl = {
      use std::ptr::null_mut;
      let engine = unsafe { cpu.engine() as *const Engine };
      let [m, k, n] = mkn;
      let [m, k, n] = [m as isize, k as isize, n as isize];
      let (x, dx) = input.dual_mut_ptr();                 
      let (w, dw) = weight.dual_mut_ptr();
      let (b, db) = bias.as_ref()
        .map_or((null_mut(), null_mut()), |b| b.dual_mut_ptr());
      if act.is_some() { unimplemented!(); }
      let (y, dy) = output.dual_mut_ptr();
      cpp!(unsafe [engine as "const dnnl::engine*",
                   m as "long int", k as "long int", n as "long int",
                   x as "float*"/*, dx as "float*"*/,
                   w as "float*"/*, dw as "float*"*/,
                   b as "float*"/*, db as "float*"*/,
                   y as "float*"/*, dy as "float*"*/] -> DenseOpImpl as "dense_op_impl" {
        using tag = dnnl::memory::format_tag;
        using dt = dnnl::memory::data_type;
        auto x_md = dnnl::memory::desc({m, k}, dt::f32, tag::ab);
        auto w_md = dnnl::memory::desc({k, n}, dt::f32, tag::ba);
        dnnl::memory::desc b_md;
        if (b) {
          b_md = dnnl::memory::desc({1, n}, dt::f32, tag::ab);
        }
        auto y_md = dnnl::memory::desc({m, n}, dt::f32, tag::ab);
        auto x_mem = dnnl::memory(x_md, *engine, x);
        auto w_mem = dnnl::memory(w_md, *engine, w);
        dnnl::memory b_mem;
        if (b) {
          b_mem = dnnl::memory(b_md, *engine, b);
        }
        auto y_mem = dnnl::memory(y_md, *engine, y);
        auto inference_d = b ?
          dnnl::matmul::desc(x_md, w_md, b_md, y_md) 
          : dnnl::matmul::desc(x_md, w_md, y_md);
        dnnl::post_ops inference_post_ops;
        inference_post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, 0.0f);
        dnnl::primitive_attr inference_attr;
        inference_attr.set_post_ops(inference_post_ops);
        auto inference_pd = dnnl::matmul::primitive_desc(inference_d, inference_attr, *engine);
        auto inference_prim = dnnl::matmul(inference_pd);
        std::unordered_map<int, dnnl::memory> inference_args;
        inference_args.insert({DNNL_ARG_SRC, x_mem});
        inference_args.insert({DNNL_ARG_WEIGHTS, w_mem});
        if (b) {
          inference_args.insert({DNNL_ARG_BIAS, b_mem});
        }
        inference_args.insert({DNNL_ARG_DST, y_mem});     
        return dense_op_impl{inference_prim, inference_args};
      })
    };
    Self{cpu, input, weight, bias, output, dense_op_impl}                
  }
  pub(super) fn forward(&self, mode: ForwardMode) {
    let stream = self.cpu.stream().borrow_mut();
    let stream = unsafe { &*stream as *const Stream };
    let dense_op_impl = unsafe { &self.dense_op_impl as *const DenseOpImpl };
    match mode {
      ForwardMode::Infer => {
        cpp!(unsafe [stream as "const dnnl::stream*", dense_op_impl as "const dense_op_impl*"] {
          dense_op_impl->inference_prim.execute(*stream, dense_op_impl->inference_args);
        });
      },
      ForwardMode::Eval => unimplemented!(),
      ForwardMode::Train => unimplemented!()
    }
  }
  pub(super) fn backward(&self, mode: BackwardMode) {
    unimplemented!()
  }
}

