use super::{AutographResult, Element, Unsigned, Activation, Autograd, ForwardMode, BackwardMode};
use std::{rc::Rc, fmt::Debug, cell::{RefCell, Ref, RefMut}};
use rc_cell::RcCell;
use cpp::*;

cpp!({
  #include <dnnl.hpp>
  
  struct dense_op_impl {
    dnnl::matmul inference_prim;
    std::unordered_map<int, dnnl::memory> inference_args;
  };
  
  struct cross_entropy_op_impl {
    dnnl::logsoftmax_forward logsoftmax_forward_prim;
    std::unordered_map<int, dnnl::memory> logsoftmax_forward_args;
    dnnl::binary mul_forward_prim;
    std::unordered_map<int, dnnl::memory> mul_forward_args;
    dnnl::pooling_forward avg_prim;
    std::unordered_map<int, dnnl::memory> avg_args;
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
  pub fn as_slice(&self) -> Ref<Vec<T>> { self.data.borrow() }
  pub fn as_mut_slice(&self) -> RefMut<Vec<T>> { self.data.borrow_mut() }
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

cpp_class!(pub unsafe struct CrossEntropyOpImpl as "cross_entropy_op_impl");

pub struct CrossEntropyOp<U: Unsigned> {
  cpu: Rc<Cpu>,
  batch_size: usize,
  nclasses: usize,
  input: DualBuffer,
  target: Buffer<U>,
  target_f32: Buffer<f32>,
  tmp_output: Buffer<f32>,
  output: Buffer<f32>,
  cross_entropy_op_impl: CrossEntropyOpImpl
}

impl<U: Unsigned> CrossEntropyOp<U> {
  pub(super) fn new(cpu: &Rc<Cpu>,
                    batch_size: usize, nclasses: usize,
                    input: DualBuffer,
                    target: Buffer<U>,
                    output: Buffer<f32>) -> Self {
    let cpu = cpu.clone();
    let target_f32 = Buffer::zeros(&cpu, batch_size*nclasses);
    let tmp_output = Buffer::zeros(&cpu, batch_size*nclasses);
    let cross_entropy_op_impl = {
      let engine = unsafe { cpu.engine() as *const Engine };
      let batch_size = batch_size as isize;
      let nclasses = nclasses as isize;
      let (x, dx) = input.dual_mut_ptr();
      let t = target_f32.as_mut_ptr();
      let tmp_y = tmp_output.as_mut_ptr();
      let y = tmp_output.as_mut_ptr();
      cpp!(unsafe [engine as "const dnnl::engine*",
                   batch_size as "long int", nclasses as "long int",
                   x as "float*"/*, dx as "float*"*/,
                   t as "float*",
                   tmp_y as "float*",
                   y as "float*"] -> CrossEntropyOpImpl as "cross_entropy_op_impl" {
        using tag = dnnl::memory::format_tag;
        using dt = dnnl::memory::data_type;
        auto x_md = dnnl::memory::desc({batch_size, nclasses}, dt::f32, tag::ab);
        auto t_md = dnnl::memory::desc({batch_size, nclasses}, dt::f32, tag::ab);
        auto tmp_y_md = dnnl::memory::desc({batch_size, nclasses}, dt::f32, tag::ab);
        auto avg_md = dnnl::memory::desc({1, 1, batch_size*nclasses}, dt::f32, tag::abc);
        auto y_md = dnnl::memory::desc({1, 1, 1}, dt::f32, tag::abc);
        auto x_mem = dnnl::memory(x_md, *engine, x);
        auto t_mem = dnnl::memory(t_md, *engine, t);
        auto tmp_y_mem = dnnl::memory(tmp_y_md, *engine, tmp_y);
        auto avg_mem = dnnl::memory(avg_md, *engine, tmp_y);
        auto y_mem = dnnl::memory(y_md, *engine, y);
        const int axis = 1;
        auto logsoftmax_forward_d = dnnl::logsoftmax_forward::desc(dnnl::prop_kind::forward_inference, x_md, axis);
        auto logsoftmax_forward_pd = dnnl::logsoftmax_forward::primitive_desc(logsoftmax_forward_d, *engine);
        auto logsoftmax_forward_prim = dnnl::logsoftmax_forward(logsoftmax_forward_pd);
        std::unordered_map<int, dnnl::memory> logsoftmax_forward_args;
        logsoftmax_forward_args.insert({DNNL_ARG_SRC, x_mem});
        logsoftmax_forward_args.insert({DNNL_ARG_DST, tmp_y_mem});
        auto mul_forward_d = dnnl::binary::desc(dnnl::algorithm::binary_mul, tmp_y_md, t_md, tmp_y_md);
        const float scale = 1.0f;
        const float alpha = -1.0f;
        const float beta = 0.0f;
        dnnl::post_ops mul_forward_ops;
        mul_forward_ops.append_eltwise(scale, dnnl::algorithm::eltwise_linear, alpha, beta);
        dnnl::primitive_attr mul_forward_attr;
        mul_forward_attr.set_post_ops(mul_forward_ops);
        auto mul_forward_pd = dnnl::binary::primitive_desc(mul_forward_d, mul_forward_attr, *engine);
        auto mul_forward_prim = dnnl::binary(mul_forward_pd);
        std::unordered_map<int, dnnl::memory> mul_forward_args;
        mul_forward_args.insert({DNNL_ARG_SRC_0, tmp_y_mem});
        mul_forward_args.insert({DNNL_ARG_SRC_1, t_mem});
        mul_forward_args.insert({DNNL_ARG_DST, tmp_y_mem});
        auto avg_d = dnnl::pooling_forward::desc(
          dnnl::prop_kind::forward_inference, 
          dnnl::algorithm::pooling_avg_include_padding,
          avg_md,
          y_md,
          {1},
          {batch_size*nclasses},
          {0},
          {0}
        );
        auto avg_pd = dnnl::pooling_forward::primitive_desc(avg_d, *engine);
        auto avg_prim = dnnl::pooling_forward(avg_pd);
        std::unordered_map<int, dnnl::memory> avg_args;
        avg_args.insert({DNNL_ARG_SRC, avg_mem});
        avg_args.insert({DNNL_ARG_SRC, y_mem}); 
        return cross_entropy_op_impl{logsoftmax_forward_prim, logsoftmax_forward_args,
                                     mul_forward_prim, mul_forward_args,
                                     avg_prim, avg_args};
      })
    };
    Self{cpu, batch_size, nclasses, input, target, target_f32, tmp_output, output, cross_entropy_op_impl}            
  }
  pub(super) fn forward(&self, mode: ForwardMode) {
    use num_traits::AsPrimitive;
    self.cpu.sync();
    let mut target_f32 = self.target_f32.as_mut_slice();
    target_f32.copy_from_slice(&vec![0.; self.batch_size*self.nclasses]);
    target_f32.chunks_mut(self.nclasses)
      .zip(self.target.as_slice().iter())
      .for_each(|(t_f32, t)| {
        t_f32[t.as_()] = 1.;
      });
    {  
      let mut stream = self.cpu.stream().borrow_mut();
      let mut stream = unsafe { &mut *stream as *mut Stream };
      let op = unsafe { &self.cross_entropy_op_impl as *const CrossEntropyOpImpl };
      cpp!(unsafe [stream as "dnnl::stream*", op as "const cross_entropy_op_impl*"] {
        op->logsoftmax_forward_prim.execute(*stream, op->logsoftmax_forward_args);
        op->mul_forward_prim.execute(*stream, op->mul_forward_args);
        //op->avg_prim.execute(*stream, op->avg_args); bugged (unable to execute)
      });
    }
    // hotfix
    self.cpu.sync();
    self.output.as_mut_slice()[0] = self.tmp_output.as_slice().iter().sum();
  }
  pub(super) fn backward(&self, mode: BackwardMode) {
    unimplemented!()
  }
}



