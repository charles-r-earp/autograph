use super::{Element, Device, Vertex, ops::{VertexOp}};
use std::{rc::Rc, cell::{RefCell, RefMut}, any::Any};
use cpp::*;

cpp!({
  #include <dnnl.hpp>
  
  struct operation {
    dnnl::primitive primitive;
    std::unordered_map<int, dnnl::memory> args;
  };
  
});


cpp_class!(pub unsafe struct Engine as "dnnl::engine");

impl Engine {
  fn new() -> Self {
    cpp!(unsafe [] -> Engine as "dnnl::engine" {
      return dnnl::engine(dnnl::engine::kind::cpu, 0);
    })
  }
}

cpp_class!(pub unsafe struct Stream as "dnnl::stream");

impl Stream {
  fn new(engine: &mut Engine) -> Self {
    let engine = unsafe { engine as *mut Engine };
    cpp!(unsafe [engine as "dnnl::engine*"] -> Stream as "dnnl::stream" {
      return dnnl::stream(*engine, dnnl::stream::flags::in_order);
    })
  }
}

cpp_class!(pub unsafe struct Operation as "operation");

impl Operation {
  fn exec(&mut self, stream: &mut Stream) {
    let mut stream = unsafe { stream as *mut Stream };
    let op = unsafe { self as *mut Operation };
    cpp!(unsafe [stream as "dnnl::stream*", op as "operation*"] {
      op->primitive.execute(*stream, op->args);
    });
  }
}

pub struct Cpu<O: VertexOp> {
  engine: RefCell<Engine>,
  stream: RefCell<Stream>,
  queue: RefCell<Vec<O>>
}

impl Cpu<O> {
  pub fn new() -> Rc<Device> {
    let mut engine = Engine::new();
    let stream = Stream::new(&mut engine);
    Rc::new(Device::Cpu(Self {
      engine: RefCell::new(engine),
      stream: RefCell::new(stream),
      queue: RefCell::new(Vec::new())
    }))
  }
  pub fn engine(&self) -> RefMut<Engine> { self.engine.borrow_mut() }
  pub fn stream(&self) -> RefMut<Stream> { self.stream.borrow_mut() }
  pub fn enq(&self, op: VertexOp) {
    self.queue.borrow_mut()
      .push(op);
  } 
  pub fn sync(&self) {
    let mut stream = self.stream.borrow_mut();
    let queue = self.queue.borrow_mut()
      .iter_mut()
      .for_each(|mut op| op.exec(&self));
    let stream = unsafe { &mut *stream as *mut Stream };
    cpp!(unsafe [stream as "dnnl::stream*"] {
      stream->wait();
    });
  }
}

impl VertexOp<Cpu> for ops::VertexGemm {
  pub fn exec(&self, device: &Cpu) {
    let ops::VertexGemm{alpha, ref a, ref b, beta, ref c};
    let trans_a = a.is_t();
    let trans_b = b.is_t();
    let m = a.dims()[0] as i64;
    let n = b.dims()[1] as i64;
    let k = a.dims()[0] as i64;
    let a = unsafe { a.buffer().cpu().unwrap().as_mut_ptr() };
    let b = unsafe { b.buffer().cpu().unwrap().as_mut_ptr() };
    let c = unsafe { c.buffer().cpu().unwrap().as_mut_ptr() };
    let mut engine = unsafe { &mut *device.engine() as *mut Engine };
    let mut stream = unsafe { &mut *device.stream() as *mut Stream };
    cpp!(unsafe [engine as "dnnl::engine*",
                 stream as "dnnl::stream*", 
                 alpha as "float", 
                 beta as "float",
                 trans_a as "bool", 
                 trans_b as "bool", 
                 m as "long int",
                 n as "long int",
                 k as "long int",
                 a as "float*",
                 b as "float*",
                 c as "float*"] -> Operation as "operation" {
      using dt = dnnl::memory::data_type;
      using tag = dnnl::memory::format_tag;
      auto a_md = dnnl::memory::desc({m, k}, dt::f32, trans_a ? tag::ba : tag::ab);
      auto b_md = dnnl::memory::desc({k, n}, dt::f32, trans_b ? tag::ba : tag::ab);
      auto c_md = dnnl::memory::desc({m, n}, dt::f32, tag::ab);
      auto a_mem = dnnl::memory(a_md, *engine, a);
      auto b_mem = dnnl::memory(b_md, *engine, b);
      auto c_mem = dnnl::memory(c_md, *engine, c);
      auto matmul_d = dnnl::matmul::desc(a_md, b_md, c_md);
      dnnl::post_ops matmul_ops;
      matmul_ops.append_eltwise(1.0, dnnl::algorithm::eltwise_linear, alpha, beta);
      dnnl::primitive_attr matmul_attr;
      matmul_attr.set_post_ops(matmul_ops);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, matmul_attr, *engine);
      auto matmul_prim = dnnl::matmul(matmul_pd);
      matmul_prim.execute(*stream, {{DNNL_ARG_SRC, a_mem}, {DNNL_ARG_WEIGHTS, b_mem}, {DNNL_ARG_DST, c_mem}});
    });
  }
} 

