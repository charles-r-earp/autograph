use super::{Element, Stack, Vertex};
use std::{rc::Rc, cell::{UnsafeCell, RefCell, RefMut}};
use cpp::*;

cpp!({
  #include <dnnl.hpp>
  
  struct device {
    dnnl::engine engine;
    dnnl::stream stream;
    device() : engine(dnnl::engine::kind::cpu, 0), stream(engine, dnnl::stream::flags::in_order) {}
  }; 
});

#[doc(hidden)]  
cpp_class!(pub unsafe struct Device as "device");


pub struct Cpu {
  device: UnsafeCell<Device>,
  stack: RefCell<Stack>
}

impl Cpu {
  pub fn new() -> Rc<super::Device> {
    Rc::new(super::Device::Cpu(Self {
      device: UnsafeCell::new(Device::default()),
      stack: RefCell::new(Stack::default())
    }))
  }
  pub unsafe fn device(&self) -> *mut Device { self.device.get() }
  pub fn stack_mut(&self) -> RefMut<Stack> { self.stack.borrow_mut() } 
  pub fn sync(&self) {
    let device = unsafe { self.device.get() };
    cpp!(unsafe [device as "device*"] {
      device->stream.wait();
    });
    self.stack.borrow_mut()
      .clear();
  }
  pub fn gemm(&self, alpha: f32, a: &Vertex<f32>, b: &Vertex<f32>, beta: f32, c: &Vertex<f32>) {
    let mut stack = self.stack.borrow_mut();
    stack.push(a.clone());
    stack.push(b.clone());
    stack.push(c.clone());
    let trans_a = a.is_t();
    let trans_b = b.is_t();
    let m = a.dims()[0] as i64;
    let n = b.dims()[1] as i64;
    let k = a.dims()[0] as i64;
    let a = unsafe { a.buffer().cpu().unwrap().as_mut_ptr() };
    let b = unsafe { b.buffer().cpu().unwrap().as_mut_ptr() };
    let c = unsafe { c.buffer().cpu().unwrap().as_mut_ptr() };
    let device = unsafe { self.device.get() };
    cpp!(unsafe [device as "device*", 
                 alpha as "float", 
                 beta as "float",
                 trans_a as "bool", 
                 trans_b as "bool", 
                 m as "long int",
                 n as "long int",
                 k as "long int",
                 a as "float*",
                 b as "float*",
                 c as "float*"] {
      using dt = dnnl::memory::data_type;
      using tag = dnnl::memory::format_tag;
      auto a_md = dnnl::memory::desc({m, k}, dt::f32, trans_a ? tag::ba : tag::ab);
      auto b_md = dnnl::memory::desc({k, n}, dt::f32, trans_b ? tag::ba : tag::ab);
      auto c_md = dnnl::memory::desc({m, n}, dt::f32, tag::ab);
      auto a_mem = dnnl::memory(a_md, device->engine, a);
      auto b_mem = dnnl::memory(b_md, device->engine, b);
      auto c_mem = dnnl::memory(c_md, device->engine, c);
      auto matmul_d = dnnl::matmul::desc(a_md, b_md, c_md);
      dnnl::post_ops matmul_ops;
      matmul_ops.append_eltwise(1.0, dnnl::algorithm::eltwise_linear, alpha, beta);
      dnnl::primitive_attr matmul_attr;
      matmul_attr.set_post_ops(matmul_ops);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, matmul_attr, device->engine);
      auto matmul_prim = dnnl::matmul(matmul_pd);
      matmul_prim.execute(device->stream, {{DNNL_ARG_SRC, a_mem}, {DNNL_ARG_WEIGHTS, b_mem}, {DNNL_ARG_DST, c_mem}});
    });
  }
} 

