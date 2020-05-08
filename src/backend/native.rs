use cpp::*;

cpp!({
  #include <dnnl.hpp>
});

cpp_class!(pub unsafe struct Engine as "dnnl::engine");

pub struct Cpu {
  engine: Engine
}

impl Cpu {
  pub fn new() -> Self {
    let engine = cpp!(unsafe [] -> Engine as "dnnl::engine" {
      return dnnl::engine(dnnl::engine::kind::cpu, 0);
    });
    Self{engine}
  }
}

impl Cpu {
  pub unsafe fn dense_forward(&self, m: i64, k: i64, n: i64, x: *const f32, w: *const f32, b: *const f32, y: *mut f32) {
    let engine = &self.engine as *const Engine;
    let x = x as *mut f32;
    let w = w as *mut f32;
    let b = b as *mut f32;
    cpp!([engine as "dnnl::engine*",
          m as "long int",
          k as "long int",
          n as "long int",
          x as "float*",
          w as "float*",
          b as "float*",
          y as "float*"] {
      using dt = dnnl::memory::data_type;
      using tag = dnnl::memory::format_tag;
      auto x_md = dnnl::memory::desc({m, k}, dt::f32, tag::ab);
      auto x_mem = dnnl::memory(x_md, *engine, x);
      auto w_md = dnnl::memory::desc({k, n}, dt::f32, tag::ba);
      auto w_mem = dnnl::memory(w_md, *engine, w);
      auto b_md = b ? 
        dnnl::memory::desc({1, n}, dt::f32, tag::ab)
        : dnnl::memory::desc();
      auto b_mem = b ? 
        dnnl::memory(b_md, *engine, b)
        : dnnl::memory();
      auto y_md = dnnl::memory::desc({m, n}, dt::f32, tag::ab);
      auto y_mem = dnnl::memory(y_md, *engine, y);
      auto matmul_d = b ? 
        dnnl::matmul::desc(x_md, w_md, b_md, y_md)
        : dnnl::matmul::desc(x_md, w_md, y_md);
      const float scale = 1.0f;
      const float alpha = 1.0f;
      const float beta = 0.0f;
      dnnl::post_ops matmul_ops;
      matmul_ops.append_eltwise(scale, dnnl::algorithm::eltwise_linear, alpha, beta);
      dnnl::primitive_attr matmul_attr;
      matmul_attr.set_post_ops(matmul_ops); 
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, matmul_attr, *engine);
      auto matmul_prim = dnnl::matmul(matmul_pd);
      std::unordered_map<int, dnnl::memory> args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_WEIGHTS, w_mem});
      if (b) {
        args.insert({DNNL_ARG_BIAS, b_mem});
      }
      args.insert({DNNL_ARG_DST, y_mem});
      auto stream = dnnl::stream(*engine);
      matmul_prim.execute(stream, args);
      stream.wait();
    });
  }
  pub unsafe fn dense_backward(&self, m: i64, k: i64, n: i64, x: *const f32, dx: *mut f32, w: *const f32, dw: *mut f32, db: *mut f32, dy: *const f32) {
    unimplemented!();
  }
}
