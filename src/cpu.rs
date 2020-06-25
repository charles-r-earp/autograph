use super::{
  Num, Unsigned, 
  Buffer,
  DataRef, DataMut, 
  Transpose, 
  Conv2dArgs, Pool2dArgs,
  TensorBase, 
  Tensor2, 
  TensorView1, TensorView4,
  TensorViewMut1, TensorViewMut4
};
use std::{sync::{Arc, Mutex}, borrow::Cow, fmt::{self, Debug}};
use ndarray::{Dimension, Ix0, Ix1, Ix2, Ix4};
use num_traits::{ToPrimitive, Bounded};
use cpp::*;

cpp!({
  #include <dnnl.hpp>
  #include <cassert>
  #include <utility>
  
  using dnnl_dim = dnnl::memory::dim;
  using dnnl_dt = dnnl::memory::data_type;
  using dnnl_tag = dnnl::memory::format_tag;
  using argmap = std::unordered_map<int, dnnl::memory>;
});

#[derive(Clone)]
pub struct CpuBuffer<T: Num> {
  data: Vec<T>
}

impl<T: Num> CpuBuffer<T> {
  pub(super) unsafe fn uninitialized(len: usize) -> Self {
    let mut data = Vec::with_capacity(len);
    data.set_len(len);
    Self{data}
  }
  pub(super) fn from_vec<'a>(vec: impl Into<Cow<'a, [T]>>) -> Self {
    let data = vec.into()
      .into_owned();
    Self{data}
  }
  pub(super) fn len(&self) -> usize {
    self.data.len()
  }
  pub(super) fn fill(&mut self, elem: T) {
    self.data.iter_mut()
      .for_each(|mut x| *x = elem); 
  }
  pub(super) fn copy_from_slice<'a>(&mut self, slice: impl Into<Cow<'a, [T]>>) {
    match slice.into() {
      Cow::Owned(vec) => {
        assert_eq!(vec.len(), self.data.len());
        self.data = vec;
      },
      Cow::Borrowed(slice) => self.data.copy_from_slice(slice)
    }
  } 
  pub(super) fn as_slice(&self) -> &[T] {
    self.data.as_slice()
  }
  pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
    self.data.as_mut_slice()
  }
  pub(super) fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }
  pub(super) fn as_mut_ptr(&mut self) -> *mut T {
    self.data.as_mut_ptr()
  }
}

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
  fn new(engine: &Engine) -> Self {
    let engine_ptr = unsafe { engine as *const Engine };
    cpp!(unsafe [engine_ptr as "const dnnl::engine*"] -> Stream as "dnnl::stream" {
      auto engine = *engine_ptr;
      return dnnl::stream(engine, dnnl::stream::flags::in_order);
    })
  }
} 

pub struct Cpu {
  engine: Engine,
  stream: Mutex<Stream>  
}

impl Cpu {
  pub fn new() -> Arc<Self> {
    let engine = Engine::new();
    let stream = Mutex::new(Stream::new(&engine));
    Arc::new(Self{engine, stream})
  }
}

impl Debug for Cpu {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Cpu")
  }
}

pub(super) fn unsigned_to_f32<T: Unsigned, S1: DataRef<Elem=T>, S2: DataMut<Elem=f32>, D: Dimension>
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D>) {
  let input = &input.as_cpu_slice()
    .unwrap();
  let mut output = output.as_mut_cpu_slice()
    .unwrap();
  let scale = T::max_value()
    .to_f32()
    .unwrap()
    .recip();
  output.iter_mut()
    .zip(input.iter())
    .for_each(|(y, &x)| *y = scale * x.to_f32().unwrap());  
}

pub(super) fn unsigned_to_one_hot_f32<T: Unsigned, S1: DataRef<Elem=T>, S2: DataMut<Elem=f32>>
  (input: &TensorBase<S1, Ix1>, output: &mut TensorBase<S2, Ix2>) {
  let (batch_size, nclasses) = output.dim();
  debug_assert_eq!(batch_size, input.dim());
  let input = &input.as_cpu_slice()
    .unwrap();
  let mut output = output.as_mut_cpu_slice()
    .unwrap();
  output.chunks_exact_mut(nclasses)
    .zip(input.iter())
    .for_each(|(y, &x)| y[x.to_usize().unwrap()] = 1.);  
}

pub(super) fn broadcast<T: Num, D: Dimension, S1: DataRef<Elem=T>, S2: DataMut<Elem=T>>
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D::Larger>) {
  let input = &input.as_cpu_slice()
    .unwrap();
  output.as_mut_cpu_slice()
    .unwrap()
    .chunks_exact_mut(input.len())
    .for_each(|mut output| {
      output.copy_from_slice(input);
    });
}

pub(super) fn broadcast_backward<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, D: Dimension>
  (input_grad: &mut TensorBase<S1, D>, output_grad: &TensorBase<S2, D::Larger>) {
  let mut input_grad = &mut input_grad.as_mut_cpu_slice()
    .unwrap();
  output_grad.as_cpu_slice()
    .unwrap()
    .chunks_exact(input_grad.len())
    .for_each(|output_grad| {
      input_grad.iter_mut()
        .zip(output_grad.iter())
        .for_each(|(dx, &dy)| *dx += dy);
    });
} 

pub(super) fn gemm<S1: DataRef<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>>
  (alpha: f32, a: &TensorBase<S1, Ix2>, trans_a: Transpose, b: &TensorBase<S2, Ix2>, trans_b: Transpose, beta: f32, c: &mut TensorBase<S3, Ix2>) {
  let (m, k1) = match trans_a {
    Transpose::No => a.dim(),
    Transpose::Yes => {
      let (k1, m) = a.dim();
      (m, k1)
    }
  };
  let (k2, n) = match trans_b {
    Transpose::No => b.dim(),
    Transpose::Yes => {
      let (n, k2) = b.dim();
      (k2, n)
    }
  };
  debug_assert_eq!(k1, k2);
  debug_assert_eq!((m, n), c.dim());
  let m = m as i64;
  let k = k1 as i64;
  let n = n as i64;
  let a = a.as_cpu_ptr().unwrap();
  let b = b.as_cpu_ptr().unwrap();
  let c = c.as_mut_cpu_ptr().unwrap();
  let t_a = trans_a == Transpose::Yes;
  let t_b = trans_b == Transpose::Yes;
  cpp!(unsafe [t_a as "bool",
               t_b as "bool",
               m as "dnnl::dim",
               k as "dnnl::dim",
               n as "dnnl::dim",
               alpha as "float",
               beta as "float",
               a as "const float*",
               b as "const float*",
               c as "float*"] {
    char trans_a = t_a ? 't' : 'n';
    char trans_b = t_b ? 't' : 'n';
    long int lda = t_a ? m : k;
    long int ldb = t_b ? k : n;
    auto status = dnnl::sgemm(
      trans_a,
      trans_b,
      m,
      n,
      k,
      alpha,
      a,
      lda,
      b,
      ldb,
      beta,
      c,
      n
    );
    assert(status == dnnl::status::success);  
  });            
}



pub(super) fn reduce_sum<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, D: Dimension>
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, Ix0>) {
  let x: &[f32] = &input.as_cpu_slice()
    .unwrap();
  let y: &mut [f32] = &mut output.as_mut_cpu_slice()
    .unwrap();
  *y.first_mut()
    .unwrap() = x.iter().sum::<f32>();
}

pub(super) fn relu<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, D: Dimension>
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D>) {
  let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream };
  let n = input.len() as i64;
  let x = input.as_cpu_ptr().unwrap();
  let y = output.as_mut_cpu_ptr().unwrap();
  
  cpp!(unsafe [engine_ptr as "const dnnl::engine*",
               stream_ptr as "dnnl::stream*",
               n as "dnnl::dim",
               x as "const float*",
               y as "float*"] {
    auto engine = *engine_ptr;
    auto stream = *stream_ptr;
    
    auto x_desc = dnnl::memory::desc({n}, dnnl_dt::f32, dnnl_tag::a);
    auto x_mem = dnnl::memory(x_desc, engine, (float*) x);
    auto y_desc = x_desc;
    auto y_mem = dnnl::memory(y_desc, engine, y);
    
    auto relu_d = dnnl::eltwise_forward::desc(
      dnnl::prop_kind::forward_inference,
      dnnl::algorithm::eltwise_relu,
      x_desc,
      0.0f,
      0.0f
    );
    dnnl::primitive_attr attr;
    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
      relu_d,
      attr,
      engine
    );
    auto relu = dnnl::eltwise_forward(relu_pd);
    argmap args;
    args.insert({DNNL_ARG_SRC, x_mem});
    args.insert({DNNL_ARG_DST, y_mem});
    
    relu.execute(stream, args);
    
    stream.wait();
  });
}

pub(super) fn relu_backward<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, S3: DataRef<Elem=f32>, D: Dimension>
  (input: &TensorBase<S1, D>, input_grad: &mut TensorBase<S2, D>, output_grad: &TensorBase<S3, D>) {
  let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream };
  let n = input.len() as i64;
  let x = input.as_cpu_ptr().unwrap();
  let dx = input_grad.as_mut_cpu_ptr().unwrap();
  let dy = output_grad.as_cpu_ptr().unwrap();
  
  cpp!(unsafe [engine_ptr as "const dnnl::engine*",
               stream_ptr as "dnnl::stream*",
               n as "dnnl::dim",
               x as "const float*",
               dx as "float*",
               dy as "const float*"] {
    auto engine = *engine_ptr;
    auto stream = *stream_ptr;
    
    auto x_desc = dnnl::memory::desc({n}, dnnl_dt::f32, dnnl_tag::a);
    auto x_mem = dnnl::memory(x_desc, engine, (float*) x);
    auto dx_desc = x_desc;
    auto dx_mem = dnnl::memory(dx_desc, engine, dx);
    auto dy_desc = x_desc;
    auto dy_mem = dnnl::memory(dy_desc, engine, (float*) dy);
    
    auto relu_d = dnnl::eltwise_forward::desc(
      dnnl::prop_kind::forward_inference,
      dnnl::algorithm::eltwise_relu,
      x_desc,
      0.0f,
      0.0f
    );
    dnnl::primitive_attr attr;
    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
      relu_d,
      attr,
      engine
    );
    
    auto relu_bw_d = dnnl::eltwise_backward::desc(
      dnnl::algorithm::eltwise_relu,
      dx_desc,
      x_desc,
      0.0f,
      0.0f
    );
    dnnl::primitive_attr bw_attr;
    auto relu_bw_pd = dnnl::eltwise_backward::primitive_desc(
      relu_bw_d,
      bw_attr,
      engine,
      relu_pd
    );
    auto relu_bw = dnnl::eltwise_backward(relu_bw_pd);
    argmap args;
    args.insert({DNNL_ARG_SRC, x_mem});
    args.insert({DNNL_ARG_DIFF_SRC, dx_mem});
    args.insert({DNNL_ARG_DIFF_DST, dy_mem});
    
    relu_bw.execute(stream, args);
    
    stream.wait();
  });
}

pub(super) fn scaled_add<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, D: Dimension>
  (lhs: &mut TensorBase<S1, D>, alpha: f32, rhs: &TensorBase<S2, D>) {
  lhs.as_mut_cpu_slice()
    .unwrap()
    .iter_mut()
    .zip(rhs.as_cpu_slice().unwrap().iter())
    .for_each(|(a, &b)| *a += alpha*b);  
} 

pub(super) fn cross_entropy<S1: DataRef<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>>
  (input: &TensorBase<S1, Ix2>, target: &TensorBase<S2, Ix2>, output: &mut TensorBase<S3, Ix2>) {
  let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream };
  let (batch_size, nclasses) = input.dim();
  let m = batch_size as i64;
  let n = nclasses as i64;
  let x = input.as_cpu_ptr().unwrap();
  let t = target.as_cpu_ptr().unwrap();
  let y = output.as_mut_cpu_ptr().unwrap();
  
  cpp!(unsafe [engine_ptr as "const dnnl::engine*",
               stream_ptr as "dnnl::stream*",
               m as "dnnl::dim",
               n as "dnnl::dim",
               x as "const float*",
               t as "const float*",
               y as "float*"] {
    
    auto engine = *engine_ptr;
    auto stream = *stream_ptr;
    
    auto x_md = dnnl::memory::desc({m, n}, dnnl_dt::f32, dnnl_tag::ab);
    auto x_mem = dnnl::memory(x_md, engine, (float*) x);
    auto t_md = dnnl::memory::desc({m, n}, dnnl_dt::f32, dnnl_tag::ab);
    auto t_mem = dnnl::memory(t_md, engine, (float*) t);
    auto y_md = dnnl::memory::desc({m, n}, dnnl_dt::f32, dnnl_tag::ab);
    auto y_mem = dnnl::memory(y_md, engine, y);   
    
    {
      int axis = 1;
      auto logsoftmax_d = dnnl::logsoftmax_forward::desc(
        dnnl::prop_kind::forward_inference,
        x_md,
        axis
      ); 
      auto logsoftmax_pd = dnnl::logsoftmax_forward::primitive_desc(logsoftmax_d, engine);
      auto logsoftmax = dnnl::logsoftmax_forward(logsoftmax_pd);
      argmap args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_DST, y_mem});
      
      logsoftmax.execute(stream, args);
    }
         
    {
      auto mul_d = dnnl::binary::desc(dnnl::algorithm::binary_mul, y_md, t_md, y_md);
      dnnl::post_ops mul_ops;
      mul_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, -1.0f, 0.0f);
      dnnl::primitive_attr mul_attr;
      mul_attr.set_post_ops(mul_ops);
      auto mul_pd = dnnl::binary::primitive_desc(mul_d, mul_attr, engine);
      auto mul = dnnl::binary(mul_pd);
      argmap args;
      
      args.insert({DNNL_ARG_SRC_0, y_mem});
      args.insert({DNNL_ARG_SRC_1, t_mem});
      args.insert({DNNL_ARG_DST, y_mem});
      
      mul.execute(stream, args);
    }
    
    stream.wait();
  });
}

pub(super) fn cross_entropy_backward<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, S3: DataRef<Elem=f32>, S4: DataRef<Elem=f32>>
  (input: &TensorBase<S1, Ix2>, input_grad: &mut TensorBase<S2, Ix2>,
   target: &TensorBase<S3, Ix2>, 
   output_grad: &TensorBase<S4, Ix0>) {
  let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream };
  let (batch_size, nclasses) = input.dim();
  let m = batch_size as i64;
  let n = nclasses as i64;
  let x = input.as_cpu_ptr().unwrap();
  let dx = input_grad.as_mut_cpu_ptr().unwrap();
  let t = target.as_cpu_ptr().unwrap();
  let dy: f32 = unsafe { *output_grad.as_cpu_ptr().unwrap() };
  
  cpp!(unsafe [engine_ptr as "const dnnl::engine*",
               stream_ptr as "dnnl::stream*",
               m as "dnnl::dim",
               n as "dnnl::dim",
               x as "const float*", dx as "float*",
               t as "const float*",
               dy as "float"] {
    
    auto engine = *engine_ptr;
    auto stream = *stream_ptr;
    
    auto x_md = dnnl::memory::desc({m, n}, dnnl_dt::f32, dnnl_tag::ab);
    auto x_mem = dnnl::memory(x_md, engine, (float*) x);
    auto dx_md = dnnl::memory::desc({m, n}, dnnl_dt::f32, dnnl_tag::ab);
    auto dx_mem = dnnl::memory(dx_md, engine, dx); 
    auto t_md = dnnl::memory::desc({m, n}, dnnl_dt::f32, dnnl_tag::ab);
    auto t_mem = dnnl::memory(t_md, engine, (float*) t);
    
    {
      std::vector<float> scales{dy, -dy};
      std::vector<dnnl::memory::desc> srcs{x_md, t_md};
      auto sum_pd = dnnl::sum::primitive_desc(scales, srcs, engine);
      auto sum = dnnl::sum(sum_pd);
      argmap args;
      args.insert({DNNL_ARG_MULTIPLE_SRC, x_mem});
      args.insert({DNNL_ARG_MULTIPLE_SRC+1, t_mem});
      args.insert({DNNL_ARG_DST, dx_mem});
      
      sum.execute(stream, args);
    } 
    
    stream.wait();
  });
}

pub(super) fn conv2d<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>>
  (input: &TensorBase<S1, Ix4>, weight: &TensorView4<f32>, bias: Option<&TensorView1<f32>>, args: &Conv2dArgs, output: &mut TensorBase<S2, Ix4>) {
  let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream }; 
  let (batch_size, inputs, ih, iw) = input.dim();
  let (outputs, _, kh, kw) = weight.dim();
  let (_, _, ow, oh) = output.dim();
  
  let bs = batch_size as i64;
  let i = inputs as i64;
  let o = outputs as i64;
  let ih = ih as i64;
  let iw = iw as i64;
  let oh = oh as i64;
  let ow = ow as i64;
  let kh = kh as i64;
  let kw = kw as i64;
  let [sh, sw] = args.strides;
  let sh = sh as i64;
  let sw = sw as i64;
  let [ph, pw] = args.padding;
  let ph = ph as i64;
  let pw = pw as i64;
  let x = input.as_cpu_ptr().unwrap();
  let w = weight.as_cpu_ptr().unwrap();
  let y = output.as_mut_cpu_ptr().unwrap();
  
  if let Some(bias) = &bias {
    let b = bias.as_cpu_ptr().unwrap();
    
    cpp!(unsafe [engine_ptr as "const dnnl::engine*",
                stream_ptr as "dnnl::stream*",
                bs as "dnnl::dim",
                i as "dnnl::dim",
                o as "dnnl::dim",
                ih as "dnnl::dim",
                iw as "dnnl::dim",
                oh as "dnnl::dim",
                ow as "dnnl::dim",
                kh as "dnnl::dim",
                kw as "dnnl::dim",
                sh as "dnnl::dim",
                sw as "dnnl::dim",
                ph as "dnnl::dim",
                pw as "dnnl::dim",
                x as "const float*",
                w as "const float*",
                b as "const float*",
                y as "float*"] {
      auto engine = *engine_ptr;
      auto stream = *stream_ptr;
      
      auto x_md = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
      auto x_mem = dnnl::memory(x_md, engine, (float*) x);
      auto w_md = dnnl::memory::desc({o, i, kh, kw}, dnnl_dt::f32, dnnl_tag::oihw);
      auto w_mem = dnnl::memory(w_md, engine, (float*) w);
      auto b_md = dnnl::memory::desc({o}, dnnl_dt::f32, dnnl_tag::a);
      auto b_mem = dnnl::memory(b_md, engine, (float*) b);
      auto y_md = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
      auto y_mem = dnnl::memory(y_md, engine, y);
      std::vector<long int> strides{sh, sw};
      std::vector<long int> pad_l{ph, pw};
      std::vector<long int> pad_r{ph, pw};
      
      auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto,
        x_md, w_md, b_md, y_md,
        strides, pad_l, pad_r
      );
      dnnl::primitive_attr attr;
      auto conv_pd = dnnl::convolution_forward::primitive_desc(
        conv_d, attr, engine
      );
      auto conv = dnnl::convolution_forward(conv_pd);
      argmap args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_WEIGHTS, w_mem});
      args.insert({DNNL_ARG_BIAS, b_mem});
      args.insert({DNNL_ARG_DST, y_mem});
      
      conv.execute(stream, args);
      stream.wait();
    });
  }
  else {
    cpp!(unsafe [engine_ptr as "const dnnl::engine*",
                stream_ptr as "dnnl::stream*",
                bs as "dnnl::dim",
                i as "dnnl::dim",
                o as "dnnl::dim",
                ih as "dnnl::dim",
                iw as "dnnl::dim",
                oh as "dnnl::dim",
                ow as "dnnl::dim",
                kh as "dnnl::dim",
                kw as "dnnl::dim",
                sh as "dnnl::dim",
                sw as "dnnl::dim",
                ph as "dnnl::dim",
                pw as "dnnl::dim",
                x as "const float*",
                w as "const float*",
                y as "float*"] {
      auto engine = *engine_ptr;
      auto stream = *stream_ptr;
      
      auto x_md = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
      auto x_mem = dnnl::memory(x_md, engine, (float*) x);
      auto w_md = dnnl::memory::desc({o, i, kh, kw}, dnnl_dt::f32, dnnl_tag::oihw);
      auto w_mem = dnnl::memory(w_md, engine, (float*) w);
      auto y_md = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
      auto y_mem = dnnl::memory(y_md, engine, y);
      std::vector<long int> strides{sh, sw};
      std::vector<long int> pad_l{ph, pw};
      std::vector<long int> pad_r{ph, pw};
      
      auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto,
        x_md, w_md, y_md,
        strides, pad_l, pad_r
      );
      dnnl::primitive_attr attr;
      auto conv_pd = dnnl::convolution_forward::primitive_desc(
        conv_d, attr, engine
      );
      auto conv = dnnl::convolution_forward(conv_pd);
      argmap args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_WEIGHTS, w_mem});
      args.insert({DNNL_ARG_DST, y_mem});
      
      conv.execute(stream, args);
      stream.wait();
    });
  }
}

pub(super) fn conv2d_backward_input<S1: DataMut<Elem=f32>>
  (input_grad: &mut TensorBase<S1, Ix4>, weight: &TensorView4<f32>, args: &Conv2dArgs, output_grad: &TensorView4<f32>) { 
  let cpu = weight.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream }; 
  let (batch_size, inputs, ih, iw) = input_grad.dim();
  let (outputs, _, kh, kw) = weight.dim();
  let (_, _, ow, oh) = output_grad.dim();
  
  let bs = batch_size as i64;
  let i = inputs as i64;
  let o = outputs as i64;
  let ih = ih as i64;
  let iw = iw as i64;
  let oh = oh as i64;
  let ow = ow as i64;
  let kh = kh as i64;
  let kw = kw as i64;
  let [sh, sw] = args.strides;
  let sh = sh as i64;
  let sw = sw as i64;
  let [ph, pw] = args.padding;
  let ph = ph as i64;
  let pw = pw as i64;
  let dx = input_grad.as_mut_cpu_ptr().unwrap();
  let w = weight.as_cpu_ptr().unwrap();
  let dy = output_grad.as_cpu_ptr().unwrap();
  
  cpp!(unsafe [engine_ptr as "const dnnl::engine*",
               stream_ptr as "dnnl::stream*",
               bs as "dnnl::dim",
               i as "dnnl::dim",
               o as "dnnl::dim",
               ih as "dnnl::dim",
               iw as "dnnl::dim",
               oh as "dnnl::dim",
               ow as "dnnl::dim",
               kh as "dnnl::dim",
               kw as "dnnl::dim",
               sh as "dnnl::dim",
               sw as "dnnl::dim",
               ph as "dnnl::dim",
               pw as "dnnl::dim",
               dx as "float*",
               w as "const float*",
               dy as "const float*"] {
    auto engine = *engine_ptr;
    auto stream = *stream_ptr;
    
    auto dx_md = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
    auto dx_mem = dnnl::memory(dx_md, engine, dx);
    auto w_md = dnnl::memory::desc({o, i, kh, kw}, dnnl_dt::f32, dnnl_tag::oihw);
    auto w_mem = dnnl::memory(w_md, engine, (float*) w);
    auto dy_md = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
    auto dy_mem = dnnl::memory(dy_md, engine, (float*) dy);
    std::vector<long int> strides{sh, sw};
    std::vector<long int> pad_l{ph, pw};
    std::vector<long int> pad_r{ph, pw};
    
    auto conv_d = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_training,
      dnnl::algorithm::convolution_auto,
      dx_md, w_md, dy_md,
      strides, pad_l, pad_r
    );
    dnnl::primitive_attr attr;
    auto conv_pd = dnnl::convolution_forward::primitive_desc(
      conv_d, attr, engine
    );
    
    auto conv_bw_d = dnnl::convolution_backward_data::desc(
      dnnl::algorithm::convolution_auto,
      dx_md, w_md, dy_md,
      strides, pad_l, pad_r
    );
    dnnl::primitive_attr bw_attr;
    auto conv_bw_pd = dnnl::convolution_backward_data::primitive_desc(
      conv_bw_d, attr, engine, conv_pd
    );
    auto conv_bw = dnnl::convolution_backward_data(conv_bw_pd);
    argmap args;
    args.insert({DNNL_ARG_DIFF_SRC, dx_mem});
    args.insert({DNNL_ARG_WEIGHTS, w_mem});
    args.insert({DNNL_ARG_DIFF_DST, dy_mem});
    
    conv_bw.execute(stream, args);
    stream.wait();
  });
}

pub(super) fn conv2d_backward_weight_bias<S1: DataRef<Elem=f32>>
  (input: &TensorBase<S1, Ix4>, weight_grad: &mut TensorViewMut4<f32>, bias_grad: Option<&mut TensorViewMut1<f32>>, args: &Conv2dArgs, output_grad: &TensorView4<f32>) {
  let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream }; 
  let (batch_size, inputs, ih, iw) = input.dim();
  let (outputs, _, kh, kw) = weight_grad.dim();
  let (_, _, ow, oh) = output_grad.dim();
  
  let bs = batch_size as i64;
  let i = inputs as i64;
  let o = outputs as i64;
  let ih = ih as i64;
  let iw = iw as i64;
  let oh = oh as i64;
  let ow = ow as i64;
  let kh = kh as i64;
  let kw = kw as i64;
  let [sh, sw] = args.strides;
  let sh = sh as i64;
  let sw = sw as i64;
  let [ph, pw] = args.padding;
  let ph = ph as i64;
  let pw = pw as i64;
  let x = input.as_cpu_ptr().unwrap();
  let dw = weight_grad.as_mut_cpu_ptr().unwrap();
  let dy = output_grad.as_cpu_ptr().unwrap();
  
  if let Some(bias_grad) = bias_grad {
    debug_assert_eq!(bias_grad.dim(), outputs);
    let db = bias_grad.as_mut_cpu_ptr().unwrap();
    
    cpp!(unsafe [engine_ptr as "const dnnl::engine*",
                 stream_ptr as "dnnl::stream*",
                 bs as "dnnl::dim",
                 i as "dnnl::dim",
                 o as "dnnl::dim",
                 ih as "dnnl::dim",
                 iw as "dnnl::dim",
                 oh as "dnnl::dim",
                 ow as "dnnl::dim",
                 kh as "dnnl::dim",
                 kw as "dnnl::dim",
                 sh as "dnnl::dim",
                 sw as "dnnl::dim",
                 ph as "dnnl::dim",
                 pw as "dnnl::dim",
                 x as "const float*",
                 dw as "float*",
                 db as "float*",
                 dy as "const float*"] {
      auto engine = *engine_ptr;
      auto stream = *stream_ptr;
      
      auto x_md = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
      auto x_mem = dnnl::memory(x_md, engine, (float*) x);
      auto dw_md = dnnl::memory::desc({o, i, kh, kw}, dnnl_dt::f32, dnnl_tag::oihw);
      auto dw_mem = dnnl::memory(dw_md, engine, dw);
      auto db_md = dnnl::memory::desc({o}, dnnl_dt::f32, dnnl_tag::a);
      auto db_mem = dnnl::memory(db_md, engine, db);
      auto dy_md = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
      auto dy_mem = dnnl::memory(dy_md, engine, (float*) dy);
      std::vector<long int> strides{sh, sw};
      std::vector<long int> pad_l{ph, pw};
      std::vector<long int> pad_r{ph, pw};
      
      auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto,
        x_md, dw_md, db_md, dy_md,
        strides, pad_l, pad_r
      );
      dnnl::primitive_attr attr;
      auto conv_pd = dnnl::convolution_forward::primitive_desc(
        conv_d, attr, engine
      );
      
      auto conv_bw_d = dnnl::convolution_backward_weights::desc(
        dnnl::algorithm::convolution_auto,
        x_md, dw_md, db_md, dy_md,
        strides, pad_l, pad_r
      );
      dnnl::primitive_attr bw_attr;
      auto conv_bw_pd = dnnl::convolution_backward_weights::primitive_desc(
        conv_bw_d, attr, engine, conv_pd
      );
      auto conv_bw = dnnl::convolution_backward_weights(conv_bw_pd);
      argmap args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_DIFF_WEIGHTS, dw_mem});
      args.insert({DNNL_ARG_DIFF_BIAS, db_mem});
      args.insert({DNNL_ARG_DIFF_DST, dy_mem});
      
      conv_bw.execute(stream, args);
      stream.wait();
    });
  }
  else {
    cpp!(unsafe [engine_ptr as "const dnnl::engine*",
                 stream_ptr as "dnnl::stream*",
                 bs as "dnnl::dim",
                 i as "dnnl::dim",
                 o as "dnnl::dim",
                 ih as "dnnl::dim",
                 iw as "dnnl::dim",
                 oh as "dnnl::dim",
                 ow as "dnnl::dim",
                 kh as "dnnl::dim",
                 kw as "dnnl::dim",
                 sh as "dnnl::dim",
                 sw as "dnnl::dim",
                 ph as "dnnl::dim",
                 pw as "dnnl::dim",
                 x as "const float*",
                 dw as "float*",
                 dy as "const float*"] {
      auto engine = *engine_ptr;
      auto stream = *stream_ptr;
      
      auto x_md = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
      auto x_mem = dnnl::memory(x_md, engine, (float*) x);
      auto dw_md = dnnl::memory::desc({o, i, kh, kw}, dnnl_dt::f32, dnnl_tag::oihw);
      auto dw_mem = dnnl::memory(dw_md, engine, dw);
      auto dy_md = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
      auto dy_mem = dnnl::memory(dy_md, engine, (float*) dy);
      std::vector<long int> strides{sh, sw};
      std::vector<long int> pad_l{ph, pw};
      std::vector<long int> pad_r{ph, pw};
      
      auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto,
        x_md, dw_md, dy_md,
        strides, pad_l, pad_r
      );
      dnnl::primitive_attr attr;
      auto conv_pd = dnnl::convolution_forward::primitive_desc(
        conv_d, attr, engine
      );
      
      auto conv_bw_d = dnnl::convolution_backward_weights::desc(
        dnnl::algorithm::convolution_auto,
        x_md, dw_md, dy_md,
        strides, pad_l, pad_r
      );
      dnnl::primitive_attr bw_attr;
      auto conv_bw_pd = dnnl::convolution_backward_weights::primitive_desc(
        conv_bw_d, attr, engine, conv_pd
      );
      auto conv_bw = dnnl::convolution_backward_weights(conv_bw_pd);
      argmap args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_DIFF_WEIGHTS, dw_mem});
      args.insert({DNNL_ARG_DIFF_DST, dy_mem});
      
      conv_bw.execute(stream, args);
      stream.wait();
    });
  }
}

pub(super) fn max_pool2d_forward<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>>
  (input: &TensorBase<S1, Ix4>, args: &Pool2dArgs, train: bool, output: &mut TensorBase<S2, Ix4>) -> Option<CpuBuffer<u8>> {
   let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream };

  let (batch_size, inputs, ih, iw) = input.dim();
  let (_, outputs, oh, ow) = output.dim();
  
  let bs = batch_size as i64;
  let i = inputs as i64;
  let o = outputs as i64;
  let ih = ih as i64;
  let iw = iw as i64;
  let oh = oh as i64;
  let ow = ow as i64;
  let kh = args.kernel[0] as i64;
  let kw = args.kernel[1] as i64;
  let sh = args.strides[0] as i64;
  let sw = args.strides[1] as i64;
  let ph = args.padding[0] as i64;
  let pw = args.padding[1] as i64;
  let x = input.as_cpu_ptr().unwrap();
  let y = output.as_mut_cpu_ptr().unwrap();
  
  if train {
    let (ws, ws_size) = cpp!(unsafe [engine_ptr as "const dnnl::engine*",
                 stream_ptr as "dnnl::stream*",
                 bs as "dnnl::dim",
                 i as "dnnl::dim",
                 o as "dnnl::dim",
                 ih as "dnnl::dim",
                 iw as "dnnl::dim",
                 oh as "dnnl::dim",
                 ow as "dnnl::dim",
                 kh as "dnnl::dim",
                 kw as "dnnl::dim",
                 sh as "dnnl::dim",
                 sw as "dnnl::dim",
                 ph as "dnnl::dim",
                 pw as "dnnl::dim",
                 x as "const float*",
                 y as "float*"] -> (*mut u8, usize) as "std::pair<unsigned char*, std::size_t>" {
      auto engine = *engine_ptr;
      auto stream = *stream_ptr;
      
      auto x_desc = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
      auto x_mem = dnnl::memory(x_desc, engine, (float*) x);
      auto y_desc = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
      auto y_mem = dnnl::memory(y_desc, engine, y);
      std::vector<long int> strides{sh, sw};
      std::vector<long int> kernel{kh, kw};
      std::vector<long int> pad_l{ph, pw};
      std::vector<long int> pad_r{ph, pw};
      
      auto pool_d = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_training,
        dnnl::algorithm::pooling_max,
        x_desc,
        y_desc,
        strides,
        kernel,
        pad_l,
        pad_r
      );
      dnnl::primitive_attr attr;
      auto pool_pd = dnnl::pooling_forward::primitive_desc(
        pool_d,
        attr,
        engine
      );
      auto ws_desc = pool_pd.workspace_desc();
      std::size_t ws_size = ws_desc.get_size();
      unsigned char* ws = (unsigned char*)malloc(ws_size);
      auto ws_mem = dnnl::memory(ws_desc, engine, ws);
      auto pool = dnnl::pooling_forward(pool_pd);
      argmap args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_DST, y_mem});
      args.insert({DNNL_ARG_WORKSPACE, ws_mem});
      
      pool.execute(stream, args);
      stream.wait();
      
      return std::make_pair(ws, ws_size);
    });
    let ws_vec = unsafe {
      Vec::from_raw_parts(ws, ws_size, ws_size)
    };
    Some(CpuBuffer::from_vec(ws_vec))
  }
  else {
    cpp!(unsafe [engine_ptr as "const dnnl::engine*",
                 stream_ptr as "dnnl::stream*",
                 bs as "dnnl::dim",
                 i as "dnnl::dim",
                 o as "dnnl::dim",
                 ih as "dnnl::dim",
                 iw as "dnnl::dim",
                 oh as "dnnl::dim",
                 ow as "dnnl::dim",
                 kh as "dnnl::dim",
                 kw as "dnnl::dim",
                 sh as "dnnl::dim",
                 sw as "dnnl::dim",
                 ph as "dnnl::dim",
                 pw as "dnnl::dim",
                 x as "const float*",
                 y as "float*"] {
      auto engine = *engine_ptr;
      auto stream = *stream_ptr;
      
      auto x_desc = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
      auto x_mem = dnnl::memory(x_desc, engine, (float*) x);
      auto y_desc = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
      auto y_mem = dnnl::memory(y_desc, engine, y);
      std::vector<long int> strides{sh, sw};
      std::vector<long int> kernel{kh, kw};
      std::vector<long int> pad_l{ph, pw};
      std::vector<long int> pad_r{ph, pw};
      
      auto pool_d = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::pooling_max,
        x_desc,
        y_desc,
        strides,
        kernel,
        pad_l,
        pad_r
      );
      dnnl::primitive_attr attr;
      auto pool_pd = dnnl::pooling_forward::primitive_desc(
        pool_d,
        attr,
        engine
      );
      auto pool = dnnl::pooling_forward(pool_pd);
      argmap args;
      args.insert({DNNL_ARG_SRC, x_mem});
      args.insert({DNNL_ARG_DST, y_mem});
      
      pool.execute(stream, args);
      stream.wait();
    });
    None
  }
}

pub(super) fn max_pool2d_backward<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>, S3: DataRef<Elem=f32>>
  (input: &TensorBase<S1, Ix4>, input_grad: &mut TensorBase<S2, Ix4>, args: &Pool2dArgs, workspace: Option<&Buffer<u8>>, output_grad: &TensorBase<S3, Ix4>) {
  let cpu = input.device()
    .cpu()
    .unwrap();
  let engine_ptr = unsafe { &cpu.engine as *const Engine };
  let mut stream = cpu.stream.lock()
    .unwrap();
  let stream_ptr = unsafe { &mut *stream as *mut Stream };

  let (batch_size, inputs, ih, iw) = input.dim();
  let (_, outputs, oh, ow) = output_grad.dim();
  
  let workspace = workspace.unwrap()
    .cpu()
    .unwrap();
  let ws = workspace.as_ptr();
  let ws_size = workspace.data.len();
  
  let bs = batch_size as i64;
  let i = inputs as i64;
  let o = outputs as i64;
  let ih = ih as i64;
  let iw = iw as i64;
  let oh = oh as i64;
  let ow = ow as i64;
  let kh = args.kernel[0] as i64;
  let kw = args.kernel[1] as i64;
  let sh = args.strides[0] as i64;
  let sw = args.strides[1] as i64;
  let ph = args.padding[0] as i64;
  let pw = args.padding[1] as i64;
  let x = input.as_cpu_ptr().unwrap();
  let dx = input_grad.as_mut_cpu_ptr().unwrap();
  let dy = output_grad.as_cpu_ptr().unwrap();
  
  cpp!(unsafe [engine_ptr as "const dnnl::engine*",
               stream_ptr as "dnnl::stream*",
               bs as "dnnl::dim",
               i as "dnnl::dim",
               o as "dnnl::dim",
               ih as "dnnl::dim",
               iw as "dnnl::dim",
               oh as "dnnl::dim",
               ow as "dnnl::dim",
               kh as "dnnl::dim",
               kw as "dnnl::dim",
               sh as "dnnl::dim",
               sw as "dnnl::dim",
               ph as "dnnl::dim",
               pw as "dnnl::dim",
               x as "const float*",
               dx as "float*",
               dy as "const float*",
               ws as "const unsigned char*",
               ws_size as "std::size_t"] {
    auto engine = *engine_ptr;
    auto stream = *stream_ptr;
    
    auto x_desc = dnnl::memory::desc({bs, i, ih, iw}, dnnl_dt::f32, dnnl_tag::nchw);
    auto x_mem = dnnl::memory(x_desc, engine, (float*) x);
    auto dx_desc = x_desc;
    auto dx_mem = dnnl::memory(dx_desc, engine, dx);
    auto dy_desc = dnnl::memory::desc({bs, o, oh, ow}, dnnl_dt::f32, dnnl_tag::nchw);
    auto dy_mem = dnnl::memory(dy_desc, engine, (float*) dy);
    std::vector<long int> strides{sh, sw};
    std::vector<long int> kernel{kh, kw};
    std::vector<long int> pad_l{ph, pw};
    std::vector<long int> pad_r{ph, pw};
    
    auto pool_d = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_training,
      dnnl::algorithm::pooling_max,
      x_desc,
      dy_desc,
      strides,
      kernel,
      pad_l,
      pad_r
    );
    dnnl::primitive_attr attr;
    auto pool_pd = dnnl::pooling_forward::primitive_desc(
      pool_d,
      attr,
      engine
    );
    
    auto ws_desc = pool_pd.workspace_desc();
    assert(ws_desc.get_size() == ws_size);
    auto ws_mem = dnnl::memory(ws_desc, engine, (void*) ws);
    
    auto pool_bw_d = dnnl::pooling_backward::desc(
      dnnl::algorithm::pooling_max,
      dx_desc,
      dy_desc,
      strides,
      kernel,
      pad_l,
      pad_r
    );
    dnnl::primitive_attr bw_attr;
    auto pool_bw_pd = dnnl::pooling_backward::primitive_desc(
      pool_bw_d,
      bw_attr,
      engine,
      pool_pd
    );
    
    auto pool_bw = dnnl::pooling_backward(pool_bw_pd);
    argmap args;
    args.insert({DNNL_ARG_SRC, x_mem});
    args.insert({DNNL_ARG_DIFF_SRC, dx_mem});
    args.insert({DNNL_ARG_DIFF_DST, dy_mem});
    args.insert({DNNL_ARG_WORKSPACE, ws_mem});
    
    pool_bw.execute(stream, args);
    stream.wait();
  });
}
