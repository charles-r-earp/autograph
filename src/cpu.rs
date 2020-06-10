use super::{Num, Unsigned, DataRef, DataMut, Transpose, TensorBase, Tensor2};
use std::{sync::{Arc, Mutex}, borrow::Cow, fmt::{self, Debug}};
use ndarray::{Dimension, Ix0, Ix1, Ix2};
use num_traits::{ToPrimitive, Bounded};
use cpp::*;

cpp!({
  #include <dnnl.hpp>
  #include <cassert>
  
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
  (input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D>) {
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
  (input_grad: &mut TensorBase<S1, D>, output_grad: &TensorBase<S2, D>) {
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
               m as "long int",
               k as "long int",
               n as "long int",
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
               m as "long int",
               n as "long int",
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
               m as "long int",
               n as "long int",
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
