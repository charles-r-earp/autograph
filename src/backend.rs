use super::{Result, ArcRwLock, Element, Device, Buffer, Gradient, Tensor, Graph, Variable, Parameter};
use std::{thread_local, cell::UnsafeCell, sync::{Once, Arc, Weak}};
use rand::{rngs::SmallRng, SeedableRng}; 

#[doc(hidden)]
pub mod native;
pub use native::Cpu;

thread_local! {
  static THREAD_RNG: UnsafeCell<Option<SmallRng>> = UnsafeCell::new(None);
}

pub fn thread_rng() -> &'static mut SmallRng { 
  THREAD_RNG.with(|t| {
    let mut rng = unsafe { &mut *t.get() };
    if rng.is_none() {
      rng.replace(SmallRng::seed_from_u64(0));
    }
    rng.as_mut().unwrap()
  })
}

static CPU_INIT: Once = Once::new();
static mut CPU: Option<Cpu> = None; 

impl Cpu {
  pub(super) fn get() -> &'static Self {
    CPU_INIT.call_once(|| unsafe {
      CPU.replace(Cpu::new());
    });
    unsafe { CPU.as_ref().unwrap() }
  } 
}

unsafe impl Send for Cpu {}
unsafe impl Sync for Cpu {}

#[doc(hidden)]
#[proxy_enum::proxy(BackwardOp)]
pub mod proxy {
  use super::{Result, DenseBackward};
  pub enum BackwardOp {
    Dense(DenseBackward)
  }
  
  impl BackwardOp {
    #[implement]
    pub(crate) fn backward(&self) -> Result<()> {}
  } 
}
pub(super) use proxy::BackwardOp;

pub struct DenseBackward {
  device: Device,
  batch_size: usize,
  input_channels: usize,
  units: usize,
  x: Option<Arc<Buffer<f32>>>,
  dx: Option<ArcRwLock<Buffer<f32>>>,
  w: Option<Arc<Buffer<f32>>>,
  dw: Option<ArcRwLock<Buffer<f32>>>,
  db: Option<ArcRwLock<Buffer<f32>>>,
  dy: ArcRwLock<Buffer<f32>>
}

impl DenseBackward {
  fn backward(&self) -> Result<()> {
    use std::ptr::{null, null_mut};
    let mut dx = self.dx.as_ref().map(|dx| dx.write().unwrap());
    let mut dw = self.dw.as_ref().map(|dw| dw.write().unwrap());
    let mut db = self.db.as_ref().map(|db| db.write().unwrap());
    let dy = self.dy.read().unwrap();
    match &self.device {
      Device::Native(cpu) => unsafe {
        cpu.dense_backward(
          self.batch_size as i64,
          self.input_channels as i64,
          self.units as i64,
          self.x.as_ref().map_or(null(), |x| x.native_ptr()),
          dx.as_mut().map_or(null_mut(), |dx| dx.native_mut_ptr()),
          self.w.as_ref().map_or(null(), |w| w.native_ptr()),
          dw.as_mut().map_or(null_mut(), |dw| dw.native_mut_ptr()),
          db.as_mut().map_or(null_mut(), |db| db.native_mut_ptr()),
          dy.native_ptr()
        );
      }
    }
    Ok(())   
  }
}

pub(super) fn dense(input: &Variable, weight: &Parameter, bias: Option<&Parameter>) -> Result<Variable> {
  use std::ptr::null;
  let device = &input.value.device;
  let graph = Weak::upgrade(&input.graph);
  let batch_size = input.dims()[0];
  let input_channels: usize = input.dims()[1..].iter().product();
  let units = weight.dims()[0];
  let x = &input.value.data;
  let w = &weight.value.data;
  let b = bias.map(|b| &b.value.data);
  let mut y = unsafe { Buffer::uninitialized(&device, batch_size*units)? };
  match device {
    Device::Native(cpu) => unsafe {
      cpu.dense_forward(
        batch_size as i64, 
        input_channels as i64,
        units as i64,
        x.native_ptr(), 
        w.native_ptr(), 
        b.map_or(null(), |b| b.native_ptr()), 
        y.native_mut_ptr()
      ); 
    }
  }
  let output = Variable::new(graph.as_ref(), unsafe { Tensor::from_dims_buffer(&device, [batch_size, units], y) });
  if let Some(graph) = graph {
    let (x, dw, db) = if let Some(dw) = &weight.grad {
      (Some(x.clone()), Some(dw.data.clone()), bias.map(|b| b.grad.as_ref().unwrap().data.clone()))
    } else { (None, None, None) };
    let (dx, w) = if let Some(dx) = &input.grad {
      (Some(dx.data.clone()), Some(w.clone()))
    } else { (None, None) };
    let dy = output.grad.as_ref().unwrap().data.clone();
    graph.backward_op(BackwardOp::Dense(DenseBackward {
      device: device.clone(),
      batch_size,
      input_channels,
      units,
      x,
      dx,
      w,
      dw,
      db,
      dy
    }));
  }
  Ok(output)
} 


