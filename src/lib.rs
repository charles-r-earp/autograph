use std::{rc::{Rc, Weak}, cell::RefCell, sync::{Arc, Mutex}, fmt, fmt::Debug};
use ndarray as nd;

pub mod functional;
pub mod init;
use init::Initializer;
pub mod optim;
use optim::Optimizer;
pub mod layer;
pub mod iter_ext;
pub mod datasets;

pub struct Graph {
  backward_ops: RefCell<Vec<Box<dyn Fn()>>>
}

impl Graph {
  pub fn new() -> Self { 
    Self{backward_ops: RefCell::new(Vec::new())}
  }
  pub fn backward_op(&self, op: impl Fn() + 'static) {
    self.backward_ops.borrow_mut()
      .push(Box::new(op));
  }
  fn exec_backward(&self) {
    self.backward_ops.borrow()
      .iter()
      .rev()
      .for_each(|op| op());
  } 
}

impl Debug for Graph {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Graph")
  }
}

#[derive(Debug, Clone)]
pub struct Var<T> {
  graph: Weak<Graph>,
  value: Rc<nd::ArrayD<T>>,
  grad: Option<Rc<RefCell<nd::ArrayD<T>>>>
}

impl<T> Var<T> {
  pub fn new<D: nd::Dimension>(graph: &Rc<Graph>, value: nd::Array<T, D>, req_grad: bool) -> Self
    where T: num_traits::Zero + Clone {
    let value = value.into_dyn();
    let grad = if req_grad {
      Some(Rc::new(RefCell::new(nd::ArrayD::zeros(value.shape()))))
    }
    else { None };
    Self{
      graph: Rc::downgrade(graph),
      value: Rc::new(value.into_dyn()),
      grad
    }
  }
  pub fn graph(&self) -> &Weak<Graph> { &self.graph }
  pub fn value(&self) -> &Rc<nd::ArrayD<T>> { &self.value }
  pub fn grad(&self) -> Option<&Rc<RefCell<nd::ArrayD<T>>>> { self.grad.as_ref() }
  pub fn req_grad(&self) -> bool { self.grad.is_some() }
  pub fn backward(&self)
    where T: num_traits::One + Clone {
    if let Some(ref grad) = self.grad {
      grad.borrow_mut()
        .iter_mut()
        .for_each(|x| *x = T::one());
    }
    self.graph.upgrade()
      .unwrap()
      .exec_backward();
  }
  pub fn into_array(self) -> nd::ArrayD<T>
    where T: Clone {
    if Rc::strong_count(self.value()) == 1 { 
      Rc::try_unwrap(self.value)
        .ok()
        .unwrap()
    }
    else {
      nd::ArrayD::clone(self.value())
    }
  }
}

#[derive(Default, Debug, Clone)]
pub struct Param<T, D: nd::Dimension> {
  value: nd::ArcArray<T, D>,
  grad: Option<Arc<Mutex<nd::Array<T, D>>>>
}

impl<T, D: nd::Dimension> Param<T, D> {
  pub fn new(value: nd::ArcArray<T, D>) -> Self {
    Self{value, grad: None}
  }
  pub fn initialize(&mut self, init: &impl Initializer<T>)
    where T: Clone {
    init.fill(&mut self.value.view_mut());
  }
  pub fn value(&self) -> &nd::ArcArray<T, D> { &self.value }
  pub fn value_mut(&mut self) -> &mut nd::ArcArray<T, D> { &mut self.value }
  pub fn grad(&self) -> Option<&Arc<Mutex<nd::Array<T, D>>>> { 
    self.grad.as_ref()
  }
  pub fn req_grad(&self) -> bool { self.grad.is_some() }
  fn zero_grad(&mut self)
    where T: num_traits::Zero + Clone {
    self.grad = Some(Arc::new(Mutex::new(nd::Array::zeros(self.value().dim()))));
  }
  fn none_grad(&mut self) { self.grad = None }
  fn step(&mut self, optim: &mut impl Optimizer<T, D>, lr: T) 
    where T: Clone {
    if let Some(grad) = self.grad.take() {
      let grad = grad.lock().unwrap();
      optim.step(&mut self.value.view_mut(), &grad, lr);
    }
  }
}

pub type Param1<T> = Param<T, nd::Ix1>;
pub type Param2<T> = Param<T, nd::Ix2>;
  
