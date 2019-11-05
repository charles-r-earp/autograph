use super::{init::Initializer, optim::Optimizer};
use std::{rc::Rc, cell::RefCell, sync::{Arc, Mutex}, fmt, fmt::Debug};
use ndarray as nd;

pub struct Tape {
  backward_ops: RefCell<Vec<Box<dyn Fn()>>>
}

impl Tape {
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

impl Debug for Tape {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Tape")
  }
}

#[derive(Debug, Clone)]
pub struct Var<T> {
  tape: Rc<Tape>,
  value: Rc<nd::ArrayD<T>>,
  grad: Option<Rc<RefCell<nd::ArrayD<T>>>>
}

impl<T> Var<T> {
  pub fn new<D: nd::Dimension>(tape: &Rc<Tape>, value: nd::Array<T, D>, req_grad: bool) -> Self
    where T: num_traits::Zero + Clone {
    let value = value.into_dyn();
    let grad = if req_grad {
      Some(Rc::new(RefCell::new(nd::ArrayD::zeros(value.shape()))))
    }
    else { None };
    Self{
      tape: Rc::clone(tape),
      value: Rc::new(value.into_dyn()),
      grad
    }
  }
  pub fn tape(&self) -> &Rc<Tape> { &self.tape }
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
    self.tape.exec_backward();
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

#[derive(Default, Debug)]
pub struct Param<T> {
  value: nd::ArcArray<T, nd::IxDyn>,
  grad: Option<Arc<Mutex<nd::ArrayD<T>>>>,
  pub initializer: Option<Box<dyn Initializer<T>>>,
  pub optimizer: Option<Box<dyn Optimizer<T>>>
}

impl<T> Param<T> {
  pub fn value(&self) -> &nd::ArcArray<T, nd::IxDyn> { &self.value }
  pub fn view(&self) -> nd::ArrayViewD<T> { self.value.view() }
  pub fn view_mut(&mut self) -> nd::ArrayViewMutD<T>
    where T: Clone { self.value.view_mut() }
  pub fn grad(&self) -> Option<&Arc<Mutex<nd::ArrayD<T>>>> { 
    self.grad.as_ref()
  }
  pub fn req_grad(&self) -> bool { self.grad.is_some() }
  pub fn zero_grad(&mut self)
    where T: num_traits::Zero + Clone {
    self.grad = Some(Arc::new(Mutex::new(nd::Array::zeros(self.value().dim()))));
  }
  pub fn none_grad(&mut self) { self.grad = None }
  pub fn initialize(&mut self, shape: impl nd::ShapeBuilder<Dim=nd::IxDyn>)
    where T: Copy + num_traits::Zero {
    self.value = unsafe { nd::ArcArray::uninitialized(shape) };
    if let Some(ref initializer) = self.initializer {
      initializer.fill(&mut self.value.view_mut());
    }
    else {
      self.value.fill(T::zero());
    }
  }
  pub fn set_optimizer(&mut self, optimizer: impl Optimizer<T> + 'static) {
    self.optimizer.replace(Box::new(optimizer));
  }
  pub fn step(&mut self, lr: T) 
    where T: 'static + num_traits::Float {
    if let Some(grad) = self.grad.take() {
      let grad = grad.lock().unwrap();
      if let Some(ref mut optimizer) = self.optimizer {
        optimizer.step(&mut self.value.view_mut(), &grad.view(), lr);
      }
      else {
        self.value.scaled_add(-lr, &grad);
      } 
    }
  }
}
