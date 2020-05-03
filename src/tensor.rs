use super::{Element, backend::{Device, Backend, Buffer}, vertex::Vertex};
use std::rc::Rc;

pub struct Tensor<T: Element> {
  backend: Rc<Backend>,
  dims: Vec<usize>,
  is_t: bool,
  vertices: Rc<Vec<Vertex>>
}
