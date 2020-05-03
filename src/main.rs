#![allow(warnings)]
use autograph::{backend::{Vertex, cpu::Cpu}};

fn main() {
  let device = Cpu::new();
  let x = Vertex::from_dims_vec(&device, [1, 1], vec![2.]).unwrap();
  let y = x.mm(&x).unwrap();
  println!("{:?}", y.to_vec().unwrap());
}
 
