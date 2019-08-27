use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .platform(ocl::Platform::first().unwrap())
    .devices(0)
    .build()
    .unwrap();
  let ws = ag::Workspace::new(context, ag::source());
  let graph = ag::Graph::new(&ws);
  let opt = ag::LearningRate::new(0.01);
  //let mut w = ag::Parameter::new(Tensor::
  //use ag::{Matmul, Sigmoid};
  
}
