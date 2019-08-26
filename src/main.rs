use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .platform(ocl::Platform::first().unwrap())
    .devices(0)
    .build()
    .unwrap();
  let ws = ag::Workspace::new(context, ag::source());
  let graph = ag::Graph::new(&ws);
  use ag::BinaryExec;
  let x = ag::Tensor::from_elem(&ws, vec![10], 1f32);
  let dx = ag::Tensor::from_elem(&graph, vec![10], 2f32);
  let mut y = &x + &dx;
  use ag::BackwardExecutor;
  (&graph).backward();
  y.read();
  println!("{:?}", y.data());
}
