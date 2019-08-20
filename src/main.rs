use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .platform(ocl::Platform::first().unwrap())
    .devices(0)
    .build()
    .unwrap();
  let ws = ag::Workspace::new(context, ag::source());
  let graph = ag::Graph::new(&ws);
  use ag::Sigmoid;
  let mut x = ag::Variable::new(ag::Tensor::new(&ws, vec![1, 2], Some(vec![1f32; 2])), Some(ag::Gradient::new(&graph, vec![1, 2], Some(vec![0f32; 2]))));
  let mut y = x.sigmoid();
  y.backward();
  x.read_grad();
  println!("{:?}", x.grad().unwrap().data());
}
