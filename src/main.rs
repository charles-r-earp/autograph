use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .platform(ocl::Platform::first().unwrap())
    .devices(0)
    .build()
    .unwrap();
  let src = ag::source(&context);
  let ws = ag::Workspace::new(context, src);
  let graph = ws.graph();
  let mut x = graph.variable::<f32>(ws.tensor(vec![10], Some(vec![1.; 10])), Some(ws.tensor(vec![10], Some(vec![0.; 10]))));
  use ag::Sigmoid;
  let mut y = x.sigmoid();
  y.read();
  println!("{:?}", y.value().data());
  y.backward();
  x.read_grad();
  y.read_grad();
  println!("{:?}", x.grad().unwrap().data());
  println!("{:?}", y.grad().unwrap().data());
}
