use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .platform(ocl::Platform::first().unwrap())
    .devices(0)
    .build()
    .unwrap();
  let ws = ag::Workspace::new(context, ag::source());
  let graph = ag::Graph::<ag::Forward>::new(&ws);
  let x = ag::Tensor::<f32>::new(&ws, vec![10], Some(vec![1.; 10]));
  
}
