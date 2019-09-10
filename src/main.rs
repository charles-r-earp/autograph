use autograph as ag;
use std::iter;
use timeit::*;

fn main() {
  let platforms = ocl::Platform::list()
    .into_iter()
    .take(1)
    .map(|p| (p, ocl::enums::DeviceSpecifier::All));
  println!("{:?}", platforms.clone().map(|(p, _)| p.name().unwrap()).collect::<Vec<_>>());
  let graph = ag::Graph::new(platforms, ag::source());
  let n = 4352;
  let x = ag::Tensor::new([n], vec![1f32; n]);
  let x = graph.variable(&x, false);
  timeit!({
    let y = &x * &x;
  });
  let x = vec![1f32; n];
  timeit!({
    let y = x.iter()
      .map(|&x| x * x)
      .collect::<Vec<_>>();
  });
}
