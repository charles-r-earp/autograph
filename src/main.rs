use autograph as ag;
use std::iter;

fn main() {
  /*let platforms = ocl::Platform::list()
    .into_iter()
    .map(|p| (p, ocl::enums::DeviceSpecifier::All));
  let graph = ag::Graph::new(platforms, ag::source());*/
  let x = ag::Tensor::new([64, 1, 28, 28], vec![1f32; 64*28*28]);
  ag::TensorRef::from(&x).batches(vec![10, 20, 30, 4])
    .for_each(|x| println!("{:?}", x.shape()));
}
