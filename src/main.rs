use autograph as ag;
use ndarray as nd;

use std::time::Instant;

fn main() {
  use ag::functional::Conv;
  let x = nd::Array::<f32, _>::ones([64, 64, 28, 28]);
  let mut w = ag::autograd::Param::default();
  w.initializer.replace(Box::new(ag::init::HeNormal));
  w.initialize(&[64, 64, 3, 3][..]);
  //println!("{:?}", w.value());
  let now = Instant::now();
  let y = x.conv(&w, None, &ag::functional::ConvArgs::default());
  println!("{:.0?}", now.elapsed());
}
