use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .platform(ocl::Platform::first().unwrap())
    .devices(0)
    .build()
    .unwrap();
  let src = ag::source(&context);
  let ws = ag::Workspace::new(context, src); 
  let x = ws.tensor::<f32>(vec![2, 2], Some(vec![1., 2., 3., 4.]));
  use ag::{Transpose};
  let mut y = x.transpose(); 
  y.read();
  println!("{:?}", &y);
}
