use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .platform(ocl::Platform::first().unwrap())
    .devices(0)
    .build()
    .unwrap();
  let src = ag::source(&context);
  //println!("{:?}", &src);
  let ws = ag::Workspace::new(context, src); 
  let x = ws.tensor::<f32>(vec![10, 2], Some(vec![1.; 20]));
  let w = ws.tensor::<f32>(vec![2, 1], Some(vec![1.; 2]));
  let b = ws.tensor::<f32>(vec![1], Some(vec![1.]));
  let t = ws.tensor::<f32>(vec![10], Some(vec![1.; 10]));
  use ag::{Matmul, Stack, Sigmoid, Square, Sum};
  let mut y = (&x.matmul(&w) + &b.stack(x.dims()[0])).sigmoid();
  let mut loss = (&y - &t).sqr().sum();  
  y.read();
  loss.read();
  println!("{:?} {:?}", &y.data(), &loss.data());
}
