use autograph as ag;
use ndarray as nd;

fn main() {
  use ag::Layer;
  let x: nd::Array2<f32> = nd::arr2(&[
    [-2., 4.],
    [4., 1.],
    [1., 6.],
    [2., 4.],
    [6., 2.]
  ]);
  let y = nd::arr1(&[-1.,-1.,1.,1.,1.]);
  let mut fc = ag::Dense::builder()
    .units(1)
    .use_bias()
    .build();
  let mut optimizer = ag::LearningRate(0.01);
  for epoch in 0 .. 100 {
    let p = fc.forward(x.clone().into_dyn());
    let loss: f32 = p.iter()
      .zip(y.iter())
      .map(|(&p, &y)| if p * y <= 0. { 1. - p * y } else { 0. })
      .sum::<f32>() / 5.;
    let grad = p.iter()
      .zip(y.iter())
      .map(|(&p, &y)| if p * y <= 0. { - y } else { 0. })
      .collect::<nd::Array1<_>>()
      .into_shape(p.dim())
      .unwrap();
    fc.backward(&grad.into_dyn());
    fc.params_mut()
      .iter_mut()
      .for_each(|p| p.step(&mut optimizer));
    println!("{}: {} {:?} {:?}", epoch, loss, p.as_slice().unwrap(), y.as_slice().unwrap());
  }
}
