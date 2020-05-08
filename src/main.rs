use autograph::{Device, Tensor, Variable, Dense, Layer, LayerBuilder};

fn main() {
  let cpu = Device::cpu();
  let model = Dense::builder()
    .input(&[1, 28, 28])
    .units(10)
    .bias()
    .build();
  let x = Variable::new(None, Tensor::from_dims_elem(&cpu, [1, 1, 28, 28], 1.).unwrap());
  let y = model.forward(&x).unwrap();
  println!("{:?}", y.value().as_slice());
}
