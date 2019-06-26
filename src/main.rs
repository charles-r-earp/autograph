use autograph as ag;

fn main() {
  let context = ocl::Context::builder()
    .build()
    .unwrap();
  let program = ocl::Program::builder()
    .src(itertools::join(&[ag::Ones::<f32>::src(), ag::Add::<f32>::src()], ""))
    .build(&context)
    .unwrap();
  let backend = ag::OpenCL::new(context, program);
  let mut graph = ag::Graph::new(&backend);
  let x: ag::Tensor<f32, ag::Ix1> = ag::Ones::op(&mut graph, ag::Shape::rmaj([10]));
  let mut y = ag::Add::op(&mut graph, &x, &x); 
  graph.exec();
  y.read(&graph);
  println!("{:?}", y);
} 
