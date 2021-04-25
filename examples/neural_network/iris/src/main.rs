#[cfg(feature = "plotters")]
use autograph::ndarray::{ArrayView1, ArrayView2};
use autograph::{
    backend::Device,
    dataset::iris,
    learn::{Fit, FitOptions, Predict},
    neural_network::{ClassificationTrainer, Dense, Sgd},
    tensor::Tensor,
    Result,
};
#[cfg(feature = "plotters")]
use plotters::prelude::*;

// Utility function used for rendering a scatter plot.
#[cfg(feature = "plotters")]
fn plot(x: &ArrayView2<f32>, y: &ArrayView1<u32>, y_pred: &ArrayView1<u32>) -> Result<()> {
    let (width, height) = (1024, 760);
    let fpath = std::path::PathBuf::from(".")
        .canonicalize()?
        .join("plot.png");
    let root = BitMapBackend::new(&fpath, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.titled(
        "Sepal Length x Petal Length x Petal Width",
        ("sans-serif", 50).into_font(),
    )?;
    let (left, right) = root.split_horizontally(width / 2);
    let mut data_chart = ChartBuilder::on(&left)
        .caption("Data", ("sans-serif", 50).into_font())
        .margin(50)
        .build_cartesian_3d(0f32..8f32, 0f32..7f32, 0f32..3f32)?;
    data_chart.configure_axes().draw()?;
    let colors = [&RED, &GREEN, &BLUE];
    data_chart.draw_series(
        x.outer_iter()
            .zip(y.iter())
            .map(|(x, y)| Circle::new((x[0], x[2], x[3]), 2, colors[*y as usize])),
    )?;
    let mut model_chart = ChartBuilder::on(&right)
        .caption("Model", ("sans-serif", 50).into_font())
        .margin(50)
        .build_cartesian_3d(0f32..8f32, 0f32..7f32, 0f32..3f32)?;
    model_chart.configure_axes().draw()?;
    model_chart.draw_series(
        x.outer_iter()
            .zip(y_pred.iter())
            .map(|(x, y)| Circle::new((x[0], x[2], x[3]), 2, colors[*y as usize])),
    )?;
    root.present()?;
    println!("plot saved to {:?}", fpath);
    Ok(())
}

// Returning a Result from main allows using the ? operator
fn main() -> Result<()> {
    // Create a device for the first Gpu
    let device = Device::new_gpu(0).expect("No gpu!")?;
    // The iris function is imported from autograph::dataset, and loads the data as a pair of
    // arrays.
    let xy_arrays = iris()?;
    // Create dense model with weight and bias
    let model = Dense::builder()
        .device(&device)
        .inputs(4)
        .outputs(3)
        .bias(true)
        .build()?;
    // Stochastic Gradient Descent with a learning rate
    let optim = Sgd::builder().learning_rate(0.01).build();
    // Construct a trainer for Classification. This implements Fit as well as Preduct.
    let mut trainer = ClassificationTrainer::from_network_optimizer(model, optim);
    // Fit the model to the dataset. Note that &Array implements Dataset which loads the data
    // into Tensors.
    // The last argument to fit is a callback which takes a reference to the model and the
    // stats for each epoch and returns whether to continue training. Returning false ends
    // training after 1 iteration.
    trainer.fit(
        &device,
        &xy_arrays,
        FitOptions::default().train_batch_size(64),
        |_trainer, _stats| Ok(false),
    )?;
    let (x_array, y_array) = xy_arrays;
    // Get the predicted classes from the model.
    let y_pred = smol::block_on(
        trainer
            .predict(Tensor::from_array(&device, x_array.view())?)?
            .to_array()?,
    )?;
    let classes = ["setosa", "versicolor", "virginica"];
    for (x, (y, y_pred)) in x_array.outer_iter().zip(y_array.iter().zip(y_pred.iter())) {
        println!(
            "dimensions: {:?} class: {} model: {}",
            x.as_slice().unwrap(),
            classes[*y as usize],
            y_pred
        );
    }
    if !cfg!(feature = "plotters") {
        println!("Feature plotters not enabled, plot not generated. Try running with:\n\tcargo run --features plotters");
    }
    #[cfg(feature = "plotters")]
    plot(&x_array.view(), &y_array.view(), &y_pred.view())?;
    Ok(())
}
