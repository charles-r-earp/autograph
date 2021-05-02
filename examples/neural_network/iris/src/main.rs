use autograph::{
    backend::Device,
    dataset::iris,
    learn::{Fit, FitOptions, Predict},
    neural_network::{ClassificationTrainer, Dense, Sgd},
    tensor::Tensor,
    ndarray::{Array2, Array1, ArrayView1, ArrayView2},
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

fn regroup(x: ArrayView2<f32>, y: ArrayView1<u32>) -> (Array2<f32>, Array1<u32>) {
    let x = x.iter().take(50)
        .chain(x.iter().skip(50).take(50))
        .chain(x.iter().skip(100))
        .copied()
        .collect::<Array1<_>>()
        .into_shape([150, 4])
        .unwrap();
    let y = y.iter().take(50)
        .chain(y.iter().skip(50).take(50))
        .chain(y.iter().skip(100))
        .copied()
        .collect();
    (x, y)
}

// TODO: Unfortunately this simple net doesn't model the dataset very well, and depending
// on the initialization, may somewhat converge after a long time but not consistently.
// It is nice to have the same dataset with a different approach. May try a deeper model to see
// if that works better, otherwise may remove this in favor of MNIST as the first NN example. 

// Returning a Result from main allows using the ? operator
fn main() -> Result<()> {
    // Create a device for the first Gpu
    let device = Device::new_gpu(0).expect("No gpu!")?;
    // The iris function is imported from autograph::dataset, and loads the data as a pair of
    // arrays.
    let (x_array, y_array) = iris()?;
    let xy_train = regroup(x_array.view(), y_array.view());
    // Create dense model with weight and bias
    let model = Dense::builder()
        .device(&device)
        .inputs(4)
        .outputs(3)
        .bias(true)
        .build()?;
    // Stochastic Gradient Descent with a learning rate (default is 0.001)
    let optim = Sgd::builder().learning_rate(0.001).build();
    // Construct a trainer for Classification. This implements Fit as well as Preduct.
    let mut trainer = ClassificationTrainer::from_network_optimizer(model, optim);
    // Fit the model to the dataset. Note that &Array implements Dataset which loads the data
    // into Tensors.
    // The last argument to fit is a callback which takes a reference to the model and the
    // stats for each epoch and returns whether to continue training.
    trainer.fit(
        &device,
        &xy_train,
        FitOptions::default().train_batch_size(64),
        move |_trainer, stats| {
            println!("epoch: {} train_loss: {:.5}", stats.get_epoch(), stats.get_train_loss());
            Ok(stats.get_epoch() < 10)
        }
    )?;
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
