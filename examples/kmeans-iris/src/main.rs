use autograph::{
    result::Result,
    device::Device,
    tensor::{Tensor, CowTensor},
    float_tensor::FloatArcTensor,
    dataset::iris::Iris,
    learn::{Train, Predict, kmeans::{KMeans, KMeansTrainer}},
};
use ndarray::{Array, s, ArrayView1, ArrayView2};
use std::{
    iter::once,
    convert::TryFrom,
};

fn plot(x: &ArrayView2<f32>, y: &ArrayView1<u32>, y_pred: &ArrayView1<u32>, centroids: &ArrayView2<f32>) -> Result<()> {
    use plotters::prelude::*;
    let (width, height) = (1024, 760);
    let fpath = std::path::PathBuf::from(".")
        .canonicalize()?
        .join("plot.png");
    let root = BitMapBackend::new(&fpath, (width, height))
        .into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(0f32..7f32, 0f32..3f32)?;
    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Petal Length")
        .y_desc("Petal Width")
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;
    let colors = [
        &RED,
        &GREEN,
        &BLUE
    ];
    chart.draw_series(
        x.outer_iter()
            .zip(y.iter())
            .map(|(x, y)| {
                Circle::new((x[0], x[1]), 1, colors[*y as usize].filled())
            })
    )?
    .label("Data")
    .legend(|(x, y)| Circle::new((x, y), 1, BLACK.filled()));
    chart.draw_series(
        centroids.outer_iter()
            .enumerate()
            .map(|(i, x)| {
                Cross::new((x[0], x[1]), 6, colors[i])
            })
    )?
    .label("Centroids")
    .legend(|(x, y)| Cross::new((x, y), 6, &BLACK));
    chart.draw_series(
        x.outer_iter()
            .zip(y_pred.iter())
            .map(|(x, y)| {
                Circle::new((x[0], x[1]), 4, colors[*y as usize])
            })
    )?
    .label("Prediction")
    .legend(|(x, y)| Circle::new((x, y), 4, &BLACK));
    chart.configure_series_labels().border_style(&BLACK).draw()?;
    root.present()?;
    println!("Plot saved to {:?}.", fpath);
    Ok(())
}

// tokio::main is used for simplicity, autograph does not depend on a runtime.
#[tokio::main]
async fn main() -> Result<()> {
    // Create the device.
    let device = Device::new()?;
    // Create the dataset.
    let iris = Iris::new();
    // The flower dimensions are the inputs to the model.
    let x_array = iris.dimensions();
    // Select only Petal Length + Petal Height
    // These are the primary dimensions and it makes plotting easier.
    let x_array = x_array.slice(&s![.., 2..]);
    // For now initialize with the first item of each type.
    let indices = [0, 50, 100];
    let init_centroids = x_array.outer_iter()
        .into_iter()
        .enumerate()
        .filter_map(|(i, x)| if indices.contains(&i) {
            Some(x)
        } else {
            None
        })
        .flatten()
        .copied()
        .collect::<Array<_, _>>()
        .into_shape([3, 2])?;
    // Load the centroids into the device.
    let centroids = Tensor::from(init_centroids)
        .into_device(device.clone())
        .await?;
    // Create the KMeans model.
    let kmeans = KMeans::from_centroids(centroids.into());
    // For small datasets, we can load the entire dataset into the device.
    // For larger datasets, the data can be streamed as an iterator.
    let x = CowTensor::from(x_array.view())
        .into_device(device)
        // Note that despite the await this will resolve immediately.
        // Host -> Device transfers are batched with other operations
        // asynchronously on the device thread.
        .await?;
    // Construct a trainer.
    let mut trainer = KMeansTrainer::from(kmeans);
    // Train the model (1 epoch).
    trainer.train(once(Ok(x.view().into())))?;
    // Get the model back.
    let kmeans = KMeans::from(trainer);
    // Get the trained centroids.
    // For multiple reads, batch them by getting the futures first.
    let centroids_fut = kmeans.centroids()
        // The centroids are in a FloatArcTensor, which can either be f32 or bf16.
        // This will convert to f32 if necessary.
        .cast_to::<f32>()?
        .read();
    // Get the predicted classes.
    let pred = kmeans.predict(&x.view().into())?
        .read()
    // Here we wait on all previous operations, including centroids_fut.
        .await?;
    // This will resolve immediately.
    let centroids = centroids_fut.await?;
    // Get the flower classes from the dataset.
    let classes = iris.classes().map(|c| *c as u32);
    // Plot the results to "plot.png".
    plot(&x_array.view(), &classes.view(), &pred.as_array(), &centroids.as_array())?;
    Ok(())
}
