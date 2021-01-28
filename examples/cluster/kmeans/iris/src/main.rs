use autograph::{
    Result,
    backend::Device,
    tensor::Tensor,
    dataset::{Iris, Dataset},
    cluster::kmeans::KMeans,
    ndarray::{ArrayView1, ArrayView2},
};
#[cfg(feature = "plotters")]
use plotters::prelude::*;

#[cfg(feature = "plotters")]
fn plot(x: &ArrayView2<f32>, y: &ArrayView1<u32>, y_pred: &ArrayView1<u32>) -> Result<()> {
    let (width, height) = (1024, 760);
    let root = BitMapBackend::new("plot.png", (width, height))
        .into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.titled("Sepal Length x Petal Length x Petal Width",  ("sans-serif", 50).into_font())?;

    let (left, right) = root.split_horizontally(width / 2);
    let mut data_chart = ChartBuilder::on(&left)
        .caption("Data", ("sans-serif", 50).into_font())
        .margin(50)
        .build_cartesian_3d(0f32..8f32, 0f32..7f32, 0f32..3f32)?;
    data_chart.configure_axes()
        .draw()?;
    let colors = [
        &RED,
        &GREEN,
        &BLUE
    ];
    data_chart.draw_series(
        x.outer_iter()
            .zip(y.iter())
            .map(|(x, y)| {
                Circle::new((x[0], x[2], x[3]), 2, colors[*y as usize])
            })
    )?;
    let mut model_chart = ChartBuilder::on(&right)
        .caption("Model", ("sans-serif", 50).into_font())
        .margin(50)
        .build_cartesian_3d(0f32..8f32, 0f32..7f32, 0f32..3f32)?;
    model_chart.configure_axes()
        .draw()?;
    model_chart.draw_series(
        x.outer_iter()
            .zip(y_pred.iter())
            .map(|(x, y)| {
                Circle::new((x[0], x[2], x[3]), 2, colors[*y as usize])
            })
    )?;
    root.present()?;
    println!("plot saved to {}", "plot.png");
    Ok(())
}

fn main() -> Result<()> {
    smol::block_on(async {

        let ref xy_dataset = Iris::new()?;
        let ref x_dataset = xy_dataset.clone().unsupervised();

        let num_samples = 150;
        let (x_array, y_array) = xy_dataset.sample(0..num_samples).unwrap().await?;
        let y_array = y_array.map(|x| *x as u32);
        let device = Device::new_gpu(0).unwrap()?;
        let mut model = KMeans::new(&device, 3)?;
        model.init_random(x_dataset)?;
        let x = Tensor::from_array(&device, x_array.view())?;
        model.train_epoch(std::iter::once(Ok(x.view())))?;
        let y_pred = model.classify(&x.view())?
            .to_array()?
            .await?;
        let classes = [
            "setosa",
            "versicolor",
            "virginica"
        ];
        for (x, (y, y_pred)) in x_array.outer_iter().zip(y_array.iter().zip(y_pred.iter())) {
            println!("dimensions: {:?} class: {} model: {}", x.as_slice().unwrap(), classes[*y as usize], y_pred);
        }
        #[cfg(feature = "plotters")]
        plot(&x_array.view(), &y_array.view(), &y_pred.view())?;
        Ok(())
    })
}
