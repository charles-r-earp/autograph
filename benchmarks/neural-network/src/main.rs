use anyhow::{bail, Result};
#[cfg(feature = "tch")]
use neural_network_benchmark::tch_trainer::Tch;
use neural_network_benchmark::{
    autograph_trainer::Autograph, DatasetDescriptor, DatasetKind, Library, NetworkDescriptor,
    NetworkKind, TrainerDescriptor, TrainerStats,
};

use argparse::{ArgumentParser, Store, StoreConst, StoreTrue};
use indicatif::{ProgressBar, ProgressStyle};
use plotters::style::{RGBColor, colors::*};

fn main() -> Result<()> {
    let mut dataset_kind = DatasetKind::Mnist;
    let mut network_kind = NetworkKind::Lenet5;
    let mut train_batch_size = 100;
    let mut test_batch_size = 1000;
    let mut epochs = 100;
    let mut autograph = false;
    let mut tch = false;
    let mut all = false;
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Neural Network Benchmark");
        ap.refer(&mut dataset_kind).add_option(
            &["--mnist"],
            StoreConst(DatasetKind::Mnist),
            "MNIST",
        );
        ap.refer(&mut network_kind).add_option(
            &["--lenet5"],
            StoreConst(NetworkKind::Lenet5),
            "LeNet5",
        );
        ap.refer(&mut train_batch_size).add_option(
            &["--train-batch-size"],
            Store,
            "The training batch size.",
        );
        ap.refer(&mut test_batch_size).add_option(
            &["--test-batch-size"],
            Store,
            "The test batch size.",
        );
        ap.refer(&mut epochs).add_option(
            &["--epochs"],
            Store,
            "The number of epochs to train for.",
        );
        ap.refer(&mut autograph)
            .add_option(&["--autograph"], StoreTrue, "Benchmark autograph.");
        ap.refer(&mut tch).add_option(
            &["--tch"],
            StoreTrue,
            "Benchmark tch. Requires --features tch",
        );
        ap.refer(&mut all)
            .add_option(&["--all"], StoreTrue, "Benchmark all libraries.");
        ap.parse_args_or_exit();
    }
    if all {
        autograph = true;
        if cfg!(feature = "tch") {
            tch = true;
        }
    }
    let mut tch_name = "tch";
    if tch {
        if !cfg!(feature = "tch") {
            bail!("Benchmarking tch requires --feature tch");
        }
        #[cfg(feature = "tch")]
        if !Tch::is_gpu_available() {
            tch_name = "tch(cpu)";
        }
    }
    if ![autograph, tch].iter().copied().any(|x| x) {
        bail!(
            "Specify at least one libary (ie --autograph) or benchmark all libraries with --all!"
        );
    }

    let title = format!("dataset = {:?}, network = {:?}, train-batch-size = {}, test-batch-size = {}, epochs = {}", dataset_kind, network_kind, train_batch_size, test_batch_size, epochs);
    println!("Benchmarking ({})", &title);
    for (lib, flag) in [("autograph", autograph), (tch_name, tch)] {
        if flag {
            println!("    - {}", lib);
        }
    }

    let dataset_descriptor = DatasetDescriptor {
        kind: dataset_kind,
        train_batch_size,
        test_batch_size,
    };
    let network_desciptor = NetworkDescriptor { kind: network_kind };
    let trainer_descriptor = TrainerDescriptor {
        dataset: dataset_descriptor,
        network: network_desciptor,
        epochs,
    };

    fn benchmark<L: Library>(name: &str, trainer: &TrainerDescriptor) -> Result<TrainerStats> {
        let style = ProgressStyle::default_bar()
            .template(&format!(
                "{} [{{bar}}] {{pos:>7}}/{{len:7}} [eta: {{eta}}]",
                name,
            ))
            .progress_chars("=> ");
        let bar = ProgressBar::new(trainer.epochs as u64).with_style(style);
        let stats = L::benchmark(trainer, |epoch| bar.set_position(epoch as u64))?;
        bar.finish();
        Ok(stats)
    }

    let mut stats = Vec::new();

    if autograph {
        stats.push(("autograph".to_string(), BLACK, benchmark::<Autograph>("autograph", &trainer_descriptor)?));
    }

    #[cfg(feature = "tch")]
    if tch {
        stats.push((tch_name.to_string(), MAGENTA, benchmark::<Tch>(tch_name, &trainer_descriptor)?));
    }

    fn plot(title: &str, stats: &[(String, RGBColor, TrainerStats)]) -> Result<()> {
        use plotters::prelude::*;

        let time_max = stats
            .iter()
            .map(|(_, _, stats)| stats.total_time.last().copied().unwrap_or(0.))
            .fold(0f32, |a, b| f32::max(a, b));
        let accuracy_min = stats
            .iter()
            .map(|(_, _, stats)| stats.test_accuracy.first().copied().unwrap_or(0.))
            .fold(1f32, |a, b| f32::min(a, b));
        let (width, height) = (1024, 760);
        let fpath = std::path::PathBuf::from(".")
            .canonicalize()?
            .join("plot.png");
        let root = BitMapBackend::new(&fpath, (width, height))
            .into_drawing_area();
        root.fill(&WHITE)?;
        let root = root.titled(title, ("sans-serif", 20))?;
        let mut chart = ChartBuilder::on(&root)
            .margin(5)
            .set_all_label_area_size(50)
            .build_cartesian_2d(0f32..time_max, accuracy_min * 100f32..100f32)?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Time (s)")
            .y_desc("Test Accuracy")
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.1}%", v))
            .draw()?;
        for (name, color, stats) in stats.iter() {
            chart
                .draw_series(LineSeries::new(
                    stats
                        .total_time
                        .iter()
                        .copied()
                        .zip(stats.test_accuracy.iter().copied().map(|a| 100. * a)),
                    &*color,
                ))?
                .label(name.to_string())
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &*color));
            chart.draw_series(
                stats
                    .total_time
                    .iter()
                    .copied()
                    .zip(stats.test_accuracy.iter().copied().map(|a| 100. * a))
                    .map(|(x, y)| Circle::new((x, y), 2, color.filled()))
            )?;
        }
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
        root.present()?;
        println!("Plot saved to {:?}.", fpath);
        Ok(())
    }
    plot(&title, &stats)?;

    Ok(())
}
