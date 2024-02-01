use autograph::krnl::{device::Device, scalar::ScalarType};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use neural_network_benches::autograph_backend;
#[cfg(feature = "tch")]
use neural_network_benches::tch_backend;
use num_format::{Locale, ToFormattedString};
use std::str::FromStr;

pub fn criterion_benchmark(c: &mut Criterion) {
    let device_index = if cfg!(feature = "device") {
        let krnl_device = std::env::var("KRNL_DEVICE");
        println!("KRNL_DEVICE = {krnl_device:?}");
        let device_index = if let Ok(krnl_device) = krnl_device.as_ref() {
            usize::from_str(krnl_device).unwrap()
        } else {
            0
        };
        println!("testing device {device_index}");
        device_index
    } else {
        0
    };

    #[cfg_attr(not(feature = "cuda"), allow(unused))]
    let cuda_device_index = if cfg!(feature = "cuda") {
        let cuda_device = std::env::var("CUDA_DEVICE");
        println!("CUDA_DEVICE = {cuda_device:?}");
        let cuda_device_index = if let Ok(cuda_device) = cuda_device.as_ref() {
            usize::from_str(cuda_device).unwrap()
        } else {
            0
        };
        println!("testing cuda device {cuda_device_index}");
        cuda_device_index
    } else {
        0
    };

    {
        // training
        let train_batch_size = 100;
        let mut g = c.benchmark_group(format!(
            "LeNet5(training, batch_size = {})",
            train_batch_size.to_formatted_string(&Locale::en)
        ));
        {
            let scalar_types = [ScalarType::BF16, ScalarType::F32];
            let devices = if cfg!(feature = "device") {
                vec![
                    Device::host(),
                    Device::builder().index(device_index).build().unwrap(),
                ]
            } else {
                vec![Device::host()]
            };
            for device in devices {
                let device_name = if device.is_device() { "device" } else { "host" };
                for scalar_type in scalar_types {
                    let scalar_name = scalar_type.name();
                    let name = format!("{scalar_name}_{device_name}");
                    let id = BenchmarkId::new("autograph", name);
                    g.bench_function(id, |b| {
                        use autograph_backend::LeNet5Classifier;
                        let mut model = LeNet5Classifier::new(device.clone(), scalar_type)
                            .unwrap()
                            .with_sgd(true);
                        b.iter(|| {
                            model.train(train_batch_size).unwrap();
                        });
                    });
                }
            }
        }
        #[cfg(feature = "tch")]
        {
            use tch::{kind::Kind, Device};

            let kinds = [Kind::BFloat16, Kind::Float];
            let devices = if cfg!(feature = "cuda") {
                vec![Device::Cpu, Device::Cuda(cuda_device_index)]
            } else {
                vec![Device::Cpu]
            };
            for device in devices {
                let device_name = if device.is_cuda() { "device" } else { "host" };
                for kind in kinds {
                    let kind_name = match kind {
                        Kind::BFloat16 => "bf16",
                        Kind::Float => "f32",
                        _ => unreachable!(),
                    };
                    let name = format!("{kind_name}_{device_name}");
                    let id = BenchmarkId::new("tch", name);
                    g.bench_function(id, |b| {
                        use tch_backend::LeNet5Classifier;
                        let mut model = LeNet5Classifier::new(device, kind)
                            .unwrap()
                            .with_sgd(true)
                            .unwrap();
                        b.iter(|| {
                            model.train(train_batch_size).unwrap();
                        });
                    });
                }
            }
        }
    }

    {
        // inference
        let infer_batch_size = 1000;
        let mut g = c.benchmark_group(format!(
            "LeNet5(inference, batch_size = {})",
            infer_batch_size.to_formatted_string(&Locale::en)
        ));

        {
            let scalar_types = [ScalarType::BF16, ScalarType::F32];
            let devices = if cfg!(feature = "device") {
                vec![
                    Device::host(),
                    Device::builder().index(device_index).build().unwrap(),
                ]
            } else {
                vec![Device::host()]
            };
            for device in devices {
                let device_name = if device.is_device() { "device" } else { "host" };
                for scalar_type in scalar_types {
                    let scalar_name = scalar_type.name();
                    let name = format!("{scalar_name}_{device_name}");
                    let id = BenchmarkId::new("autograph", name);
                    g.bench_function(id, |b| {
                        use autograph_backend::LeNet5Classifier;
                        let model = LeNet5Classifier::new(device.clone(), scalar_type).unwrap();
                        b.iter(|| {
                            model.infer(infer_batch_size).unwrap();
                        });
                    });
                }
            }
        }

        #[cfg(feature = "tch")]
        {
            use tch::{kind::Kind, Device};

            let kinds = [Kind::BFloat16, Kind::Float];
            let devices = if cfg!(feature = "cuda") {
                vec![Device::Cpu, Device::Cuda(cuda_device_index)]
            } else {
                vec![Device::Cpu]
            };
            for device in devices {
                let device_name = if device.is_cuda() { "device" } else { "host" };
                for kind in kinds {
                    let kind_name = match kind {
                        Kind::BFloat16 => "bf16",
                        Kind::Float => "f32",
                        _ => unreachable!(),
                    };
                    let name = format!("{kind_name}_{device_name}");
                    let id = BenchmarkId::new("tch", name);
                    g.bench_function(id, |b| {
                        use tch_backend::LeNet5Classifier;
                        let model = LeNet5Classifier::new(device, kind).unwrap();
                        b.iter(|| {
                            model.infer(infer_batch_size).unwrap();
                        });
                    });
                }
            }
        }
    }
    if cfg!(all(feature = "device", feature = "tch")) {
        eprintln!("warning: sig abort in torch on exit when vulkan is used");
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
