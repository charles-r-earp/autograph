use autograph::krnl::{device::Device, scalar::ScalarType};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use neural_network_benches::autograph_backend;
#[cfg(feature = "candle")]
use neural_network_benches::candle_backend;
#[cfg(feature = "tch")]
use neural_network_benches::tch_backend;
use num_format::{Locale, ToFormattedString};
use std::str::FromStr;

fn autograph_devices(index: usize) -> impl IntoIterator<Item = Device> {
    [
        Device::host(),
        #[cfg(feature = "device")]
        Device::builder().index(index).build().unwrap(),
    ]
}

#[cfg(feature = "tch")]
fn tch_devices(
    #[cfg_attr(not(feature = "cuda"), allow(unused))] index: usize,
) -> impl IntoIterator<Item = tch::Device> {
    use tch::Device;

    [
        Device::Cpu,
        #[cfg(feature = "cuda")]
        Device::Cuda(index),
    ]
}

#[cfg(feature = "candle")]
fn candle_devices(
    #[cfg_attr(not(feature = "cuda"), allow(unused))] index: usize,
) -> impl IntoIterator<Item = candle_core::Device> {
    use candle_core::Device;
    #[cfg(feature = "cuda")]
    use candle_core::{backend::BackendDevice, CudaDevice};

    [
        Device::Cpu,
        #[cfg(feature = "cuda")]
        Device::Cuda(CudaDevice::new(index).unwrap()),
    ]
}

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
            for device in autograph_devices(device_index) {
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
            use tch::kind::Kind;

            let kinds = [Kind::BFloat16, Kind::Float];
            for device in tch_devices(cuda_device_index) {
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
        #[cfg(feature = "candle")]
        {
            use candle_core::DType;

            let dtypes = [/* Not Supported DType::BF16,*/ DType::F32];
            for device in candle_devices(cuda_device_index) {
                let device_name = if device.is_cuda() { "device" } else { "host" };
                for dtype in dtypes {
                    let scalar_name = match dtype {
                        //DType::BF16 => "bf16",
                        DType::F32 => "f32",
                        _ => unreachable!(),
                    };
                    let name = format!("{scalar_name}_{device_name}");
                    let id = BenchmarkId::new("candle", name);
                    g.bench_function(id, |b| {
                        use candle_backend::LeNet5Classifier;
                        let mut model = LeNet5Classifier::new(device.clone(), dtype)
                            .unwrap()
                            .with_sgd(false)
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
            for device in autograph_devices(device_index) {
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
            use tch::kind::Kind;

            let kinds = [Kind::BFloat16, Kind::Float];
            for device in tch_devices(cuda_device_index) {
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
        #[cfg(feature = "candle")]
        {
            use candle_core::DType;

            let dtypes = [/* Not Supported DType::BF16,*/ DType::F32];
            for device in candle_devices(cuda_device_index) {
                let device_name = if device.is_cuda() { "device" } else { "host" };
                for dtype in dtypes {
                    let scalar_name = match dtype {
                        //DType::BF16 => "bf16",
                        DType::F32 => "f32",
                        _ => unreachable!(),
                    };
                    let name = format!("{scalar_name}_{device_name}");
                    let id = BenchmarkId::new("candle", name);
                    g.bench_function(id, |b| {
                        use candle_backend::LeNet5Classifier;
                        let model = LeNet5Classifier::new(device.clone(), dtype)
                            .unwrap()
                            .with_sgd(false)
                            .unwrap();
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
