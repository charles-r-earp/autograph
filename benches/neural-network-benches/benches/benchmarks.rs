use autograph::{device::Device, scalar::ScalarType};
use criterion::{criterion_group, criterion_main, Criterion};
use neural_network_benches::autograph_backend;
#[cfg(feature = "tch")]
use neural_network_benches::tch_backend;
use std::str::FromStr;

pub fn criterion_benchmark(c: &mut Criterion) {
    #[cfg_attr(not(feature = "cuda"), allow(unused))]
    let krnl_device = if cfg!(feature = "device") {
        let krnl_device = std::env::var("KRNL_DEVICE");
        println!("KRNL_DEVICE = {krnl_device:?}");
        let krnl_device_index = if let Ok(krnl_device) = krnl_device.as_ref() {
            usize::from_str(krnl_device).unwrap()
        } else {
            0
        };
        println!("testing device {krnl_device_index}");
        krnl_device_index
    } else {
        0
    };

    #[cfg_attr(not(feature = "cuda"), allow(unused))]
    let tch_device_index = if cfg!(feature = "tch") {
        let tch_device = std::env::var("TCH_DEVICE");
        println!("TCH_DEVICE = {tch_device:?}");
        let tch_device_index = if let Ok(tch_device) = tch_device.as_ref() {
            usize::from_str(tch_device).unwrap()
        } else {
            0
        };
        #[cfg(feature = "tch")]
        if tch::utils::has_cuda() {
            println!("testing tch device {tch_device_index}");
        }
        tch_device_index
    } else {
        0
    };

    let train_batch_size = 100;
    let infer_batch_size = 1000;

    c.bench_function("autograph_linear_classifier_infer_host", |b| {
        use autograph_backend::LinearClassifier;

        let model = LinearClassifier::new(Device::host(), ScalarType::F32, 28 * 28, 10).unwrap();
        b.iter(|| {
            model.infer(infer_batch_size).unwrap();
        });
    });

    c.bench_function("autograph_linear_classifier_train_host", |b| {
        use autograph_backend::LinearClassifier;

        let mut model = LinearClassifier::new(Device::host(), ScalarType::F32, 28 * 28, 10)
            .unwrap()
            .with_sgd(true);
        b.iter(|| {
            model.train(train_batch_size).unwrap();
        });
    });

    #[cfg(feature = "tch")]
    {
        use tch::{kind::Kind, Device};

        let mut devices = vec![Device::Cpu];
        if tch::utils::has_cuda() {
            devices.push(Device::Cuda(tch_device_index));
        }

        for device in [Device::Cpu, Device::Cuda(tch_device_index)] {
            let device_name = if device.is_cuda() { "device" } else { "host" };

            c.bench_function(&format!("tch_linear_classifier_infer_{device_name}"), |b| {
                use tch_backend::LinearClassifier;

                let model = LinearClassifier::new(device, Kind::Float, 28 * 28, 10).unwrap();
                b.iter(|| {
                    model.infer(infer_batch_size).unwrap();
                });
            });

            c.bench_function(&format!("tch_linear_classifier_train_{device_name}"), |b| {
                use tch_backend::LinearClassifier;

                let mut model = LinearClassifier::new(device, Kind::Float, 28 * 28, 10)
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
