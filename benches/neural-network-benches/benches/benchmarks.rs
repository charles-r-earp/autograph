use autograph::{device::Device, scalar::ScalarType};
use criterion::{criterion_group, criterion_main, Criterion};
use neural_network_benches::autograph_backend;
#[cfg(feature = "tch")]
use neural_network_benches::tch_backend;
use std::str::FromStr;

pub fn criterion_benchmark(c: &mut Criterion) {
    #[cfg_attr(not(feature = "cuda"), allow(unused))]
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

    let mut devices = vec![Device::host()];

    if cfg!(feature = "device") {
        devices.push(Device::builder().index(device_index).build().unwrap());
    }

    for device in devices {
        let device_name = if device.is_device() { "device" } else { "host" };

        c.bench_function(
            &format!("autograph_infer_{infer_batch_size}_{device_name}"),
            |b| {
                use autograph_backend::Lenet5Classifier;

                let model = Lenet5Classifier::new(device.clone(), ScalarType::F32).unwrap();
                b.iter(|| {
                    model.infer(infer_batch_size).unwrap();
                });
            },
        );

        c.bench_function(
            &format!("autograph_train_{train_batch_size}_{device_name}"),
            |b| {
                use autograph_backend::Lenet5Classifier;

                let mut model = Lenet5Classifier::new(device.clone(), ScalarType::F32)
                    .unwrap()
                    .with_sgd(true);
                b.iter(|| {
                    model.train(train_batch_size).unwrap();
                });
            },
        );
    }

    #[cfg(feature = "tch")]
    {
        use tch::{kind::Kind, Device};

        let mut devices = vec![Device::Cpu];
        if tch::utils::has_cuda() {
            let device = Device::Cuda(tch_device_index);
            devices.push(device);
        }

        for device in [Device::Cpu, Device::Cuda(tch_device_index)] {
            let device_name = if device.is_cuda() { "device" } else { "host" };

            c.bench_function(
                &format!("tch_infer_{infer_batch_size}_{device_name}"),
                |b| {
                    use tch_backend::Lenet5Classifier;

                    let model = Lenet5Classifier::new(device, Kind::Float).unwrap();
                    b.iter(|| {
                        model.infer(infer_batch_size).unwrap();
                    });
                },
            );

            c.bench_function(
                &format!("tch_train_{train_batch_size}_{device_name}"),
                |b| {
                    use tch_backend::Lenet5Classifier;

                    let mut model = Lenet5Classifier::new(device, Kind::Float)
                        .unwrap()
                        .with_sgd(true)
                        .unwrap();
                    b.iter(|| {
                        model.train(train_batch_size).unwrap();
                    });
                },
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
