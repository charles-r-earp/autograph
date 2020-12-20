use autograph::backend::Device;
use autograph::tensor::{Num, Tensor};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, ArrayView2};
use std::any::type_name;

fn gemm_benches<T: Num>(device: &Device, c: &mut Criterion) {
    fn bench_dot_nn<T: Num>(
        device: &Device,
        c: &mut Criterion,
        x1: &ArrayView2<T>,
        x2: &ArrayView2<T>,
    ) {
        c.bench_function(
            &format!(
                "{:?} Tensor Dot {:?} x {:?} {}",
                device,
                x1.raw_dim(),
                x2.raw_dim(),
                type_name::<T>()
            ),
            move |b| {
                let x1 = Tensor::from_array(device, x1.view()).unwrap();
                let x2 = Tensor::from_array(device, x2.view()).unwrap();
                b.iter(move || {
                    x1.dot(&x2).unwrap();
                    smol::block_on(device.synchronize().unwrap()).unwrap();
                });
            },
        );
    }
    fn bench_dot_nt<T: Num>(
        device: &Device,
        c: &mut Criterion,
        x1: &ArrayView2<T>,
        x2: &ArrayView2<T>,
    ) {
        c.bench_function(
            &format!(
                "{:?} Tensor Dot {:?} x {:?}T {}",
                device,
                x1.raw_dim(),
                x2.raw_dim(),
                type_name::<T>()
            ),
            move |b| {
                let x1 = Tensor::from_array(device, x1.view()).unwrap();
                let x2 = Tensor::from_array(device, x2.view()).unwrap();
                b.iter(move || {
                    x1.dot(&x2.t()).unwrap();
                    smol::block_on(device.synchronize().unwrap()).unwrap();
                });
            },
        );
    }
    for n in [10, 100, 256].iter().copied() {
        bench_dot_nn::<T>(
            device,
            c,
            &Array::ones([n; 2]).view(),
            &Array::ones([n; 2]).view(),
        );
        bench_dot_nt::<T>(
            device,
            c,
            &Array::ones([n; 2]).view(),
            &Array::ones([n; 2]).view(),
        );
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    for gpu in Device::list_gpus() {
        gemm_benches::<f32>(&gpu, c);
        gemm_benches::<u32>(&gpu, c);
        gemm_benches::<i32>(&gpu, c);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
