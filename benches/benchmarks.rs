use autograph::backend::Device;
use autograph::tensor::{linalg::gemm, Num, Tensor};
use criterion::{criterion_group, criterion_main, Criterion};
use half::bf16;
use std::any::type_name;
use std::fmt::Debug;
use std::time::Instant;

// Note: 16 and 64 bit types are not fully supported, so are commented out

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
enum Transpose {
    N,
    T,
}

fn gemm_benches<X: Num>(device: &Device, c: &mut Criterion) {
    fn bench_gemm<T: Num>(
        device: &Device,
        c: &mut Criterion,
        m: usize,
        k: usize,
        n: usize,
        a_t: Transpose,
        b_t: Transpose,
    ) {
        let name = format!(
            "{:?} tensor_gemm_{}_m{}_k{}_n{}_{:?}_{:?}",
            device,
            type_name::<T>(),
            m,
            k,
            n,
            a_t,
            b_t
        );
        let alpha = T::one();
        let beta = T::one();
        c.bench_function(&name, |b| {
            let device = device.clone();
            let x1_dim = match a_t {
                Transpose::N => [m, k],
                Transpose::T => [k, m],
            };
            let x2_dim = match a_t {
                Transpose::N => [k, n],
                Transpose::T => [n, k],
            };
            let y_dim = [x1_dim[0], x2_dim[1]];
            let x1 = Tensor::ones(&device, x1_dim).unwrap();
            let x2 = Tensor::ones(&device, x2_dim).unwrap();
            let mut y = Tensor::ones(&device, y_dim).unwrap();
            b.iter_custom(move |n| {
                let x1 = match a_t {
                    Transpose::N => x1.view(),
                    Transpose::T => x1.t(),
                };
                let x2 = match b_t {
                    Transpose::N => x2.view(),
                    Transpose::T => x2.t(),
                };
                let mut y = y.view_mut();
                let start = Instant::now();
                smol::block_on(async {
                    for _ in 0..n as usize {
                        gemm(alpha, &x1, &x2, beta, &mut y).unwrap();
                    }
                    device.synchronize().unwrap().await.unwrap();
                });
                start.elapsed()
            });
        });
    }
    use Transpose::*;
    for n in [32, 64, 100, 256, 1024].iter().copied() {
        for (a_t, b_t) in [(N, N), (N, T), (T, N), (N, N)].iter().copied() {
            bench_gemm::<X>(device, c, n, n, n, a_t, b_t);
        }
    }
}

fn num_bench<T: Num>(device: &Device, c: &mut Criterion) {
    gemm_benches::<T>(device, c);
}

pub fn run_num_benches(c: &mut Criterion) {
    for device in Device::list() {
        //num_bench::<bf16>(&device, c);
        num_bench::<u32>(&device, c);
        num_bench::<i32>(&device, c);
        num_bench::<f32>(&device, c);
        //num_bench::<u64>(&device, c);
        //num_bench::<i64>(&device, c);
        //num_bench::<f64>(&device, c);
    }
}

criterion_group!(num_benches, run_num_benches);
criterion_main!(num_benches);
