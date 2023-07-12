#![allow(warnings)]

use anyhow::Result;
use autograph::tensor::{Tensor, TensorView};
use dry::macro_for;
use half::{bf16, f16};
#[cfg(feature = "device")]
use krnl::buffer::Buffer;
use krnl::{buffer::Slice, device::Device, scalar::Scalar};
#[cfg(not(target_arch = "wasm32"))]
use krnl::{device::Features, scalar::ScalarType};
#[cfg(not(target_arch = "wasm32"))]
use libtest_mimic::{Arguments, Trial};
use ndarray::{Array, Array1, Axis, Dimension, IntoDimension, RemoveAxis};
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use std::str::FromStr;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[cfg(all(target_arch = "wasm32", run_in_browser))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = Arguments::from_args();
    let tests = if cfg!(feature = "device") && !cfg!(miri) {
        let devices: Vec<_> = [Device::builder().build().unwrap()]
            .into_iter()
            .chain((1..).map_while(|i| Device::builder().index(i).build().ok()))
            .collect();
        if devices.is_empty() {
            panic!("No device!");
        }
        let device_infos: Vec<_> = devices.iter().map(|x| x.info().unwrap()).collect();
        println!("devices: {device_infos:#?}");
        let krnl_device = std::env::var("KRNL_DEVICE");
        let device_index = if let Ok(krnl_device) = krnl_device.as_ref() {
            usize::from_str(krnl_device).unwrap()
        } else {
            0
        };
        println!("KRNL_DEVICE = {krnl_device:?}");
        println!("testing device {device_index}");
        let device = devices.get(device_index).unwrap();
        tests(&Device::host())
            .into_iter()
            .chain(tests(device))
            .collect()
    } else {
        tests(&Device::host()).into_iter().collect()
    };
    libtest_mimic::run(&args, tests).exit()
}

#[cfg(not(target_arch = "wasm32"))]
fn device_test(device: &Device, name: &str, f: impl Fn(&Device) + Send + Sync + 'static) -> Trial {
    let name = format!(
        "{name}_{}",
        if device.is_host() { "host" } else { "device" }
    );
    let device = device.clone();
    Trial::test(name, move || {
        f(&device);
        Ok(())
    })
}

fn features_for_scalar_size(size: usize) -> Features {
    Features::empty()
        .with_shader_int8(size == 1)
        .with_shader_int16(size == 2)
        .with_shader_int64(size == 8)
}

fn features_for_scalar(scalar_type: ScalarType) -> Features {
    features_for_scalar_size(scalar_type.size()).with_shader_float64(scalar_type == ScalarType::F64)
}

#[cfg(not(target_arch = "wasm32"))]
fn tests(device: &Device) -> Vec<Trial> {
    tensor_tests(device)
}

#[cfg(not(target_arch = "wasm32"))]
fn tensor_tests(device: &Device) -> Vec<Trial> {
    let features = device
        .info()
        .map(|x| x.features())
        .unwrap_or(Features::empty());
    let mut tests = Vec::new();

    tests.extend([
        Trial::test("tensor_from_array0", || {
            tensor_from_array(Array::from_elem((), 1));
            Ok(())
        }),
        Trial::test("tensor_from_array1", || {
            tensor_from_array(Array::from_shape_vec(3, (1..=3).into_iter().collect()).unwrap());
            Ok(())
        }),
        Trial::test("tensor_from_array2", || {
            tensor_from_array(
                Array::from_shape_vec([2, 3], (1..=6).into_iter().collect()).unwrap(),
            );
            Ok(())
        }),
        Trial::test("tensor_from_array3", || {
            tensor_from_array(
                Array::from_shape_vec([2, 3, 4], (1..=24).into_iter().collect()).unwrap(),
            );
            Ok(())
        }),
        Trial::test("tensor_from_array4", || {
            tensor_from_array(
                Array::from_shape_vec([2, 3, 4, 5], (1..=120).into_iter().collect()).unwrap(),
            );
            Ok(())
        }),
        Trial::test("tensor_from_array4", || {
            tensor_from_array(
                Array::from_shape_vec([2, 3, 4, 5, 6], (1..=120 * 6).into_iter().collect())
                    .unwrap(),
            );
            Ok(())
        }),
        Trial::test("tensor_from_array5", || {
            tensor_from_array(
                Array::from_shape_vec([2, 3, 4, 5, 6], (1..=120 * 6).into_iter().collect())
                    .unwrap(),
            );
            Ok(())
        }),
        Trial::test("tensor_from_array6", || {
            tensor_from_array(
                Array::from_shape_vec([2, 3, 4, 5, 6, 7], (1..=120 * 6 * 7).into_iter().collect())
                    .unwrap(),
            );
            Ok(())
        }),
        Trial::test("tensor_from_arrayD", || {
            tensor_from_array(
                Array::from_shape_vec(
                    [2, 3, 4, 5, 6, 7, 8].as_ref(),
                    (1..=120 * 6 * 7 * 8).into_iter().collect(),
                )
                .unwrap(),
            );
            Ok(())
        }),
    ]);
    tests.extend(
        linalg::linalg_tests(device)
            .into_iter()
            .chain(reorder::reorder_tests(device))
            .chain(reduce::reduce_tests(device))
            .chain(ops::ops_tests(device)),
    );
    #[cfg(feature = "learn")]
    tests.extend(learn::learn_tests(device));
    tests
}

fn tensor_from_array<D: Dimension>(x: Array<u32, D>) {
    let y = TensorView::try_from(x.view()).unwrap();
    assert_eq!(x.view(), y.as_array().unwrap());
    let y_t = TensorView::try_from(x.t()).unwrap();
    assert_eq!(x.t(), y_t.as_array().unwrap());
}

mod linalg {
    use super::*;
    use approx::assert_relative_eq;
    use autograph::tensor::CowTensor;
    use ndarray::{linalg::Dot, Array2};
    use std::fmt::{self, Display};

    pub fn linalg_tests(device: &Device) -> Vec<Trial> {
        let mut tests = Vec::new();
        let features = if let Some(info) = device.info() {
            info.features()
        } else {
            Features::empty()
        };
        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            let scalar_type = $T::scalar_type();
            let type_name = scalar_type.name();
            let ignore = device.is_device() &&
                    !features.contains(&features_for_scalar(scalar_type));
            for n in [2, 4, 5, 8, 16, 32, 64, 128] {
                let [m, k, n] = [n; 3];
                use Transpose::*;
                for (ta, tb) in [(N, N), (T, N), (N, T), (T, T)] {
                    let name = format!("tensor_dot_{type_name}_m{m}_k{k}_n{n}_{ta}{tb}");
                    tests.push(device_test(device, &name, move |device| {
                        tensor_dot::<$T>(device, [m, k, n], [ta, tb])
                    }).with_ignored_flag(ignore));
                }
            }
        });
        tests
    }

    fn gen_array<T: Scalar>(dim: [usize; 2]) -> Array2<T> {
        let n = dim[0] * dim[1];
        let vec: Vec<T> = (1..10)
            .cycle()
            .map(|x| {
                if std::mem::size_of::<T>() == 1 {
                    T::from_u8((x == 1) as u8).unwrap()
                } else {
                    T::from_usize(x).unwrap()
                }
            })
            .take(n)
            .collect();
        Array2::from_shape_vec(dim, vec).unwrap()
    }

    #[allow(unused)]
    #[derive(Clone, Copy, Debug)]
    enum Transpose {
        N,
        T,
    }

    impl Display for Transpose {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let c = match self {
                Self::N => 'n',
                Self::T => 't',
            };
            write!(f, "{c}")
        }
    }

    fn tensor_dot<T: Scalar>(device: &Device, [m, k, n]: [usize; 3], [a_t, b_t]: [Transpose; 2]) {
        let dim1 = match a_t {
            Transpose::N => [m, k],
            Transpose::T => [k, m],
        };
        let dim2 = match b_t {
            Transpose::N => [k, n],
            Transpose::T => [n, k],
        };
        let a1 = gen_array::<T>(dim1);
        let t1 = CowTensor::from(a1.view())
            .into_device(device.clone())
            .unwrap();
        let (a1, t1) = match a_t {
            Transpose::N => (a1.view(), t1.view()),
            Transpose::T => (a1.t(), t1.t()),
        };
        let a2 = gen_array::<T>(dim2);
        let t2 = CowTensor::from(a2.view())
            .into_device(device.clone())
            .unwrap();
        let (a2, t2) = match b_t {
            Transpose::N => (a2.view(), t2.view()),
            Transpose::T => (a2.t(), t2.t()),
        };
        let a_true = a1.dot(&a2);
        let a_out = t1.dot(&t2).unwrap().into_array().unwrap();
        let scalar_type = T::scalar_type();
        if matches!(scalar_type, ScalarType::F16 | ScalarType::BF16) {
            let a_true = a_true.map(|x| x.to_f32().unwrap());
            let a_out = a_out.map(|x| x.to_f32().unwrap());
            let epsilon = k as f32;
            assert_relative_eq!(a_true, a_out, epsilon = epsilon);
        } else if scalar_type == ScalarType::F32 {
            let a_true = a_true.map(|x| x.to_f32().unwrap());
            let a_out = a_out.map(|x| x.to_f32().unwrap());
            assert_relative_eq!(a_true, a_out);
        } else if scalar_type == ScalarType::F64 {
            let a_true = a_true.map(|x| x.to_f64().unwrap());
            let a_out = a_out.map(|x| x.to_f64().unwrap());
            assert_relative_eq!(a_true, a_out);
        } else {
            assert_eq!(a_out, a_true);
        }
    }
}

mod ops {
    use super::*;
    use ndarray::{Array1, IntoDimension};

    pub fn ops_tests(device: &Device) -> Vec<Trial> {
        let mut tests = Vec::new();
        let features = if let Some(info) = device.info() {
            info.features()
        } else {
            Features::empty()
        };
        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                let scalar_type = $T::scalar_type();
                let ignore = device.is_device() &&
                    !features.contains(&features_for_scalar(scalar_type));
                let ty = scalar_type.name();
                let lens = [7, 64, 300];
                tests.extend([
                    device_test(device, &format!("scaled_add_{ty}"), |device| {
                        for n in [7, 64, 300] {
                            scaled_add::<$T>(device, &[n]);
                        }
                        scaled_add::<$T>(device, &[3, 5]);
                        scaled_add::<$T>(device, &[21, 14]);
                    }),
                ].into_iter().map(|trial| trial.with_ignored_flag(ignore)));
        });

        tests
    }

    fn scaled_add<T: Scalar>(device: &Device, shape: &[usize]) {
        let alpha = T::from_u32(2).unwrap();
        let shape = shape.into_dimension();
        let x_array = (1..10)
            .cycle()
            .take(shape.size())
            .map(|x| T::from_usize(x).unwrap())
            .collect::<Array1<_>>()
            .into_shape(shape.clone())
            .unwrap();
        let mut y_array = (11..20)
            .cycle()
            .take(x_array.len())
            .map(|x| T::from_usize(x).unwrap())
            .collect::<Array1<_>>()
            .into_shape(shape.clone())
            .unwrap();
        let x = Tensor::from(x_array.clone())
            .into_device(device.clone())
            .unwrap();
        let mut y = Tensor::from(y_array.clone())
            .into_device(device.clone())
            .unwrap();
        y_array.scaled_add(alpha, &x_array);
        y.scaled_add(alpha, &x).unwrap();
        let y = y.into_array().unwrap();
        assert_eq!(y, y_array);
    }
}

mod reorder {
    use super::*;
    use ndarray::IntoDimension;

    pub fn reorder_tests(device: &Device) -> Vec<Trial> {
        let mut tests = Vec::new();

        let features = if let Some(info) = device.info() {
            info.features()
        } else {
            Features::empty()
        };
        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                let scalar_type = $T::scalar_type();
                let ignore = device.is_device() &&
                    !features.contains(&features_for_scalar(scalar_type));
                let ty = scalar_type.name();
                tests.extend([
                    device_test(device, &format!("into_standard_layout2_{ty}"), |device| {
                        into_standard_layout::<$T, _>(device, [3, 3], [1, 0]);
                        into_standard_layout::<$T, _>(device, [21, 30], [1, 0]);
                    }),
                    device_test(device, &format!("into_standard_layout4_{ty}"), |device| {
                        into_standard_layout::<$T, _>(device, [1, 2, 3, 3], [0, 2, 3, 1]);
                        into_standard_layout::<$T, _>(device, [2, 21, 3, 30], [0, 3, 1, 2]);
                    }),
                ].into_iter().map(|trial| trial.with_ignored_flag(ignore)));
        });

        tests
    }

    fn into_standard_layout<T: Scalar, E: IntoDimension>(device: &Device, shape: E, axes: E) {
        let shape = shape.into_dimension();
        let x_vec = (1..100)
            .cycle()
            .take(shape.size())
            .map(|x| T::from_usize(x).unwrap())
            .collect();
        let x_array = Array::from_shape_vec(shape, x_vec).unwrap();
        let axes = E::Dim::from_dimension(&axes.into_dimension()).unwrap();
        let y_array = x_array
            .view()
            .permuted_axes(axes.clone())
            .as_standard_layout()
            .to_owned();
        let x = Tensor::from(x_array.clone())
            .into_device(device.clone())
            .unwrap();
        let y = x
            .permuted_axes(axes)
            .into_standard_layout()
            .unwrap()
            .into_array()
            .unwrap();
        assert_eq!(y, y_array);
    }
}

mod reduce {
    use super::*;

    pub fn reduce_tests(device: &Device) -> Vec<Trial> {
        let mut tests = Vec::new();
        tests.extend([
            device_test(device, "sum_f32", |device| {
                for n in [4, 11, 33, 517, 1021] {
                    sum::<f32, _>(device, n);
                }
            }),
            device_test(device, "sum_axis_f32", |device| {
                for n in [4, 11, 33, 517, 1021] {
                    for axis in [0, 1] {
                        sum_axis::<f32, _>(device, [n / 2, n], Axis(axis));
                    }
                }
            }),
        ]);
        tests
    }

    fn sum<T: Scalar, E: IntoDimension>(device: &Device, shape: E) {
        let shape = shape.into_dimension();
        let x_array = (1..10)
            .cycle()
            .take(shape.size())
            .map(|x| T::from_usize(x).unwrap())
            .collect::<Array1<_>>()
            .into_shape(shape.clone())
            .unwrap();
        let y_array = x_array.sum();
        let x = Tensor::from(x_array).into_device(device.clone()).unwrap();
        let y = x.sum().unwrap();
        assert_eq!(y, y_array);
    }

    fn sum_axis<T: Scalar, E: IntoDimension>(device: &Device, shape: E, axis: Axis)
    where
        E::Dim: RemoveAxis,
    {
        let shape = shape.into_dimension();
        let x_array = (1..10)
            .cycle()
            .take(shape.size())
            .map(|x| T::from_usize(x).unwrap())
            .collect::<Array1<_>>()
            .into_shape(shape.clone())
            .unwrap();
        let y_array = x_array.sum_axis(axis);
        let x = Tensor::from(x_array).into_device(device.clone()).unwrap();
        let y = x.sum_axis(axis).unwrap().into_array().unwrap();
        assert_eq!(y, y_array, "{:?} {:?}", shape.slice(), axis);
    }
}

#[cfg(feature = "learn")]
mod learn {
    use super::*;

    pub fn learn_tests(device: &Device) -> Vec<Trial> {
        let mut tests = Vec::new();
        #[cfg(feature = "neural-network")]
        {
            tests.extend(neural_network::neural_network_tests(device));
        }
        tests
    }

    #[cfg(feature = "neural-network")]
    mod neural_network {
        use super::*;
        use autograph::{
            learn::neural_network::{
                autograd::Variable,
                layer::{Forward, MaxPool2},
            },
            tensor::Tensor1,
        };

        pub fn neural_network_tests(device: &Device) -> Vec<Trial> {
            let mut tests = Vec::new();

            if device.is_device() {
                tests.push(device_test(device, &format!("max_pool2_f32"), |device| {
                    let pool = MaxPool2::builder().size([2, 2]).strides([2, 2]).build();
                    max_pool2::<f32>(device, [1, 1, 4, 4], &pool);
                    max_pool2::<f32>(device, [1, 1, 12, 12], &pool);
                    max_pool2::<f32>(device, [2, 3, 4, 4], &pool);
                    max_pool2::<f32>(device, [1, 1, 24, 24], &pool);
                }));
            }
            tests
        }

        fn max_pool2<T: Scalar>(device: &Device, input_shape: [usize; 4], pool: &MaxPool2) {
            let len = input_shape.iter().product();
            let x_vec: Vec<T> = (0..10u8)
                .map(|x| T::from_u8(x).unwrap())
                .cycle()
                .take(len)
                .collect();
            let x_array = Array::from(x_vec).into_shape(input_shape).unwrap();
            let x_host = Tensor::from(x_array);
            let x_device = x_host.to_device(device.clone()).unwrap();
            let y_host = pool
                .forward(Variable::from(x_host))
                .unwrap()
                .into_value()
                .into_owned()
                .unwrap()
                .try_into_tensor::<T>()
                .unwrap();
            let y_device = pool
                .forward(Variable::from(x_device))
                .unwrap()
                .into_value()
                .into_owned()
                .unwrap()
                .try_into_tensor::<T>()
                .unwrap();
            assert_eq!(y_host.into_array().unwrap(), y_device.into_array().unwrap());
        }
    }
}

/*
fn buffer_test_lengths() -> impl ExactSizeIterator<Item = usize> {
    [0, 1, 3, 4, 16, 67, 157].into_iter()
}
fn buffer_transfer_test_lengths() -> impl ExactSizeIterator<Item = usize> {
    #[cfg(not(miri))]
    {
        [0, 1, 3, 4, 16, 345, 9_337_791].into_iter()
    }
    #[cfg(miri)]
    {
        [0, 1, 3, 4, 16, 345].into_iter()
    }
}

fn buffer_from_vec(device: Device) {
    let n = buffer_transfer_test_lengths().last().unwrap();
    let x = (10..20).cycle().take(n).collect::<Vec<_>>();
    for n in buffer_transfer_test_lengths() {
        let x = &x[..n];
        let y = Slice::from(x)
            .to_device(device.clone())
            .unwrap()
            .into_vec()
            .unwrap();
        assert_eq!(y.len(), n);
        if x != y.as_slice() {
            for (x, y) in x.iter().zip(y) {
                assert_eq!(&y, x);
            }
        }
    }
}

#[cfg(feature = "device")]
fn device_buffer_too_large(device: Device) {
    use krnl::buffer::error::DeviceBufferTooLarge;
    let error = unsafe {
        Buffer::<u32>::uninit(device, (i32::MAX / 4 + 1).try_into().unwrap())
    }.err().unwrap();
    error.downcast_ref::<DeviceBufferTooLarge>().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
fn buffer_transfer(device: Device, device2: Device) {
    let n = buffer_transfer_test_lengths().last().unwrap();
    let x = (10..20).cycle().take(n).collect::<Vec<_>>();
    for n in buffer_transfer_test_lengths() {
        let x = &x[..n];
        let y = Slice::from(x)
            .to_device(device.clone())
            .unwrap()
            .to_device(device2.clone())
            .unwrap()
            .into_vec()
            .unwrap();
        if x != y.as_slice() {
            for (x, y) in x.iter().zip(y) {
                assert_eq!(&y, x);
            }
        }
    }
}

fn buffer_fill<T: Scalar>(device: Device) {
    let elem = T::one();
    let n = buffer_test_lengths().last().unwrap();
    let x = (10..20)
        .cycle()
        .map(|x| T::from_u32(x).unwrap())
        .take(n)
        .collect::<Vec<_>>();
    for n in buffer_test_lengths() {
        let x = &x[..n];
        let mut y = Slice::from(x).to_device(device.clone()).unwrap();
        y.fill(elem).unwrap();
        let y: Vec<T> = y.into_vec().unwrap();
        for y in y.into_iter() {
            assert_eq!(y, elem);
        }
    }
}

fn buffer_cast<X: Scalar, Y: Scalar>(device: Device) {
    let n = buffer_test_lengths().last().unwrap();
    let x = (10..20)
        .cycle()
        .map(|x| X::from_u32(x).unwrap())
        .take(n)
        .collect::<Vec<_>>();
    for n in buffer_test_lengths() {
        let x = &x[..n];
        let y = Slice::<X>::from(x)
            .into_device(device.clone())
            .unwrap()
            .cast_into::<Y>()
            .unwrap()
            .into_vec()
            .unwrap();
        for (x, y) in x.iter().zip(y.iter()) {
            assert_eq!(*y, x.cast::<Y>());
        }
    }
}

fn buffer_bitcast<X: Scalar, Y: Scalar>(device: Device) {
    let x_host = &[X::default(); 16];
    let x = Slice::from(x_host.as_ref()).to_device(device).unwrap();
    for i in 0..=16 {
        for range in [i..16, 0..i] {
            let bytemuck_result =
                bytemuck::try_cast_slice::<X, Y>(&x_host[range.clone()]).map(|_| ());
            let result = x.slice(range).unwrap().bitcast::<Y>().map(|_| ());
            #[cfg(miri)]
            let _ = (bytemuck_result, result);
            #[cfg(not(miri))]
            assert_eq!(result, bytemuck_result);
        }
    }
}

#[test]
fn buffer_from_vec_host() {
    buffer_from_vec(Device::host());
}

#[cfg(target_arch = "wasm32")]
macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        #[test]
        fn [<buffer_fill_ $T _host>]() {
            buffer_fill::<$T>(Device::host());
        }
    }
});

macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            #[test]
            fn [<buffer_cast_ $X _ $Y _host>]() {
                buffer_cast::<$X, $Y>(Device::host());
            }
            #[test]
            fn [<buffer_bitcast_ $X _ $Y _host>]() {
                buffer_bitcast::<$X, $Y>(Device::host());
            }
        }
    });
});
*/
