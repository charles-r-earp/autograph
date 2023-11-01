[![LicenseBadge]][License]
[![DocsBadge]][Docs]
[![build](https://github.com/charles-r-earp/autograph/actions/workflows/ci.yml/badge.svg)](https://github.com/charles-r-earp/autograph/actions/workflows/ci.yml)

[License]: https://github.com/charles-r-earp/autograph/blob/main/LICENSE-APACHE
[LicenseBadge]: https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg

[Docs]: https://docs.rs/autograph
[DocsBadge]: https://docs.rs/autograph/badge.svg


# **autograph**
A machine learning library for Rust.

GPGPU kernels implemented with [**krnl**](https://github.com/charles-r-earp/krnl).
- Host / Device execution.
- Tensors emulate [ndarray](https://github.com/rust-ndarray/ndarray)
   - Host tensors can be borrowed as arrays.
- Tensors / Models / Optimizers can be serialized with [serde](https://github.com/serde-rs/serde).
   - Portable between platforms.
   - Save / resume training progress.
- Fully extensible, in Rust.

## Neural Networks
```rust
#[derive(Layer, Forward, Debug)]
#[autograph(forward(Variable4, Output=Variable2))]
struct LeNet5 {
    #[layer]
    conv1: Conv2<Relu>,
    #[layer]
    pool1: MaxPool2,
    #[layer]
    conv2: Conv2<Relu>,
    #[layer]
    pool2: MaxPool2,
    #[layer]
    flatten: Flatten,
    #[layer]
    dense1: Dense<Relu>,
    #[layer]
    dense2: Dense<Relu>,
    #[layer]
    dense3: Dense,
}

impl LeNet5 {
    fn new(device: Device, scalar_type: ScalarType) -> Result<Self> {
        let conv1 = Conv2::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(1)
            .outputs(6)
            .filter([5, 5])
            .activation(Relu)
            .build()?;
        let pool1 = MaxPool2::builder().filter([2, 2]).build();
        let conv2 = Conv2::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(6)
            .outputs(16)
            .filter([5, 5])
            .activation(Relu)
            .build()?;
        let pool2 = MaxPool2::builder().filter([2, 2]).build();
        let flatten = Flatten;
        let dense1 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(16 * 4 * 4)
            .outputs(128)
            .activation(Relu)
            .build()?;
        let dense2 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(128)
            .outputs(84)
            .activation(Relu)
            .build()?;
        let dense3 = Dense::builder()
            .device(device.clone())
            .scalar_type(scalar_type)
            .inputs(84)
            .outputs(10)
            .bias(true)
            .build()?;
        Ok(Self {
            conv1,
            pool1,
            conv2,
            pool2,
            flatten,
            dense1,
            dense2,
            dense3,
        })
    }
}

let mut model = LeNet5::new(device.clone(), ScalarType::F32)?;
model.set_training(true)?;
let y = model.forward(x)?;
let loss = y.cross_entropy_loss(t)?;
loss.backward()?;
for parameter in model.parameters_mut()? {
    optimizer.update(learning_rate, parameter)?;
}
```
See the [Neural Network MNIST](examples/neural-network-mnist) example.

# Benchmarks
*NVIDIA GeForce GTX 1060 with Max-Q Design*

## LeNet5(training, batch_size = 100)

|                   | `autograph`               | `tch`                            |
|:------------------|:--------------------------|:-------------------------------- |
| **`bf16_host`**   | `591.07 ms` (✅ **1.00x**) | `76.58 ms` (🚀 **7.72x faster**)  |
| **`f32_host`**    | `16.60 ms` (✅ **1.00x**)  | `3.18 ms` (🚀 **5.22x faster**)   |
| **`bf16_device`** | `7.44 ms` (✅ **1.00x**)   | `21.09 ms` (❌ *2.84x slower*)    |
| **`f32_device`**  | `1.56 ms` (✅ **1.00x**)   | `3.94 ms` (❌ *2.52x slower*)     |

## LeNet5(inference, batch_size = 1,000)

|                   | `autograph`               | `tch`                              |
|:------------------|:--------------------------|:---------------------------------- |
| **`bf16_host`**   | `2.14 s` (✅ **1.00x**)    | `196.52 ms` (🚀 **10.87x faster**)  |
| **`f32_host`**    | `104.09 ms` (✅ **1.00x**) | `9.15 ms` (🚀 **11.38x faster**)    |
| **`bf16_device`** | `4.31 ms` (✅ **1.00x**)   | `48.74 ms` (❌ *11.31x slower*)     |
| **`f32_device`**  | `4.34 ms` (✅ **1.00x**)   | `1.85 ms` (🚀 **2.35x faster**)     |

See the [Neural Network](benches/neural-network-benches) benchmark.

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.

# Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
