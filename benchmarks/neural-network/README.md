# Neural Network Benchmark
This benchmark compares training performance of autograph and other libraries.

# Datasets
- Mnist

# Networks
- Lenet5

# Usage
```
cargo run --release -- --autograph
/// Run all the benchmarks with --all
cargo run --release -- --all
/// Run for 10 epochs (default 100)
cargo run --release -- --all --epochs 100
```

# tch
Rust bindings for the C++ API of PyTorch: https://github.com/LaurentMazare/tch-rs.
```
/// Run tch benchmark
cargo run --release --features tch -- --tch
```

# Results
NVIDIA GeForce GTX 1060 with Max-Q Design

![NVIDIA GeForce GTX 1060 with Max-Q Design](nv_gtx1060.png)

# Final Notes
If you have any issues with this example please create an issue at https://github.com/charles-r-earp/autograph.
