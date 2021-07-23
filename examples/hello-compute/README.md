# Hello Compute!
This example shows the basics of how to write your own shader code.

# Nightly
[rust-gpu](https://github.com/EmbarkStudios/rust-gpu) compiles rust to spirv, and currently requires nightly rust. This example will only run on nightly, however **autograph** itself uses a separate crate to compile shaders and injects them into the source, to eliminate this dependency.

## Install Nightly
See the [rust-gpu toolchain](https://github.com/EmbarkStudios/rust-gpu/blob/main/rust-toolchain). The current toolchain can be installed with:
```
rustup toolchain install nightly-2021-06-09 --component rust-src rustc-dev llvm-tools-preview
```

## Set Override
Since this example will run on a specific channel, set the override so that it doesn't have to be specified each time:
```
rustup override set nightly-2021-06-09
```

# Run!
```
cargo run
```
This should print out:
```
[2] + [2] = [4]
```

# Final Notes
If you have any issues with this example please create an issue at https://github.com/charles-r-earp/autograph.
