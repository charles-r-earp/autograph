# Hello Compute!
This guide will explain how to write and execute your own shaders, in rust!

# Prerequisites
- [basic-setup](basic-setup.md)

# Setup

## Create the crate
Our crate will compute 2 + 2 = 4 on the device. First start a new crate:
- `cargo new hello-compute`
- `cd hello-compute`

Add **autograph** as a dependency and a build dependency:
- `cargo add autograph --git https://github.com/charles-r-earp/autograph/tree/engine`
- `cargo add autograph --build --git https://github.com/charles-r-earp/autograph/tree/engine`

## Create a builder crate
The builder crate will use [spirv-builder](https://github.com/EmbarkStudios/rust-gpu/tree/main/crates/spirv-builder) to compile the rust source to spirv.
- `cargo new builder --lib`
- `cd builder`
- `cargo add spirv-builder`

Add

## Create a shader crate
This will have our shader code.
- `cargo new shader --lib`
