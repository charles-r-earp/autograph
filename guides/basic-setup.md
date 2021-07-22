# Basic Setup
This guide will explain how to use **autograph** in your crate.

# Install drivers
First, ensure that your graphics drivers are up-to-date.

If using Vulkan on Debian platforms, you can use vulkan-utils to check that the device is visible. If vulkan-utils is not installed you can install it with `sudo apt install vulkan-utils`.

Here we are piping the output to less so that we can scroll from the top.
```
vulkaninfo | less
```
Scroll down until you find something like this:
```
Device Properties and Extensions :
==================================
GPU0
VkPhysicalDeviceProperties:
===========================
        apiVersion     = 0x402085  (1.2.133)
        driverVersion  = 1889386688 (0x709dc0c0)
        vendorID       = 0x10de
        deviceID       = 0x1c20
        deviceType     = DISCRETE_GPU
        deviceName     = GeForce GTX 1060 with Max-Q Design
```
Press q to exit less.

# Install Rust
See https://www.rust-lang.org/learn/get-started.

Once you have Cargo installed, you should be able to run:
```
cargo --version
```

# Install Cargo-Edit
A handy tool for adding dependencies https://github.com/killercup/cargo-edit.
```
cargo install cargo-edit
```

# Create a new crate
```
cargo new basic-setup
```

# Add **autograph** as a dependency
```
cargo add autograph --git https://github.com/charles-r-earp/autograph/tree/engine
```
This will modify the Cargo.toml:
```
[dependencies]
autograph = { git = "https://github.com/charles-r-earp/autograph/tree/engine" }
```

# Create a device.
In main.rs, replace the generated code with the following:
```
use autograph::{Result, device::Device};

fn main() -> Result<()> {
    let device = Device::new()?;
    dbg!(&device);
    dbg!(device.info());
    Ok(())
}
```

# Compile and run!
```
cargo run
```
You should see something like:
```
[src/main.rs:5] &device = Device(0)
[src/main.rs:6] device.info() = Some(
    DeviceInfo {
        api: Vulkan,
        adapter_info: AdapterInfo {
            name: "GeForce GTX 1060 with Max-Q Design",
            vendor: 4318,
            device: 7200,
            device_type: DiscreteGpu,
        },
        device_memory: 65536000000,
    },
)
```
`Device` implements `Debug` by printing out either `Host` or `Device(id)`, where id is a unique identifier. The `.info()` method returns a `DeviceInfo` struct which has useful information.

# Optional: Create a device with a builder.
For more control, or to get all the devices available, use `Device::builder_iter()`.
Replace the body of main with:
```
let builders: Vec<_> = Device::builder_iter().collect();
dbg!(builders);
OK(())
```
Now with `cargo run`, it should print a list. To get the second device for example:
```
let device = Device::builder_iter()
    .skip(1)
    .next()
    .expect("Device not found!")
    .build()?;
Ok(())
```

# Final Notes
More information can be found in the docs for the `device` module. If you have an problems with this guide please create an issue!
