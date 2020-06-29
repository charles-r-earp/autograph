#![allow(warnings)]
use std::{env, ffi::CString, process::Command};

fn main() {
    {
        // dnnl
        let dst = cmake::Config::new("oneDNN")
            .define("DNNL_LIBRARY_TYPE", "STATIC")
            .define("DNNL_BUILD_EXAMPLES", "OFF")
            .define("DNNL_BUILD_TESTS", "OFF")
            .build();
        println!(
            "cargo:rustc-link-search=native={}",
            dst.join("lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            dst.join("lib64").display()
        );
        println!("cargo:rustc-link-lib=static=dnnl");
        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
        cpp_build::Config::new()
            .include(dst.join("include").display().to_string())
            .build("src/lib.rs");
    }
    #[cfg(feature = "cuda")]
    {
        {
            // compile custom cuda source
            println!("cargo:rustc-rerun-if-changed=src/cuda/kernels.cu");
            let status = Command::new("nvcc")
                .arg("src/cuda/kernels.cu")
                .arg("--ptx")
                .arg("-odir")
                .arg(env::var("OUT_DIR").unwrap())
                .status()
                .unwrap();
            assert!(status.success());
        }
    }
}
