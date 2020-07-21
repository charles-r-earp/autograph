#![allow(warnings)]
use std::{env, ffi::CString, process::Command};

fn main() {
    { // onednn cpp 
        cpp_build::Config::new()
            .include(env::var("DEP_DNNL_INCLUDE").unwrap())
            .build("src/lib.rs");
        println!("cargo:rustc-link-lib=static=dnnl");
        if let Ok(omp_lib) = env::var("DEP_DNNL_OMP_LIB") {
            println!("cargo:rustc-link-search={}", omp_lib);
        }
        if let Ok(omp) = env::var("DEP_DNNL_OMP") {
            println!("cargo:rustc-link-lib=dylib={}", omp);
        }
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
    println!("cargo:rustc-rerun-if-changed=build.rs");
}
