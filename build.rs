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
        {
            // check if OpenMP is available
            cc::Build::new()
                .file(
                    dst.join("build")
                        .join("CMakeFiles")
                        .join("FindOpenMP")
                        .join("OpenMPTryFlag.c"),
                )
                .flag_if_supported("-fopenmp")
                // Don't link to the crate
                .cargo_metadata(false)
                .try_compile("openmp_try_flag")
                .map(|_| {
                    println!("cargo:rustc-link-lib=dylib=gomp");
                });
        }
        /*if cfg!(target_family = "unix") {
            if let Ok(output) = Command::new("locate")
                .arg("libomp.so")
                .output() {
                let output = String::from_utf8(output.stdout).unwrap();
                let mut found = false;
                for line in output.lines() {
                    if !line.is_empty() {
                        // this will be a possible path to libomp.so
                        found = true;
                        break;
                    }
                }
                if found {
                    println!("cargo:rustc-link-lib=dylib=gomp");
                }
                else {
                    println!("cargo:warning=Unable to located OpenMP.")
                }
            }
            else {
                println!("cargo:warning=Command locate libomp.so failed, unable to locate OpenMP.")
            }
        }
        else if cfg!(target_family = "windows") {
            if let Ok(output) = Command::new("dir")
                .arg("libomp.dll")
                .arg("/s")
                .output() {
                let output = String::from_utf8(output.stdout).unwrap();
                let mut found = false;
                for line in output.lines() {
                    if !line.is_empty() {
                        // this will be a possible path to libomp.dll
                        found = true;
                        break;
                    }
                }
                if found {
                    println!("cargo:rustc-link-lib=dylib=gomp");
                }
                else {
                    println!("cargo:warning=Unable to located OpenMP.")
                }
            }
        }
        else {
            // probably unreachable
            println!("cargo:warning=Unknown platform, unable to locate OpenMP.");
        }*/
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
    println!("cargo:rustc-rerun-if-changed=build.rs");
}
