#![allow(warnings)]
use std::{env, process::Command, ffi::CString};
use static_assertions::assert_cfg;

fn main() {
  { // dnnl
    let dst = cmake::Config::new("oneDNN")
      .define("DNNL_LIBRARY_TYPE", "STATIC")
      .define("DNNL_BUILD_EXAMPLES", "OFF")
      .define("DNNL_BUILD_TESTS", "OFF")
      .pic(true)
      .build();
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=dnnl");
    cpp_build::Config::new()
      .include(dst.join("include").display().to_string())
      .build("src/lib.rs");
  }
  {
    // gomp for dnnl
    assert_cfg!(target_os = "linux", "Only Linux supported!");
    if cfg!(target_os = "linux") {
      let machine = Command::new("gcc")
        .arg("-dumpmachine")
        .output()
        .unwrap()
        .stdout;
      let machine = CString::new(machine)
        .unwrap()
        .into_string()
        .unwrap();
      let machine = machine.lines()
        .next()
        .unwrap();
      let version = Command::new("gcc")
        .arg("-dumpversion")
        .output()
        .unwrap()
        .stdout;
      let version = CString::new(version)
        .unwrap()
        .into_string()
        .unwrap();
      let version = version.lines()
        .next()
        .unwrap();
      println!("cargo:rustc-link-search=native=/usr/lib/gcc/{}/{}", machine, version);
      println!("cargo:rustc-link-lib=static=gomp");
    }
  }
    
  #[cfg(feature="cuda")]
  {
    //println!("cargo:rustc-link-lib=dylib=culibos");
    //println!("cargo:rustc-link-lib=dylib=cudart");
    { // compile custom cuda source
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
