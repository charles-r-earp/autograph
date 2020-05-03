

fn main() {
  let dst = cmake::Config::new("oneDNN")
    .define("DNNL_BUILD_EXAMPLES", "OFF")
    .define("DNNL_BUILD_TESTS", "OFF")
    .build();
  println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
  println!("cargo:rustc-link-lib=dylib=dnnl");
  
  cpp_build::Config::new()
    .include(dst.join("include").display().to_string())
    .build("src/lib.rs");
}
