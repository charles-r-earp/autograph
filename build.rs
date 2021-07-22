//#![cfg_attr(feature = "compile-shaders", feature(rustc_private))] // Restricts to nightly compiler

use std::{env, fs, path::PathBuf};
use walkdir::WalkDir;

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
/*
#[cfg(feature = "compile-shaders")]
mod shaders {
    use super::Result;
    use std::{
        path::{Path, PathBuf},
        fs,
        process::Command,
        env,
    };

    /*static GLSL_SHADERS: &[&'static str] = &[
        "accuracy",
        "bias_backward",
        "binary",
        "buffer_macros",
        "cast",
        "cross_entropy_loss",
        "cross_entropy_loss_backward",
        "fill_u32",
        "fill_u64",
        "gemm",
        "index_select",
        "kmeans_distance",
        "kmeans_accumulate_next_centroids",
        "kmeans_update_centroids",
        "one_hot",
        "reduce_final",
    ];

    fn compile_glsl() -> Result<()> {
        let glsl_shaders_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("glsl-shaders");
        println!(
            "cargo:rerun-if-changed={}",
            glsl_shaders_path.join("build.rs").to_str().unwrap()
        );
        for shader in GLSL_SHADERS {
            let path = glsl_shaders_path
                .join("src")
                .join("glsl")
                .join(shader)
                .with_extension("spv");
            println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
        }
        let status = Command::new("cargo")
            .arg("build")
            .current_dir(glsl_shaders_path)
            .env("AUTOGRAPH_DIR", env::var("CARGO_MANIFEST_DIR")?)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err("Compiling glsl-shaders failed!".into())
        }
    }*/
    fn compile_rust() -> Result<()> {
        fn rust_toolchain(rust_shaders_path: &Path) -> Result<String> {
            let output = Command::new("cargo")
                .arg("run")
                .arg("-vv")
                .current_dir(rust_shaders_path.join("rust-toolchain"))
                .output()?;
            if output.status.success() {
                String::from_utf8(output.stdout)?
                    .strip_suffix('\n')
                    .ok_or("rust-toolchain invalid output!".into())
                    .map(str::to_string)
            } else {
                let err = String::from_utf8(output.stderr)?;
                let err = err.strip_prefix("Error: \"")
                    .ok_or_else(|| err.as_str())?
                    .strip_suffix("\"\n")
                    .ok_or_else(|| err.as_str())?;
                Err(err.into())
            }
        }
        let rust_shaders_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("rust-shaders");
        println!(
            "cargo:rerun-if-changed={}",
            rust_shaders_path.join("build.rs").to_str().unwrap()
        );
        println!(
            "cargo:rerun-if-changed={}",
            rust_shaders_path
                .join("src")
                .join("lib.rs")
                .to_str()
                .unwrap()
        );
        let toolchain = rust_toolchain(&rust_shaders_path)?;
        let status = Command::new("cargo")
            .arg(format!("+{}", toolchain))
            .arg("build")
            .current_dir(rust_shaders_path)
            .env("AUTOGRAPH_DIR", env::var("CARGO_MANIFEST_DIR")?)
            .status()?;
        if !status.success() {
            return Err("Compiling rust-shaders failed! ".into());
        }
        Ok(())
    }
    pub fn compile_shaders() -> Result<()> {
        //compile_glsl()?;
        compile_rust()?;
        generate_modules()?;
        Ok(())
    }
}*/

#[path = "src/device/shader.rs"]
#[allow(unused)]
mod shader;
use shader::Module;

fn generate_modules() -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    fs::create_dir_all(out_dir.join("shaders").join("rust"))?;
    for entry in WalkDir::new("src/shaders/rust")
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let ext = e.path().extension()?.to_str()?;
            if ext == "spv" {
                Some(e)
            } else {
                None
            }
        })
    {
        let spirv_path = entry.path();
        let module_path = out_dir
            .join(spirv_path.strip_prefix("src")?)
            .with_extension("module");
        let module = Module::from_spirv(fs::read(spirv_path)?)?;
        fs::write(module_path, bincode::serialize(&module)?)?;
    }
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn main() -> Result<()> {
    /*#[cfg(feature = "compile-shaders")]
    shaders::compile_shaders()?;*/
    generate_modules()?;
    Ok(())
}
