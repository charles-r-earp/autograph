type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

#[cfg(feature = "compile-shaders")]
mod shaders {
    use super::Result;

    static GLSL_SHADERS: &[&'static str] = &[
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

    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

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
    }
    pub fn compile_shaders() -> Result<()> {
        compile_glsl()?;
        Ok(())
    }
}

#[allow(clippy::unnecessary_wraps)]
fn main() -> Result<()> {
    #[cfg(feature = "compile-shaders")]
    shaders::compile_shaders()?;
    Ok(())
}
