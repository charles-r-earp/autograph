type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

#[cfg(feature = "compile-shaders")]
mod shaders {
    use super::Result;

    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    fn compile_glsl() -> Result<()> {
        let glsl_shaders_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("glsl-shaders");
        println!(
            "cargo:rerun-if-changed={}",
            glsl_shaders_path.join("build.rs").to_str().unwrap()
        );
        for shader in ["buffer_macros", "fill_u32", "fill_u64", "gemm"].iter() {
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

fn main() -> Result<()> {
    #[cfg(feature = "compile-shaders")]
    shaders::compile_shaders()?;

    Ok(())
}
