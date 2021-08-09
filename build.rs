use std::{collections::HashMap, env, fs, path::PathBuf};
use walkdir::WalkDir;

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

#[path = "src/device/shader.rs"]
#[allow(unused)]
mod shader;
use shader::Module;

fn generate_modules() -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    fs::create_dir_all(out_dir.join("shaders").join("glsl"))?;
    let mut glsl_modules = HashMap::new();
    for entry in WalkDir::new("src/shaders/glsl")
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let spirv_path = entry.path();
        let name: &str = spirv_path
            .components()
            .last()
            .unwrap()
            .as_os_str()
            .to_str()
            .unwrap()
            .strip_suffix(".spv")
            .unwrap();
        let module = Module::from_spirv(fs::read(spirv_path)?)?.with_name(name);
        glsl_modules.insert(name.to_string(), module);
    }
    //panic!("{:#?}", glsl_modules.keys().collect::<Vec<_>>());
    let glsl_modules_path = out_dir
        .join("shaders")
        .join("glsl")
        .join("modules")
        .with_extension("bincode");
    fs::write(glsl_modules_path, bincode::serialize(&glsl_modules)?)?;
    fs::create_dir_all(out_dir.join("shaders").join("rust"))?;
    for entry in WalkDir::new("src/shaders/rust")
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let spirv_path = entry.path();
        let module_path = out_dir
            .join(spirv_path.strip_prefix("src")?)
            .with_extension("bincode");
        let module = Module::from_spirv(fs::read(spirv_path)?)?;
        fs::write(module_path, bincode::serialize(&module)?)?;
    }
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn main() -> Result<()> {
    generate_modules()?;
    Ok(())
}
