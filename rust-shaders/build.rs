use spirv_builder::{SpirvBuilder, ModuleResult, Capability};
use std::{
    error::Error,
    path::PathBuf,
    fs,
};

fn validate_spirv(name: &str, spirv: &[u32]) -> Result<(), Box<dyn Error>> {
    use spirv_cross::{spirv, hlsl, msl};
    let module = spirv::Module::from_words(spirv);
    spirv::Ast::<msl::Target>::parse(&module)
        .map_err(|e| format!("Metal parsing of {name} failed: {e}", name=name, e=e))?
        .compile()
        .map_err(|e| format!("Metal compilation of {name} failed: {e}", name=name, e=e))?;
    spirv::Ast::<hlsl::Target>::parse(&module)
        .map_err(|e| format!("HLSL parsing of {name} failed: {e}", name=name, e=e))?
        .compile()
        .map_err(|e| format!("HLSL compilation of {name} failed: {e}", name=name, e=e))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let result = SpirvBuilder::new("shaders", "spirv-unknown-vulkan1.1")
        .capability(Capability::VulkanMemoryModelDeviceScopeKHR)
        .build()?;
    match result.module {
        ModuleResult::SingleModule(path) => {
            let rust_path = PathBuf::from("../")
                .join("src")
                .join("shaders")
                .join("rust");
            validate_spirv("core", bytemuck::cast_slice(&fs::read(&path)?))?;
            if !cfg!(feature = "dry-run") {
                fs::create_dir_all(&rust_path)?;
                let rust_shader_path = rust_path.join("core.spv");
                println!("cargo:rerun-if-changed={}", rust_shader_path.to_str().unwrap());
                fs::copy(path, rust_shader_path)?;
            }
        }
        e => unreachable!("{:?}", &e),
    }
    Ok(())
}
