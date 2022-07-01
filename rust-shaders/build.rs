use spirv_builder::{SpirvBuilder, Capability, MetadataPrintout};
use std::{
    error::Error,
    path::PathBuf,
    fs,
};

fn validate_spirv(name: &str, spirv: &[u32]) -> Result<(), Box<dyn Error>> {
    use spirv_cross::{spirv::{self, ExecutionModel}, glsl, msl, hlsl};
    let module = spirv::Module::from_words(spirv);
    let shader_dir = PathBuf::from(std::env::var("OUT_DIR")?).join("shaders").join(name);
    fs::create_dir_all(&shader_dir)?;
    let mut glsl_ast = spirv::Ast::<glsl::Target>::parse(&module)
        .map_err(|e| format!("GLSL parsing of {name} failed: {e}", name=name, e=e))?;
    let mut glsl_compiler_options = glsl::CompilerOptions::default();
    //glsl_compiler_options.version = glsl::Version::V1_1;
    for entry_point in glsl_ast.get_entry_points()? {
        glsl_compiler_options.entry_point.replace((entry_point.name.clone(), ExecutionModel::GlCompute));
        glsl_ast.set_compiler_options(&glsl_compiler_options)?;
        let glsl = glsl_ast.compile()
            .map_err(|e| format!("GLSL compilation of {name}::{entry} failed: {e}", name=name, entry=&entry_point.name, e=e))?;
        let entry_name = entry_point.name.replace("::", "__");
        fs::write(shader_dir.join(entry_name).with_extension("glsl"), &glsl)?;
    }
    let mut metal_ast = spirv::Ast::<msl::Target>::parse(&module)
        .map_err(|e| format!("Metal parsing of {name} failed: {e}", name=name, e=e))?;
    let mut metal_compiler_options = msl::CompilerOptions::default();
    metal_compiler_options.version = msl::Version::V1_2;
    for entry_point in metal_ast.get_entry_points()? {
        metal_compiler_options.entry_point.replace((entry_point.name.clone(), ExecutionModel::GlCompute));
        metal_ast.set_compiler_options(&metal_compiler_options)?;
        let metal = metal_ast.compile()
            .map_err(|e| format!("Metal compilation of {name}::{entry} failed: {e}", name=name, entry=&entry_point.name, e=e))?;
        let entry_name = entry_point.name.replace("::", "__");
        fs::write(shader_dir.join(entry_name).with_extension("metal"), &metal)?;
    }
    let mut hlsl_ast = spirv::Ast::<hlsl::Target>::parse(&module)
        .map_err(|e| format!("HLSL parsing of {name} failed: {e}", name=name, e=e))?;
    let mut hlsl_compiler_options = hlsl::CompilerOptions::default();
    hlsl_compiler_options.shader_model = hlsl::ShaderModel::V5_1;
    for entry_point in hlsl_ast.get_entry_points()? {
        hlsl_compiler_options.entry_point.replace((entry_point.name.clone(), ExecutionModel::GlCompute));
        hlsl_ast.set_compiler_options(&hlsl_compiler_options)?;
        let hlsl = hlsl_ast.compile()
            .map_err(|e| format!("HLSL compilation of {name}::{entry} failed: {e}", name=name, entry=&entry_point.name, e=e))?;
        let entry_name = entry_point.name.replace("::", "__");
        fs::write(shader_dir.join(entry_name).with_extension("hlsl"), &hlsl)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let result = SpirvBuilder::new("shaders", "spirv-unknown-vulkan1.1")
        .capability(Capability::VulkanMemoryModelDeviceScopeKHR)
        .multimodule(true)
        .print_metadata(MetadataPrintout::DependencyOnly)
        .build()?;
    for (name, path) in result.module.unwrap_multi() {
        let rust_path = PathBuf::from("../")
            .join("src")
            .join("shaders")
            .join("rust");
        validate_spirv(&name, bytemuck::cast_slice(&fs::read(&path)?))?;
        fs::create_dir_all(&rust_path)?;
        let rust_shader_path = rust_path.join(&name.replace("::", "__")).with_extension("spv");
        println!("cargo:rerun-if-changed={}", rust_shader_path.to_str().unwrap());
        fs::copy(path, rust_shader_path)?;
    }
    Ok(())
}
