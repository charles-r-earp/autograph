use spirv_builder::{SpirvBuilder, ModuleResult};
/*
use rspirv::{
    dr::Loader,
    binary::{Parser, Disassemble},
};*/
use std::{
    error::Error,
    path::PathBuf,
    fs,
};

fn main() -> Result<(), Box<dyn Error>> {
    let result = SpirvBuilder::new("shaders", "spirv-unknown-vulkan1.0")
        .bindless(true)
        .build()?;
    match result.module {
        ModuleResult::SingleModule(path) => {
            let rust_path = PathBuf::from("../")
                .join("src")
                .join("shaders")
                .join("rust");
            fs::create_dir_all(&rust_path)?;
            let rust_shader_path = rust_path.join("core.spv");
            fs::copy(path, rust_shader_path)?;
        }
        e => unreachable!("{:?}", &e),
    }
    /*
    let mut loader = Loader::new();
    let spirv = std::fs::read("target/spirv-builder/spirv-unknown-vulkan1.0/release/deps/shaders.spv.dir/module")?;
    Parser::new(&spirv, &mut loader)
        .parse()
        .unwrap();
    let module = loader.module();
    println!("{}", module.disassemble());
    */
    Ok(())
}
