use autograph::{
    result::Result,
    device::shader::Module,
};
use spirv_builder::{SpirvBuilder, ModuleResult};
use std::{path::PathBuf, fs, env};

fn main() -> Result<()> {
    // Compile to spirv.
    let result = SpirvBuilder::new("shader", "spirv-unknown-vulkan1.1")
        .build()?;
    match result.module {
        // Get the module path.
        ModuleResult::SingleModule(path) => {
            let spirv = fs::read(path)?;
            /*{
                use rspirv::{
                    dr::Loader,
                    binary::{Parser, Disassemble},
                };
                let mut loader = Loader::new();
                Parser::new(&spirv, &mut loader)
                    .parse()
                    .unwrap();
                let module = loader.module();
                panic!("{}", module.disassemble());
            }*/
            // Create a Module from the spirv.
            let module = Module::from_spirv(spirv)?.with_name("shader");
            // Write the module to the target directory. It will be loaded in the crate via the include_bytes! macro.
            let shaders_path = PathBuf::from(env::var("OUT_DIR")?)
                .join("shaders");
            fs::create_dir_all(&shaders_path)?;
            fs::write(shaders_path.join("shader.bincode"), bincode::serialize(&module)?)?;
        }
        e => unreachable!("{:?}", e),
    }
    Ok(())
}
