use std::path::PathBuf;
use std::{env, fs};
use shaderc::{Compiler, CompileOptions, ShaderKind};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

struct GlslCompiler {
    glsl_dir: PathBuf,
    compiler: Compiler,
    options: CompileOptions<'static>,
}

impl GlslCompiler {
    fn new() -> Result<Self> {
        let glsl_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
            .join("src")
            .join("glsl");
        let compiler = Compiler::new()
            .ok_or("Unable to create Compiler!")?;
        let options = CompileOptions::new()
            .ok_or("Unable to create CompileOptions!")?;
        Ok(Self {
            glsl_dir,
            compiler,
            options,
        })
    }
    fn compile(&mut self, name: impl AsRef<str>) -> Result<()> {
        let name = name.as_ref();
        let relative_path: PathBuf = name.split("::")
            .collect();
        let src_path = self.glsl_dir.join(relative_path).with_extension("comp");
        let src = String::from_utf8(
            fs::read(&src_path)
                .or(Err(format!("Unable to read src_path: {:?}", src_path)))?
        )?;
        let artifact = self.compiler.compile_into_spirv(
            &src,
            ShaderKind::Compute,
            name,
            "main",
            Some(&self.options)
        )?;
        let glsl_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
            .join("target")
            .join("glsl");
        fs::create_dir_all(&glsl_path)?;
        let fname = name.replace("::", "__");
        let fpath = glsl_path.join(&fname);
        fs::write(&fpath, artifact.as_binary_u8())?;
        let name = format!("glsl::{}", &name);
        println!(
            "cargo:rustc-env={}={}", 
            name, 
            fpath.to_str().unwrap_or(&format!("Unable to convert path to str: {:?}", fpath))
        );
        Ok(())   
    } 
}

fn main() -> Result<()> {
    
    let mut compiler = GlslCompiler::new()?;
    
    let glsl_modules = [
        "fill::fill_u32"
    ];
    
    for module in glsl_modules.iter() {
        compiler.compile(module)?;
    }
      
    Ok(()) 
}
