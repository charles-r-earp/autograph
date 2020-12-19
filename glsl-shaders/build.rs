use shaderc::{CompileOptions, Compiler, ShaderKind, SourceLanguage};
use std::path::PathBuf;
use std::{env, fs};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

fn glsl_options() -> CompileOptions<'static> {
    let mut options = CompileOptions::new().unwrap();
    options.set_auto_bind_uniforms(true);
    options.set_source_language(SourceLanguage::GLSL);
    options
} 

fn compile_glsl(
    compiler: &mut Compiler,
    src: &str,
    name: &str,
    options: Option<&CompileOptions>,
) -> Result<()> {
    let artifact = compiler.compile_into_spirv(src, ShaderKind::Compute, name, "main", options)?;
    let glsl_path = PathBuf::from(env::var("AUTOGRAPH_DIR")?)
        .join("src")
        .join("shaders")
        .join("glsl");
    eprintln!("{:?}", glsl_path);
    fs::create_dir_all(&glsl_path)?;
    let fpath = glsl_path.join(&name).with_extension("spv");
    fs::write(&fpath, artifact.as_binary_u8())?;
    Ok(())
}

fn glsl_fill(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/fill.comp");
    // Note that these can be used for types of the same size, ie u32 and f32
    for (rust_ty, c_ty) in [("f32", "float"), ("f64", "double")].iter() {
        let mut options = glsl_options();
        options.add_macro_definition("T", Some(c_ty));
        compile_glsl(compiler, src, &format!("fill_{}", rust_ty), Some(&options))?;
    }
    Ok(())
}

fn glsl_gemm(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/gemm.comp");
    for (rust_ty, c_ty) in [("f32", "float"), ("f64", "double"), ("i32", "int")].iter() {
        let mut options = glsl_options();
        options.add_macro_definition("T", Some(c_ty));
        compile_glsl(compiler, src, &format!("gemm_{}", rust_ty), Some(&options))?;
    }
    for (rust_ty, c_ty) in [("f32", "float")].iter() {
        {
            // Relu
            let mut options = glsl_options();
            options.add_macro_definition("T", Some(c_ty));
            options.add_macro_definition("RELU", None);
            compile_glsl(
                compiler,
                src,
                &format!("gemm_relu_{}", rust_ty),
                Some(&options),
            )?;
        }
        {
            // Bias
            let mut options = glsl_options();
            options.add_macro_definition("T", Some(c_ty));
            options.add_macro_definition("BIAS", None);
            compile_glsl(
                compiler,
                src,
                &format!("gemm_bias_{}", rust_ty),
                Some(&options),
            )?;
        }
        {
            // Bias + Relu
            let mut options = glsl_options();
            options.add_macro_definition("T", Some(c_ty));
            options.add_macro_definition("BIAS", None);
            options.add_macro_definition("RELU", None);
            compile_glsl(
                compiler,
                src,
                &format!("gemm_bias_relu_{}", rust_ty),
                Some(&options),
            )?;
        }
    }
    Ok(())
}

pub fn main() -> Result<()> {
    let mut compiler = Compiler::new().unwrap();

    glsl_fill(&mut compiler)?;
    glsl_gemm(&mut compiler)?;

    Ok(())
}

