use shaderc::{CompileOptions, Compiler, ShaderKind, SourceLanguage};
use std::path::PathBuf;
use std::{env, fs};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

static NUM_TYPES: &[(&'static str, &'static str)] = &[
    ("bf16_as_f32", "uint16_t"), 
    ("u32", "uint"), 
    ("i32", "int"),
    ("f32", "float"), 
    ("u64", "uint64_t"),
    ("i64", "int64_t"),
    ("f64", "float64_t"),
];

static NN_TYPES: &[(&'static str, &'static str)] = &[
    ("bf16_as_f32", "uint16_t"), 
    ("f32", "float"), 
];

fn glsl_options() -> CompileOptions<'static> {
    let mut options = CompileOptions::new().unwrap();
    options.set_auto_bind_uniforms(true);
    options.set_source_language(SourceLanguage::GLSL);
    options
} 

#[allow(unused)]
#[derive(Clone, Copy)]
enum Behavior {
    Enable,
    Require,
    Warn,
    Disable
}

impl Behavior {
    fn to_str(&self) -> &'static str {
        use Behavior::*;
        match self {
            Enable => "enable",
            Require => "require",
            Warn => "warn",
            Disable => "disable",
        }
    }
}

fn glsl_extension(src: &mut String, extension: &str, behavior: Behavior) {
    let version_str = "#version 450";
    let index = src.find(version_str)
        .map(|i| i + version_str.len())
        .expect(src);
    src.insert_str(index, &format!("\n#extension {} : {}\n", extension, behavior.to_str()));     
}

fn add_load_store_macros(options: &mut CompileOptions, rust_ty: &str) {
    if rust_ty == "bf16_as_f32" {
        options.add_macro_definition("LOAD(x)",  Some("uintBitsToFloat(uint(x) << 16)"));
        options.add_macro_definition("STORE(x)", Some("uint16_t(floatBitsToUint(x) >> 16)")); 
    } else {
        options.add_macro_definition("LOAD(x)",  Some("x"));
        options.add_macro_definition("STORE(x)", Some("x"));        
    }
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
    for &(rust_ty, c_ty) in [("u8", "uint8_t"), ("u16", "uint16_t"), ("u32", "uint"), ("u64", "uint64_t")].iter() {
        let mut src = String::from(src);
        let mut options = glsl_options();
        if c_ty == "uint8_t" {
            glsl_extension(&mut src, "GL_EXT_shader_8bit_storage", Behavior::Require);
        } else if c_ty == "uint16_t" {
            glsl_extension(&mut src, "GL_EXT_shader_16bit_storage", Behavior::Require); 
        } else if c_ty == "uint64_t" {
            glsl_extension(&mut src, "GL_EXT_shader_explicit_arithmetic_types_int64", Behavior::Require); 
            options.add_macro_definition("Ts", Some("uint64_t"));
        } 
        options.add_macro_definition("T", Some(c_ty));
        compile_glsl(compiler, &src, &format!("fill_{}", rust_ty), Some(&options))?;
    }
    Ok(())
}

fn glsl_gemm(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/gemm.comp");
    for &(rust_ty, c_ty) in NUM_TYPES.iter() {
        let mut src = String::from(src);
        let mut options = glsl_options();
        if c_ty == "uint16_t" {
            glsl_extension(&mut src, "GL_EXT_shader_16bit_storage", Behavior::Require); 
        } else if c_ty == "uint64_t" || c_ty == "int64_t" {
            glsl_extension(&mut src, "GL_EXT_shader_explicit_arithmetic_types_int64", Behavior::Require); 
        } else if c_ty == "float64_t" {
            glsl_extension(&mut src, "GL_EXT_shader_explicit_arithmetic_types_float64", Behavior::Require); 
        } 
        if rust_ty == "bf16_as_f32" {
            options.add_macro_definition("BF16_AS_F32", None);
        } else if rust_ty == "u64" || rust_ty == "i64" || rust_ty == "f64" {
            options.add_macro_definition("T_64", None);
        }
        options.add_macro_definition("T", Some(c_ty));
        add_load_store_macros(&mut options, rust_ty);
        compile_glsl(compiler, &src, &format!("gemm_{}", rust_ty), Some(&options))?;
    }
    for &(rust_ty, c_ty) in NN_TYPES.iter() {
        let mut src = String::from(src);
        if c_ty == "uint16_t" {
            glsl_extension(&mut src, "GL_AMD_gpu_shader_int16", Behavior::Require); 
        } 
        {
            // Relu
            let mut options = glsl_options();
            if rust_ty == "bf16_as_f32" {
                options.add_macro_definition("BF16_AS_F32", None);
            }
            options.add_macro_definition("T", Some(c_ty));
            add_load_store_macros(&mut options, rust_ty);
            options.add_macro_definition("RELU", None);
            compile_glsl(
                compiler,
                &src,
                &format!("gemm_relu_{}", rust_ty),
                Some(&options),
            )?;
        }
        {
            // Bias
            let mut options = glsl_options();
            if rust_ty == "bf16_as_f32" {
                options.add_macro_definition("BF16_AS_F32", None);
            }
            options.add_macro_definition("T", Some(c_ty));
            add_load_store_macros(&mut options, rust_ty);
            options.add_macro_definition("BIAS", None);
            compile_glsl(
                compiler,
                &src,
                &format!("gemm_bias_{}", rust_ty),
                Some(&options),
            )?;
        }
        {
            // Bias + Relu
            let mut options = glsl_options();
            if rust_ty == "bf16_as_f32" {
                options.add_macro_definition("BF16_AS_F32", None);
            }
            options.add_macro_definition("T", Some(c_ty));
            add_load_store_macros(&mut options, rust_ty);
            options.add_macro_definition("BIAS", None);
            options.add_macro_definition("RELU", None);
            compile_glsl(
                compiler,
                &src,
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

