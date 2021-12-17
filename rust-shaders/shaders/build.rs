use std::{fs, io, env, path::PathBuf};
use anyhow::Result;

fn buffer_load_types(ty: &'static str) -> [&'static str; 2] {
    match ty {
        "u8" => ["u8x4", "u32"],
        "u16" => ["u16x2", "u32"],
        "bf16" => ["bf16x2", "f32"],
        "u32" | "i32" | "f32" => [ty, ty],
        _ => todo!(),
    }
}

fn gen_scale_impls() -> Result<()> {
    let mut s = String::new();
    s.push_str("impl_scale!{\n");
    for x in ["u8", "u16", "bf16", "u32", "i32", "f32"] {
        let [x_buf, x_load] = buffer_load_types(x);
        for y in ["bf16", "u32", "i32", "f32"] {
            let [y_buf, y_load] = buffer_load_types(y);
            s.push_str(&format!("    scale_{x}_{y}<{x_buf}, {x_load}, {y_buf}, {y_load}>,\n"));
        }
    }
    s.push_str("}");
    let out_dir = env::var("OUT_DIR")?;
    let path = PathBuf::from(out_dir)
        .join("scale_impls.rs");
    fs::write(path, s)?;
    Ok(())
}

fn main() -> Result<()> {
    gen_scale_impls()?;
    Ok(())
}
