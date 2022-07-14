use crate::{
    device::{builders::ComputePassBuilder, shader::Module},
    result::Result,
};
use anyhow::anyhow;
use once_cell::sync::OnceCell;
use std::collections::HashMap;

static MODULES: OnceCell<HashMap<&'static str, Module>> = OnceCell::new();

fn modules() -> Result<&'static HashMap<&'static str, Module>> {
    Ok(MODULES.get_or_try_init(|| {
        bincode::deserialize(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rust/modules.bincode",
        )))
    })?)
}

pub(crate) fn module(name: impl AsRef<str>) -> Result<&'static Module> {
    let name = name.as_ref();
    let module = modules()?
        .get(name)
        .ok_or_else(|| anyhow!("Module {:?} not found!", name))?;
    Ok(module)
}

pub(crate) fn compute_pass(name: impl AsRef<str>) -> Result<ComputePassBuilder<'static, 'static>> {
    module(name.as_ref())?.compute_pass(name.as_ref())
}
