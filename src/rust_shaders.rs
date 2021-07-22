use crate::{device::shader::Module, result::Result};
use once_cell::sync::OnceCell;

static CORE: OnceCell<Module> = OnceCell::new();

pub(crate) fn core() -> Result<&'static Module> {
    Ok(CORE.get_or_try_init(|| {
        bincode::deserialize(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rust/core.module"
        )))
    })?)
}
