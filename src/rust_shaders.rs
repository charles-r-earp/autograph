use crate::{device::shader::Module, result::Result};
use once_cell::sync::OnceCell;

static CORE: OnceCell<Module> = OnceCell::new();

#[doc(hidden)]
pub fn core() -> Result<&'static Module> {
    Ok(CORE.get_or_try_init(|| {
        bincode::deserialize(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rust/core.bincode"
        )))
    })?)
}
