use crate::{device::shader::Module, result::Result};
use once_cell::sync::OnceCell;

static CORE: OnceCell<Module> = OnceCell::new();

pub(crate) fn core() -> Result<&'static Module> {
    Ok(CORE.get_or_try_init(|| {
        bincode::deserialize(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rust/core.bincode"
        )))
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_to_metal() -> Result<()> {
        core()?.to_metal()?;
        Ok(())
    }

    #[test]
    fn core_to_hlsl() -> Result<()> {
        core()?.to_hlsl()?;
        Ok(())
    }
}
