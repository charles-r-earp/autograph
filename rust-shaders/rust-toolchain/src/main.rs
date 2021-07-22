use serde::Deserialize;
use std::{
    process::Command,
    result::Result,
    error::Error,
};

#[derive(Debug, Deserialize)]
struct Config {
    toolchain: Toolchain,
}

#[derive(Debug, Deserialize)]
struct Toolchain {
    channel: String,
    components: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let body = reqwest::get("https://raw.githubusercontent.com/EmbarkStudios/rust-gpu/main/rust-toolchain")
        .await?
        .text()
        .await?;
    dbg!(&body);
    let config: Config = toml::from_str(&body)?;
    let toolchain = config.toolchain;
    let output = Command::new("rustup")
        .arg("component")
        .arg("list")
        .arg("--installed")
        .arg("--toolchain")
        .arg(&toolchain.channel)
        .output()?;
    if output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        if !toolchain.components.iter()
            .any(|component| !stdout.lines().any(|x| x.starts_with::<&str>(component.as_str()))) {
            println!("{}", &toolchain.channel);
            return Ok(());
        }
    }
    Err(format!("run: rustup toolchain install {} --component {}",
        &toolchain.channel,
        toolchain.components.join(" "),
    ).into())
}
