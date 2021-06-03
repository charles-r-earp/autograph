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
        let components: Vec<String> = toolchain.components.into_iter()
            .filter(|component| {
                !stdout.lines().any(|x| x.starts_with::<&str>(component.as_str()))
            })
            .collect();
        if components.is_empty() {
            println!("{}", &toolchain.channel);
            Ok(())
        } else {
            Err(format!("run: rustup toolchain install {} --component {}",
                &toolchain.channel,
                components.join(" "),
            ).into())
        }
    } else {
        Err(String::from_utf8(output.stderr)?.into())
    }
}
