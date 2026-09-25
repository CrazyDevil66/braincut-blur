mod catalog;
mod install;

pub use catalog::{get_catalog, refresh_catalog};
pub use install::install_model_bg;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use serde_json::{json, Value};

use crate::config::Config;

pub type InstallMap = Arc<Mutex<HashMap<String, Value>>>;

pub fn new_install_map() -> InstallMap {
    Arc::new(Mutex::new(HashMap::new()))
}

fn models_config_path(cfg: &Config) -> std::path::PathBuf {
    cfg.models_path.join("config.json")
}

pub fn load_model_config(cfg: &Config) -> Value {
    let p = models_config_path(cfg);
    if p.exists() {
        if let Ok(s) = std::fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                return v;
            }
        }
    }
    json!({ "face_model": "builtin-centerface", "plate_model": null })
}

pub fn save_model_config(cfg: &Config, val: &Value) -> Result<()> {
    std::fs::create_dir_all(&cfg.models_path)?;
    std::fs::write(models_config_path(cfg), serde_json::to_string_pretty(val)?)?;
    Ok(())
}

pub fn get_installed_models(cfg: &Config) -> Value {
    let mut result = json!({ "builtin-centerface": { "builtin": true, "size_mb": 5.3 } });
    if cfg.models_path.exists() {
        if let Ok(entries) = std::fs::read_dir(&cfg.models_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "onnx").unwrap_or(false) {
                    let id = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                    let size_mb = path.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1_048_576.0;
                    result[id] = json!({ "file": path.to_string_lossy(), "size_mb": (size_mb * 10.0).round() / 10.0 });
                }
            }
        }
    }
    result
}

pub fn validate_model_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 80
        && id.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '_' || c == '-')
}
