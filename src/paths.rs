use std::path::PathBuf;
use crate::config::Config;

pub fn remap(path: &str, cfg: &Config) -> String {
    if path.starts_with(&cfg.media_host_path) {
        path.replacen(&cfg.media_host_path, &cfg.container_root, 1)
    } else {
        path.to_string()
    }
}

pub fn validate_data_path(path: &str, cfg: &Config) -> anyhow::Result<String> {
    let p = PathBuf::from(path);
    let canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
    let root = PathBuf::from(&cfg.container_root)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(&cfg.container_root));
    if canonical.starts_with(&root) || path.starts_with(&cfg.container_root) {
        Ok(path.to_string())
    } else {
        anyhow::bail!("Pfad außerhalb von {}: {}", cfg.container_root, path)
    }
}

pub fn fix_status_url(url: &str, cfg: &Config) -> String {
    if url.is_empty() {
        return String::new();
    }
    if url.contains("127.0.0.1") || url.contains("localhost") {
        if !cfg.n8n_ip.is_empty() {
            return url
                .replace("127.0.0.1", &cfg.n8n_ip)
                .replace("localhost", &cfg.n8n_ip);
        }
    }
    url.to_string()
}
