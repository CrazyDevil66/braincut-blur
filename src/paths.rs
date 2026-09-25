use std::path::{Component, Path, PathBuf};
use crate::config::Config;

pub fn remap(path: &str, cfg: &Config) -> String {
    let host = cfg.media_host_path.trim_end_matches('/');
    match path.strip_prefix(host) {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => format!("{}{}", cfg.container_root, rest),
        _ => path.to_string(),
    }
}

// Löst `.` und `..` rein textuell auf, ohne das Dateisystem zu befragen.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => { out.pop(); }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// Wie Python `Path.resolve()`: Symlinks im existierenden Teil des Pfads auflösen,
// der noch nicht existierende Rest (z. B. eine neue Ausgabedatei) wird angehängt.
fn resolve(path: &Path) -> PathBuf {
    let norm = normalize(path);
    let mut existing = norm.as_path();
    let mut rest: Vec<&std::ffi::OsStr> = Vec::new();
    loop {
        if let Ok(canon) = existing.canonicalize() {
            let mut out = canon;
            for part in rest.iter().rev() { out.push(part); }
            return out;
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => { rest.push(name); existing = parent; }
            _ => return norm,
        }
    }
}

/// Prüft, dass `path` (bereits auf Container-Pfade umgesetzt) innerhalb von `container_root` liegt,
/// und liefert den aufgelösten Pfad zurück.
pub fn validate_data_path(path: &str, cfg: &Config) -> anyhow::Result<String> {
    if path.is_empty() {
        anyhow::bail!("Leerer Pfad nicht erlaubt");
    }
    let p = Path::new(path);
    if !p.is_absolute() {
        anyhow::bail!("Relativer Pfad nicht erlaubt: {path}");
    }
    let resolved = resolve(p);
    let root = resolve(Path::new(&cfg.container_root));
    if resolved == root || resolved.starts_with(&root) {
        Ok(resolved.to_string_lossy().into_owned())
    } else {
        anyhow::bail!("Pfad außerhalb von {} nicht erlaubt: {}", cfg.container_root, path)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_mit_root(root: &str) -> Config {
        let mut cfg = Config::from_env();
        cfg.container_root = root.to_string();
        cfg.media_host_path = "/mnt/user/n8n_automation/BrainCut".into();
        cfg
    }

    #[test]
    fn remap_setzt_host_pfad_um() {
        let cfg = cfg_mit_root("/data");
        assert_eq!(remap("/mnt/user/n8n_automation/BrainCut/01_upload/a.mp4", &cfg), "/data/01_upload/a.mp4");
        assert_eq!(remap("/mnt/user/n8n_automation/BrainCut", &cfg), "/data");
    }

    #[test]
    fn remap_ignoriert_aehnliche_praefixe() {
        let cfg = cfg_mit_root("/data");
        assert_eq!(remap("/mnt/user/n8n_automation/BrainCutAlt/a.mp4", &cfg), "/mnt/user/n8n_automation/BrainCutAlt/a.mp4");
    }

    #[test]
    fn validate_erlaubt_pfade_unter_root() {
        let tmp = std::env::temp_dir().join("bc-paths-test-erlaubt");
        std::fs::create_dir_all(tmp.join("01_upload")).unwrap();
        let cfg = cfg_mit_root(tmp.to_str().unwrap());
        let root = tmp.canonicalize().unwrap();
        let ok = validate_data_path(tmp.join("01_upload/neu.mp4").to_str().unwrap(), &cfg).unwrap();
        assert_eq!(PathBuf::from(ok), root.join("01_upload/neu.mp4"));
        let tief = validate_data_path(tmp.join("03_output/unter/neu.mp4").to_str().unwrap(), &cfg).unwrap();
        assert_eq!(PathBuf::from(tief), root.join("03_output/unter/neu.mp4"));
    }

    #[test]
    fn validate_verbietet_ausbruch_und_aehnliche_praefixe() {
        let tmp = std::env::temp_dir().join("bc-paths-test-verbot");
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = cfg_mit_root(tmp.to_str().unwrap());
        let root = tmp.to_str().unwrap();
        assert!(validate_data_path(&format!("{root}/../etc/passwd"), &cfg).is_err());
        assert!(validate_data_path(&format!("{root}/a/../../x.mp4"), &cfg).is_err());
        assert!(validate_data_path(&format!("{root}base/x.mp4"), &cfg).is_err());
        assert!(validate_data_path("relativ/x.mp4", &cfg).is_err());
        assert!(validate_data_path("", &cfg).is_err());
    }

    #[test]
    fn validate_loest_punkt_punkt_innerhalb_auf() {
        let tmp = std::env::temp_dir().join("bc-paths-test-punkte");
        std::fs::create_dir_all(tmp.join("a")).unwrap();
        let cfg = cfg_mit_root(tmp.to_str().unwrap());
        let ok = validate_data_path(&format!("{}/a/../b.mp4", tmp.to_str().unwrap()), &cfg).unwrap();
        assert_eq!(PathBuf::from(ok), tmp.canonicalize().unwrap().join("b.mp4"));
    }
}
