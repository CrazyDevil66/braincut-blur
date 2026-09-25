use std::io::{Read, Write};

use anyhow::Result;
use serde_json::json;

use crate::{config::Config, state::SharedState};

use super::InstallMap;

pub(super) const MAX_MODEL_BYTES: u64 = 500 * 1024 * 1024;

pub fn install_model_bg(
    model_id: String,
    url: String,
    hf_token: String,
    cfg: Config,
    progress: InstallMap,
    state: SharedState,
) {
    std::thread::spawn(move || {
        {
            let mut p = progress.lock().unwrap();
            p.insert(model_id.clone(), json!({ "status": "downloading", "pct": 0, "error": "" }));
        }

        let target     = cfg.models_path.join(format!("{model_id}.onnx"));
        let target_tmp = cfg.models_path.join(format!("{model_id}.onnx.tmp"));
        let zip_tmp    = cfg.models_path.join(format!("{model_id}.zip.tmp"));

        let result = (|| -> Result<()> {
            std::fs::create_dir_all(&cfg.models_path)?;

            let dl_path = if url.ends_with(".zip") { &zip_tmp } else { &target_tmp };
            let (is_zip, _) = download_to_file(&url, &hf_token, dl_path, &progress, &model_id)?;

            if is_zip {
                extract_onnx_from_zip(&zip_tmp, &target_tmp)?;
                let _ = std::fs::remove_file(&zip_tmp);
            }

            std::fs::rename(&target_tmp, &target)?;
            let size_bytes = target.metadata().map(|m| m.len()).unwrap_or(0);
            if size_bytes < 512 * 1024 {
                let _ = std::fs::remove_file(&target);
                anyhow::bail!(
                    "Heruntergeladene Datei zu klein ({} KB) – wahrscheinlich ein LFS-Pointer statt echtem Modell. Bitte direkte Download-URL verwenden.",
                    size_bytes / 1024
                );
            }
            let size_mb = size_bytes as f64 / 1_048_576.0;
            crate::state::log(&state, &format!("Modell installiert: {model_id} ({:.1} MB)", size_mb));
            Ok(())
        })();

        let _ = std::fs::remove_file(&target_tmp);
        let _ = std::fs::remove_file(&zip_tmp);

        let val = match result {
            Ok(()) => json!({ "status": "done", "pct": 100, "error": "" }),
            Err(e) => {
                let msg = e.to_string();
                let msg = &msg[..msg.len().min(300)];
                crate::state::log(&state, &format!("Modell-Installation fehlgeschlagen ({model_id}): {msg}"));
                json!({ "status": "error", "pct": 0, "error": msg })
            }
        };
        progress.lock().unwrap().insert(model_id, val);
    });
}

fn download_to_file(
    url: &str,
    hf_token: &str,
    dest: &std::path::Path,
    progress: &InstallMap,
    model_id: &str,
) -> Result<(bool, String)> {
    let mut req = ureq::get(url).timeout(std::time::Duration::from_secs(120));
    if !hf_token.is_empty() {
        req = req.set("Authorization", &format!("Bearer {hf_token}"));
    }

    let resp = req.call().map_err(|e| match &e {
        ureq::Error::Status(401, _) => {
            anyhow::anyhow!("401 Unauthorized – HuggingFace-Token erforderlich. Token im Einstellungen-Panel eingeben.")
        }
        ureq::Error::Status(code, r) => anyhow::anyhow!("HTTP {code}: {}", r.status_text()),
        _ => anyhow::anyhow!("{e}"),
    })?;

    let content_type = resp.content_type().to_owned();
    if content_type.contains("text/html") {
        anyhow::bail!("Unerwarteter Content-Type: {content_type} – kein Modell?");
    }
    let is_zip = url.ends_with(".zip") || content_type.contains("zip");

    let total: u64 = resp.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    if total > 0 && total > MAX_MODEL_BYTES {
        anyhow::bail!("Modell zu groß: {} MB (Limit: 500 MB)", total / 1_048_576);
    }

    let mut reader = resp.into_reader();
    let mut downloaded: u64 = 0;
    let mut chunk = vec![0u8; 65536];
    let mut file = std::fs::File::create(dest)?;

    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 { break; }
        file.write_all(&chunk[..n])?;
        downloaded += n as u64;
        if downloaded > MAX_MODEL_BYTES {
            anyhow::bail!("Modell überschreitet 500 MB – Download abgebrochen");
        }
        if total > 0 {
            let pct = ((downloaded as f64 / total as f64) * 99.0) as u64;
            progress.lock().unwrap().insert(model_id.to_owned(), json!({ "status": "downloading", "pct": pct, "error": "" }));
        }
    }
    Ok((is_zip, content_type))
}

fn extract_onnx_from_zip(
    zip_path: &std::path::Path,
    target_tmp: &std::path::Path,
) -> Result<()> {
    let zfile = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(zfile)?;
    let names: Vec<String> = archive.file_names().map(String::from).collect();
    let Some(name) = choose_onnx(&names) else {
        anyhow::bail!("Keine .onnx-Datei im ZIP-Archiv gefunden");
    };
    let mut entry = archive.by_name(&name)?;
    let mut out = std::fs::File::create(target_tmp)?;
    std::io::copy(&mut entry, &mut out)?;
    Ok(())
}

/// InsightFace-Pakete (buffalo_l, buffalo_sc) enthalten mehrere Modelle – der Gesichtsdetektor heißt `det_*.onnx`.
/// Ohne solchen Eintrag wird die einzige bzw. erste ONNX-Datei genommen.
fn choose_onnx(names: &[String]) -> Option<String> {
    let onnx: Vec<&String> = names.iter().filter(|n| n.ends_with(".onnx") && !n.contains("__MACOSX")).collect();
    let basename = |n: &str| n.rsplit('/').next().unwrap_or(n).to_owned();
    onnx.iter().find(|n| basename(n).starts_with("det_")).or_else(|| onnx.first()).map(|n| n.to_string())
}

#[cfg(test)]
mod tests {
    use super::choose_onnx;

    #[test]
    fn aus_buffalo_wird_der_detektor_gewaehlt() {
        let names: Vec<String> = ["genderage.onnx", "2d106det.onnx", "det_10g.onnx", "1k3d68.onnx", "w600k_r50.onnx"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(choose_onnx(&names).as_deref(), Some("det_10g.onnx"));
        let sc: Vec<String> = ["buffalo_sc/w600k_mbf.onnx", "buffalo_sc/det_500m.onnx"].iter().map(|s| s.to_string()).collect();
        assert_eq!(choose_onnx(&sc).as_deref(), Some("buffalo_sc/det_500m.onnx"));
    }

    #[test]
    fn ohne_detektor_die_erste_onnx() {
        let names: Vec<String> = ["readme.txt", "__MACOSX/x.onnx", "plates.onnx"].iter().map(|s| s.to_string()).collect();
        assert_eq!(choose_onnx(&names).as_deref(), Some("plates.onnx"));
        assert_eq!(choose_onnx(&["a.txt".to_string()]), None);
    }
}
