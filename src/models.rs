use std::{
    collections::HashMap,
    io::{Read, Write},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use serde_json::{json, Value};

use crate::{config::Config, state::SharedState};

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

pub fn get_catalog(cfg: &Config) -> Value {
    let p = cfg.models_path.join("catalog.json");
    if p.exists() {
        if let Ok(s) = std::fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                return v;
            }
        }
    }
    builtin_catalog()
}

pub fn refresh_catalog(cfg: &Config, state: &SharedState) -> Value {
    let url = std::env::var("MODEL_CATALOG_URL").unwrap_or_default();
    let cat = if url.is_empty() {
        let mut c = builtin_catalog();
        c["source"] = json!("integriert");
        c
    } else {
        match ureq::get(&url).timeout(std::time::Duration::from_secs(15)).call() {
            Ok(resp) => match resp.into_json::<Value>() {
                Ok(mut cat) => {
                    cat["source"] = json!(url);
                    let n = cat.get("models").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                    crate::state::log(state, &format!("Katalog aktualisiert: {} Modelle", n));
                    cat
                }
                Err(e) => {
                    crate::state::log(state, &format!("Katalog-Parse fehlgeschlagen: {e}"));
                    let mut c = builtin_catalog();
                    c["source"] = json!("integriert (Parse-Fehler)");
                    c
                }
            },
            Err(e) => {
                crate::state::log(state, &format!("Katalog-Abruf fehlgeschlagen: {e}"));
                let mut c = builtin_catalog();
                c["source"] = json!("integriert (Fehler beim Abruf)");
                c
            }
        }
    };

    if let Ok(s) = serde_json::to_string_pretty(&cat) {
        let _ = std::fs::create_dir_all(&cfg.models_path);
        let _ = std::fs::write(cfg.models_path.join("catalog.json"), s);
    }
    cat
}

pub fn get_installed_models(cfg: &Config) -> Value {
    let mut result = json!({
        "builtin-centerface": { "builtin": true, "size_mb": 5.3 }
    });
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

const MAX_MODEL_BYTES: u64 = 500 * 1024 * 1024;

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

        let target = cfg.models_path.join(format!("{model_id}.onnx"));
        let target_tmp = cfg.models_path.join(format!("{model_id}.onnx.tmp"));

        let result = (|| -> Result<()> {
            std::fs::create_dir_all(&cfg.models_path)?;

            let mut req = ureq::get(&url).timeout(std::time::Duration::from_secs(120));
            if !hf_token.is_empty() {
                req = req.set("Authorization", &format!("Bearer {hf_token}"));
            }

            let resp = req.call().map_err(|e| match &e {
                ureq::Error::Status(401, _) => {
                    anyhow::anyhow!("401 Unauthorized – HuggingFace-Token erforderlich. Token im Einstellungen-Panel eingeben.")
                }
                ureq::Error::Status(code, r) => {
                    anyhow::anyhow!("HTTP {code}: {}", r.status_text())
                }
                _ => anyhow::anyhow!("{e}"),
            })?;

            let content_type = resp.content_type().to_owned();
            if content_type.contains("text/html") || content_type.contains("text/plain") {
                anyhow::bail!("Unerwarteter Content-Type: {content_type} – kein Modell?");
            }

            let total: u64 = resp
                .header("content-length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);

            if total > 0 && total > MAX_MODEL_BYTES {
                anyhow::bail!("Modell zu groß: {} MB (Limit: 500 MB)", total / 1_048_576);
            }

            let mut reader = resp.into_reader();
            let mut file = std::fs::File::create(&target_tmp)?;
            let mut downloaded: u64 = 0;
            let mut chunk = vec![0u8; 65536];

            loop {
                let n = reader.read(&mut chunk)?;
                if n == 0 {
                    break;
                }
                file.write_all(&chunk[..n])?;
                downloaded += n as u64;
                if downloaded > MAX_MODEL_BYTES {
                    anyhow::bail!("Modell überschreitet 500 MB – Download abgebrochen");
                }
                if total > 0 {
                    let pct = ((downloaded as f64 / total as f64) * 99.0) as u64;
                    progress.lock().unwrap().insert(
                        model_id.clone(),
                        json!({ "status": "downloading", "pct": pct, "error": "" }),
                    );
                }
            }

            std::fs::rename(&target_tmp, &target)?;
            let size_mb = target.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1_048_576.0;
            crate::state::log(&state, &format!("Modell installiert: {model_id} ({:.1} MB)", size_mb));
            Ok(())
        })();

        let _ = std::fs::remove_file(&target_tmp);

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

fn builtin_catalog() -> Value {
    json!({
        "models": [
            // ── Gesichtsmodelle ──────────────────────────────────────────────
            {
                "id": "builtin-centerface",
                "name": "CenterFace (integriert)",
                "type": "face",
                "format": "CenterFace",
                "builtin": true,
                "size_mb": 5.3,
                "description": "Integriertes Gesichtsmodell – kein Download nötig. Gut für Frontalgesichter, bei Seitenprofil und weiten Entfernungen eingeschränkt."
            },
            {
                "id": "scrfd-10g",
                "name": "SCRFD 10G – Gesicht (empfohlen)",
                "type": "face",
                "format": "SCRFD",
                "size_mb": 16.9,
                "description": "InsightFace SCRFD_10G: deutlich besser bei kleinen, seitlichen und weit entfernten Gesichtern. Ideal für Dashcam-Material.",
                "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/det_10g.onnx"
            },
            {
                "id": "scrfd-500m",
                "name": "SCRFD 500M – Gesicht (schnell)",
                "type": "face",
                "format": "SCRFD",
                "size_mb": 1.7,
                "description": "InsightFace SCRFD_500M: sehr schnell, weniger präzise bei kleinen/seitlichen Gesichtern. Für Echtzeit mit begrenzter GPU.",
                "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_sc/det_500m.onnx"
            },
            {
                "id": "yolov8n-face",
                "name": "YOLOv8 Nano – Gesicht",
                "type": "face",
                "format": "YOLOv8",
                "size_mb": 6.2,
                "description": "Schnelles YOLO-Gesichtsmodell (Nano). Ähnliche Qualität wie CenterFace, anderer Ansatz."
            },
            {
                "id": "yolov8s-face",
                "name": "YOLOv8 Small – Gesicht",
                "type": "face",
                "format": "YOLOv8",
                "size_mb": 22.5,
                "description": "YOLO-Gesichtsmodell (Small): besser als Nano bei schwierigen Lichtverhältnissen und Winkeln."
            },
            // ── Kennzeichenmodelle ───────────────────────────────────────────
            {
                "id": "yolov8s-plates-eu",
                "name": "YOLOv8 Small – EU-Kennzeichen",
                "type": "plate",
                "format": "YOLOv8",
                "size_mb": 22.5,
                "description": "EU-Kennzeichenerkennung (YOLOv8 Small) – gutes Preis-Leistungs-Verhältnis."
            },
            {
                "id": "yolov8n-plates-eu",
                "name": "YOLOv8 Nano – EU-Kennzeichen",
                "type": "plate",
                "format": "YOLOv8",
                "size_mb": 6.2,
                "description": "EU-Kennzeichenerkennung (YOLOv8 Nano) – schneller, etwas ungenauer bei kleinen/weit entfernten Kennzeichen."
            },
            {
                "id": "yolov8m-plates-eu",
                "name": "YOLOv8 Medium – EU-Kennzeichen",
                "type": "plate",
                "format": "YOLOv8",
                "size_mb": 52.0,
                "description": "EU-Kennzeichenerkennung (YOLOv8 Medium) – höchste Genauigkeit, etwas mehr GPU-Bedarf."
            }
        ]
    })
}
