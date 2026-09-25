use serde_json::{json, Value};

use crate::{config::Config, state::SharedState};

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
                    let mut c = builtin_catalog(); c["source"] = json!("integriert (Parse-Fehler)"); c
                }
            },
            Err(e) => {
                crate::state::log(state, &format!("Katalog-Abruf fehlgeschlagen: {e}"));
                let mut c = builtin_catalog(); c["source"] = json!("integriert (Fehler beim Abruf)"); c
            }
        }
    };

    if let Ok(s) = serde_json::to_string_pretty(&cat) {
        let _ = std::fs::create_dir_all(&cfg.models_path);
        let _ = std::fs::write(cfg.models_path.join("catalog.json"), s);
    }
    cat
}

pub fn builtin_catalog() -> Value {
    json!({
        "models": [
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
                "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
            },
            {
                "id": "scrfd-500m",
                "name": "SCRFD 500M – Gesicht (schnell)",
                "type": "face",
                "format": "SCRFD",
                "size_mb": 1.7,
                "description": "InsightFace SCRFD_500M: sehr schnell, weniger präzise bei kleinen/seitlichen Gesichtern. Für Echtzeit mit begrenzter GPU.",
                "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
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
