use axum::{extract::{Path, State}, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;

use crate::models;

use super::{json_err, json_ok, App};

pub async fn api_models_get(State(app): State<App>) -> Json<serde_json::Value> {
    let catalog   = models::get_catalog(&app.cfg);
    let installed = models::get_installed_models(&app.cfg);
    let cfg_val   = models::load_model_config(&app.cfg);
    let progress  = app.install_progress.lock().unwrap().clone();
    Json(json!({
        "catalog": catalog.get("models").cloned().unwrap_or(json!([])),
        "catalog_source": catalog.get("source").and_then(|v| v.as_str()).unwrap_or("integriert"),
        "installed": installed, "config": cfg_val, "install_progress": progress,
    }))
}

pub async fn api_models_refresh(State(app): State<App>) -> Json<serde_json::Value> {
    let cat = models::refresh_catalog(&app.cfg, &app.status);
    let count = cat.get("models").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    Json(json!({ "ok": true, "count": count, "source": cat.get("source") }))
}

pub async fn api_models_install(State(app): State<App>, Json(data): Json<serde_json::Value>) -> impl IntoResponse {
    let model_id = data.get("id").and_then(|v| v.as_str()).unwrap_or("").trim().to_owned();
    let url      = data.get("url").and_then(|v| v.as_str()).unwrap_or("").trim().to_owned();
    let hf_token = data.get("hf_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_owned();

    if model_id.is_empty() || url.is_empty() {
        return json_err(StatusCode::BAD_REQUEST, "id und url erforderlich");
    }
    if !models::validate_model_id(&model_id) {
        return json_err(StatusCode::BAD_REQUEST, "Ungültige model_id");
    }

    models::install_model_bg(model_id.clone(), url, hf_token, (*app.cfg).clone(), app.install_progress.clone(), app.status.clone());
    json_ok(json!({ "ok": true, "id": model_id }))
}

pub async fn api_models_activate(State(app): State<App>, Json(data): Json<serde_json::Value>) -> impl IntoResponse {
    let model_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let model_id   = data.get("id").cloned();

    if model_type != "face" && model_type != "plate" {
        return json_err(StatusCode::BAD_REQUEST, "type muss 'face' oder 'plate' sein");
    }
    if let Some(ref v) = model_id {
        if let Some(id) = v.as_str() {
            if id != "builtin-centerface" {
                let path = app.cfg.models_path.join(format!("{id}.onnx"));
                if !path.exists() {
                    return json_err(StatusCode::NOT_FOUND, &format!("Modell '{id}' nicht installiert – zuerst herunterladen"));
                }
            }
        }
    }

    let mut cfg_val = models::load_model_config(&app.cfg);
    if model_type == "face" {
        cfg_val["face_model"] = model_id.clone().unwrap_or(json!("builtin-centerface"));
    } else {
        cfg_val["plate_model"] = model_id.clone().unwrap_or(json!(null));
    }
    if let Err(e) = models::save_model_config(&app.cfg, &cfg_val) {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }

    let id_str = model_id.and_then(|v| v.as_str().map(|s| s.to_owned())).unwrap_or_else(|| "null".into());
    crate::state::log(&app.status, &format!("Aktives {model_type}-Modell geändert: {id_str}"));
    json_ok(json!({ "ok": true, "config": cfg_val }))
}

pub async fn api_models_delete(State(app): State<App>, Path(model_id): Path<String>) -> impl IntoResponse {
    if !models::validate_model_id(&model_id) {
        return json_err(StatusCode::BAD_REQUEST, "Ungültige model_id");
    }
    if model_id == "builtin-centerface" {
        return json_err(StatusCode::BAD_REQUEST, "Integriertes Modell kann nicht gelöscht werden");
    }
    let target = app.cfg.models_path.join(format!("{model_id}.onnx"));
    if !target.exists() {
        return json_err(StatusCode::NOT_FOUND, "Modell nicht gefunden");
    }
    if let Err(e) = std::fs::remove_file(&target) {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }

    let mut cfg_val = models::load_model_config(&app.cfg);
    let mut changed = false;
    if cfg_val.get("face_model").and_then(|v| v.as_str()) == Some(&model_id) {
        cfg_val["face_model"] = json!("builtin-centerface"); changed = true;
    }
    if cfg_val.get("plate_model").and_then(|v| v.as_str()) == Some(&model_id) {
        cfg_val["plate_model"] = json!(null); changed = true;
    }
    if changed { let _ = models::save_model_config(&app.cfg, &cfg_val); }

    crate::state::log(&app.status, &format!("Modell gelöscht: {model_id}"));
    json_ok(json!({ "ok": true }))
}

pub async fn api_config_get(State(app): State<App>) -> Json<serde_json::Value> {
    let cfg_val = models::load_model_config(&app.cfg);
    Json(json!({
        "detection_interval": cfg_val.get("detection_interval").and_then(|v| v.as_u64()).unwrap_or(app.cfg.detection_interval as u64),
        "plate_conf_thresh": cfg_val.get("plate_conf_thresh").and_then(|v| v.as_f64()).unwrap_or(app.cfg.plate_conf_thresh as f64),
        "face_conf_thresh": cfg_val.get("face_conf_thresh").and_then(|v| v.as_f64()).unwrap_or(app.cfg.face_conf_thresh as f64),
        "plate_tile_cols": cfg_val.get("plate_tile_cols").and_then(|v| v.as_u64()).unwrap_or(app.cfg.plate_tile_cols as u64),
        "plate_tile_rows": cfg_val.get("plate_tile_rows").and_then(|v| v.as_u64()).unwrap_or(app.cfg.plate_tile_rows as u64),
    }))
}

pub async fn api_config_set(State(app): State<App>, Json(data): Json<serde_json::Value>) -> impl IntoResponse {
    let mut cfg_val = models::load_model_config(&app.cfg);
    if let Some(v) = data.get("detection_interval").and_then(|v| v.as_u64()) {
        cfg_val["detection_interval"] = json!(v.clamp(1, 8));
    }
    if let Some(v) = data.get("plate_conf_thresh").and_then(|v| v.as_f64()) {
        cfg_val["plate_conf_thresh"] = json!((v.clamp(0.3, 0.8) * 100.0).round() / 100.0);
    }
    if let Some(v) = data.get("face_conf_thresh").and_then(|v| v.as_f64()) {
        cfg_val["face_conf_thresh"] = json!((v.clamp(0.05, 0.9) * 100.0).round() / 100.0);
    }
    for key in ["plate_tile_cols", "plate_tile_rows"] {
        if let Some(v) = data.get(key).and_then(|v| v.as_u64()) {
            cfg_val[key] = json!(v.clamp(1, 4));
        }
    }
    if let Err(e) = models::save_model_config(&app.cfg, &cfg_val) {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    json_ok(json!({ "ok": true }))
}
