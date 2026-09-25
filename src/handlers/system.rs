use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    Json,
};
use serde_json::json;

use super::{now_unix, App};

static UI_HTML: &str = include_str!("../../ui/index.html");

pub async fn ui_handler() -> Html<&'static str> {
    Html(UI_HTML)
}

pub async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

pub async fn status_handler(State(app): State<App>) -> Json<serde_json::Value> {
    let s = app.status.lock().unwrap();
    let ts = s.started_at_ts;
    let elapsed = if ts > 0.0 && s.state != "idle" { (now_unix() - ts) as u64 } else { 0 };
    let log_vec: Vec<&str> = s.log.iter().map(|l| l.as_str()).collect();
    Json(json!({
        "state": s.state, "current": s.current, "total": s.total,
        "name": s.name, "error": s.error,
        "frame_current": s.frame_current, "frame_total": s.frame_total,
        "frame_pct": s.frame_pct, "eta_seconds": s.eta_seconds,
        "face_count": s.face_count, "plate_count": s.plate_count,
        "hw_nvdec": s.hw_nvdec, "hw_nvenc": s.hw_nvenc, "hw_trt": s.hw_trt,
        "sub_state": s.sub_state, "out_name": s.out_name,
        "started_at": s.started_at, "started_at_ts": s.started_at_ts,
        "elapsed_seconds": elapsed, "logs": log_vec,
    }))
}

pub async fn api_frame_handler(State(app): State<App>) -> impl IntoResponse {
    let jpeg = app.status.lock().unwrap().preview_jpeg.clone();
    if jpeg.is_empty() {
        return (StatusCode::NO_CONTENT, [("content-type", "image/jpeg")], Vec::new());
    }
    (StatusCode::OK, [("content-type", "image/jpeg")], jpeg)
}
