mod blur;
mod config;
mod detection;
mod image_ops;
mod models;
mod paths;
mod state;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use blur::{new_cancel_flag, process_jobs, CancelFlag};
use config::Config;
use models::{new_install_map, InstallMap};
use state::{new_state, SharedState};

// ── App state ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct App {
    status: SharedState,
    cancel: CancelFlag,
    install_progress: InstallMap,
    cfg: Arc<Config>,
}

// ── HTML UI (embedded) ────────────────────────────────────────────────────────

static UI_HTML: &str = include_str!("../ui/index.html");

// ── Helpers ───────────────────────────────────────────────────────────────────

fn json_ok(v: Value) -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(v))
}

fn json_err(code: StatusCode, msg: &str) -> (StatusCode, Json<Value>) {
    (code, Json(json!({ "error": msg })))
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn ui_handler() -> Html<&'static str> {
    Html(UI_HTML)
}

async fn health_handler() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn status_handler(State(app): State<App>) -> Json<Value> {
    let s = app.status.lock().unwrap();
    let ts = s.started_at_ts;
    let elapsed = if ts > 0.0 && s.state != "idle" {
        (now_unix() - ts) as u64
    } else {
        0
    };
    let log_vec: Vec<&str> = s.log.iter().map(|l| l.as_str()).collect();
    Json(json!({
        "state": s.state,
        "current": s.current,
        "total": s.total,
        "name": s.name,
        "error": s.error,
        "frame_current": s.frame_current,
        "frame_total": s.frame_total,
        "frame_pct": s.frame_pct,
        "eta_seconds": s.eta_seconds,
        "face_count": s.face_count,
        "plate_count": s.plate_count,
        "hw_nvdec": s.hw_nvdec,
        "hw_nvenc": s.hw_nvenc,
        "hw_trt": s.hw_trt,
        "sub_state": s.sub_state,
        "out_name": s.out_name,
        "started_at": s.started_at,
        "started_at_ts": s.started_at_ts,
        "elapsed_seconds": elapsed,
        "logs": log_vec,
    }))
}

async fn blur_handler(
    State(app): State<App>,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    let jobs = data.get("jobs").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let resume_url = data.get("resumeUrl").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let status_url = data.get("statusUrl").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let full_job = data.get("fullJob").cloned();

    {
        let mut s = app.status.lock().unwrap();
        if s.state != "idle" {
            let state_val = s.state.clone();
            return (
                StatusCode::CONFLICT,
                Json(json!({ "error": "Job läuft bereits", "state": state_val })),
            );
        }
        s.state = "queued".into();
    }

    state::log(&app.status, &format!("Blur-Auftrag empfangen: {} Job(s)", jobs.len()));

    let count = jobs.len();
    let app2 = app.clone();
    std::thread::spawn(move || {
        process_jobs(
            jobs,
            resume_url,
            status_url,
            full_job,
            app2.status,
            app2.cancel,
            (*app2.cfg).clone(),
        );
    });

    (StatusCode::OK, Json(json!({ "status": "queued", "count": count })))
}

async fn cancel_handler(State(app): State<App>) -> Json<Value> {
    use std::sync::atomic::Ordering;
    app.cancel.store(true, Ordering::Relaxed);
    state::log(&app.status, "Abbruch angefordert");
    Json(json!({ "status": "cancel_requested" }))
}

async fn run_shell_handler() -> impl IntoResponse {
    (
        StatusCode::GONE,
        Json(json!({ "error": "/run-shell wurde aus Sicherheitsgründen entfernt." })),
    )
}

// Legacy render endpoint – wraps blur pipeline for render-mode jobs
async fn render_handler(
    State(app): State<App>,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    {
        let mut s = app.status.lock().unwrap();
        if s.state != "idle" {
            let state_val = s.state.clone();
            return (
                StatusCode::CONFLICT,
                Json(json!({ "error": "Job läuft bereits", "state": state_val })),
            );
        }
        s.state = "queued".into();
    }
    // Re-use blur pipeline with jobs wrapped in render format
    let jobs = data.get("jobs").and_then(|v| v.as_array()).cloned().unwrap_or_else(|| {
        // Legacy single-job format
        if data.get("input_path").is_some() { vec![data.clone()] } else { vec![] }
    });
    let resume_url = data.get("resumeUrl").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let status_url = data.get("statusUrl").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let count = jobs.len();
    let app2 = app.clone();
    std::thread::spawn(move || {
        process_jobs(jobs, resume_url, status_url, None, app2.status, app2.cancel, (*app2.cfg).clone());
    });
    (StatusCode::OK, Json(json!({ "status": "queued", "count": count })))
}

async fn job_control_handler(
    State(app): State<App>,
    Json(data): Json<Value>,
) -> Json<Value> {
    let action = data.get("action").and_then(|v| v.as_str()).unwrap_or("status");
    if action == "cancel" {
        use std::sync::atomic::Ordering;
        app.cancel.store(true, Ordering::Relaxed);
        state::log(&app.status, "Abbruch angefordert (job-control)");
        let s = app.status.lock().unwrap();
        return Json(json!({ "action": "cancel", "state": s.state, "name": s.name }));
    }
    let s = app.status.lock().unwrap();
    Json(json!({ "action": "status", "state": s.state, "name": s.name }))
}

// ── Model management ──────────────────────────────────────────────────────────

async fn api_models_get(State(app): State<App>) -> Json<Value> {
    let catalog = models::get_catalog(&app.cfg);
    let installed = models::get_installed_models(&app.cfg);
    let cfg_val = models::load_model_config(&app.cfg);
    let progress = app.install_progress.lock().unwrap().clone();
    Json(json!({
        "catalog": catalog.get("models").cloned().unwrap_or(json!([])),
        "catalog_source": catalog.get("source").and_then(|v| v.as_str()).unwrap_or("integriert"),
        "installed": installed,
        "config": cfg_val,
        "install_progress": progress,
    }))
}

async fn api_models_refresh(State(app): State<App>) -> Json<Value> {
    let cat = models::refresh_catalog(&app.cfg, &app.status);
    let count = cat.get("models").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    Json(json!({ "ok": true, "count": count, "source": cat.get("source") }))
}

async fn api_models_install(
    State(app): State<App>,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    let model_id = data.get("id").and_then(|v| v.as_str()).unwrap_or("").trim().to_owned();
    let url = data.get("url").and_then(|v| v.as_str()).unwrap_or("").trim().to_owned();
    let hf_token = data.get("hf_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_owned();

    if model_id.is_empty() || url.is_empty() {
        return json_err(StatusCode::BAD_REQUEST, "id und url erforderlich");
    }
    if !models::validate_model_id(&model_id) {
        return json_err(StatusCode::BAD_REQUEST, "Ungültige model_id");
    }

    models::install_model_bg(
        model_id.clone(),
        url,
        hf_token,
        (*app.cfg).clone(),
        app.install_progress.clone(),
        app.status.clone(),
    );

    json_ok(json!({ "ok": true, "id": model_id }))
}

async fn api_models_activate(
    State(app): State<App>,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    let model_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let model_id = data.get("id").cloned();

    if model_type != "face" && model_type != "plate" {
        return json_err(StatusCode::BAD_REQUEST, "type muss 'face' oder 'plate' sein");
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
    state::log(&app.status, &format!("Aktives {model_type}-Modell geändert: {id_str}"));
    json_ok(json!({ "ok": true, "config": cfg_val }))
}

async fn api_models_delete(
    State(app): State<App>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
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
        cfg_val["face_model"] = json!("builtin-centerface");
        changed = true;
    }
    if cfg_val.get("plate_model").and_then(|v| v.as_str()) == Some(&model_id) {
        cfg_val["plate_model"] = json!(null);
        changed = true;
    }
    if changed {
        let _ = models::save_model_config(&app.cfg, &cfg_val);
    }
    state::log(&app.status, &format!("Modell gelöscht: {model_id}"));
    json_ok(json!({ "ok": true }))
}

async fn api_config_get(State(app): State<App>) -> Json<Value> {
    let cfg_val = models::load_model_config(&app.cfg);
    Json(json!({
        "detection_interval": cfg_val.get("detection_interval").and_then(|v| v.as_u64()).unwrap_or(app.cfg.detection_interval as u64),
        "plate_conf_thresh": cfg_val.get("plate_conf_thresh").and_then(|v| v.as_f64()).unwrap_or(app.cfg.plate_conf_thresh as f64),
        "face_conf_thresh": cfg_val.get("face_conf_thresh").and_then(|v| v.as_f64()).unwrap_or(app.cfg.face_conf_thresh as f64),
    }))
}

async fn api_config_set(
    State(app): State<App>,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    let mut cfg_val = models::load_model_config(&app.cfg);
    if let Some(v) = data.get("detection_interval").and_then(|v| v.as_u64()) {
        cfg_val["detection_interval"] = json!(v.clamp(1, 8));
    }
    if let Some(v) = data.get("plate_conf_thresh").and_then(|v| v.as_f64()) {
        let clamped = (v.clamp(0.3, 0.8) * 100.0).round() / 100.0;
        cfg_val["plate_conf_thresh"] = json!(clamped);
    }
    if let Some(v) = data.get("face_conf_thresh").and_then(|v| v.as_f64()) {
        let clamped = (v.clamp(0.3, 0.9) * 100.0).round() / 100.0;
        cfg_val["face_conf_thresh"] = json!(clamped);
    }
    if let Err(e) = models::save_model_config(&app.cfg, &cfg_val) {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    json_ok(json!({ "ok": true }))
}

// ── Startup ───────────────────────────────────────────────────────────────────

fn startup_log(status: &SharedState, cfg: &Config) {
    state::log(status, &format!("BrainCut Blur Service (Rust) gestartet"));
    state::log(status, &format!("Pfad-Mapping: {} → {}", cfg.media_host_path, cfg.container_root));
    state::log(status, &format!("Modell-Pfad: {}", cfg.models_path.display()));
    state::log(status, &format!("CenterFace-Modell: {}", cfg.centerface_model.display()));
    if !cfg.n8n_ip.is_empty() {
        state::log(status, &format!("N8N-Server: {}:{}", cfg.n8n_ip, cfg.n8n_port));
    } else {
        state::log(status, "N8N_SERVER_IP nicht gesetzt – Completion-Webhook deaktiviert");
    }
    if detection::check_nvdec() {
        state::log(status, "GPU: NVDEC (cuda hwaccel) verfügbar");
    } else {
        state::log(status, "GPU: NVDEC nicht verfügbar – Software-Decode");
    }
    if detection::check_nvenc() {
        state::log(status, "GPU: NVENC (h264_nvenc) verfügbar");
    } else {
        state::log(status, "GPU: NVENC nicht verfügbar – libx264");
    }
    let cf_exists = cfg.centerface_model.exists();
    state::log(status, &format!("CenterFace-ONNX: {}", if cf_exists { "gefunden" } else { "NICHT GEFUNDEN" }));
    let model_cfg = models::load_model_config(cfg);
    state::log(status, &format!("Aktives Gesichts-Modell: {}", model_cfg.get("face_model").and_then(|v| v.as_str()).unwrap_or("builtin-centerface")));
    state::log(status, &format!("Aktives Kennzeichen-Modell: {}", model_cfg.get("plate_model").and_then(|v| v.as_str()).unwrap_or("–")));
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Initialize ORT (idempotent)
    // commit() returns bool in ort rc.12
    ort::init().with_name("BrainCutBlur").commit();

    let cfg = Config::from_env();
    let status = new_state();
    let cancel = new_cancel_flag();
    let install_progress = new_install_map();

    startup_log(&status, &cfg);

    let app_state = App {
        status,
        cancel,
        install_progress,
        cfg: Arc::new(cfg),
    };

    let router = Router::new()
        .route("/", get(ui_handler))
        .route("/status", get(status_handler))
        .route("/health", get(health_handler))
        .route("/blur", post(blur_handler))
        .route("/cancel", post(cancel_handler))
        .route("/render", post(render_handler))
        .route("/job-control", post(job_control_handler))
        .route("/run-shell", post(run_shell_handler))
        .route("/api/models", get(api_models_get))
        .route("/api/models/refresh", post(api_models_refresh))
        .route("/api/models/install", post(api_models_install))
        .route("/api/models/activate", post(api_models_activate))
        .route("/api/models/:id", delete(api_models_delete))
        .route("/api/config", get(api_config_get).post(api_config_set))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let addr = "0.0.0.0:8080";
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
