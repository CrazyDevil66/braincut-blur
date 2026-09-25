mod blur;
mod config;
mod detection;
mod handlers;
mod image_ops;
mod models;
mod paths;
mod state;

use std::sync::Arc;

use axum::{
    routing::{delete, get, post},
    Router,
};
use tower_http::cors::CorsLayer;

use blur::new_cancel_flag;
use config::Config;
use handlers::{
    App,
    jobs::{blur_handler, cancel_handler, job_control_handler, render_handler},
    models::{api_config_get, api_config_set, api_models_activate, api_models_delete, api_models_get, api_models_install, api_models_refresh},
    system::{api_frame_handler, health_handler, status_handler, ui_handler},
};
use models::new_install_map;
use state::new_state;

fn build_router(app_state: App) -> Router {
    Router::new()
        .route("/",                     get(ui_handler))
        .route("/status",               get(status_handler))
        .route("/health",               get(health_handler))
        .route("/blur",                 post(blur_handler))
        .route("/cancel",               post(cancel_handler))
        .route("/render",               post(render_handler))
        .route("/job-control",          post(job_control_handler))
        .route("/api/models",           get(api_models_get))
        .route("/api/models/refresh",   post(api_models_refresh))
        .route("/api/models/install",   post(api_models_install))
        .route("/api/models/activate",  post(api_models_activate))
        .route("/api/models/:id",       delete(api_models_delete))
        .route("/api/frame",            get(api_frame_handler))
        .route("/api/config",           get(api_config_get).post(api_config_set))
        .layer(CorsLayer::permissive())
        .with_state(app_state)
}

fn startup_log(status: &state::SharedState, cfg: &Config) {
    state::log(status, "BrainCut Blur Service (Rust) gestartet");
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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    ort::init().with_name("BrainCutBlur").commit();

    let cfg = Config::from_env();
    let status = new_state();
    let cancel = new_cancel_flag();
    let install_progress = new_install_map();

    startup_log(&status, &cfg);

    let app_state = App { status, cancel, install_progress, cfg: Arc::new(cfg) };
    let router = build_router(app_state);

    let addr = "0.0.0.0:8080";
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
