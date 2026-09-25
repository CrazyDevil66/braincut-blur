use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;

use crate::blur::process_jobs;

use super::App;

pub async fn blur_handler(State(app): State<App>, Json(data): Json<serde_json::Value>) -> impl IntoResponse {
    let jobs = data.get("jobs").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let resume_url = data.get("resumeUrl").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let status_url = data.get("statusUrl").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let full_job = data.get("fullJob").cloned();

    {
        let mut s = app.status.lock().unwrap();
        if s.state != "idle" {
            let state_val = s.state.clone();
            return (StatusCode::CONFLICT, Json(json!({ "error": "Job läuft bereits", "state": state_val })));
        }
        s.state = "queued".into();
    }

    crate::state::log(&app.status, &format!("Blur-Auftrag empfangen: {} Job(s)", jobs.len()));
    let count = jobs.len();
    let app2 = app.clone();
    std::thread::spawn(move || {
        process_jobs(jobs, resume_url, status_url, full_job, app2.status, app2.cancel, (*app2.cfg).clone());
    });
    (StatusCode::OK, Json(json!({ "status": "queued", "count": count })))
}

pub async fn cancel_handler(State(app): State<App>) -> Json<serde_json::Value> {
    use std::sync::atomic::Ordering;
    app.cancel.store(true, Ordering::Relaxed);
    crate::state::log(&app.status, "Abbruch angefordert");
    Json(json!({ "status": "cancel_requested" }))
}

/// Rendert synchron wie der frühere Python-Service: WF3 wartet auf die Antwort.
/// Fehler kommen als `{ success: false, error }` mit Status 200, damit WF3 sie auswerten kann.
pub async fn render_handler(State(app): State<App>, Json(data): Json<serde_json::Value>) -> impl IntoResponse {
    {
        let mut s = app.status.lock().unwrap();
        if s.state != "idle" {
            let state_val = s.state.clone();
            return (StatusCode::CONFLICT, Json(json!({ "error": "Job läuft bereits", "state": state_val })));
        }
        s.state = "queued".into();
    }
    let (status, cancel, cfg) = (app.status.clone(), app.cancel.clone(), app.cfg.clone());
    let result = tokio::task::spawn_blocking(move || crate::render::run_render(&data, &status, &cancel, &cfg)).await;
    let body = result.unwrap_or_else(|e| {
        let msg = format!("Render-Task abgestürzt: {e}");
        {
            let mut s = app.status.lock().unwrap_or_else(|p| p.into_inner());
            s.state = "idle".into();
            s.error = msg.clone();
        }
        crate::state::log(&app.status, &msg);
        json!({ "success": false, "error": msg })
    });
    (StatusCode::OK, Json(body))
}

pub async fn job_control_handler(State(app): State<App>, Json(data): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let action = data.get("action").and_then(|v| v.as_str()).unwrap_or("status");
    if action == "cancel" {
        use std::sync::atomic::Ordering;
        app.cancel.store(true, Ordering::Relaxed);
        crate::state::log(&app.status, "Abbruch angefordert (job-control)");
        let s = app.status.lock().unwrap();
        return Json(json!({ "action": "cancel", "state": s.state, "name": s.name }));
    }
    let mut v = super::system::status_json(&app.status.lock().unwrap());
    v["action"] = json!("status");
    Json(v)
}
