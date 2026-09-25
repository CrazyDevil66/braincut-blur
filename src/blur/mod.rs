mod deface;
mod frame;
mod pipeline;
mod preview;
mod probe;
mod progress;

pub use deface::run_deface;

use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use serde_json::Value;

use crate::{
    paths::{fix_status_url, remap, validate_data_path},
    state::{log as slog, SharedState},
};
use pipeline::post_status;
use probe::wakeup_disk;

pub type CancelFlag = Arc<AtomicBool>;

pub fn new_cancel_flag() -> CancelFlag {
    Arc::new(AtomicBool::new(false))
}

pub fn process_jobs(
    jobs: Vec<Value>,
    resume_url: String,
    status_url: String,
    full_job: Option<Value>,
    state: SharedState,
    cancel: CancelFlag,
    cfg: crate::config::Config,
) {
    cancel.store(false, Ordering::Relaxed);

    let status_url = fix_status_url(&status_url, &cfg);
    let total = jobs.len();
    let mut errors: Vec<Value> = Vec::new();
    let mut was_cancelled = false;

    {
        let mut s = state.lock().unwrap();
        s.state = "blur".into();
        s.current = 0;
        s.total = total as u32;
        s.error = String::new();
        s.frame_current = 0;
        s.frame_total = 0;
        s.frame_pct = 0;
        s.eta_seconds = 0;
        s.started_at = chrono::Local::now().format("%H:%M:%S").to_string();
        s.started_at_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
    }
    slog(&state, &format!("Blur gestartet: {} Video(s)", total));
    post_status(&status_url, &serde_json::json!({ "event": "start", "total": total }));

    'jobs: for (i, job) in jobs.iter().enumerate() {
        let idx = i + 1;

        let raw_in  = job.get("input_path").and_then(|v| v.as_str()).unwrap_or("");
        let raw_out = job.get("output_path").and_then(|v| v.as_str()).unwrap_or("");

        let input_path = match validate_data_path(&remap(raw_in, &cfg), &cfg) {
            Ok(p) => p,
            Err(e) => {
                let msg = e.to_string();
                slog(&state, &format!("[{idx}/{total}] Pfadfehler: {msg}"));
                errors.push(serde_json::json!({ "input": raw_in, "error": msg }));
                post_status(&status_url, &serde_json::json!({
                    "event": "error", "current": idx, "total": total, "error": msg
                }));
                continue;
            }
        };
        let output_path = match validate_data_path(&remap(raw_out, &cfg), &cfg) {
            Ok(p) => p,
            Err(e) => {
                let msg = e.to_string();
                slog(&state, &format!("[{idx}/{total}] Pfadfehler: {msg}"));
                errors.push(serde_json::json!({ "input": raw_in, "error": msg }));
                continue;
            }
        };

        let blur_faces  = job.get("blur_faces").and_then(|v| v.as_bool()).unwrap_or(false);
        let blur_plates = job.get("blur_plates").and_then(|v| v.as_bool()).unwrap_or(false);
        let detection_res = job.get("detection_resolution").and_then(|v| v.as_str()).unwrap_or("720p");
        let name = Path::new(&input_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        {
            let mut s = state.lock().unwrap();
            s.current = idx as u32;
            s.name = name.clone();
            s.frame_current = 0;
            s.frame_total = 0;
            s.frame_pct = 0;
        }
        slog(&state, &format!("[{idx}/{total}] Starte: {name} (faces={blur_faces}, plates={blur_plates})"));
        post_status(&status_url, &serde_json::json!({
            "event": "progress_start", "current": idx, "total": total, "name": name
        }));

        wakeup_disk(&input_path);

        if let Some(parent) = Path::new(&output_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let mode = match (blur_faces, blur_plates) {
            (true, true)  => "both",
            (true, false) => "faces",
            (false, true) => "plates",
            _ => {
                let _ = std::fs::copy(&input_path, &output_path);
                slog(&state, &format!("[{idx}/{total}] Kein Blur angefordert – Datei kopiert"));
                post_status(&status_url, &serde_json::json!({
                    "event": "progress_done", "current": idx, "total": total, "name": name
                }));
                continue;
            }
        };

        match run_deface(&input_path, &output_path, mode, &status_url, &name, detection_res, state.clone(), cancel.clone(), &cfg) {
            Ok(()) => {
                slog(&state, &format!("[{idx}/{total}] Fertig: {name}"));
                post_status(&status_url, &serde_json::json!({
                    "event": "progress_done", "current": idx, "total": total, "name": name
                }));
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("cancelled") {
                    cancel.store(false, Ordering::Relaxed);
                    slog(&state, &format!("[{idx}/{total}] Job abgebrochen."));
                    errors.push(serde_json::json!({ "input": input_path, "error": "Abgebrochen" }));
                    post_status(&status_url, &serde_json::json!({
                        "event": "cancelled", "current": idx, "total": total, "name": name
                    }));
                    was_cancelled = true;
                    break 'jobs;
                }
                slog(&state, &format!("[{idx}/{total}] FEHLER: {}", &msg[..msg.len().min(300)]));
                errors.push(serde_json::json!({ "input": input_path, "error": &msg[..msg.len().min(500)] }));
                post_status(&status_url, &serde_json::json!({
                    "event": "error", "current": idx, "total": total, "name": name,
                    "error": &msg[..msg.len().min(500)]
                }));
            }
        }
    }

    slog(&state, &format!("Blur abgeschlossen. Fehler: {}", errors.len()));

    {
        let mut s = state.lock().unwrap();
        s.state = "idle".into();
        s.sub_state = String::new();
        s.error = if was_cancelled {
            "Abgebrochen".into()
        } else {
            errors.first()
                .and_then(|e| e.get("error"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .chars().take(200).collect()
        };
        s.frame_current = 0;
        s.frame_total = 0;
        s.frame_pct = 0;
        s.eta_seconds = 0;
        s.started_at_ts = 0.0;
    }

    post_status(&status_url, &serde_json::json!({ "event": "done", "total": total, "errors": errors }));

    if !resume_url.is_empty() {
        match ureq::post(&resume_url)
            .set("Content-Type", "application/json")
            .send_json(serde_json::json!({ "status": "done", "errors": errors }))
        {
            Ok(_)  => slog(&state, "Blur-Callback gesendet"),
            Err(e) => slog(&state, &format!("Blur-Callback fehlgeschlagen: {e}")),
        }
    }

    if !cfg.completion_webhook.is_empty() {
        let mut payload = serde_json::json!({
            "status": if was_cancelled { "cancelled" } else { "done" },
            "errors": errors
        });
        if let Some(fj) = full_job { payload["fullJob"] = fj; }
        match ureq::post(&cfg.completion_webhook)
            .set("Content-Type", "application/json")
            .send_json(payload)
        {
            Ok(_)  => slog(&state, "Completion-Webhook gesendet"),
            Err(e) => slog(&state, &format!("Completion-Webhook fehlgeschlagen: {e}")),
        }
    }
}
