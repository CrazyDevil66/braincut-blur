use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::Value;

use crate::{
    config::Config,
    state::{log as slog, SharedState},
};

use super::pipeline::post_status;

pub struct ResolvedModels {
    pub face_model_name: String,
    pub face_yolo_path: Option<std::path::PathBuf>,
    pub face_scrfd_path: Option<std::path::PathBuf>,
    pub plate_model_path: Option<std::path::PathBuf>,
    pub use_centerface: bool,
}

pub fn resolve_models(model_cfg: &Value, mode: &str, cfg: &Config, state: &SharedState) -> Result<ResolvedModels> {
    let plate_model_path = if mode == "plates" || mode == "both" {
        match model_cfg.get("plate_model").and_then(|v| v.as_str()) {
            None => {
                if mode == "plates" {
                    anyhow::bail!("Kennzeichen-Blur angefordert, aber kein Kennzeichen-Modell aktiv.");
                }
                slog(state, "Kein Kennzeichen-Modell aktiv – nur Gesichter.");
                None
            }
            Some(id) => {
                let p = cfg.models_path.join(format!("{id}.onnx"));
                if !p.exists() {
                    if mode == "plates" {
                        anyhow::bail!("Kennzeichen-Modell '{id}' nicht gefunden.");
                    }
                    slog(state, &format!("Kennzeichen-Modell nicht gefunden ({id}) – nur Gesichter."));
                    None
                } else {
                    Some(p)
                }
            }
        }
    } else {
        None
    };

    let face_model_name = model_cfg
        .get("face_model")
        .and_then(|v| v.as_str())
        .unwrap_or("builtin-centerface")
        .to_owned();

    let is_scrfd = face_model_name.starts_with("scrfd-");
    let (face_yolo_path, face_scrfd_path) = if mode == "faces" || mode == "both" {
        if face_model_name == "builtin-centerface" {
            (None, None)
        } else {
            let p = cfg.models_path.join(format!("{}.onnx", face_model_name));
            if p.exists() {
                if is_scrfd { (None, Some(p)) } else { (Some(p), None) }
            } else {
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    let use_centerface = (mode == "faces" || mode == "both")
        && face_yolo_path.is_none()
        && face_scrfd_path.is_none();

    if use_centerface && face_model_name != "builtin-centerface" {
        slog(state, &format!(
            "WARNUNG: Gesichts-Modell '{}' nicht gefunden – Fallback auf CenterFace",
            face_model_name
        ));
    }

    Ok(ResolvedModels { face_model_name, face_yolo_path, face_scrfd_path, plate_model_path, use_centerface })
}

#[allow(clippy::too_many_arguments)]
pub fn update_frame_progress(
    state: &SharedState,
    frame_idx: u64,
    total_frames: u64,
    start: &Instant,
    last_log: &mut Instant,
    last_milestone: &mut u64,
    status_url: &str,
    job_name: &str,
    mode: &str,
    total_faces: u64,
    total_plates: u64,
) {
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed <= 0.0 { return; }

    let fps_actual = frame_idx as f64 / elapsed;
    let pct = if total_frames > 0 { (frame_idx * 100 / total_frames).min(100) } else { 0 };
    let eta = if fps_actual > 0.0 && total_frames > frame_idx {
        ((total_frames - frame_idx) as f64 / fps_actual) as u64
    } else { 0 };

    {
        let mut s = state.lock().unwrap();
        s.frame_current = frame_idx;
        s.frame_total = total_frames;
        s.frame_pct = pct as u8;
        s.eta_seconds = eta;
        s.face_count = total_faces;
        s.plate_count = total_plates;
    }

    if last_log.elapsed() >= Duration::from_secs(10) {
        slog(state, &format!("  {}% | {}/{} Frames | {:.1}fps | ~{}s verbleibend",
            pct, frame_idx, total_frames, fps_actual, eta));
        *last_log = Instant::now();
    }

    let milestone = (pct / 10) * 10;
    if milestone > 0 && milestone > *last_milestone {
        *last_milestone = milestone;
        post_status(status_url, &serde_json::json!({
            "event": "frame_progress",
            "name": job_name,
            "mode": mode,
            "pct": milestone,
            "frame_current": frame_idx,
            "frame_total": total_frames,
            "eta_seconds": eta,
        }));
    }
}
