use std::{
    collections::HashMap,
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::{
    config::Config,
    detection::{check_nvenc, BBox, Detectors},
    image_ops::{gaussian_blur_roi, pixelate_roi},
    models::load_model_config,
    paths::{fix_status_url, remap, validate_data_path},
    state::{log as slog, SharedState},
};

pub type CancelFlag = Arc<AtomicBool>;

pub fn new_cancel_flag() -> CancelFlag {
    Arc::new(AtomicBool::new(false))
}

// ── Video probe ──────────────────────────────────────────────────────────────

struct VideoMeta {
    width: usize,
    height: usize,
    total_frames: u64,
    fps_str: String,
    fps: f64,
    rotation: i32,
    codec: String,
}

fn probe(path: &str) -> VideoMeta {
    let out = Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            path,
        ])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice::<Value>(&o.stdout).ok())
        .unwrap_or_default();

    let mut meta = VideoMeta {
        width: 0, height: 0, total_frames: 0,
        fps_str: "30/1".into(), fps: 30.0,
        rotation: 0, codec: String::new(),
    };

    for s in out.get("streams").and_then(|v| v.as_array()).unwrap_or(&vec![]) {
        if s.get("codec_type").and_then(|v| v.as_str()) != Some("video") {
            continue;
        }
        meta.codec = s.get("codec_name").and_then(|v| v.as_str()).unwrap_or("").to_owned();
        meta.width = s.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        meta.height = s.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let rfr = s.get("r_frame_rate").and_then(|v| v.as_str()).unwrap_or("30/1");
        meta.fps_str = rfr.to_owned();
        if let Some((n, d)) = rfr.split_once('/') {
            let dn: f64 = n.parse().unwrap_or(30.0);
            let dd: f64 = d.parse().unwrap_or(1.0);
            meta.fps = if dd > 0.0 { dn / dd } else { 30.0 };
        }

        let nb = s.get("nb_frames").and_then(|v| v.as_str()).unwrap_or("");
        if let Ok(n) = nb.parse::<u64>() {
            meta.total_frames = n;
        } else if let Some(dur) = s.get("duration").and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok()) {
            meta.total_frames = (dur * meta.fps) as u64;
        }

        // Rotation from tags or side_data
        if let Some(tags) = s.get("tags") {
            if let Some(r) = tags.get("rotate").and_then(|v| v.as_str()).and_then(|s| s.parse::<i32>().ok()) {
                meta.rotation = r;
            }
        }
        if let Some(sdl) = s.get("side_data_list").and_then(|v| v.as_array()) {
            for sd in sdl {
                if sd.get("side_data_type").and_then(|v| v.as_str()) == Some("Display Matrix") {
                    if let Some(r) = sd.get("rotation").and_then(|v| v.as_i64()) {
                        meta.rotation = (-(r as i32)).rem_euclid(360);
                    }
                }
            }
        }
        break;
    }
    meta
}

// ── Disk wakeup ──────────────────────────────────────────────────────────────

fn wakeup_disk(path: &str) {
    let _ = std::fs::metadata(path);
}

// ── Status HTTP post ─────────────────────────────────────────────────────────

fn post_status(url: &str, body: &Value) {
    if url.is_empty() {
        return;
    }
    let url = url.to_owned();
    let body = body.clone();
    std::thread::spawn(move || {
        let _ = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(body);
    });
}

// ── Top-level job processing ─────────────────────────────────────────────────

pub fn process_jobs(
    jobs: Vec<Value>,
    resume_url: String,
    status_url: String,
    full_job: Option<Value>,
    state: SharedState,
    cancel: CancelFlag,
    cfg: Config,
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

        let raw_in = job.get("input_path").and_then(|v| v.as_str()).unwrap_or("");
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

        let blur_faces = job.get("blur_faces").and_then(|v| v.as_bool()).unwrap_or(false);
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

        // Create output directory
        if let Some(parent) = Path::new(&output_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let mode = match (blur_faces, blur_plates) {
            (true, true) => "both",
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

        match run_deface(
            &input_path,
            &output_path,
            mode,
            &status_url,
            &name,
            detection_res,
            state.clone(),
            cancel.clone(),
            &cfg,
        ) {
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
                .chars()
                .take(200)
                .collect()
        };
        s.frame_current = 0;
        s.frame_total = 0;
        s.frame_pct = 0;
        s.eta_seconds = 0;
        s.started_at_ts = 0.0;
    }

    post_status(&status_url, &serde_json::json!({
        "event": "done", "total": total, "errors": errors
    }));

    // N8N resume callback
    if !resume_url.is_empty() {
        match ureq::post(&resume_url)
            .set("Content-Type", "application/json")
            .send_json(serde_json::json!({ "status": "done", "errors": errors }))
        {
            Ok(_) => slog(&state, "Blur-Callback gesendet"),
            Err(e) => slog(&state, &format!("Blur-Callback fehlgeschlagen: {e}")),
        }
    }

    // Completion webhook (auch bei Abbruch senden damit N8N zurücksetzen kann)
    if !cfg.completion_webhook.is_empty() {
        let mut payload = serde_json::json!({
            "status": if was_cancelled { "cancelled" } else { "done" },
            "errors": errors
        });
        if let Some(fj) = full_job {
            payload["fullJob"] = fj;
        }
        match ureq::post(&cfg.completion_webhook)
            .set("Content-Type", "application/json")
            .send_json(payload)
        {
            Ok(_) => slog(&state, "Completion-Webhook gesendet"),
            Err(e) => slog(&state, &format!("Completion-Webhook fehlgeschlagen: {e}")),
        }
    }
}

// ── run_deface ────────────────────────────────────────────────────────────────

pub fn run_deface(
    input_path: &str,
    output_path: &str,
    mode: &str,
    status_url: &str,
    job_name: &str,
    detection_res: &str,
    state: SharedState,
    cancel: CancelFlag,
    cfg: &Config,
) -> Result<()> {
    let model_cfg = load_model_config(cfg);

    let det_interval = model_cfg
        .get("detection_interval")
        .and_then(|v| v.as_u64())
        .unwrap_or(cfg.detection_interval as u64) as u64;
    let conf_thresh = model_cfg
        .get("plate_conf_thresh")
        .and_then(|v| v.as_f64())
        .unwrap_or(cfg.plate_conf_thresh as f64) as f32;

    // Resolve plate model
    let plate_model_path = if mode == "plates" || mode == "both" {
        let pid = model_cfg.get("plate_model").and_then(|v| v.as_str());
        match pid {
            None => {
                if mode == "plates" {
                    bail!("Kennzeichen-Blur angefordert, aber kein Kennzeichen-Modell aktiv.");
                }
                slog(&state, "Kein Kennzeichen-Modell aktiv – nur Gesichter.");
                None
            }
            Some(id) => {
                let p = cfg.models_path.join(format!("{id}.onnx"));
                if !p.exists() {
                    if mode == "plates" {
                        bail!("Kennzeichen-Modell '{id}' nicht gefunden.");
                    }
                    slog(&state, &format!("Kennzeichen-Modell nicht gefunden ({id}) – nur Gesichter."));
                    None
                } else {
                    Some(p)
                }
            }
        }
    } else {
        None
    };

    // Resolve face model
    let face_model_cfg = model_cfg
        .get("face_model")
        .and_then(|v| v.as_str())
        .unwrap_or("builtin-centerface");

    let face_yolo_path = if mode == "faces" || mode == "both" {
        if face_model_cfg == "builtin-centerface" {
            None
        } else {
            let p = cfg.models_path.join(format!("{face_model_cfg}.onnx"));
            if p.exists() { Some(p) } else { None }
        }
    } else {
        None
    };

    let use_centerface = (mode == "faces" || mode == "both")
        && face_model_cfg == "builtin-centerface"
        && face_yolo_path.is_none();

    // CenterFace detection resolution
    let (in_h, in_w) = match detection_res {
        "1080p" => (1080, 1920),
        "native" => (0, 0), // filled in after probe
        _ => (720, 1280),   // default 720p
    };

    slog(&state, &format!("deface [{mode}] startet: {}", Path::new(input_path).file_name().unwrap_or_default().to_string_lossy()));
    { state.lock().unwrap().sub_state = "probe".into(); }

    if !std::path::Path::new(input_path).exists() {
        bail!("Datei nicht gefunden: {input_path}\nPrüfe ob das Volume korrekt gemountet ist.");
    }

    // Probe
    let mut meta = probe(input_path);
    slog(&state, &format!("Video: {}x{} @ {:.1}fps, {} Frames, Codec: {}", meta.width, meta.height, meta.fps, meta.total_frames, meta.codec));

    if meta.rotation % 90 != 0 {
        meta.rotation = 0;
    }
    if meta.rotation != 0 {
        slog(&state, &format!("Video-Rotation erkannt: {}° – FFmpeg autorotiert", meta.rotation));
    }
    // After FFmpeg autorotate, displayed dimensions are swapped for 90/270°
    let (w, h) = if meta.rotation == 90 || meta.rotation == 270 {
        (meta.height, meta.width)
    } else {
        (meta.width, meta.height)
    };

    let (in_h, in_w) = if detection_res == "native" { (h, w) } else { (in_h, in_w) };

    // TRT dauert beim ersten Lauf ohne Cache-Engine 5–10 Min → grundsätzlich deaktiviert.
    // Kann per ORT_ENABLE_TRT=1 aktiviert werden (nur sinnvoll mit persistentem Cache-Volume).
    let use_trt = std::env::var("ORT_ENABLE_TRT").map(|v| v == "1").unwrap_or(false);

    {
        let mut s = state.lock().unwrap();
        s.hw_trt = use_trt;
    }

    // Load detectors
    let cf_path = if use_centerface && cfg.centerface_model.exists() {
        slog(&state, &format!("CenterFace geladen: {} (in_shape={}x{})", cfg.centerface_model.display(), in_h, in_w));
        Some(cfg.centerface_model.as_path())
    } else if use_centerface {
        slog(&state, &format!("WARNUNG: CenterFace-Modell nicht gefunden: {}", cfg.centerface_model.display()));
        None
    } else {
        None
    };

    { state.lock().unwrap().sub_state = "load_models".into(); }
    let mut detectors = Detectors::load(
        cf_path,
        face_yolo_path.as_deref(),
        plate_model_path.as_deref(),
        in_h,
        in_w,
        use_trt,
    ).context("Detektoren laden")?;

    // FFmpeg hardware setup
    wakeup_disk(input_path);
    let use_nvenc = check_nvenc();

    let cuvid_map = [("h264", "h264_cuvid"), ("hevc", "hevc_cuvid"), ("vp9", "vp9_cuvid"), ("av1", "av1_cuvid")];
    let cuvid = cuvid_map.iter().find(|(k, _)| *k == meta.codec).map(|(_, v)| *v).unwrap_or("");

    let tmp_output = format!("{output_path}.enc.tmp.mp4");
    let final_tmp = format!("{output_path}.final.tmp.mp4");

    // Decode command
    let mut dec_args: Vec<String> = vec![
        "-loglevel".into(), "error".into(),
        "-hwaccel".into(), "cuda".into(),
    ];
    if !cuvid.is_empty() {
        dec_args.extend(["-c:v".into(), cuvid.into()]);
    }
    dec_args.extend(["-i".into(), input_path.into(), "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "bgr24".into(), "pipe:1".into()]);

    // Encode command
    let mut enc_args: Vec<String> = vec![
        "-loglevel".into(), "error".into(), "-y".into(),
        "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "bgr24".into(),
        "-s".into(), format!("{w}x{h}"),
        "-r".into(), meta.fps_str.clone(),
        "-i".into(), "pipe:0".into(),
    ];
    if use_nvenc {
        enc_args.extend(["-c:v".into(), "h264_nvenc".into(), "-preset".into(), "p4".into(), "-cq".into(), "18".into()]);
    } else {
        enc_args.extend(["-c:v".into(), "libx264".into(), "-crf".into(), "18".into(), "-preset".into(), "fast".into()]);
    }
    enc_args.extend(["-pix_fmt".into(), "yuv420p".into(), "-an".into(), tmp_output.clone()]);

    slog(&state, &format!("Hardware: NVDEC={}, NVENC={}", if cuvid.is_empty() { "–" } else { cuvid }, if use_nvenc { "h264_nvenc" } else { "– (libx264)" }));
    {
        let mut s = state.lock().unwrap();
        s.hw_nvdec = !cuvid.is_empty();
        s.hw_nvenc = use_nvenc;
        s.frame_total = meta.total_frames;
        s.frame_current = 0;
    }

    let mut proc_dec = Command::new("ffmpeg")
        .args(&dec_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("FFmpeg Decoder starten")?;

    let mut proc_enc = Command::new("ffmpeg")
        .args(&enc_args)
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("FFmpeg Encoder starten")?;

    let frame_size = w * h * 3;
    let mut buf = vec![0u8; frame_size];

    let mut frame_idx: u64 = 0;
    let start = Instant::now();
    let mut last_log = Instant::now();
    let mut last_milestone = 0u64;
    let mut cancelled = false;

    // Plate buffer: key=(x/grid,y/grid,x2/grid,y2/grid), value=(bbox, ttl)
    let mut plate_buf: HashMap<(usize, usize, usize, usize), (BBox, u32)> = HashMap::new();
    const PLATE_TTL: u32 = 80;

    let mut total_faces: u64 = 0;
    let mut total_plates: u64 = 0;
    let mut last_face_dets: Vec<BBox> = Vec::new();

    let grid = (cfg.plate_grid as usize).max(1);

    let dec_stdout = proc_dec.stdout.take().expect("decoder stdout");
    let enc_stdin = proc_enc.stdin.take().expect("encoder stdin");

    // Detection log (CSV next to output file)
    let det_log_path = format!("{output_path}.detections.csv");
    let mut det_log: Option<std::io::BufWriter<std::fs::File>> = std::fs::File::create(&det_log_path)
        .ok()
        .map(std::io::BufWriter::new);
    if let Some(ref mut log) = det_log {
        let _ = writeln!(log, "frame,model,x,y,w,h,rel_w_pct,rel_h_pct,applied");
    }

    // Frame loop – runs in this thread (blocking I/O)
    { state.lock().unwrap().sub_state = "blur_loop".into(); }
    let result: Result<()> = (|| {
        let mut dec_reader = std::io::BufReader::with_capacity(frame_size, dec_stdout);
        let mut enc_writer = enc_stdin;

        loop {
            match dec_reader.read_exact(&mut buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("Decoder lesen"),
            }

            if cancel.load(Ordering::Relaxed) {
                cancelled = true;
                break;
            }

            frame_idx += 1;
            let should_detect = (frame_idx - 1) % det_interval == 0;

            if should_detect {
                if mode == "faces" || mode == "both" {
                    last_face_dets = detectors.detect_faces(&buf, w, h);
                    total_faces += last_face_dets.len() as u64;
                    if let Some(ref mut log) = det_log {
                        for &BBox { x, y, x2, y2 } in &last_face_dets {
                            let dw = x2 - x; let dh = y2 - y;
                            let applied = dw >= 50 && dh >= 50 && dw <= w/5 && dh <= h/5;
                            let _ = writeln!(log, "{frame_idx},face,{x},{y},{dw},{dh},{:.1},{:.1},{}",
                                dw as f32 / w as f32 * 100.0, dh as f32 / h as f32 * 100.0,
                                if applied { 1 } else { 0 });
                        }
                    }
                }
                if mode == "plates" || mode == "both" {
                    // Decay existing plate buffer
                    plate_buf.retain(|_, (_, ttl)| { *ttl = ttl.saturating_sub(1); *ttl > 0 });
                    let new_plates = detectors.detect_plates(&buf, w, h, conf_thresh);
                    for bp in new_plates {
                        let key = (bp.x / grid, bp.y / grid, bp.x2 / grid, bp.y2 / grid);
                        if let Some(ref mut log) = det_log {
                            let dw = bp.x2 - bp.x; let dh = bp.y2 - bp.y;
                            let _ = writeln!(log, "{frame_idx},plate,{},{},{dw},{dh},{:.1},{:.1},1",
                                bp.x, bp.y,
                                dw as f32 / w as f32 * 100.0, dh as f32 / h as f32 * 100.0);
                        }
                        plate_buf.insert(key, (bp, PLATE_TTL));
                    }
                    total_plates += plate_buf.len() as u64;
                }
            } else if mode == "plates" || mode == "both" {
                // Decay plate buffer every frame
                plate_buf.retain(|_, (_, ttl)| { *ttl = ttl.saturating_sub(1); *ttl > 0 });
            }

            // Apply face blur
            let max_face_w = w / 5;
            let max_face_h = h / 5;
            for &BBox { x, y, x2, y2 } in &last_face_dets {
                let rw = x2 - x;
                let rh = y2 - y;
                if rw < 50 || rh < 50 {
                    continue;
                }
                if rw > max_face_w || rh > max_face_h {
                    continue;
                }
                gaussian_blur_roi(&mut buf, w, x, y, x2, y2);
            }

            // Apply plate pixelation
            for (_, (bp, _)) in &plate_buf {
                pixelate_roi(&mut buf, w, bp.x, bp.y, bp.x2, bp.y2);
            }

            enc_writer.write_all(&buf).context("Encoder schreiben")?;

            // Progress
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                let fps_actual = frame_idx as f64 / elapsed;
                let pct = if meta.total_frames > 0 {
                    (frame_idx * 100 / meta.total_frames).min(100)
                } else {
                    0
                };
                let eta = if fps_actual > 0.0 && meta.total_frames > frame_idx {
                    ((meta.total_frames - frame_idx) as f64 / fps_actual) as u64
                } else {
                    0
                };

                {
                    let mut s = state.lock().unwrap();
                    s.frame_current = frame_idx;
                    s.frame_total = meta.total_frames;
                    s.frame_pct = pct as u8;
                    s.eta_seconds = eta;
                    s.face_count = total_faces;
                    s.plate_count = total_plates;
                }

                if last_log.elapsed() >= Duration::from_secs(10) {
                    slog(&state, &format!("  {}% | {}/{} Frames | {:.1}fps | ~{}s verbleibend",
                        pct, frame_idx, meta.total_frames, fps_actual, eta));
                    last_log = Instant::now();
                }

                let milestone = (pct / 10) * 10;
                if milestone > 0 && milestone > last_milestone {
                    last_milestone = milestone;
                    post_status(status_url, &serde_json::json!({
                        "event": "frame_progress",
                        "name": job_name,
                        "mode": mode,
                        "pct": milestone,
                        "frame_current": frame_idx,
                        "frame_total": meta.total_frames,
                        "eta_seconds": eta,
                    }));
                }
            }
        }

        enc_writer.flush().context("Encoder flush")?;
        Ok(())
    })();

    // Teardown
    {
        let mut s = state.lock().unwrap();
        s.state = "render".into();
        s.sub_state = "encode".into();
        s.out_name = Path::new(output_path).file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
    }

    let _ = proc_enc.wait();
    let _ = proc_dec.kill();
    let _ = proc_dec.wait();

    slog(&state, &format!(
        "Detektion: {} Gesichts-Erkennungen, {} Kennzeichen-Erkennungen über {} Frames",
        total_faces, total_plates, frame_idx
    ));
    if mode == "faces" || mode == "both" {
        if total_faces == 0 {
            slog(&state, "WARNUNG: Kein Gesicht erkannt! CenterFace evtl. fehlerhaft.");
            post_status(status_url, &serde_json::json!({
                "event": "warning_no_detections", "name": job_name, "mode": mode
            }));
        }
    }

    // Propagate frame-loop error
    result?;

    if cancelled {
        cancel.store(false, Ordering::Relaxed);
        slog(&state, &format!("  Abbruch – {}/{} Frames verarbeitet", frame_idx, meta.total_frames));
        let _ = std::fs::remove_file(&tmp_output);
        bail!("cancelled");
    }

    // Validate encoder output
    if !std::path::Path::new(&tmp_output).exists()
        || std::fs::metadata(&tmp_output).map(|m| m.len()).unwrap_or(0) < 1024
    {
        bail!("Encoder fehlgeschlagen: Ausgabedatei fehlt oder leer ({tmp_output})");
    }

    { state.lock().unwrap().sub_state = "mux".into(); }
    slog(&state, "Audio-Mux läuft...");

    // Mux audio from original (encode used -an)
    let mux_ok = Command::new("ffmpeg")
        .args([
            "-loglevel", "error", "-y",
            "-i", &tmp_output,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c", "copy",
            &final_tmp,
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    // Decide final source: prefer mux result, fall back to video-only
    let src = if mux_ok && std::path::Path::new(&final_tmp).exists()
        && std::fs::metadata(&final_tmp).map(|m| m.len()).unwrap_or(0) > 1024
    {
        let _ = std::fs::remove_file(&tmp_output);
        final_tmp.clone()
    } else {
        slog(&state, "Audio-Mux fehlgeschlagen – Video ohne Audio");
        let _ = std::fs::remove_file(&final_tmp);
        tmp_output.clone()
    };

    if std::fs::metadata(&src).map(|m| m.len()).unwrap_or(0) < 1024 {
        bail!("Ausgabedatei nach Mux ist leer: {src}");
    }

    if std::path::Path::new(output_path).exists() {
        let _ = std::fs::remove_file(output_path);
    }
    std::fs::rename(&src, output_path).context("Ausgabedatei umbenennen")?;

    Ok(())
}
