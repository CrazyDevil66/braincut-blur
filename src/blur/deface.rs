use std::{
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    time::Instant,
};

use anyhow::{bail, Context, Result};

use crate::{
    detection::{check_nvenc, Detectors},
    image_ops::{gaussian_blur_roi, pixelate_roi},
    models::load_model_config,
    state::{log as slog, SharedState},
};

use super::{
    CancelFlag,
    frame::{face_wird_verpixelt, process_detection_frame, Track},
    pipeline::{build_dec_args, build_enc_args, mux_and_finalize, post_status},
    preview::make_preview,
    probe::{probe, wakeup_disk},
    progress::{resolve_models, update_frame_progress},
};

pub fn run_deface(
    input_path: &str,
    output_path: &str,
    mode: &str,
    status_url: &str,
    job_name: &str,
    detection_res: &str,
    state: SharedState,
    cancel: CancelFlag,
    cfg: &crate::config::Config,
) -> Result<()> {
    let model_cfg = load_model_config(cfg);

    let det_interval = model_cfg.get("detection_interval").and_then(|v| v.as_u64())
        .unwrap_or(cfg.detection_interval as u64);
    let conf_thresh = model_cfg.get("plate_conf_thresh").and_then(|v| v.as_f64())
        .unwrap_or(cfg.plate_conf_thresh as f64) as f32;
    let plate_tiles = (
        model_cfg.get("plate_tile_cols").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(cfg.plate_tile_cols).clamp(1, 4),
        model_cfg.get("plate_tile_rows").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(cfg.plate_tile_rows).clamp(1, 4),
    );

    let face_tiles = (
        model_cfg.get("face_tile_cols").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(cfg.face_tile_cols).clamp(1, 4),
        model_cfg.get("face_tile_rows").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(cfg.face_tile_rows).clamp(1, 4),
    );

    let models = resolve_models(&model_cfg, mode, cfg, &state)?;
    // CenterFace liefert Werte auf einer anderen Skala als SCRFD/YOLO – ohne eigene Einstellung je Modell passender Standard.
    let face_conf_thresh = model_cfg.get("face_conf_thresh").and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(if models.use_centerface { cfg.face_conf_thresh } else { 0.5 });

    let (in_h, in_w) = match detection_res {
        "1080p"  => (1080, 1920),
        "native" => (0, 0),
        _        => (720, 1280),
    };

    slog(&state, &format!("deface [{mode}] startet: {}", Path::new(input_path).file_name().unwrap_or_default().to_string_lossy()));
    { state.lock().unwrap().sub_state = "probe".into(); }

    if !std::path::Path::new(input_path).exists() {
        bail!("Datei nicht gefunden: {input_path}\nPrüfe ob das Volume korrekt gemountet ist.");
    }

    let mut meta = probe(input_path);
    slog(&state, &format!("Video: {}x{} @ {:.1}fps, {} Frames, Codec: {}", meta.width, meta.height, meta.fps, meta.total_frames, meta.codec));

    if meta.rotation % 90 != 0 { meta.rotation = 0; }
    if meta.rotation != 0 {
        slog(&state, &format!("Video-Rotation erkannt: {}° – FFmpeg autorotiert", meta.rotation));
    }
    let (w, h) = if meta.rotation == 90 || meta.rotation == 270 {
        (meta.height, meta.width)
    } else {
        (meta.width, meta.height)
    };
    let (in_h, in_w) = if detection_res == "native" { (h, w) } else { (in_h, in_w) };

    let use_trt = std::env::var("ORT_ENABLE_TRT").map(|v| v == "1").unwrap_or(false);
    { state.lock().unwrap().hw_trt = use_trt; }

    let cf_path = if models.use_centerface && cfg.centerface_model.exists() {
        slog(&state, &format!("CenterFace geladen: {} (in_shape={}x{})", cfg.centerface_model.display(), in_h, in_w));
        Some(cfg.centerface_model.as_path())
    } else if models.use_centerface {
        slog(&state, &format!("WARNUNG: CenterFace-Modell nicht gefunden: {}", cfg.centerface_model.display()));
        None
    } else {
        None
    };

    { state.lock().unwrap().sub_state = "load_models".into(); }
    let mut detectors = Detectors::load(
        cf_path,
        models.face_yolo_path.as_deref(),
        models.face_scrfd_path.as_deref(),
        models.plate_model_path.as_deref(),
        in_h, in_w, use_trt, face_conf_thresh,
    ).context("Detektoren laden")?;
    detectors.face_tiles = face_tiles;
    if models.face_scrfd_path.is_some() && (mode == "faces" || mode == "both") {
        slog(&state, &format!("Gesichtserkennung SCRFD: Gesamtbild + {}×{} Kacheln", face_tiles.0, face_tiles.1));
    }

    wakeup_disk(input_path);
    let use_nvenc = check_nvenc();
    let tmp_output = format!("{output_path}.enc.tmp.mp4");
    let final_tmp  = format!("{output_path}.final.tmp.mp4");

    let (dec_args, cuvid) = build_dec_args(input_path, &meta.codec);
    let enc_args = build_enc_args(&tmp_output, w, h, &meta.fps_str, use_nvenc);

    if mode == "plates" || mode == "both" {
        slog(&state, &format!("Kennzeichen-Erkennung: Gesamtbild + {}×{} Kacheln", plate_tiles.0, plate_tiles.1));
    }
    slog(&state, &format!("Gesichts-Schwelle: {face_conf_thresh:.2}"));
    slog(&state, &format!("Hardware: NVDEC={}, NVENC={}",
        if cuvid.is_empty() { "–" } else { &cuvid },
        if use_nvenc { "h264_nvenc" } else { "– (libx264)" }));
    {
        let mut s = state.lock().unwrap();
        s.hw_nvdec = !cuvid.is_empty();
        s.hw_nvenc = use_nvenc;
        s.frame_total = meta.total_frames;
        s.frame_current = 0;
    }

    let mut proc_dec = Command::new("ffmpeg").args(&dec_args)
        .stdout(Stdio::piped()).stderr(Stdio::inherit())
        .spawn().context("FFmpeg Decoder starten")?;
    let mut proc_enc = Command::new("ffmpeg").args(&enc_args)
        .stdin(Stdio::piped()).stderr(Stdio::inherit())
        .spawn().context("FFmpeg Encoder starten")?;

    let frame_size = w * h * 3;
    let mut buf = vec![0u8; frame_size];
    let mut frame_idx: u64 = 0;
    let start = Instant::now();
    let mut last_log = Instant::now();
    let mut last_preview = Instant::now();
    let mut last_milestone = 0u64;
    let mut cancelled = false;
    let mut total_faces: u64 = 0;
    let mut total_plates: u64 = 0;

    let mut plate_buf: Vec<Track> = Vec::new();
    let mut face_buf:  Vec<Track> = Vec::new();

    let dec_stdout = proc_dec.stdout.take().expect("decoder stdout");
    let enc_stdin  = proc_enc.stdin.take().expect("encoder stdin");

    let det_log_path = format!("{output_path}.detections.csv");
    let mut det_log: Option<std::io::BufWriter<std::fs::File>> = std::fs::File::create(&det_log_path)
        .ok().map(std::io::BufWriter::new);
    if let Some(ref mut log) = det_log {
        let _ = writeln!(log, "frame,model,x,y,w,h,rel_w_pct,rel_h_pct,score,applied");
    }

    use std::time::Duration;
    use std::sync::atomic::Ordering;

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

            if cancel.load(Ordering::Relaxed) { cancelled = true; break; }

            frame_idx += 1;
            let should_detect = (frame_idx - 1) % det_interval == 0;

            let (nf, np) = process_detection_frame(
                &mut detectors, &buf, w, h, mode, should_detect,
                &mut face_buf, &mut plate_buf, conf_thresh, plate_tiles, frame_idx, &mut det_log,
            );
            total_faces  += nf;
            total_plates += np;

            for t in face_buf.iter().filter(|t| face_wird_verpixelt(&t.bbox)) {
                let bf = t.bbox;
                gaussian_blur_roi(&mut buf, w, bf.x, bf.y, bf.x2, bf.y2);
            }
            for t in &plate_buf {
                let bp = t.bbox;
                pixelate_roi(&mut buf, w, bp.x, bp.y, bp.x2, bp.y2);
            }

            enc_writer.write_all(&buf).context("Encoder schreiben")?;

            if last_preview.elapsed() >= Duration::from_secs(1) {
                last_preview = Instant::now();
                let jpeg = make_preview(&buf, w, h, &face_buf, &plate_buf);
                if !jpeg.is_empty() { state.lock().unwrap().preview_jpeg = jpeg; }
            }

            update_frame_progress(
                &state, frame_idx, meta.total_frames, &start,
                &mut last_log, &mut last_milestone,
                status_url, job_name, mode, total_faces, total_plates,
            );
        }

        enc_writer.flush().context("Encoder flush")?;
        Ok(())
    })();

    {
        let mut s = state.lock().unwrap();
        s.state = "render".into();
        s.sub_state = "encode".into();
        s.out_name = Path::new(output_path).file_name()
            .map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    }

    let _ = proc_enc.wait();
    let _ = proc_dec.kill();
    let _ = proc_dec.wait();
    state.lock().unwrap().preview_jpeg = Vec::new();

    slog(&state, &format!(
        "Detektion: {} Gesichts-Erkennungen, {} Kennzeichen-Erkennungen über {} Frames",
        total_faces, total_plates, frame_idx
    ));
    if (mode == "faces" || mode == "both") && total_faces == 0 {
        slog(&state, &format!("WARNUNG: Kein Gesicht erkannt! (Modell: {})", models.face_model_name));
        post_status(status_url, &serde_json::json!({
            "event": "warning_no_detections", "name": job_name, "mode": mode
        }));
    }

    result?;

    if cancelled {
        cancel.store(false, Ordering::Relaxed);
        slog(&state, &format!("  Abbruch – {}/{} Frames verarbeitet", frame_idx, meta.total_frames));
        let _ = std::fs::remove_file(&tmp_output);
        bail!("cancelled");
    }

    if !std::path::Path::new(&tmp_output).exists()
        || std::fs::metadata(&tmp_output).map(|m| m.len()).unwrap_or(0) < 1024
    {
        bail!("Encoder fehlgeschlagen: Ausgabedatei fehlt oder leer ({tmp_output})");
    }

    mux_and_finalize(&tmp_output, input_path, &final_tmp, output_path, &state)
}
