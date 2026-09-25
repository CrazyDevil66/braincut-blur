use anyhow::{Context, Result};
use serde_json::Value;
use std::process::Command;

use crate::state::SharedState;

pub fn post_status(url: &str, body: &Value) {
    if url.is_empty() { return; }
    let url = url.to_owned();
    let body = body.clone();
    std::thread::spawn(move || {
        let _ = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(body);
    });
}

pub fn build_dec_args(input_path: &str, codec: &str) -> (Vec<String>, String) {
    let mut args = vec!["-loglevel".into(), "error".into(), "-hwaccel".into(), "cuda".into()];
    let cuvid_map = [("h264","h264_cuvid"),("hevc","hevc_cuvid"),("vp9","vp9_cuvid"),("av1","av1_cuvid")];
    let cuvid = cuvid_map.iter().find(|(k,_)| *k == codec).map(|(_,v)| *v).unwrap_or("").to_owned();
    if !cuvid.is_empty() {
        args.extend(["-c:v".into(), cuvid.clone()]);
    }
    args.extend(["-i".into(), input_path.into(), "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "bgr24".into(), "pipe:1".into()]);
    (args, cuvid)
}

pub fn build_enc_args(output_path: &str, w: usize, h: usize, fps_str: &str, use_nvenc: bool) -> Vec<String> {
    let mut args = vec![
        "-loglevel".into(), "error".into(), "-y".into(),
        "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "bgr24".into(),
        "-s".into(), format!("{w}x{h}"),
        "-r".into(), fps_str.into(),
        "-i".into(), "pipe:0".into(),
    ];
    if use_nvenc {
        args.extend(["-c:v".into(), "h264_nvenc".into(), "-preset".into(), "p4".into(), "-cq".into(), "18".into()]);
    } else {
        args.extend(["-c:v".into(), "libx264".into(), "-crf".into(), "18".into(), "-preset".into(), "fast".into()]);
    }
    args.extend(["-pix_fmt".into(), "yuv420p".into(), "-an".into(), output_path.into()]);
    args
}

pub fn mux_and_finalize(
    enc_tmp: &str,
    input_path: &str,
    final_tmp: &str,
    output_path: &str,
    state: &SharedState,
) -> Result<()> {
    { state.lock().unwrap().sub_state = "mux".into(); }
    crate::state::log(state, "Audio-Mux läuft...");

    let mux_ok = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y",
               "-i", enc_tmp, "-i", input_path,
               "-map", "0:v:0", "-map", "1:a?", "-c", "copy", final_tmp])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    let src = if mux_ok && std::path::Path::new(final_tmp).exists()
        && std::fs::metadata(final_tmp).map(|m| m.len()).unwrap_or(0) > 1024
    {
        let _ = std::fs::remove_file(enc_tmp);
        final_tmp.to_owned()
    } else {
        crate::state::log(state, "Audio-Mux fehlgeschlagen – Video ohne Audio");
        let _ = std::fs::remove_file(final_tmp);
        enc_tmp.to_owned()
    };

    if std::fs::metadata(&src).map(|m| m.len()).unwrap_or(0) < 1024 {
        anyhow::bail!("Ausgabedatei nach Mux ist leer: {src}");
    }
    if std::path::Path::new(output_path).exists() {
        let _ = std::fs::remove_file(output_path);
    }
    std::fs::rename(&src, output_path).context("Ausgabedatei umbenennen")?;
    Ok(())
}
