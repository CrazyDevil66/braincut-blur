use serde_json::Value;
use std::process::Command;

pub struct VideoMeta {
    pub width: usize,
    pub height: usize,
    pub total_frames: u64,
    pub fps_str: String,
    pub fps: f64,
    pub rotation: i32,
    pub codec: String,
}

pub fn probe(path: &str) -> VideoMeta {
    let out = Command::new("ffprobe")
        .args(["-v", "quiet", "-print_format", "json", "-show_streams", path])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice::<Value>(&o.stdout).ok())
        .unwrap_or_default();

    let mut meta = VideoMeta {
        width: 0, height: 0, total_frames: 0,
        fps_str: "30/1".into(), fps: 30.0, rotation: 0, codec: String::new(),
    };

    for s in out.get("streams").and_then(|v| v.as_array()).unwrap_or(&vec![]) {
        if s.get("codec_type").and_then(|v| v.as_str()) != Some("video") { continue; }
        meta.codec  = s.get("codec_name").and_then(|v| v.as_str()).unwrap_or("").to_owned();
        meta.width  = s.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
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

pub fn wakeup_disk(path: &str) {
    let _ = std::fs::metadata(path);
}
