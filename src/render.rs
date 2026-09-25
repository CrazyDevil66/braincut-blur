//! Zusammenschnitt der fertigen Clips für WF3 (`POST /render`): Geschwindigkeit, Lautstärke,
//! Musik und Verkettung per FFmpeg. Port von `render_core.py` aus dem Python-Service –
//! Anfrage und Antwort sind unverändert.

use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
    sync::atomic::Ordering,
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::{
    blur::CancelFlag,
    config::Config,
    paths::{remap, validate_data_path},
    state::{log as slog, SharedState},
};

const ABGEBROCHEN: &str = "cancelled";

#[derive(Debug, Clone, PartialEq)]
pub struct ClipSpec {
    pub speed_factor: f64,
    pub mute_audio: bool,
    pub volume_factor: f64,
    pub has_audio: bool,
    pub duration: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MusicSpec {
    pub replace: bool,
    pub volume: f64,
    pub loop_music: bool,
}

/// `atempo` akzeptiert nur 0.5–2.0 je Stufe, größere Faktoren werden verkettet.
pub fn atempo_chain(factor: f64) -> Vec<String> {
    let mut parts = Vec::new();
    let mut f = factor;
    while f > 2.0 {
        parts.push("atempo=2.0".to_string());
        f /= 2.0;
    }
    while f < 0.5 {
        parts.push("atempo=0.5".to_string());
        f *= 2.0;
    }
    parts.push(format!("atempo={f:.4}"));
    parts
}

/// Baut den `-filter_complex`-Graphen und liefert ihn zusammen mit der Audio-Spur für `-map`.
/// Die Musik ist – falls vorhanden – der Eingang direkt nach den Clips.
pub fn build_filter_graph(clips: &[ClipSpec], music: Option<&MusicSpec>) -> (String, String) {
    let mut filters: Vec<String> = Vec::new();
    let mut pairs = String::new();
    // Bei „Musik statt Ton“ wird der Originalton nicht gebraucht.
    let replace_audio = music.map(|m| m.replace).unwrap_or(false);
    let total_duration: f64 = if clips.iter().all(|c| c.duration > 0.0) {
        clips.iter().map(|c| c.duration / c.speed_factor).sum()
    } else {
        0.0
    };

    for (i, c) in clips.iter().enumerate() {
        let sf = c.speed_factor;
        let sf_int = sf.round();
        if sf_int >= 2.0 && (sf - sf_int).abs() < 0.01 {
            filters.push(format!(
                "[{i}:v]select='not(mod(n,{}))',setpts=N/FR/TB,scale=1920:-2,setsar=1,fps=30,format=yuv420p[v{i}]",
                sf_int as i64
            ));
        } else {
            filters.push(format!(
                "[{i}:v]setpts={:.6}*(PTS-STARTPTS),scale=1920:-2,setsar=1,fps=30,format=yuv420p[v{i}]",
                1.0 / sf
            ));
        }

        if replace_audio {
            pairs.push_str(&format!("[v{i}]"));
            continue;
        }
        if c.mute_audio || !c.has_audio {
            // Stille muss so lang sein wie das beschleunigte Video, sonst verrutscht die Verkettung.
            let trim = if c.duration > 0.0 {
                format!(",atrim=duration={:.4},asetpts=PTS-STARTPTS", c.duration / sf)
            } else {
                String::new()
            };
            filters.push(format!(
                "anullsrc=r=48000:cl=stereo{trim},aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
            ));
        } else {
            let vol = c.volume_factor.clamp(0.0, 3.0);
            let mut af: Vec<String> = Vec::new();
            if sf != 1.0 {
                af.extend(atempo_chain(sf));
            }
            if vol != 1.0 {
                af.push(format!("volume={vol:.3}"));
            }
            af.extend(
                ["asetpts=PTS-STARTPTS", "aresample=48000", "aformat=sample_fmts=fltp:channel_layouts=stereo"]
                    .map(String::from),
            );
            filters.push(format!("[{i}:a]{}[a{i}]", af.join(",")));
        }
        pairs.push_str(&format!("[v{i}][a{i}]"));
    }

    if replace_audio {
        filters.push(format!("{pairs}concat=n={}:v=1:a=0[vout]", clips.len()));
    } else {
        filters.push(format!("{pairs}concat=n={}:v=1:a=1[vout][aorig]", clips.len()));
    }

    let mut audio_map = "[aorig]".to_string();
    if let Some(m) = music {
        let idx = clips.len();
        let mut mf = format!("[{idx}:a]volume={:.3}", m.volume.clamp(0.0, 3.0));
        if !m.loop_music {
            mf.push_str(",apad");
        }
        // Eine endlose Tonspur (Schleife oder apad) als Ausgabe lässt FFmpeg 6.1 trotz -shortest
        // mit „No space left on device“ abbrechen – daher auf die Videolänge kürzen.
        if m.replace && total_duration > 0.0 {
            mf.push_str(&format!(",atrim=duration={total_duration:.4}"));
        }
        mf.push_str("[amusic]");
        filters.push(mf);
        if m.replace {
            audio_map = "[amusic]".to_string();
        } else {
            filters.push("[aorig][amusic]amix=inputs=2:duration=first:dropout_transition=2[aout]".to_string());
            audio_map = "[aout]".to_string();
        }
    }

    (filters.join(";"), audio_map)
}

pub fn format_bytes(b: u64) -> String {
    if b > 1_073_741_824 {
        format!("{:.2} GB", b as f64 / 1_073_741_824.0)
    } else if b > 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", b as f64 / 1024.0)
    }
}

fn tail(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

fn file_name(path: &str) -> String {
    Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Liefert (hat Tonspur, Dauer in Sekunden). Bei ffprobe-Fehlern wie in Python: Ton angenommen, Dauer 0.
fn probe_clip(path: &str, state: &SharedState) -> (bool, f64) {
    let parsed = Command::new("ffprobe")
        .args(["-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice::<Value>(&o.stdout).ok());
    let Some(v) = parsed else {
        slog(state, &format!("ffprobe Fehler ({})", file_name(path)));
        return (true, 0.0);
    };

    let format_duration = v.pointer("/format/duration").and_then(Value::as_str);
    let mut has_audio = false;
    let mut duration = 0.0;
    for s in v.get("streams").and_then(Value::as_array).into_iter().flatten() {
        match s.get("codec_type").and_then(Value::as_str) {
            Some("video") => {
                let d = s.get("duration").and_then(Value::as_str).or(format_duration);
                if let Some(d) = d.and_then(|d| d.parse::<f64>().ok()) {
                    duration = d;
                }
            }
            Some("audio") => has_audio = true,
            _ => {}
        }
    }
    (has_audio, duration)
}

/// Führt FFmpeg aus und beendet es bei gesetztem Abbruch-Flag. Liefert (Erfolg, stderr, abgebrochen).
fn run_ffmpeg(args: &[String], cancel: &CancelFlag) -> Result<(bool, String, bool)> {
    let mut child = Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("FFmpeg starten")?;
    let mut stderr = child.stderr.take().context("FFmpeg-stderr fehlt")?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    });

    let mut cancelled = false;
    let status = loop {
        if let Some(st) = child.try_wait().context("FFmpeg-Status")? {
            break st;
        }
        if cancel.load(Ordering::Relaxed) {
            cancelled = true;
            let _ = child.kill();
            break child.wait().context("FFmpeg beenden")?;
        }
        std::thread::sleep(Duration::from_millis(250));
    };
    let err = reader.join().unwrap_or_default();
    if cancelled {
        cancel.store(false, Ordering::Relaxed);
    }
    Ok((status.success(), err, cancelled))
}

fn verschiebe_quellen(sources: &[String], move_sources_to: &str, state: &SharedState, cfg: &Config) {
    let dest_dir = match validate_data_path(&remap(move_sources_to, cfg), cfg) {
        Ok(d) => d,
        Err(e) => {
            slog(state, &format!("Quelldateien verschieben fehlgeschlagen: {e}"));
            return;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&dest_dir) {
        slog(state, &format!("Quelldateien verschieben fehlgeschlagen: {e}"));
        return;
    }
    let mut seen: Vec<&String> = Vec::new();
    for src in sources {
        if seen.contains(&src) || !Path::new(src).exists() {
            continue;
        }
        seen.push(src);
        let base = file_name(src);
        let mut dest = Path::new(&dest_dir).join(&base);
        if dest.exists() {
            let p = Path::new(&base);
            let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
            let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
            dest = Path::new(&dest_dir).join(format!("{stem}_{ts}{ext}"));
        }
        match std::fs::rename(src, &dest) {
            Ok(()) => slog(state, &format!("Verschoben: {base} → {}", file_name(&dest_dir))),
            Err(e) => slog(state, &format!("Quelldatei verschieben fehlgeschlagen ({base}): {e}")),
        }
    }
}

fn bereinige(cleanup_paths: &[Value], state: &SharedState, cfg: &Config) {
    for p in cleanup_paths.iter().filter_map(Value::as_str) {
        match validate_data_path(&remap(p, cfg), cfg) {
            Ok(rp) => {
                if Path::new(&rp).exists() {
                    match std::fs::remove_file(&rp) {
                        Ok(()) => slog(state, &format!("Bereinigt: {}", file_name(&rp))),
                        Err(e) => slog(state, &format!("Bereinigung fehlgeschlagen: {e}")),
                    }
                }
            }
            Err(e) => slog(state, &format!("Bereinigung fehlgeschlagen: {e}")),
        }
    }
}

fn render(data: &Value, state: &SharedState, cancel: &CancelFlag, cfg: &Config) -> Result<Value> {
    let clips = data.get("clips").and_then(Value::as_array).cloned().unwrap_or_default();
    if clips.is_empty() {
        bail!("Keine Clips im Render-Auftrag.");
    }
    let audio = data.get("audio").cloned().unwrap_or_else(|| json!({ "mode": "original" }));
    let output_raw = data.get("output_path").and_then(Value::as_str).unwrap_or("");
    let output_path = validate_data_path(&remap(output_raw, cfg), cfg)?;
    let overwrite = data.get("overwrite").and_then(Value::as_bool).unwrap_or(true);
    let out_name = file_name(&output_path);
    // FFmpeg mit `-n` endet bei vorhandener Datei ohne Fehlercode – daher vorher prüfen.
    if !overwrite && Path::new(&output_path).exists() {
        bail!("Ausgabedatei existiert bereits: {out_name}");
    }

    {
        let mut s = state.lock().unwrap();
        s.state = "render".into();
        s.name = out_name.clone();
        s.out_name = out_name.clone();
        s.error = String::new();
        s.started_at = chrono::Local::now().format("%H:%M:%S").to_string();
        s.started_at_ts = crate::handlers::now_unix();
    }
    slog(state, &format!("Render gestartet: {} Clip(s) → {out_name}", clips.len()));

    let mut inputs: Vec<String> = Vec::new();
    let mut specs: Vec<ClipSpec> = Vec::new();
    for c in &clips {
        let raw = c.get("path").and_then(Value::as_str).unwrap_or("");
        let p = validate_data_path(&remap(raw, cfg), cfg).map_err(|e| anyhow!("Ungültiger Clip-Pfad: {e}"))?;
        let speed_factor = c.get("speed_factor").and_then(Value::as_f64).unwrap_or(1.0);
        if !(speed_factor > 0.0) {
            bail!("Ungültige Geschwindigkeit {speed_factor} für {}", file_name(&p));
        }
        let (has_audio, duration) = probe_clip(&p, state);
        specs.push(ClipSpec {
            speed_factor,
            mute_audio: c.get("mute_audio").and_then(Value::as_bool).unwrap_or(false),
            volume_factor: c.get("volume_factor").and_then(Value::as_f64).unwrap_or(1.0),
            has_audio,
            duration,
        });
        inputs.push(p);
    }

    let mode = audio.get("mode").and_then(Value::as_str).unwrap_or("original");
    let music_raw = audio.pointer("/musicFile/path").and_then(Value::as_str).unwrap_or("");
    let use_music = (mode == "replace_with_music" || mode == "mix_music") && !music_raw.is_empty();

    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-nostats".into(), "-loglevel".into(), "error".into(),
        if overwrite { "-y" } else { "-n" }.into(),
    ];
    for p in &inputs {
        args.extend(["-i".to_string(), p.clone()]);
    }
    let music = if use_music {
        let mp = validate_data_path(&remap(music_raw, cfg), cfg).map_err(|e| anyhow!("Ungültiger Musik-Pfad: {e}"))?;
        let loop_music = audio.get("loop").and_then(Value::as_bool).unwrap_or(false);
        if loop_music {
            args.extend(["-stream_loop".to_string(), "-1".to_string()]);
        }
        args.extend(["-i".to_string(), mp]);
        Some(MusicSpec {
            replace: mode == "replace_with_music",
            volume: audio.get("musicVolume").and_then(Value::as_f64).unwrap_or(0.35),
            loop_music,
        })
    } else {
        None
    };

    let (graph, audio_map) = build_filter_graph(&specs, music.as_ref());
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).context("Ausgabeordner anlegen")?;
    }
    args.extend(
        [
            "-filter_complex", graph.as_str(), "-map", "[vout]", "-map", audio_map.as_str(),
            "-c:v", "libx264", "-crf", "22", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", output_path.as_str(),
        ]
        .map(String::from),
    );
    slog(state, &format!("FFmpeg startet ({} Clips, audio={mode})", specs.len()));

    let (ok, stderr, cancelled) = run_ffmpeg(&args, cancel)?;
    if cancelled {
        let _ = std::fs::remove_file(&output_path);
        bail!(ABGEBROCHEN);
    }
    if !ok {
        let _ = std::fs::remove_file(&output_path);
        let err = if stderr.trim().is_empty() { "FFmpeg fehlgeschlagen".to_string() } else { tail(stderr.trim(), 800) };
        slog(state, &format!("FFmpeg Fehler: {}", tail(&err, 300)));
        bail!(err);
    }

    let size_bytes = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
    if size_bytes < 1024 {
        bail!("FFmpeg: Ausgabedatei fehlt oder leer");
    }
    let size = format_bytes(size_bytes);
    slog(state, &format!("Render fertig: {out_name} ({size})"));

    if let Some(dest) = data.get("move_sources_to").and_then(Value::as_str).filter(|d| !d.is_empty()) {
        verschiebe_quellen(&inputs, dest, state, cfg);
    }
    if let Some(paths) = data.get("cleanup_paths").and_then(Value::as_array) {
        bereinige(paths, state, cfg);
    }

    Ok(json!({
        "success": true,
        "out_path": output_path,
        "out_name": out_name,
        "size": size,
        "size_bytes": size_bytes,
    }))
}

/// Blockierend: rendert den Auftrag und setzt den Status danach immer auf `idle` zurück.
pub fn run_render(data: &Value, state: &SharedState, cancel: &CancelFlag, cfg: &Config) -> Value {
    cancel.store(false, Ordering::Relaxed);
    let result = render(data, state, cancel, cfg);

    let antwort = match &result {
        Ok(v) => v.clone(),
        Err(e) if e.to_string() == ABGEBROCHEN => json!({ "success": false, "error": ABGEBROCHEN }),
        Err(e) => json!({ "success": false, "error": tail(&e.to_string(), 800) }),
    };
    let fehler = match &result {
        Ok(_) => String::new(),
        Err(e) if e.to_string() == ABGEBROCHEN => "Render abgebrochen".to_string(),
        Err(e) => e.to_string().chars().take(200).collect(),
    };
    {
        let mut s = state.lock().unwrap();
        s.state = "idle".into();
        s.sub_state = String::new();
        s.name = String::new();
        s.error = fehler.clone();
        s.started_at_ts = 0.0;
    }
    match &result {
        Ok(_) => {}
        Err(e) if e.to_string() == ABGEBROCHEN => slog(state, "Render abgebrochen durch Benutzer"),
        Err(_) => slog(state, &format!("Render-Fehler: {}", tail(&fehler, 300))),
    }
    antwort
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(sf: f64, mute: bool, vol: f64, has_audio: bool, dur: f64) -> ClipSpec {
        ClipSpec { speed_factor: sf, mute_audio: mute, volume_factor: vol, has_audio, duration: dur }
    }

    #[test]
    fn atempo_kette_teilt_grosse_und_kleine_faktoren() {
        assert_eq!(atempo_chain(1.5), vec!["atempo=1.5000"]);
        assert_eq!(atempo_chain(4.0), vec!["atempo=2.0", "atempo=2.0000"]);
        assert_eq!(atempo_chain(8.0), vec!["atempo=2.0", "atempo=2.0", "atempo=2.0000"]);
        assert_eq!(atempo_chain(0.25), vec!["atempo=0.5", "atempo=0.5000"]);
    }

    #[test]
    fn graph_einzelner_clip_originalton() {
        let (g, map) = build_filter_graph(&[clip(1.0, false, 1.0, true, 10.0)], None);
        assert_eq!(g, "[0:v]setpts=1.000000*(PTS-STARTPTS),scale=1920:-2,setsar=1,fps=30,format=yuv420p[v0];\
[0:a]asetpts=PTS-STARTPTS,aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a0];\
[v0][a0]concat=n=1:v=1:a=1[vout][aorig]");
        assert_eq!(map, "[aorig]");
    }

    #[test]
    fn graph_ganzzahliger_faktor_stumm_mit_passender_stille() {
        let (g, _) = build_filter_graph(&[clip(4.0, true, 1.0, true, 100.0)], None);
        assert!(g.starts_with("[0:v]select='not(mod(n,4))',setpts=N/FR/TB,scale=1920:-2"));
        assert!(g.contains("anullsrc=r=48000:cl=stereo,atrim=duration=25.0000,asetpts=PTS-STARTPTS,aresample=48000"));
    }

    #[test]
    fn graph_krummer_faktor_mit_lautstaerke() {
        let (g, _) = build_filter_graph(&[clip(1.5, false, 0.5, true, 10.0)], None);
        assert!(g.contains("[0:v]setpts=0.666667*(PTS-STARTPTS)"));
        assert!(g.contains("[0:a]atempo=1.5000,volume=0.500,asetpts=PTS-STARTPTS"));
    }

    #[test]
    fn graph_ohne_tonspur_und_ohne_dauer_ohne_trim() {
        let (g, _) = build_filter_graph(&[clip(1.0, false, 1.0, false, 0.0)], None);
        assert!(g.contains("anullsrc=r=48000:cl=stereo,aresample=48000"));
        assert!(!g.contains("atrim"));
    }

    #[test]
    fn graph_zwei_clips_und_musik_gemischt() {
        let clips = [clip(1.0, false, 1.0, true, 5.0), clip(2.0, false, 1.0, true, 5.0)];
        let music = MusicSpec { replace: false, volume: 0.35, loop_music: true };
        let (g, map) = build_filter_graph(&clips, Some(&music));
        assert!(g.contains("[v0][a0][v1][a1]concat=n=2:v=1:a=1[vout][aorig]"));
        assert!(g.contains("[2:a]volume=0.350[amusic]"));
        assert!(g.ends_with("[aorig][amusic]amix=inputs=2:duration=first:dropout_transition=2[aout]"));
        assert_eq!(map, "[aout]");
    }

    #[test]
    fn graph_musik_ersetzt_ton_ohne_schleife() {
        let music = MusicSpec { replace: true, volume: 0.35, loop_music: false };
        let clips = [clip(1.0, false, 1.0, true, 5.0), clip(4.0, true, 1.0, true, 8.0)];
        let (g, map) = build_filter_graph(&clips, Some(&music));
        assert!(!g.contains("[0:a]") && !g.contains("anullsrc"));
        assert!(g.contains("[v0][v1]concat=n=2:v=1:a=0[vout]"));
        assert!(g.ends_with("[2:a]volume=0.350,apad,atrim=duration=7.0000[amusic]"));
        assert_eq!(map, "[amusic]");
    }

    #[test]
    fn graph_musik_schleife_wird_auf_videolaenge_gekuerzt() {
        let music = MusicSpec { replace: true, volume: 0.35, loop_music: true };
        let (g, _) = build_filter_graph(&[clip(2.0, true, 1.0, false, 4.0)], Some(&music));
        assert!(g.ends_with("[1:a]volume=0.350,atrim=duration=2.0000[amusic]"));
    }

    #[test]
    fn bytes_formatierung_wie_python() {
        assert_eq!(format_bytes(512 * 1024), "512 KB");
        assert_eq!(format_bytes(5 * 1_048_576 + 1), "5.0 MB");
        assert_eq!(format_bytes(3 * 1_073_741_824 + 1), "3.00 GB");
    }
}
