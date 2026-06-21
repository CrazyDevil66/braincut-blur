import json
import os
import subprocess
from datetime import datetime

import state
from paths import _remap, validate_data_path
from utils import format_bytes


def _probe_clip(path: str) -> dict:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(r.stdout)
        has_audio = False
        duration = 0.0
        width, height, fps = 1920, 1080, 30.0
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                width = int(s.get("width", 1920))
                height = int(s.get("height", 1080))
                try:
                    n, d = s.get("r_frame_rate", "30/1").split("/")
                    fps = int(n) / max(1, int(d))
                except Exception:
                    fps = 30.0
                for key in ("duration",):
                    v = s.get(key) or data.get("format", {}).get(key)
                    if v:
                        try:
                            duration = float(v)
                        except Exception:
                            pass
            elif s.get("codec_type") == "audio":
                has_audio = True
        return {"has_audio": has_audio, "duration": duration,
                "width": width, "height": height, "fps": fps}
    except Exception as exc:
        state._log(f"ffprobe Fehler ({os.path.basename(path)}): {exc}")
        return {"has_audio": True, "duration": 0.0,
                "width": 1920, "height": 1080, "fps": 30.0}


def _atempo_chain(factor: float) -> list:
    parts = []
    f = factor
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f *= 2.0
    parts.append(f"atempo={f:.4f}")
    return parts


def _run_render_task(render_data: dict) -> dict:
    clips = render_data.get("clips", [])
    audio = render_data.get("audio", {"mode": "original"})
    output_path_raw = render_data.get("output_path", "")
    overwrite = render_data.get("overwrite", True)
    move_sources_to = render_data.get("move_sources_to", "")
    cleanup_paths = render_data.get("cleanup_paths", [])

    try:
        output_path = validate_data_path(_remap(output_path_raw))
    except RuntimeError as exc:
        state._set(state="idle", error=str(exc)[:200])
        return {"success": False, "error": str(exc)}

    state._set(state="render", out_name=os.path.basename(output_path), error="")
    state._log(f"Render gestartet: {len(clips)} Clip(s) → {os.path.basename(output_path)}")

    try:
        probes: list = []
        for c in clips:
            try:
                p = validate_data_path(_remap(c.get("path", "")))
            except RuntimeError as exc:
                raise RuntimeError(f"Ungültiger Clip-Pfad: {exc}") from exc
            probes.append((p, _probe_clip(p)))

        use_music = (
            audio.get("mode") in ("replace_with_music", "mix_music")
            and bool(audio.get("musicFile", {}).get("path"))
        )
        music_idx = len(clips)

        input_args: list = []
        for path, _ in probes:
            input_args += ["-i", path]
        if use_music:
            try:
                music_path = validate_data_path(_remap(audio["musicFile"]["path"]))
            except RuntimeError as exc:
                raise RuntimeError(f"Ungültiger Musik-Pfad: {exc}") from exc
            if audio.get("loop"):
                input_args += ["-stream_loop", "-1"]
            input_args += ["-i", music_path]

        filters: list = []
        pairs: list = []
        for i, (c, (path, probe)) in enumerate(zip(clips, probes)):
            sf = float(c.get("speed_factor", 1.0))
            vol = 0.0 if c.get("mute_audio") else max(0.0, min(3.0, float(c.get("volume_factor", 1.0))))
            dur = probe.get("duration", 0.0)

            sf_int = round(sf)
            if sf_int >= 2 and abs(sf - sf_int) < 0.01:
                vf = (f"[{i}:v]select='not(mod(n,{sf_int}))',setpts=N/FR/TB,"
                      f"scale=1920:-2,setsar=1,fps=30,format=yuv420p[v{i}]")
            else:
                vf = (f"[{i}:v]setpts={1/sf:.6f}*(PTS-STARTPTS),"
                      f"scale=1920:-2,setsar=1,fps=30,format=yuv420p[v{i}]")
            filters.append(vf)

            if c.get("mute_audio") or not probe.get("has_audio"):
                dur_trim = (f",atrim=duration={dur:.4f},asetpts=PTS-STARTPTS" if dur > 0 else "")
                filters.append(
                    f"anullsrc=r=48000:cl=stereo{dur_trim},"
                    f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
                )
            else:
                af: list = []
                if sf != 1.0:
                    af += _atempo_chain(sf)
                if vol != 1.0:
                    af.append(f"volume={vol:.3f}")
                af += [
                    "asetpts=PTS-STARTPTS",
                    "aresample=48000",
                    "aformat=sample_fmts=fltp:channel_layouts=stereo",
                ]
                filters.append(f"[{i}:a]{','.join(af)}[a{i}]")

            pairs.append(f"[v{i}][a{i}]")

        filters.append(f"{''.join(pairs)}concat=n={len(clips)}:v=1:a=1[vout][aorig]")

        audio_map = "[aorig]"
        if use_music:
            mv = max(0.0, min(3.0, float(audio.get("musicVolume", 0.35))))
            mf = f"[{music_idx}:a]volume={mv:.3f}"
            if not audio.get("loop"):
                mf += ",apad"
            mf += "[amusic]"
            filters.append(mf)
            if audio.get("mode") == "replace_with_music":
                audio_map = "[amusic]"
            else:
                filters.append("[aorig][amusic]amix=inputs=2:duration=first:dropout_transition=2[aout]")
                audio_map = "[aout]"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            "ffmpeg", "-hide_banner",
            "-y" if overwrite else "-n",
            *input_args,
            "-filter_complex", ";".join(filters),
            "-map", "[vout]",
            "-map", audio_map,
            "-c:v", "libx264", "-crf", "22", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        state._log(f"FFmpeg startet ({len(clips)} Clips, audio={audio.get('mode', 'original')})")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        state.set_render_process(proc)
        _, stderr = proc.communicate()
        rc = proc.returncode
        state.set_render_process(None)

        cancelled = state.check_and_clear_cancel()

        if cancelled:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            state._set(state="idle", error="Render abgebrochen")
            state._log("Render abgebrochen durch Benutzer")
            return {"success": False, "error": "cancelled"}

        if rc != 0:
            err = (stderr or "FFmpeg fehlgeschlagen")[-800:]
            state._log(f"FFmpeg Fehler (rc={rc}): {err[-300:]}")
            state._set(state="idle", error=err[-200:])
            return {"success": False, "error": err}

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            err = "FFmpeg: Ausgabedatei fehlt oder leer"
            state._set(state="idle", error=err)
            return {"success": False, "error": err}

        size_bytes = os.path.getsize(output_path)
        size_str = format_bytes(size_bytes)
        state._log(f"Render fertig: {os.path.basename(output_path)} ({size_str})")

        if move_sources_to:
            try:
                dest_dir = validate_data_path(_remap(move_sources_to))
                os.makedirs(dest_dir, exist_ok=True)
                seen: set = set()
                for c in clips:
                    src = _remap(c.get("path", ""))
                    if not src or src in seen or not os.path.exists(src):
                        continue
                    seen.add(src)
                    base = os.path.basename(src)
                    dest = os.path.join(dest_dir, base)
                    if os.path.exists(dest):
                        stem, ext = os.path.splitext(base)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dest = os.path.join(dest_dir, f"{stem}_{ts}{ext}")
                    os.rename(src, dest)
                    state._log(f"Verschoben: {base} → 04_done")
            except Exception as exc:
                state._log(f"Quelldateien verschieben fehlgeschlagen: {exc}")

        for p in cleanup_paths:
            try:
                rp = validate_data_path(_remap(p))
                if os.path.exists(rp):
                    os.remove(rp)
                    state._log(f"Bereinigt: {os.path.basename(rp)}")
            except Exception as exc:
                state._log(f"Bereinigung fehlgeschlagen: {exc}")

        state._set(state="idle", error="")
        return {
            "success": True,
            "out_path": output_path,
            "out_name": os.path.basename(output_path),
            "size": size_str,
            "size_bytes": size_bytes,
        }

    except Exception as exc:
        err = str(exc)[:500]
        state._log(f"Render-Fehler: {err}")
        state._set(state="idle", error=err[:200])
        return {"success": False, "error": err}
