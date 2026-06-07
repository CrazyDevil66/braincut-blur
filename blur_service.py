import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from threading import Lock

import requests
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="BrainCut Blur Service")

# ── Globaler Status ───────────────────────────────────────────────────────────
_lock = Lock()
_cancel_flag = False
_n8n_ip = os.getenv("N8N_SERVER_IP", "")
COMPLETION_WEBHOOK = f"http://{_n8n_ip}:5678/webhook/blur-done" if _n8n_ip else ""

_status = {
    "state": "idle",
    "operation": "",
    "current": 0,
    "total": 0,
    "name": "",
    "out_name": "",
    "size": "",
    "error": "",
    "started_at": "",
    "frame_current": 0,
    "frame_total": 0,
    "frame_pct": 0,
    "eta_seconds": 0,
    "logs": [],
}

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    log.info(msg)
    with _lock:
        _status["logs"].append(entry)
        if len(_status["logs"]) > 200:
            _status["logs"] = _status["logs"][-200:]

def _set(**kwargs):
    with _lock:
        _status.update(kwargs)

def _format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "–"
    if seconds < 60:
        return f"{seconds} Sek"
    m, s = divmod(seconds, 60)
    return f"{m} Min {s} Sek" if s else f"{m} Min"

# ── Startup-Check ─────────────────────────────────────────────────────────────
def _startup_check():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        _log(f"ONNX Runtime verfügbar. Provider: {providers}")
        if "CUDAExecutionProvider" in providers:
            _log("GPU: CUDAExecutionProvider aktiv – GPU wird genutzt")
        else:
            _log("GPU: CUDA nicht verfügbar – läuft auf CPU")
    except Exception as exc:
        _log(f"ONNX-Check fehlgeschlagen: {exc}")

    try:
        from deface.centerface import CenterFace
        _log("deface CenterFace-Modell geladen")
    except Exception as exc:
        _log(f"deface-Import fehlgeschlagen: {exc}")

@app.on_event("startup")
async def on_startup():
    _startup_check()

# ── Web-GUI ───────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainCut Blur Service</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 24px; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 20px; color: #f8fafc; }
  .card { background: #1e2330; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .row { display: flex; align-items: center; gap: 12px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .dot.idle    { background: #64748b; }
  .dot.running { background: #22c55e; box-shadow: 0 0 8px #22c55e; animation: pulse 1.2s infinite; }
  .dot.error   { background: #ef4444; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .state-label { font-size: 1.1rem; font-weight: 600; }
  .sub { color: #94a3b8; font-size: 0.85rem; margin-top: 4px; }
  .progress-wrap { background: #0f1117; border-radius: 8px; height: 10px; margin-top: 14px; overflow: hidden; }
  .progress-bar { height: 100%; background: linear-gradient(90deg,#6366f1,#22c55e); border-radius: 8px; transition: width .4s; }
  .progress-labels { display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.75rem; color: #64748b; }
  .log-box { background: #0f1117; border-radius: 8px; padding: 12px; height: 340px; overflow-y: auto;
             font-family: monospace; font-size: 0.78rem; color: #94a3b8; }
  .log-box .entry { padding: 2px 0; border-bottom: 1px solid #1e2330; }
  .log-box .entry:last-child { color: #e2e8f0; }
  .badge { display: inline-block; background: #334155; border-radius: 6px; padding: 2px 8px;
           font-size: 0.75rem; margin-left: 8px; }
  .refresh { color: #475569; font-size: 0.75rem; margin-top: 8px; }
</style>
</head>
<body>
<h1>BrainCut Blur Service</h1>

<div class="card" id="statusCard">
  <div class="row">
    <div class="dot idle" id="dot"></div>
    <div>
      <div class="state-label" id="stateLabel">Laden...</div>
      <div class="sub" id="subLabel"></div>
    </div>
  </div>
  <div id="progressSection" style="display:none">
    <div class="progress-wrap">
      <div class="progress-bar" id="progressBar" style="width:0%"></div>
    </div>
    <div class="progress-labels">
      <span id="progressLeft"></span>
      <span id="progressRight"></span>
    </div>
  </div>
</div>

<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-weight:600;font-size:.9rem">Protokoll</span>
    <span class="badge" id="logCount">0 Einträge</span>
  </div>
  <div class="log-box" id="logBox"></div>
  <div class="refresh" id="refreshTs"></div>
</div>

<script>
function formatEta(s) {
  if (!s || s <= 0) return '';
  if (s < 60) return s + ' Sek';
  const m = Math.floor(s / 60), r = s % 60;
  return r ? m + ' Min ' + r + ' Sek' : m + ' Min';
}

async function refresh() {
  try {
    const r = await fetch('/status');
    const d = await r.json();

    const dot = document.getElementById('dot');
    const label = document.getElementById('stateLabel');
    const sub = document.getElementById('subLabel');
    const section = document.getElementById('progressSection');
    const bar = document.getElementById('progressBar');
    const pLeft = document.getElementById('progressLeft');
    const pRight = document.getElementById('progressRight');

    dot.className = 'dot ' + (d.state === 'idle' ? 'idle' : d.error ? 'error' : 'running');

    if (d.state === 'idle') {
      label.textContent = 'Bereit';
      sub.textContent = d.error ? 'Letzter Fehler: ' + d.error : 'Warte auf Auftrag...';
      section.style.display = 'none';

    } else if (d.state === 'blur') {
      label.textContent = 'Blur läuft – ' + (d.name || '');
      sub.textContent = 'Video ' + d.current + ' von ' + d.total;
      section.style.display = 'block';

      if (d.frame_total > 0) {
        bar.style.width = d.frame_pct + '%';
        pLeft.textContent = d.frame_pct + '% – ' + d.frame_current.toLocaleString('de-DE') + ' / ' + d.frame_total.toLocaleString('de-DE') + ' Frames';
        const eta = formatEta(d.eta_seconds);
        pRight.textContent = eta ? '~' + eta + ' verbleibend' : '';
      } else {
        bar.style.width = (d.total ? Math.round((d.current / d.total) * 100) : 0) + '%';
        pLeft.textContent = '';
        pRight.textContent = '';
      }

    } else if (d.state === 'render') {
      label.textContent = 'FFmpeg rendert – ' + (d.out_name || '');
      sub.textContent = d.started_at ? 'Gestartet: ' + d.started_at : '';
      section.style.display = 'block';
      bar.style.width = '100%';
      pLeft.textContent = 'Rendering...';
      pRight.textContent = '';
    }

    const box = document.getElementById('logBox');
    const wasAtBottom = box.scrollHeight - box.clientHeight <= box.scrollTop + 20;
    box.innerHTML = d.logs.map(l => '<div class="entry">' + l + '</div>').join('');
    if (wasAtBottom) box.scrollTop = box.scrollHeight;

    document.getElementById('logCount').textContent = d.logs.length + ' Einträge';
    document.getElementById('refreshTs').textContent =
      'Aktualisiert: ' + new Date().toLocaleTimeString('de-DE');
  } catch(e) {
    document.getElementById('stateLabel').textContent = 'Verbindung verloren';
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTML


@app.get("/status")
def status():
    with _lock:
        return dict(_status)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/blur")
def blur(data: dict, bg: BackgroundTasks):
    jobs = data.get("jobs", [])
    resume_url = data.get("resumeUrl", "")
    status_url = data.get("statusUrl", "")
    _log(f"Blur-Auftrag empfangen: {len(jobs)} Job(s)")
    bg.add_task(process_jobs, jobs, resume_url, status_url)
    return {"status": "queued", "count": len(jobs)}


@app.post("/cancel")
def cancel_job():
    global _cancel_flag
    with _lock:
        _cancel_flag = True
    _log("⚠️ Abbruch angefordert")
    return {"status": "cancel_requested"}


@app.post("/job-control")
def job_control(data: dict):
    global _cancel_flag
    action = data.get("action", "status")
    if action == "cancel":
        with _lock:
            _cancel_flag = True
        _log("⚠️ Abbruch angefordert (job-control)")
        with _lock:
            return {"action": "cancel", "state": _status.get("state"), "name": _status.get("name", "")}
    else:
        with _lock:
            s = dict(_status)
        s["action"] = "status"
        return s


@app.post("/render")
def render(data: dict, bg: BackgroundTasks):
    cmd = data.get("cmd", "")
    resume_url = data.get("resumeUrl", "")
    status_url = data.get("statusUrl", "")
    out_name = data.get("outName", "")
    out_path = data.get("outPath", "")
    _log(f"Render-Auftrag empfangen: {out_name}")
    bg.add_task(run_render, cmd, resume_url, status_url, out_name, out_path)
    return {"status": "queued"}


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def post_status(status_url: str, data: dict):
    if not status_url:
        return
    try:
        requests.post(status_url, json=data, timeout=5)
    except Exception as exc:
        _log(f"Status-Update fehlgeschlagen: {exc}")


def format_bytes(b: int) -> str:
    if b > 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GB"
    if b > 1_048_576:
        return f"{b / 1_048_576:.1f} MB"
    return f"{b / 1024:.0f} KB"


def wakeup_disk(path: str):
    try:
        with open(path, "rb") as f:
            f.read(4096)
        _log(f"Datenträger aktiv: {os.path.basename(path)}")
    except Exception as exc:
        _log(f"Disk-Wakeup fehlgeschlagen: {exc}")


# ── Blur-Verarbeitung ─────────────────────────────────────────────────────────
def process_jobs(jobs: list, resume_url: str, status_url: str = ""):
    total = len(jobs)
    errors = []
    was_cancelled = False

    _set(state="blur", current=0, total=total, error="",
         frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0,
         started_at=datetime.now().strftime("%H:%M:%S"))
    _log(f"Blur gestartet: {total} Video(s)")
    post_status(status_url, {"event": "start", "total": total})

    for i, job in enumerate(jobs, 1):
        input_path = job.get("input_path", "")
        output_path = job.get("output_path", "")
        blur_faces = job.get("blur_faces", False)
        blur_plates = job.get("blur_plates", False)
        name = os.path.basename(input_path)

        _set(current=i, name=name, frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0)
        _log(f"[{i}/{total}] Starte: {name} (faces={blur_faces}, plates={blur_plates})")
        post_status(status_url, {"event": "progress_start", "current": i, "total": total, "name": name})

        wakeup_disk(input_path)
        detection_resolution = job.get("detection_resolution", "720p")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if blur_faces and blur_plates:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    run_deface(input_path, tmp_path, mode="faces", status_url=status_url, job_name=name, detection_resolution=detection_resolution)
                    run_deface(tmp_path, output_path, mode="plates", status_url=status_url, job_name=name, detection_resolution=detection_resolution)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            elif blur_faces:
                run_deface(input_path, output_path, mode="faces", status_url=status_url, job_name=name, detection_resolution=detection_resolution)
            elif blur_plates:
                run_deface(input_path, output_path, mode="plates", status_url=status_url, job_name=name, detection_resolution=detection_resolution)
            else:
                subprocess.run(["cp", "--", input_path, output_path], check=True)

            _log(f"[{i}/{total}] Fertig: {name}")
            post_status(status_url, {"event": "progress_done", "current": i, "total": total, "name": name})
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower():
                global _cancel_flag
                with _lock:
                    _cancel_flag = False
                _log(f"[{i}/{total}] Job abgebrochen.")
                errors.append({"input": input_path, "error": "Abgebrochen"})
                post_status(status_url, {"event": "cancelled", "current": i, "total": total, "name": name})
                was_cancelled = True
                break
            err = str(exc)
            _log(f"[{i}/{total}] FEHLER: {err[:300]}")
            errors.append({"input": input_path, "error": err})
            post_status(status_url, {"event": "error", "current": i, "total": total, "name": name, "error": err[:500]})
        except Exception as exc:
            err = str(exc)
            _log(f"[{i}/{total}] FEHLER: {err[:300]}")
            errors.append({"input": input_path, "error": err})
            post_status(status_url, {"event": "error", "current": i, "total": total, "name": name, "error": err[:500]})

    _log(f"Blur abgeschlossen. Fehler: {len(errors)}")
    _set(state="idle", error=errors[0]["error"][:200] if errors else "",
         frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0)
    post_status(status_url, {"event": "done", "total": total, "errors": errors})

    if resume_url:
        try:
            requests.post(resume_url, json={"status": "done", "errors": errors}, timeout=15)
            _log("Blur-Callback gesendet")
        except Exception as exc:
            _log(f"Blur-Callback fehlgeschlagen: {exc}")

    if COMPLETION_WEBHOOK and not was_cancelled:
        try:
            requests.post(COMPLETION_WEBHOOK, json={"status": "done", "errors": errors}, timeout=10)
            _log("Completion-Webhook gesendet")
        except Exception as exc:
            _log(f"Completion-Webhook fehlgeschlagen: {exc}")


# ── Render-Verarbeitung ───────────────────────────────────────────────────────
def run_render(cmd: str, resume_url: str, status_url: str, out_name: str, out_path: str):
    _set(state="render", out_name=out_name, error="",
         started_at=datetime.now().strftime("%H:%M:%S"))
    _log(f"FFmpeg startet: {out_name}")
    post_status(status_url, {"event": "render_start", "outName": out_name})

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=7200,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

        size = "unbekannt"
        m = re.search(r"__BRAINCUT_SIZE__=(\d+)", stdout)
        if m:
            size = format_bytes(int(m.group(1)))

        success = exit_code == 0 and bool(m)

        if success:
            _log(f"FFmpeg fertig: {out_name} ({size})")
            _set(state="idle", size=size, error="")
            post_status(status_url, {"event": "render_done", "outName": out_name, "size": size})
        else:
            err = stderr[-800:] if stderr else stdout[-800:]
            _log(f"FFmpeg Fehler (exit {exit_code}): {err[:300]}")
            _set(state="idle", error=err[:200])
            post_status(status_url, {"event": "render_error", "error": err[:400]})

        if resume_url:
            requests.post(resume_url, json={
                "success": success, "exitCode": exit_code,
                "stdout": stdout[-2000:],
                "stderr": stderr[-1200:] if not success else "",
                "size": size, "outName": out_name, "outPath": out_path,
            }, timeout=15)
            _log("Render-Callback gesendet")

    except subprocess.TimeoutExpired:
        err = "FFmpeg-Timeout nach 2 Stunden."
        _log(err); _set(state="idle", error=err)
        post_status(status_url, {"event": "render_error", "error": err})
        if resume_url:
            requests.post(resume_url, json={"success": False, "error": err, "outName": out_name}, timeout=15)
    except Exception as exc:
        err = str(exc)[:500]
        _log(f"Render-Fehler: {err}"); _set(state="idle", error=err)
        post_status(status_url, {"event": "render_error", "error": err})
        if resume_url:
            requests.post(resume_url, json={"success": False, "error": err, "outName": out_name}, timeout=15)


# ── deface: direkt per Python API ────────────────────────────────────────────
def run_deface(input_path: str, output_path: str, mode: str = "faces",
               status_url: str = "", job_name: str = "",
               detection_resolution: str = "720p"):
    _log(f"deface [{mode}] startet: {os.path.basename(input_path)}")

    if mode == "plates":
        weights = os.environ.get("PLATE_MODEL_PATH", "")
        if not (weights and os.path.exists(weights)):
            _log("Kein Kennzeichen-Modell – Datei wird kopiert.")
            subprocess.run(["cp", "--", input_path, output_path], check=True)
            return

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            _log("deface: nutze GPU (CUDA)")
        else:
            providers = ["CPUExecutionProvider"]
            _log("deface: CUDA nicht verfügbar – nutze CPU")
    except ImportError:
        providers = None
        _log("deface: onnxruntime nicht gefunden – OpenCV-Fallback")

    import cv2
    from deface.centerface import CenterFace

    res_map = {"720p": (1280, 720), "1080p": (1920, 1080), "native": None}
    in_shape = res_map.get(detection_resolution, (1280, 720))
    _log(f"deface: Erkennungsauflösung {detection_resolution} → in_shape={in_shape}")

    cf = CenterFace(in_shape=in_shape, backend="onnxrt")
    if providers is not None:
        try:
            cf.sess.set_providers(providers)
        except Exception:
            pass

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Kann Video nicht öffnen: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _log(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} Frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    start_time = time.time()
    last_log_time = 0.0
    last_milestone = 0  # letzte gemeldete 25%-Schwelle

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with _lock:
            if _cancel_flag:
                _log(f"  ⚠️ Abbruch – {frame_idx}/{total_frames} Frames verarbeitet")
                cap.release()
                out.release()
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise RuntimeError("cancelled")
        frame_idx += 1

        dets, _ = cf(frame, threshold=0.2)
        for x, y, x2, y2, _ in dets:
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
            roi = frame[y:y2, x:x2]
            if roi.size > 0:
                bw = max(1, (x2 - x) // 10)
                bh = max(1, (y2 - y) // 10)
                blurred = cv2.resize(cv2.resize(roi, (bw, bh)), (x2 - x, y2 - y))
                frame[y:y2, x:x2] = blurred

        out.write(frame)

        now = time.time()
        elapsed = now - start_time

        if total_frames > 0 and elapsed > 0:
            pct = int(frame_idx / total_frames * 100)
            fps_actual = frame_idx / elapsed
            remaining = max(0, total_frames - frame_idx)
            eta = int(remaining / fps_actual)

            _set(frame_current=frame_idx, frame_total=total_frames,
                 frame_pct=pct, eta_seconds=eta)

            # Log alle 10 Sekunden
            if now - last_log_time >= 10:
                _log(f"  {pct}% | {frame_idx:,}/{total_frames:,} Frames | ~{_format_eta(eta)} verbleibend")
                last_log_time = now

            # 25%-Meilensteine an statusUrl
            milestone = (pct // 25) * 25
            if milestone > 0 and milestone > last_milestone:
                last_milestone = milestone
                post_status(status_url, {
                    "event": "frame_progress",
                    "name": job_name,
                    "mode": mode,
                    "pct": milestone,
                    "frame_current": frame_idx,
                    "frame_total": total_frames,
                    "eta_seconds": eta,
                    "eta_human": _format_eta(eta),
                })

    cap.release()
    out.release()

    # mp4v → h264 re-encodieren
    tmp_out = output_path + ".tmp.mp4"
    os.rename(output_path, tmp_out)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_out, "-c:v", "libx264", "-crf", "18",
         "-preset", "fast", "-c:a", "copy", output_path],
        capture_output=True, text=True
    )
    os.remove(tmp_out)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg re-encode Fehler: {result.stderr[-400:]}")

    _log(f"deface [{mode}] abgeschlossen: {frame_idx:,} Frames verarbeitet")
