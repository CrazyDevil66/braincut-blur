import logging
import time

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import config
import state
from blur_core import process_jobs
from detection import _yolov8_detect
from models import (
    _get_catalog,
    _get_installed_models,
    _install_lock,
    _install_model_bg,
    _install_progress,
    _load_model_config,
    _refresh_catalog_sync,
    _save_model_config,
    _validate_model_id,
)
from paths import DATA_ROOT, _remap, validate_data_path
from render_core import _run_render_task
from ui import HTML

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="BrainCut Blur Service")


def _startup_check() -> None:
    state._log(f"Pfad-Mapping: {config._media_host_path} → {config._CONTAINER_ROOT}")
    state._log(f"Modell-Pfad: {config._MODELS_PATH}")
    if config._MODEL_CATALOG_URL:
        state._log(f"Katalog-URL: {config._MODEL_CATALOG_URL}")
    if config._n8n_ip:
        state._log(f"N8N-Server: {config._n8n_ip}:{config._n8n_port}")
    else:
        state._log("N8N_SERVER_IP nicht gesetzt – Completion-Webhook deaktiviert")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        state._log(f"ONNX Runtime verfügbar. Provider: {providers}")
        if "CUDAExecutionProvider" in providers:
            state._log("GPU: CUDAExecutionProvider aktiv – GPU wird genutzt")
        else:
            state._log("GPU: CUDA nicht verfügbar – läuft auf CPU")
    except Exception as exc:
        state._log(f"ONNX-Check fehlgeschlagen: {exc}")
    try:
        from deface.centerface import CenterFace  # noqa: F401
        state._log("deface CenterFace-Modell geladen")
    except Exception as exc:
        state._log(f"deface-Import fehlgeschlagen: {exc}")
    cfg = _load_model_config()
    state._log(f"Aktives Gesichts-Modell: {cfg.get('face_model', 'builtin-centerface')}")
    state._log(f"Aktives Kennzeichen-Modell: {cfg.get('plate_model') or '–'}")


@app.on_event("startup")
async def on_startup():
    _startup_check()


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTML


@app.get("/status")
def status():
    with state._lock:
        s = dict(state._status)
    ts = s.get("started_at_ts", 0.0)
    s["elapsed_seconds"] = int(time.time() - ts) if ts > 0 and s.get("state") != "idle" else 0
    return s


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/blur")
def blur(data: dict, bg: BackgroundTasks):
    jobs = data.get("jobs", [])
    resume_url = data.get("resumeUrl", "")
    status_url = data.get("statusUrl", "")
    full_job = data.get("fullJob", {})
    with state._lock:
        if state._status["state"] != "idle":
            return JSONResponse(
                status_code=409,
                content={"error": "Job läuft bereits", "state": state._status["state"]},
            )
        state._status["state"] = "queued"
    state._log(f"Blur-Auftrag empfangen: {len(jobs)} Job(s)")
    bg.add_task(process_jobs, jobs, resume_url, status_url, full_job)
    return {"status": "queued", "count": len(jobs)}


@app.post("/cancel")
def cancel_job():
    state.request_cancel()
    state._log("⚠️ Abbruch angefordert")
    proc = state.get_render_process()
    if proc is not None:
        try:
            proc.terminate()
            state._log("FFmpeg-Prozess beendet (SIGTERM)")
        except Exception as exc:
            state._log(f"FFmpeg beenden fehlgeschlagen: {exc}")
    return {"status": "cancel_requested"}


@app.post("/run-shell")
def run_shell(data: dict):
    return JSONResponse(
        status_code=410,
        content={"error": "/run-shell wurde aus Sicherheitsgründen entfernt. Bitte N8N-Workflow aktualisieren."},
    )


@app.post("/render")
def render_endpoint(data: dict):
    with state._lock:
        if state._status["state"] != "idle":
            return JSONResponse(
                status_code=409,
                content={"error": "Job läuft bereits", "state": state._status["state"]},
            )
        state._status["state"] = "queued"
    return _run_render_task(data)


@app.post("/job-control")
def job_control(data: dict):
    action = data.get("action", "status")
    if action == "cancel":
        state.request_cancel()
        state._log("⚠️ Abbruch angefordert (job-control)")
        with state._lock:
            return {"action": "cancel", "state": state._status.get("state"), "name": state._status.get("name", "")}
    with state._lock:
        s = dict(state._status)
    s["action"] = "status"
    return s


@app.get("/api/models")
def api_models():
    catalog = _get_catalog()
    installed = _get_installed_models()
    cfg = _load_model_config()
    with _install_lock:
        progress = dict(_install_progress)
    return {
        "catalog": catalog.get("models", []),
        "catalog_source": catalog.get("source", "integriert"),
        "installed": installed,
        "config": cfg,
        "install_progress": progress,
    }


@app.post("/api/models/refresh")
def api_models_refresh():
    cat = _refresh_catalog_sync()
    return {"ok": True, "count": len(cat.get("models", [])), "source": cat.get("source", "")}


@app.post("/api/models/install")
def api_models_install(data: dict, bg: BackgroundTasks):
    model_id = data.get("id", "").strip()
    url = data.get("url", "").strip()
    hf_token = data.get("hf_token", "").strip()
    if not model_id or not url:
        return {"error": "id und url erforderlich"}
    if not _validate_model_id(model_id):
        return {"error": "Ungültige model_id – nur A-Z, a-z, 0-9, ., _, - erlaubt (max. 80 Zeichen)"}
    bg.add_task(_install_model_bg, model_id, url, hf_token)
    return {"ok": True, "id": model_id}


@app.post("/api/models/activate")
def api_models_activate(data: dict):
    model_type = data.get("type", "")
    model_id = data.get("id")
    if model_type not in ("face", "plate"):
        return {"error": "type muss 'face' oder 'plate' sein"}
    cfg = _load_model_config()
    if model_type == "face":
        cfg["face_model"] = model_id
    else:
        cfg["plate_model"] = model_id
    _save_model_config(cfg)
    state._log(f"Aktives {model_type}-Modell geändert: {model_id}")
    return {"ok": True, "config": cfg}


@app.get("/api/config")
def api_config_get():
    cfg = _load_model_config()
    return {
        "detection_interval": cfg.get("detection_interval", config._DETECTION_INTERVAL),
        "plate_conf_thresh": cfg.get("plate_conf_thresh", config._PLATE_CONF_THRESH),
    }


@app.post("/api/config")
def api_config_set(data: dict):
    cfg = _load_model_config()
    if "detection_interval" in data:
        cfg["detection_interval"] = max(1, min(8, int(data["detection_interval"])))
    if "plate_conf_thresh" in data:
        cfg["plate_conf_thresh"] = round(max(0.3, min(0.8, float(data["plate_conf_thresh"]))), 2)
    _save_model_config(cfg)
    return {"ok": True}


@app.post("/api/models/test")
def api_models_test(data: dict):
    import cv2
    model_id = data.get("model_id", "").strip()
    image_path_raw = data.get("image", "").strip()
    if not model_id or not image_path_raw:
        return {"error": "model_id und image erforderlich"}
    if not _validate_model_id(model_id):
        return {"error": "Ungültige model_id"}
    try:
        image_path = validate_data_path(_remap(image_path_raw))
    except RuntimeError as exc:
        return {"error": str(exc)}
    model_file = config._MODELS_PATH / f"{model_id}.onnx"
    if not model_file.exists():
        return {"error": f"Modell nicht gefunden: {model_id}"}
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(model_file), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        frame = cv2.imread(image_path)
        if frame is None:
            return {"error": f"Bild konnte nicht geladen werden: {image_path}"}
        boxes = _yolov8_detect(sess, frame, conf_thresh=0.3)
        return {"detections": len(boxes), "boxes": [[int(v) for v in b] for b in boxes]}
    except Exception as exc:
        return {"error": str(exc)[:300]}


@app.delete("/api/models/{model_id}")
def api_models_delete(model_id: str):
    if not _validate_model_id(model_id):
        return {"error": "Ungültige model_id"}
    if model_id == "builtin-centerface":
        return {"error": "Integriertes Modell kann nicht gelöscht werden"}
    target = (config._MODELS_PATH / f"{model_id}.onnx").resolve()
    if DATA_ROOT not in target.parents:
        return {"error": "Ungültiger Pfad"}
    if not target.exists():
        return {"error": "Modell nicht gefunden"}
    target.unlink()
    cfg = _load_model_config()
    changed = False
    if cfg.get("face_model") == model_id:
        cfg["face_model"] = "builtin-centerface"
        changed = True
    if cfg.get("plate_model") == model_id:
        cfg["plate_model"] = None
        changed = True
    if changed:
        _save_model_config(cfg)
    state._log(f"Modell gelöscht: {model_id}")
    return {"ok": True}
