import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
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
_n8n_port = os.getenv("N8N_SERVER_PORT", "5678")
COMPLETION_WEBHOOK = f"http://{_n8n_ip}:{_n8n_port}/webhook/blur-done" if _n8n_ip else ""

# ── Pfad-Remapping (Docker-Volume) ────────────────────────────────────────────
_media_host_path = os.getenv("MEDIA_HOST_PATH", "/mnt/user/n8n_automation/BrainCut").rstrip("/")
_CONTAINER_ROOT = "/data"

def _remap(path: str) -> str:
    if path.startswith(_media_host_path):
        return _CONTAINER_ROOT + path[len(_media_host_path):]
    return path

def _fix_status_url(url: str) -> str:
    if _n8n_ip and "localhost" in url:
        return url.replace("localhost", _n8n_ip)
    return url

# ── Modell-Verwaltung ─────────────────────────────────────────────────────────
_MODELS_PATH = Path(os.getenv("MODELS_PATH", "/app/.cache/models"))
_MODEL_CATALOG_URL = os.getenv("MODEL_CATALOG_URL", "")
_install_progress: dict = {}
_install_lock = Lock()

_BUILTIN_CATALOG: dict = {
    "version": "1.0",
    "updated": "2026-06-08",
    "models": [
        {
            "id": "builtin-centerface",
            "name": "CenterFace (Standard)",
            "type": "face",
            "format": "centerface",
            "builtin": True,
            "url": "",
            "size_mb": 5.3,
            "description": "Frontale Gesichtserkennung. Im Container integriert – kein Download nötig.",
        },
        {
            "id": "yolov8n-plates-eu",
            "name": "YOLOv11n Kennzeichen (schnell)",
            "type": "plate",
            "format": "yolov8",
            "builtin": False,
            "url": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.onnx",
            "size_mb": 10.5,
            "description": "YOLOv11n Kennzeichenerkennung – schnell, GPU-optimiert. (morsetechlab, MIT)",
        },
        {
            "id": "yolov8s-plates-eu",
            "name": "YOLOv11s Kennzeichen (genau)",
            "type": "plate",
            "format": "yolov8",
            "builtin": False,
            "url": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.onnx",
            "size_mb": 37.8,
            "description": "YOLOv11s Kennzeichenerkennung – höhere Genauigkeit, etwas langsamer. (morsetechlab, MIT)",
        },
    ],
}


def _models_config_path() -> Path:
    return _MODELS_PATH / "config.json"


def _load_model_config() -> dict:
    p = _models_config_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"face_model": "builtin-centerface", "plate_model": None}


def _save_model_config(cfg: dict):
    _MODELS_PATH.mkdir(parents=True, exist_ok=True)
    _models_config_path().write_text(json.dumps(cfg, indent=2))


def _get_catalog() -> dict:
    cached = _MODELS_PATH / "catalog.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text())
        except Exception:
            pass
    return dict(_BUILTIN_CATALOG)


def _refresh_catalog_sync() -> dict:
    if not _MODEL_CATALOG_URL:
        cat = dict(_BUILTIN_CATALOG)
        cat["source"] = "integriert"
    else:
        try:
            r = requests.get(_MODEL_CATALOG_URL, timeout=15)
            r.raise_for_status()
            cat = r.json()
            cat["source"] = _MODEL_CATALOG_URL
            _log(f"Katalog aktualisiert: {len(cat.get('models', []))} Modelle")
        except Exception as exc:
            _log(f"Katalog-Abruf fehlgeschlagen: {exc} – nutze integrierten Katalog")
            cat = dict(_BUILTIN_CATALOG)
            cat["source"] = "integriert (Fehler beim Abruf)"
    _MODELS_PATH.mkdir(parents=True, exist_ok=True)
    (_MODELS_PATH / "catalog.json").write_text(json.dumps(cat, indent=2))
    return cat


def _get_installed_models() -> dict:
    result: dict = {"builtin-centerface": {"builtin": True, "size_mb": 5.3}}
    if _MODELS_PATH.exists():
        for f in _MODELS_PATH.glob("*.onnx"):
            result[f.stem] = {"file": str(f), "size_mb": round(f.stat().st_size / 1_048_576, 1)}
    return result


def _install_model_bg(model_id: str, url: str, hf_token: str = ""):
    with _install_lock:
        _install_progress[model_id] = {"status": "downloading", "pct": 0, "error": ""}
    try:
        _MODELS_PATH.mkdir(parents=True, exist_ok=True)
        target = _MODELS_PATH / f"{model_id}.onnx"
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        r = requests.get(url, stream=True, timeout=120, allow_redirects=True, headers=headers)
        if r.status_code == 401:
            raise RuntimeError("401 Unauthorized – HuggingFace-Token erforderlich. Token im Einstellungen-Panel eingeben.")
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(99, int(downloaded / total * 100))
                        with _install_lock:
                            _install_progress[model_id]["pct"] = pct
        with _install_lock:
            _install_progress[model_id] = {"status": "done", "pct": 100, "error": ""}
        _log(f"Modell installiert: {model_id} ({round(target.stat().st_size/1_048_576,1)} MB)")
    except Exception as exc:
        with _install_lock:
            _install_progress[model_id] = {"status": "error", "pct": 0, "error": str(exc)[:300]}
        _log(f"Modell-Installation fehlgeschlagen ({model_id}): {exc}")


# ── Globaler Blur-Status ──────────────────────────────────────────────────────
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
    "started_at_ts": 0.0,
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
    _log(f"Pfad-Mapping: {_media_host_path} → {_CONTAINER_ROOT}")
    _log(f"Modell-Pfad: {_MODELS_PATH}")
    if _MODEL_CATALOG_URL:
        _log(f"Katalog-URL: {_MODEL_CATALOG_URL}")
    if _n8n_ip:
        _log(f"N8N-Server: {_n8n_ip}:{_n8n_port}")
    else:
        _log("N8N_SERVER_IP nicht gesetzt – Completion-Webhook deaktiviert")
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

    cfg = _load_model_config()
    _log(f"Aktives Gesichts-Modell: {cfg.get('face_model', 'builtin-centerface')}")
    _log(f"Aktives Kennzeichen-Modell: {cfg.get('plate_model') or '–'}")

@app.on_event("startup")
async def on_startup():
    _startup_check()

# ── Web-GUI ───────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainCut Blur</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #000;
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}
.container { max-width: 680px; margin: 0 auto; padding: 32px 20px; }
.header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px; }
.brand { display: flex; align-items: center; gap: 12px; }
.logo {
  width: 40px; height: 40px;
  background: linear-gradient(135deg,#0a84ff,#5e5ce6);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem;
}
.brand-text h1 { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.02em; }
.brand-text p  { font-size: 0.72rem; color: rgba(255,255,255,.38); margin-top: 1px; }

.status-pill {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 14px; background: #1c1c1e; border-radius: 20px;
  font-size: 0.78rem; font-weight: 500;
  border: 1px solid rgba(255,255,255,.08); transition: all .3s;
}
.status-dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
  background: rgba(255,255,255,.25);
}
.status-dot.active { background: #30d158; }
.status-dot.error  { background: #ff453a; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.pulse { animation: pulse 1.6s ease-in-out infinite; }

.gear-btn {
  width: 36px; height: 36px; margin-left: 8px;
  background: #1c1c1e; border: 1px solid rgba(255,255,255,.08);
  border-radius: 10px; color: rgba(255,255,255,.5);
  font-size: 1.05rem; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all .2s; flex-shrink: 0;
}
.gear-btn:hover { border-color: rgba(255,255,255,.2); color: #fff; }
.gear-btn.active { border-color: #0a84ff; color: #0a84ff; background: rgba(10,132,255,.1); }

.card {
  background: #1c1c1e; border-radius: 16px;
  padding: 18px 20px; margin-bottom: 12px;
  border: 1px solid rgba(255,255,255,.06);
}
.card-title {
  font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: rgba(255,255,255,.38); margin-bottom: 14px;
}

.err-banner {
  background: rgba(255,69,58,.1); border: 1px solid rgba(255,69,58,.28);
  border-radius: 12px; padding: 12px 16px; color: #ff453a;
  font-size: 0.82rem; margin-bottom: 12px; gap: 8px; align-items: flex-start;
}

.job-name {
  font-size: 0.95rem; font-weight: 600; margin-bottom: 3px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.job-sub { font-size: 0.75rem; color: rgba(255,255,255,.38); margin-bottom: 16px; }

.progress-track {
  height: 4px; background: rgba(255,255,255,.1); border-radius: 2px;
  overflow: hidden; margin-bottom: 8px;
}
.progress-fill {
  height: 100%; background: #0a84ff; border-radius: 2px;
  transition: width .5s ease; position: relative; overflow: hidden;
}
.progress-fill::after {
  content: ''; position: absolute; top: 0; left: -40px; width: 40px; height: 100%;
  background: linear-gradient(90deg,transparent,rgba(255,255,255,.4),transparent);
  animation: shimmer 1.6s linear infinite;
}
@keyframes shimmer { to { left: 110%; } }
.progress-meta {
  display: flex; justify-content: space-between;
  font-size: 0.72rem; color: rgba(255,255,255,.32);
}

.stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 14px; }
.stat { background: #2c2c2e; border-radius: 10px; padding: 10px 12px; }
.stat-lbl {
  font-size: 0.62rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .06em; color: rgba(255,255,255,.32); margin-bottom: 4px;
}
.stat-val { font-size: 1rem; font-weight: 600; }
.stat-val.blue  { color: #0a84ff; }
.stat-val.green { color: #30d158; }

.cancel-btn {
  background: transparent; border: 1px solid rgba(255,69,58,.4);
  color: #ff453a; padding: 5px 14px; border-radius: 8px;
  font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: all .2s;
}
.cancel-btn:hover:not(:disabled) { background: rgba(255,69,58,.1); border-color: #ff453a; }
.cancel-btn:disabled { opacity: .4; cursor: not-allowed; }

.render-row { display: flex; align-items: center; gap: 14px; }
.spinner {
  width: 20px; height: 20px; flex-shrink: 0;
  border: 2px solid rgba(255,255,255,.1); border-top-color: #0a84ff;
  border-radius: 50%; animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.render-name { font-size: 0.9rem; font-weight: 600; }
.render-sub  { font-size: 0.72rem; color: rgba(255,255,255,.38); margin-top: 2px; }

.log-box {
  background: #000; border-radius: 10px; height: 300px; overflow-y: auto;
  font-family: 'SF Mono','Menlo','Monaco','Consolas',monospace;
  font-size: 0.71rem; color: rgba(255,255,255,.28); padding: 12px 14px;
}
.log-box::-webkit-scrollbar { width: 4px; }
.log-box::-webkit-scrollbar-thumb { background: #3a3a3c; border-radius: 2px; }
.le { padding: 1.5px 0; line-height: 1.7; }
.le:last-child { color: rgba(255,255,255,.75); }
.le .ts { color: rgba(255,255,255,.2); }
.le.w { color: #ffd60a; }
.le.e { color: #ff453a; }
.log-foot {
  display: flex; justify-content: space-between; align-items: center;
  margin-top: 8px; font-size: 0.68rem; color: rgba(255,255,255,.25);
}
.as-btn { cursor: pointer; transition: color .2s; }
.as-btn.on { color: #0a84ff; }

/* ── Settings Drawer (CSS-transform-based, KEIN display:none toggle) ── */
.scrim {
  position: fixed; inset: 0; background: rgba(0,0,0,.6);
  z-index: 50; opacity: 0; pointer-events: none; transition: opacity .3s;
}
.scrim.open { opacity: 1; pointer-events: auto; }

.drawer {
  position: fixed; top: 0; right: 0;
  width: 420px; max-width: 100vw; height: 100%;
  background: #1c1c1e; z-index: 51;
  transform: translateX(100%);
  transition: transform .35s cubic-bezier(.4,0,.2,1);
  overflow-y: auto; display: flex; flex-direction: column;
  pointer-events: none;
}
.drawer.open { transform: translateX(0); pointer-events: auto; }

.drawer-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 20px 16px;
  border-bottom: 1px solid rgba(255,255,255,.06);
  flex-shrink: 0; position: sticky; top: 0; background: #1c1c1e; z-index: 1;
}
.drawer-title { font-size: 1rem; font-weight: 600; }
.drawer-close {
  background: #2c2c2e; border: none; color: rgba(255,255,255,.6);
  width: 28px; height: 28px; border-radius: 50%;
  font-size: 0.75rem; cursor: pointer;
  display: flex; align-items: center; justify-content: center; transition: all .2s;
}
.drawer-close:hover { background: #3a3a3c; color: #fff; }
.drawer-body { padding: 16px 20px; flex: 1; }

.catalog-bar {
  display: flex; align-items: center; justify-content: space-between;
  background: #2c2c2e; border-radius: 10px; padding: 10px 14px; margin-bottom: 16px;
}
.catalog-info { font-size: 0.75rem; color: rgba(255,255,255,.45); flex: 1; min-width: 0; }
.refresh-btn {
  background: transparent; border: 1px solid rgba(255,255,255,.15);
  color: rgba(255,255,255,.65); padding: 4px 12px; border-radius: 7px;
  font-size: 0.72rem; font-weight: 500; cursor: pointer; transition: all .2s;
  flex-shrink: 0; margin-left: 10px; font-family: inherit;
}
.refresh-btn:hover:not(:disabled) { border-color: #0a84ff; color: #0a84ff; }
.refresh-btn:disabled { opacity: .4; cursor: not-allowed; }

.seg-ctrl {
  display: flex; background: #2c2c2e; border-radius: 9px; padding: 2px; margin-bottom: 14px;
}
.seg-btn {
  flex: 1; padding: 7px 12px; border-radius: 7px; border: none;
  background: transparent; color: rgba(255,255,255,.45);
  font-size: 0.8rem; font-weight: 500; cursor: pointer; transition: all .2s;
  text-align: center; font-family: inherit;
}
.seg-btn.active { background: #3a3a3c; color: #fff; font-weight: 600; }

.model-list { display: flex; flex-direction: column; gap: 8px; }
.model-row {
  display: flex; align-items: center; gap: 12px;
  background: #2c2c2e; border-radius: 12px; padding: 12px 14px;
  border: 1px solid transparent; transition: border-color .2s;
}
.model-row.active { border-color: #0a84ff; }
.model-icon {
  width: 36px; height: 36px; border-radius: 8px; background: #3a3a3c;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; flex-shrink: 0;
}
.model-icon.fi { background: rgba(10,132,255,.15); }
.model-icon.pi { background: rgba(48,209,88,.15); }
.model-info { flex: 1; min-width: 0; }
.model-name { font-size: 0.85rem; font-weight: 600; }
.model-desc {
  font-size: 0.72rem; color: rgba(255,255,255,.38); margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.model-meta { font-size: 0.68rem; color: rgba(255,255,255,.22); margin-top: 2px; }
.active-badge {
  display: inline-block; background: rgba(10,132,255,.2); color: #0a84ff;
  font-size: 0.6rem; font-weight: 700; letter-spacing: .05em;
  padding: 2px 6px; border-radius: 5px; text-transform: uppercase;
  margin-left: 6px; vertical-align: middle;
}
.model-actions { display: flex; flex-direction: column; gap: 5px; flex-shrink: 0; }
.m-btn {
  padding: 5px 12px; border-radius: 7px; border: none;
  font-size: 0.72rem; font-weight: 600; cursor: pointer;
  transition: all .15s; white-space: nowrap; font-family: inherit;
}
.m-btn:disabled { opacity: .4; cursor: not-allowed; }
.m-btn-install  { background: rgba(10,132,255,.2); color: #0a84ff; }
.m-btn-install:hover:not(:disabled) { background: rgba(10,132,255,.35); }
.m-btn-activate { background: rgba(48,209,88,.15); color: #30d158; }
.m-btn-activate:hover:not(:disabled) { background: rgba(48,209,88,.28); }
.m-btn-delete   { background: rgba(255,69,58,.1); color: #ff453a; }
.m-btn-delete:hover:not(:disabled)   { background: rgba(255,69,58,.22); }

.dl-bar { height: 3px; background: rgba(255,255,255,.1); border-radius: 2px; margin-top: 6px; overflow: hidden; }
.dl-bar-fill { height: 100%; background: #0a84ff; border-radius: 2px; transition: width .3s; }
</style>
</head>
<body>

<!-- Scrim -->
<div class="scrim" id="scrim"></div>

<!-- Settings Drawer -->
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <span class="drawer-title">Einstellungen</span>
    <button class="drawer-close" id="drawerClose">&#x2715;</button>
  </div>
  <div class="drawer-body">
    <div class="catalog-bar">
      <span class="catalog-info" id="catalogInfo">Lade Katalog&hellip;</span>
      <button class="refresh-btn" id="refreshCatalogBtn">&#x21bb; Aktualisieren</button>
    </div>
    <div style="margin-bottom:14px">
      <label style="display:block;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:rgba(255,255,255,.35);margin-bottom:6px">HuggingFace Token <span style="font-weight:400;text-transform:none;letter-spacing:0">(für gated Models)</span></label>
      <div style="display:flex;gap:8px;align-items:center">
        <input type="password" id="hfTokenInput" placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
          style="flex:1;background:#2c2c2e;border:1px solid rgba(255,255,255,.12);border-radius:8px;
                 padding:8px 12px;color:#fff;font-size:.8rem;font-family:inherit;outline:none;">
        <button id="hfTokenSaveBtn" class="m-btn m-btn-activate" style="white-space:nowrap">Speichern</button>
      </div>
      <div id="hfTokenStatus" style="font-size:.68rem;color:rgba(255,255,255,.3);margin-top:5px"></div>
    </div>
    <div class="seg-ctrl">
      <button class="seg-btn active" id="tabFace">Gesicht</button>
      <button class="seg-btn" id="tabPlate">Kennzeichen</button>
    </div>
    <div class="model-list" id="faceModelList">
      <div style="color:rgba(255,255,255,.3);font-size:.8rem;padding:8px 0">Lade&hellip;</div>
    </div>
    <div class="model-list" id="plateModelList" style="display:none">
      <div style="color:rgba(255,255,255,.3);font-size:.8rem;padding:8px 0">Lade&hellip;</div>
    </div>
  </div>
</div>

<!-- Main -->
<div class="container">

  <div class="header">
    <div class="brand">
      <div class="logo">&#9986;</div>
      <div class="brand-text">
        <h1>BrainCut Blur</h1>
        <p>Face &amp; Plate Detection</p>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:8px">
      <div class="status-pill" id="pill">
        <div class="status-dot" id="sdot"></div>
        <span id="pillText">Laden&hellip;</span>
      </div>
      <button class="gear-btn" id="gearBtn" title="Einstellungen">&#9881;</button>
    </div>
  </div>

  <div class="err-banner" id="errBanner" style="display:none">
    &#9888;&nbsp;<span id="errText"></span>
  </div>

  <div class="card" id="jobCard" style="display:none">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <span class="card-title" style="margin-bottom:0">Aktiver Job</span>
      <button class="cancel-btn" id="cancelBtn">Abbrechen</button>
    </div>
    <div class="job-name" id="jobName"></div>
    <div class="job-sub"  id="jobSub"></div>
    <div class="progress-track"><div class="progress-fill" id="pBar" style="width:0%"></div></div>
    <div class="progress-meta"><span id="pLeft"></span><span id="pRight"></span></div>
    <div class="stats">
      <div class="stat">
        <div class="stat-lbl">Geschwindigkeit</div>
        <div class="stat-val blue" id="sFps">&#8211;</div>
      </div>
      <div class="stat">
        <div class="stat-lbl">Verstrichen</div>
        <div class="stat-val" id="sElapsed">&#8211;</div>
      </div>
      <div class="stat">
        <div class="stat-lbl">Verbleibend</div>
        <div class="stat-val green" id="sEta">&#8211;</div>
      </div>
    </div>
  </div>

  <div class="card" id="renderCard" style="display:none">
    <div class="card-title">FFmpeg Re-Encode</div>
    <div class="render-row">
      <div class="spinner"></div>
      <div>
        <div class="render-name" id="renderName"></div>
        <div class="render-sub">NVENC H.264 &ndash; Encoding l&auml;uft&hellip;</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <span class="card-title" style="margin-bottom:0">Protokoll</span>
      <span id="logCount" style="font-size:.68rem;color:rgba(255,255,255,.22)">0 Eintr&auml;ge</span>
    </div>
    <div class="log-box" id="logBox"></div>
    <div class="log-foot">
      <span id="refreshTs"></span>
      <span class="as-btn on" id="asBtn">&#8595; Auto-Scroll</span>
    </div>
  </div>

</div>

<script>
var lastF=0,lastFT=Date.now(),smoothFps=0,autoScroll=true;
var settingsData=null,activeTab='face';
var hfToken=localStorage.getItem('hf_token')||'';

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}

function fmt(s){
  if(!s||s<=0)return '–';
  if(s<60)return s+' Sek';
  var m=Math.floor(s/60),r=s%60;
  return r?m+' min '+r+' s':m+' min';
}

function calcFps(cf){
  var now=Date.now(),dt=(now-lastFT)/1000;
  if(dt>=1.8&&cf>lastF){
    var raw=(cf-lastF)/dt;
    smoothFps=smoothFps===0?raw:smoothFps*.55+raw*.45;
    lastF=cf;lastFT=now;
  }
  return smoothFps>0?smoothFps.toFixed(1)+' fps':'–';
}

function logClass(l){
  if(/FEHLER|ERROR/i.test(l))return 'e';
  if(/Warnung|Warning|warn/i.test(l))return 'w';
  return '';
}
function renderLog(l){
  return esc(l).replace(/(\\[\\d{2}:\\d{2}:\\d{2}\\])/,'<span class="ts">$1</span>');
}

function toggleAs(){
  autoScroll=!autoScroll;
  document.getElementById('asBtn').className='as-btn'+(autoScroll?' on':'');
}

async function doCancel(){
  var btn=document.getElementById('cancelBtn');
  btn.disabled=true;btn.textContent='Wird abgebrochen…';
  try{await fetch('/cancel',{method:'POST'});}catch(e){}
}

function saveHfToken(){
  hfToken=document.getElementById('hfTokenInput').value.trim();
  localStorage.setItem('hf_token',hfToken);
  var st=document.getElementById('hfTokenStatus');
  st.textContent=hfToken?'Token gespeichert.':'Token geleert.';
  st.style.color=hfToken?'#30d158':'rgba(255,255,255,.3)';
  setTimeout(function(){st.textContent='';},3000);
}

function openSettings(){
  document.getElementById('drawer').classList.add('open');
  document.getElementById('scrim').classList.add('open');
  document.getElementById('gearBtn').classList.add('active');
  var inp=document.getElementById('hfTokenInput');
  if(inp)inp.value=hfToken;
  loadSettings();
}
function closeSettings(){
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('scrim').classList.remove('open');
  document.getElementById('gearBtn').classList.remove('active');
}
function toggleSettings(){
  if(document.getElementById('drawer').classList.contains('open'))closeSettings();
  else openSettings();
}

function switchTab(tab){
  activeTab=tab;
  document.getElementById('tabFace').className='seg-btn'+(tab==='face'?' active':'');
  document.getElementById('tabPlate').className='seg-btn'+(tab==='plate'?' active':'');
  document.getElementById('faceModelList').style.display=tab==='face'?'':'none';
  document.getElementById('plateModelList').style.display=tab==='plate'?'':'none';
}

async function loadSettings(){
  try{
    var r=await fetch('/api/models');
    settingsData=await r.json();
    renderSettings();
  }catch(e){
    document.getElementById('faceModelList').innerHTML='<div style="color:#ff453a;font-size:.8rem">Fehler beim Laden.</div>';
  }
}

function renderSettings(){
  var d=settingsData;
  if(!d)return;
  document.getElementById('catalogInfo').textContent=
    (d.catalog_source||'integriert')+' · '+d.catalog.length+' Modelle';
  renderModelList('face',d);
  renderModelList('plate',d);
}

function renderModelList(type,d){
  var listId=type==='face'?'faceModelList':'plateModelList';
  var models=d.catalog.filter(function(m){return m.type===type;});
  var activeId=type==='face'?d.config.face_model:d.config.plate_model;
  var iconCls=type==='face'?'fi':'pi';
  var iconChar=type==='face'?'👁':'🚗';

  if(models.length===0){
    document.getElementById(listId).innerHTML='<div style="color:rgba(255,255,255,.3);font-size:.8rem;padding:8px 0">Keine Modelle.</div>';
    return;
  }

  var html='';
  models.forEach(function(m){
    var isActive=m.id===activeId;
    var isInstalled=!!(d.installed[m.id]);
    var prog=d.install_progress[m.id];
    var isDl=prog&&prog.status==='downloading';

    html+='<div class="model-row'+(isActive?' active':'')+'">';
    html+='<div class="model-icon '+iconCls+'">'+iconChar+'</div>';
    html+='<div class="model-info">';
    html+='<div class="model-name">'+esc(m.name);
    if(isActive)html+='<span class="active-badge">Aktiv</span>';
    html+='</div>';
    html+='<div class="model-desc">'+esc(m.description)+'</div>';
    html+='<div class="model-meta">'+(m.builtin?'Integriert':m.size_mb+' MB')+' · '+esc(m.format)+'</div>';
    if(isDl){
      html+='<div class="dl-bar"><div class="dl-bar-fill" style="width:'+prog.pct+'%"></div></div>';
      html+='<div style="font-size:.66rem;color:rgba(255,255,255,.3);margin-top:3px">'+prog.pct+'% heruntergeladen…</div>';
    }
    if(prog&&prog.status==='error'){
      html+='<div style="font-size:.68rem;color:#ff453a;margin-top:4px">⚠ '+esc(prog.error)+'</div>';
    }
    html+='</div>';
    html+='<div class="model-actions">';
    if(isDl){
      html+='<button class="m-btn m-btn-install" disabled>Lädt…</button>';
    }else if(!m.builtin&&!isInstalled&&m.url){
      html+='<button class="m-btn m-btn-install" data-action="install" data-id="'+esc(m.id)+'" data-url="'+esc(m.url)+'">Installieren</button>';
    }
    if((isInstalled||m.builtin)&&!isActive){
      html+='<button class="m-btn m-btn-activate" data-action="activate" data-id="'+esc(m.id)+'" data-type="'+type+'">Aktivieren</button>';
    }
    if(isInstalled&&!m.builtin){
      html+='<button class="m-btn m-btn-delete" data-action="delete" data-id="'+esc(m.id)+'">Löschen</button>';
    }
    html+='</div></div>';
  });

  document.getElementById(listId).innerHTML=html;
}

async function settingsRefreshCatalog(){
  var btn=document.getElementById('refreshCatalogBtn');
  btn.disabled=true;btn.textContent='Aktualisiert…';
  try{
    await fetch('/api/models/refresh',{method:'POST'});
    await loadSettings();
  }catch(e){}
  btn.disabled=false;btn.textContent='↻ Aktualisieren';
}

async function installModel(id,url){
  try{
    await fetch('/api/models/install',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:id,url:url,hf_token:hfToken})
    });
    var iv=setInterval(async function(){
      await loadSettings();
      if(settingsData&&settingsData.install_progress[id]){
        var s=settingsData.install_progress[id].status;
        if(s==='done'||s==='error')clearInterval(iv);
      }
    },800);
  }catch(e){}
}

async function activateModel(id,type){
  try{
    await fetch('/api/models/activate',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:id,type:type})
    });
    await loadSettings();
  }catch(e){}
}

async function deleteModel(id){
  if(!confirm('Modell wirklich löschen?'))return;
  try{
    await fetch('/api/models/'+id,{method:'DELETE'});
    await loadSettings();
  }catch(e){}
}

async function refresh(){
  try{
    var r=await fetch('/status'),d=await r.json();
    var sdot=document.getElementById('sdot');
    var pillText=document.getElementById('pillText');

    if(d.error){
      document.getElementById('errBanner').style.display='flex';
      document.getElementById('errText').textContent=d.error;
    }else{
      document.getElementById('errBanner').style.display='none';
    }

    if(d.state==='idle'){
      sdot.className='status-dot'+(d.error?' error':'');
      pillText.textContent=d.error?'Fehler':'Bereit';
      document.getElementById('jobCard').style.display='none';
      document.getElementById('renderCard').style.display='none';
      lastF=0;lastFT=Date.now();smoothFps=0;
      var cb=document.getElementById('cancelBtn');
      cb.disabled=false;cb.textContent='Abbrechen';
    }else if(d.state==='blur'){
      sdot.className='status-dot active pulse';
      pillText.textContent='Verarbeitet';
      document.getElementById('jobCard').style.display='block';
      document.getElementById('renderCard').style.display='none';
      document.getElementById('jobName').textContent=d.name||'–';
      document.getElementById('jobSub').textContent='Video '+d.current+' von '+d.total;
      var pct=d.frame_pct||0;
      document.getElementById('pBar').style.width=pct+'%';
      document.getElementById('pLeft').textContent=
        pct+'% · '+(d.frame_current||0).toLocaleString('de-DE')+
        ' / '+(d.frame_total||0).toLocaleString('de-DE')+' Frames';
      document.getElementById('sFps').textContent=calcFps(d.frame_current||0);
      document.getElementById('sElapsed').textContent=fmt(d.elapsed_seconds);
      document.getElementById('sEta').textContent=fmt(d.eta_seconds);
    }else if(d.state==='render'){
      sdot.className='status-dot active pulse';
      pillText.textContent='Encodiert';
      document.getElementById('jobCard').style.display='none';
      document.getElementById('renderCard').style.display='block';
      document.getElementById('renderName').textContent=d.out_name||'';
    }

    var box=document.getElementById('logBox');
    var atBottom=box.scrollHeight-box.clientHeight<=box.scrollTop+32;
    box.innerHTML=d.logs.map(function(l){
      return '<div class="le '+logClass(l)+'">'+renderLog(l)+'</div>';
    }).join('');
    if(autoScroll&&(atBottom||d.state!=='idle'))box.scrollTop=box.scrollHeight;

    document.getElementById('logCount').textContent=d.logs.length+' Einträge';
    document.getElementById('refreshTs').textContent=
      'Aktualisiert: '+new Date().toLocaleTimeString('de-DE');
  }catch(e){
    document.getElementById('sdot').className='status-dot error';
    document.getElementById('pillText').textContent='Verbindung verloren';
  }
}

function handleModelClick(e){
  var btn=e.target.closest('button[data-action]');
  if(!btn||btn.disabled)return;
  var action=btn.dataset.action;
  var id=btn.dataset.id;
  if(action==='install')installModel(id,btn.dataset.url);
  else if(action==='activate')activateModel(id,btn.dataset.type);
  else if(action==='delete')deleteModel(id);
}

document.getElementById('gearBtn').addEventListener('click',toggleSettings);
document.getElementById('scrim').addEventListener('click',closeSettings);
document.getElementById('drawerClose').addEventListener('click',closeSettings);
document.getElementById('hfTokenSaveBtn').addEventListener('click',saveHfToken);
document.getElementById('refreshCatalogBtn').addEventListener('click',settingsRefreshCatalog);
document.getElementById('tabFace').addEventListener('click',function(){switchTab('face');});
document.getElementById('tabPlate').addEventListener('click',function(){switchTab('plate');});
document.getElementById('cancelBtn').addEventListener('click',doCancel);
document.getElementById('asBtn').addEventListener('click',toggleAs);
document.getElementById('faceModelList').addEventListener('click',handleModelClick);
document.getElementById('plateModelList').addEventListener('click',handleModelClick);

refresh();
setInterval(refresh,2000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTML


@app.get("/status")
def status():
    with _lock:
        s = dict(_status)
    ts = s.get("started_at_ts", 0.0)
    s["elapsed_seconds"] = int(time.time() - ts) if ts > 0 and s.get("state") != "idle" else 0
    return s


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/blur")
def blur(data: dict, bg: BackgroundTasks):
    jobs = data.get("jobs", [])
    resume_url = data.get("resumeUrl", "")
    status_url = data.get("statusUrl", "")
    full_job = data.get("fullJob", {})
    _log(f"Blur-Auftrag empfangen: {len(jobs)} Job(s)")
    bg.add_task(process_jobs, jobs, resume_url, status_url, full_job)
    return {"status": "queued", "count": len(jobs)}


@app.post("/cancel")
def cancel_job():
    global _cancel_flag
    with _lock:
        _cancel_flag = True
    _log("⚠️ Abbruch angefordert")
    return {"status": "cancel_requested"}


@app.post("/run-shell")
def run_shell(data: dict):
    """FFmpeg-Kommando im Container ausführen (ersetzt N8N SSH-Node)."""
    cmd = data.get("cmd", "").strip()
    if not cmd:
        return {"stdout": "", "stderr": "Kein Befehl angegeben", "exitCode": 1}
    cmd_remapped = cmd.replace(_media_host_path, _CONTAINER_ROOT)
    _log(f"Shell-Befehl gestartet ({len(cmd_remapped)} Zeichen)")
    try:
        result = subprocess.run(
            ["bash", "-c", cmd_remapped],
            capture_output=True, text=True, timeout=3600
        )
        _log(f"Shell-Befehl beendet: exitCode={result.returncode}")
        if result.returncode != 0:
            _log(f"stderr: {result.stderr[-500:]}")
        return {
            "stdout": result.stdout,
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "exitCode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timeout nach 3600s", "exitCode": 1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc)[:500], "exitCode": 1}


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


# ── Modell-API ────────────────────────────────────────────────────────────────
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
    _log(f"Aktives {model_type}-Modell geändert: {model_id}")
    return {"ok": True, "config": cfg}


@app.delete("/api/models/{model_id}")
def api_models_delete(model_id: str):
    if model_id == "builtin-centerface":
        return {"error": "Integriertes Modell kann nicht gelöscht werden"}
    target = _MODELS_PATH / f"{model_id}.onnx"
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
    _log(f"Modell gelöscht: {model_id}")
    return {"ok": True}


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


# ── YOLOv8 ONNX Inferenz ──────────────────────────────────────────────────────
def _yolov8_detect(sess, frame, conf_thresh: float = 0.5, aspect_filter: bool = False) -> list:
    """YOLOv8/YOLOv11 ONNX Detection. Gibt [(x1,y1,x2,y2), ...] zurück."""
    import cv2
    h_orig, w_orig = frame.shape[:2]
    inp = sess.get_inputs()[0]

    # Eingabegröße ermitteln (unterstützt dynamische Shapes)
    raw_h, raw_w = inp.shape[2], inp.shape[3]
    in_h = raw_h if isinstance(raw_h, int) and raw_h > 0 else 640
    in_w = raw_w if isinstance(raw_w, int) and raw_w > 0 else 640

    img = cv2.resize(frame, (in_w, in_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = img.transpose(2, 0, 1)[np.newaxis]

    out = sess.run(None, {inp.name: blob})[0]

    # Ausgabe auf [anchors, 4+nc] normalisieren
    if out.ndim == 3:
        # [1, 4+nc, anchors] → [anchors, 4+nc]
        if out.shape[1] <= out.shape[2]:
            out = out.transpose(0, 2, 1)
        out = out[0]
    # out ist jetzt [anchors, 4+nc]

    boxes = []
    for row in out:
        score = float(row[4:].max())
        if score < conf_thresh:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = int((cx - bw / 2) / in_w * w_orig)
        y1 = int((cy - bh / 2) / in_h * h_orig)
        x2 = int((cx + bw / 2) / in_w * w_orig)
        y2 = int((cy + bh / 2) / in_h * h_orig)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if aspect_filter:
            wb, hb = x2 - x1, y2 - y1
            if hb == 0 or wb < 20 or hb < 8:
                continue
            if not (1.5 <= wb / hb <= 7.0):
                continue
        boxes.append((x1, y1, x2, y2, score))

    # NMS
    boxes.sort(key=lambda b: b[4], reverse=True)
    kept = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        discard = False
        for kb in kept:
            ix1, iy1 = max(x1, kb[0]), max(y1, kb[1])
            ix2, iy2 = min(x2, kb[2]), min(y2, kb[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area = (x2-x1)*(y2-y1) + (kb[2]-kb[0])*(kb[3]-kb[1]) - inter
                if area > 0 and inter / area > 0.45:
                    discard = True
                    break
        if not discard:
            kept.append(b)

    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in kept]


# ── Blur-Verarbeitung ─────────────────────────────────────────────────────────
def process_jobs(jobs: list, resume_url: str, status_url: str = "", full_job: dict = None):
    status_url = _fix_status_url(status_url)
    total = len(jobs)
    errors = []
    was_cancelled = False

    _set(state="blur", current=0, total=total, error="",
         frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0,
         started_at=datetime.now().strftime("%H:%M:%S"),
         started_at_ts=time.time())
    _log(f"Blur gestartet: {total} Video(s)")
    post_status(status_url, {"event": "start", "total": total})

    try:
        for i, job in enumerate(jobs, 1):
            input_path = _remap(job.get("input_path", ""))
            output_path = _remap(job.get("output_path", ""))
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
                    run_deface(input_path, output_path, mode="both", status_url=status_url, job_name=name, detection_resolution=detection_resolution)
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
    finally:
        _log(f"Blur abgeschlossen. Fehler: {len(errors)}")
        _set(state="idle", error=errors[0]["error"][:200] if errors else "",
             frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0, started_at_ts=0.0)
        post_status(status_url, {"event": "done", "total": total, "errors": errors})

        if resume_url:
            try:
                requests.post(resume_url, json={"status": "done", "errors": errors}, timeout=15)
                _log("Blur-Callback gesendet")
            except Exception as exc:
                _log(f"Blur-Callback fehlgeschlagen: {exc}")

        if COMPLETION_WEBHOOK and not was_cancelled:
            try:
                payload = {"status": "done", "errors": errors}
                if full_job:
                    payload["fullJob"] = full_job
                requests.post(COMPLETION_WEBHOOK, json=payload, timeout=10)
                _log(f"Completion-Webhook gesendet (fullJob={'ja' if full_job else 'nein'})")
            except Exception as exc:
                _log(f"Completion-Webhook fehlgeschlagen: {exc}")


# ── deface: direkt per Python API ────────────────────────────────────────────
_FRAME_BUFFER = 32
_PLATE_CONF_THRESH = 0.45
_PLATE_GRID = 30
_DETECTION_INTERVAL = 4


def _load_centerface(in_shape: tuple, providers: list):
    from deface.centerface import CenterFace
    try:
        cf = CenterFace(in_shape=in_shape)
        _log(f"CenterFace geladen mit in_shape={in_shape}")
    except Exception as e1:
        _log(f"CenterFace(in_shape) fehlgeschlagen ({e1}), versuche CenterFace() ohne Argumente")
        try:
            cf = CenterFace()
            _log("CenterFace ohne in_shape geladen")
        except Exception as e2:
            _log(f"CenterFace komplett fehlgeschlagen: {e2}")
            raise RuntimeError(f"CenterFace konnte nicht geladen werden: {e2}") from e2
    for attr in ("centerface", "sess", "session", "ort_session"):
        sess = getattr(cf, attr, None)
        if hasattr(sess, "set_providers"):
            try:
                sess.set_providers(providers)
                _log(f"CenterFace GPU-Provider gesetzt über cf.{attr}")
            except Exception as exc:
                _log(f"CenterFace Provider-Setzen fehlgeschlagen (cf.{attr}): {exc}")
            break
    else:
        _log("CenterFace: kein Session-Attribut gefunden – läuft auf Standard-Provider")
    return cf


def _check_nvdec() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=5
        )
        return "cuda" in r.stdout
    except Exception:
        return False


def _check_nvenc() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


def run_deface(input_path: str, output_path: str, mode: str = "faces",
               status_url: str = "", job_name: str = "",
               detection_resolution: str = "720p"):
    global _cancel_flag

    cfg = _load_model_config()

    # ── Kennzeichen-Modell prüfen ─────────────────────────────────────────────
    plate_model_id = None
    plate_model_file = None
    if mode in ("plates", "both"):
        plate_model_id = cfg.get("plate_model")
        if not plate_model_id:
            if mode == "plates":
                _log("Kein Kennzeichen-Modell aktiv – Datei wird kopiert.")
                subprocess.run(["cp", "--", input_path, output_path], check=True)
                return
            else:
                _log("Kein Kennzeichen-Modell aktiv – nur Gesichter werden verarbeitet.")
        else:
            plate_model_file = _MODELS_PATH / f"{plate_model_id}.onnx"
            if not plate_model_file.exists():
                if mode == "plates":
                    _log(f"Kennzeichen-Modell nicht gefunden ({plate_model_id}) – Datei wird kopiert.")
                    subprocess.run(["cp", "--", input_path, output_path], check=True)
                    return
                else:
                    _log(f"Kennzeichen-Modell nicht gefunden ({plate_model_id}) – nur Gesichter.")
                    plate_model_file = None

    _log(f"deface [{mode}] startet: {os.path.basename(input_path)}")

    if not os.path.exists(input_path):
        raise RuntimeError(
            f"Datei nicht gefunden: {input_path}\n"
            "Prüfe ob das Volume im Docker-Container korrekt gemountet ist."
        )

    # ── Video-Eigenschaften + Decoder via PyAV ───────────────────────────────
    import av as _av

    # NVDEC (h264_cuvid) versuchen, sonst CPU-Multithreading
    use_nvdec = False
    try:
        _av.codec.Codec('h264_cuvid', 'r')
        in_container = _av.open(input_path, options={'video_codec': 'h264_cuvid'})
        use_nvdec = True
    except Exception:
        in_container = _av.open(input_path)

    in_vs = in_container.streams.video[0]
    if not use_nvdec:
        in_vs.codec_context.thread_count = 0
    w = in_vs.width
    h = in_vs.height
    fps_rate = in_vs.average_rate  # Fraction (z.B. 100/1)
    fps = float(fps_rate)
    total_frames = in_vs.frames or 0
    if not total_frames and in_vs.duration and fps > 0:
        total_frames = int(float(in_vs.duration) * float(in_vs.time_base) * fps)
    _log(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} Frames")

    # ── ONNX Runtime ──────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            # TRT nur für YOLO (feste Input-Shapes) – CenterFace braucht dynamische Shapes
            providers_yolo = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers_face = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            _log("deface: YOLO→TensorRT, CenterFace→CUDA (TRT unterstützt keine dyn. Shapes)")
        elif "CUDAExecutionProvider" in available:
            providers_yolo = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers_face = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            _log("deface: nutze GPU (CUDA)")
        else:
            providers_yolo = ["CPUExecutionProvider"]
            providers_face = ["CPUExecutionProvider"]
            _log("deface: CUDA nicht verfügbar – nutze CPU")
    except ImportError:
        providers_yolo = ["CPUExecutionProvider"]
        providers_face = ["CPUExecutionProvider"]
        _log("deface: onnxruntime nicht gefunden – nutze CPU")

    import cv2

    # ── Gesichts-Detektor ─────────────────────────────────────────────────────
    cf = None
    face_yolo_sess = None
    if mode in ("faces", "both"):
        face_model_id = cfg.get("face_model", "builtin-centerface")
        if face_model_id == "builtin-centerface":
            res_map = {"720p": (720, 1280), "1080p": (1080, 1920)}
            in_shape = (h, w) if detection_resolution == "native" else res_map.get(detection_resolution, (720, 1280))
            _log(f"deface: CenterFace, in_shape={in_shape}")
            cf = _load_centerface(in_shape, providers_face)
        else:
            model_path = str(_MODELS_PATH / f"{face_model_id}.onnx")
            if not os.path.exists(model_path):
                _log(f"Gesichts-Modell nicht gefunden ({face_model_id}) – Fallback auf CenterFace")
                in_shape = {"720p": (720, 1280), "1080p": (1080, 1920)}.get(detection_resolution, (720, 1280))
                cf = _load_centerface(in_shape, providers_face)
            else:
                try:
                    face_yolo_sess = ort.InferenceSession(model_path, providers=providers_yolo)
                    _log(f"deface: YOLOv8 Gesichts-Modell geladen: {face_model_id}")
                except Exception as exc:
                    _log(f"Gesichts-Modell Ladefehler: {exc} – Fallback auf CenterFace")
                    in_shape = {"720p": (720, 1280), "1080p": (1080, 1920)}.get(detection_resolution, (720, 1280))
                    cf = _load_centerface(in_shape, providers_face)

    # ── Kennzeichen-Detektor ──────────────────────────────────────────────────
    plate_yolo_sess = None
    if mode in ("plates", "both") and plate_model_file:
        try:
            plate_yolo_sess = ort.InferenceSession(str(plate_model_file), providers=providers_yolo)
            _log(f"deface: Kennzeichen-Modell geladen: {plate_model_id}")
        except Exception as exc:
            _log(f"Kennzeichen-Modell Ladefehler: {exc}")
            if mode == "plates":
                subprocess.run(["cp", "--", input_path, output_path], check=True)
                return

    # ── PyAV Ausgabe-Container ────────────────────────────────────────────────
    wakeup_disk(input_path)
    tmp_output = output_path + ".enc.tmp.mp4"
    out_container = _av.open(tmp_output, 'w')

    use_nvenc = False
    try:
        out_video = out_container.add_stream('h264_nvenc', rate=fps_rate)
        out_video.options = {'preset': 'p4', 'cq': '18'}
        use_nvenc = True
    except Exception:
        out_video = out_container.add_stream('libx264', rate=fps_rate)
        out_video.options = {'crf': '18', 'preset': 'fast'}
    out_video.width = w
    out_video.height = h
    out_video.pix_fmt = 'yuv420p'
    codec_label = "PyAV+NVENC" if use_nvenc else "PyAV+libx264"
    _log(f"Hardware: PyAV-Decoder ({'NVDEC/h264_cuvid' if use_nvdec else 'CPU multithreaded'}), NVENC={'ja' if use_nvenc else 'nein'}")

    in_audio_streams = list(in_container.streams.audio)
    out_audio = None
    if in_audio_streams:
        try:
            out_audio = out_container.add_stream(template=in_audio_streams[0])
        except Exception:
            pass

    frame_idx = 0
    start_time = time.time()
    last_log_time = 0.0
    last_milestone = 0
    cancelled = False
    _PLATE_TTL = 10
    plate_buffer: dict = {}
    total_face_detections = 0
    total_plate_detections = 0
    face_bbox_sample: list = []
    face_bbox_areas: list = []
    last_face_dets: list = []

    from concurrent.futures import ThreadPoolExecutor
    _executor = ThreadPoolExecutor(max_workers=2)

    # Einmalig Callables erzeugen (nicht pro Frame neu)
    _face_callable = None
    if mode in ("faces", "both"):
        if face_yolo_sess is not None:
            def _face_callable(f, _s=face_yolo_sess):
                return _yolov8_detect(_s, f)
        elif cf is not None:
            def _face_callable(f, _cf=cf, _fw=w, _fh=h):
                result = _cf(f, threshold=0.1)
                raw_dets = result[0] if (result is not None and result[0] is not None) else []
                expanded = []
                for dx, dy, dx2, dy2, _ in raw_dets:
                    ex = max(2, int((dx2 - dx) * 0.30))
                    ey = max(2, int((dy2 - dy) * 0.30))
                    expanded.append((
                        max(0, int(dx) - ex), max(0, int(dy) - ey),
                        min(_fw - 1, int(dx2) + ex), min(_fh - 1, int(dy2) + ey)
                    ))
                return expanded

    _plate_callable = None
    if mode in ("plates", "both") and plate_yolo_sess is not None:
        def _plate_callable(f, _s=plate_yolo_sess):
            return _yolov8_detect(_s, f, conf_thresh=_PLATE_CONF_THRESH, aspect_filter=True)

    def _iter_decoded():
        for _pkt in in_container.demux():
            if _pkt.stream.type == 'audio' and out_audio is not None:
                _pkt.stream = out_audio
                out_container.mux(_pkt)
                continue
            if _pkt.stream.type != 'video':
                continue
            for _avf in _pkt.decode():
                yield _avf.to_ndarray(format='bgr24'), _avf

    try:
        for frame, _av_frame in _iter_decoded():
            with _lock:
                if _cancel_flag:
                    cancelled = True
                    break

            frame_idx += 1
            should_detect = (frame_idx % _DETECTION_INTERVAL == 1)

            # ── Parallele Detektion mit Frame-Skip ────────────────────────
            face_future = None
            plate_future = None
            if should_detect:
                if _face_callable is not None:
                    face_future = _executor.submit(_face_callable, frame)
                if _plate_callable is not None:
                    plate_future = _executor.submit(_plate_callable, frame)

            # Gesichts-Ergebnisse sammeln
            face_dets: list = []
            if face_future is not None:
                try:
                    face_dets = face_future.result()
                    if face_dets:
                        if len(face_bbox_sample) < 3:
                            face_bbox_sample.append((frame_idx, *face_dets[0]))
                        for fx, fy, fx2, fy2 in face_dets:
                            face_bbox_areas.append((fx2 - fx) * (fy2 - fy))
                except Exception as exc:
                    _log(f"CenterFace Fehler Frame {frame_idx}: {exc}")
                last_face_dets = face_dets
            else:
                face_dets = last_face_dets

            # Kennzeichen TTL-Verfall (jeden Frame) + neue Ergebnisse
            for k in list(plate_buffer):
                box, ttl = plate_buffer[k]
                if ttl <= 1:
                    del plate_buffer[k]
                else:
                    plate_buffer[k] = (box, ttl - 1)
            if plate_future is not None:
                try:
                    raw_plates = plate_future.result()
                except Exception:
                    raw_plates = []
                for box in raw_plates:
                    key = (
                        box[0] // _PLATE_GRID, box[1] // _PLATE_GRID,
                        box[2] // _PLATE_GRID, box[3] // _PLATE_GRID,
                    )
                    plate_buffer[key] = (box, _PLATE_TTL)
            plate_dets: list = [box for box, _ in plate_buffer.values()]

            total_face_detections += len(face_dets)
            total_plate_detections += len(plate_dets)

            # ── Blur anwenden ──────────────────────────────────────────────
            for x, y, x2, y2 in face_dets:
                rw, rh = x2 - x, y2 - y
                if rw < 4 or rh < 4:
                    continue
                roi = frame[y:y2, x:x2]
                if roi.size == 0:
                    continue
                ksize = max(31, (rw // 2) | 1)
                frame[y:y2, x:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)

            for x, y, x2, y2 in plate_dets:
                roi = frame[y:y2, x:x2]
                if roi.size > 0:
                    bw = max(1, (x2 - x) // 10)
                    bh = max(1, (y2 - y) // 10)
                    frame[y:y2, x:x2] = cv2.resize(
                        cv2.resize(roi, (bw, bh)), (x2 - x, y2 - y)
                    )

            _out_frame = _av.VideoFrame.from_ndarray(frame, format='bgr24')
            _out_frame.pts = _av_frame.pts if _av_frame.pts is not None else frame_idx - 1
            _out_frame.time_base = in_vs.time_base
            for _enc_pkt in out_video.encode(_out_frame):
                out_container.mux(_enc_pkt)

            now = time.time()
            elapsed = now - start_time
            _set(frame_current=frame_idx, frame_total=total_frames)
            if total_frames > 0 and elapsed > 0:
                pct = int(frame_idx / total_frames * 100)
                fps_actual = frame_idx / elapsed
                remaining = max(0, total_frames - frame_idx)
                eta = int(remaining / fps_actual)
                _set(frame_pct=pct, eta_seconds=eta)

                if now - last_log_time >= 10:
                    _log(f"  {pct}% | {frame_idx:,}/{total_frames:,} Frames | ~{_format_eta(eta)} verbleibend")
                    last_log_time = now

                milestone = (pct // 10) * 10
                if milestone > 0 and milestone > last_milestone:
                    last_milestone = milestone
                    post_status(status_url, {
                        "event": "frame_progress", "name": job_name, "mode": mode,
                        "pct": milestone, "frame_current": frame_idx,
                        "frame_total": total_frames, "eta_seconds": eta,
                        "eta_human": _format_eta(eta),
                    })
            elif now - last_log_time >= 30:
                _log(f"  {frame_idx:,} Frames verarbeitet (Länge unbekannt)")
                last_log_time = now

    finally:
        _executor.shutdown(wait=False)
        _log(
            f"Detektion-Zusammenfassung: {total_face_detections} Gesichts-Erkennungen, "
            f"{total_plate_detections} Kennzeichen-Erkennungen über {frame_idx} Frames "
            f"(Modus: {mode})"
        )
        if mode in ("faces", "both"):
            if total_face_detections == 0:
                _log("WARNUNG: Kein Gesicht erkannt! CenterFace evtl. fehlerhaft geladen.")
            else:
                if face_bbox_areas:
                    avg_a = sum(face_bbox_areas) / len(face_bbox_areas)
                    min_a = min(face_bbox_areas)
                    max_a = max(face_bbox_areas)
                    _log(
                        f"Gesicht BBox-Fläche: min={min_a}px² avg={avg_a:.0f}px² max={max_a}px² "
                        f"(Framegröße: {w}x{h}={w * h}px²)"
                    )
                for s in face_bbox_sample:
                    fi, fx, fy, fx2, fy2 = s
                    _log(f"Gesicht Beispiel Frame {fi}: ({fx},{fy})-({fx2},{fy2}) → {fx2 - fx}x{fy2 - fy}px")

        # Encoder flushen und Container schließen
        _set(state="render", out_name=os.path.basename(output_path))
        try:
            for _enc_pkt in out_video.encode(None):
                out_container.mux(_enc_pkt)
        except Exception:
            pass
        try:
            out_container.close()
        except Exception:
            pass
        try:
            in_container.close()
        except Exception:
            pass
        _set(state="blur")

    if cancelled:
        with _lock:
            _cancel_flag = False
        _log(f"  ⚠️ Abbruch – {frame_idx}/{total_frames} Frames verarbeitet")
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise RuntimeError("cancelled")

    if not os.path.exists(tmp_output) or os.path.getsize(tmp_output) < 1024:
        raise RuntimeError(f"Encoder fehlgeschlagen: Ausgabedatei fehlt oder leer ({tmp_output})")

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_output, output_path)

    _log(f"deface [{mode}] abgeschlossen: {frame_idx:,} Frames ({codec_label})")
