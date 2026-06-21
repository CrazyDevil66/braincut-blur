import json
import os
import re
from pathlib import Path
from threading import Lock

import requests

import config
import state

_install_progress: dict = {}
_install_lock = Lock()
_MAX_MODEL_BYTES = 500 * 1024 * 1024


def _models_config_path() -> Path:
    return config._MODELS_PATH / "config.json"


def _load_model_config() -> dict:
    p = _models_config_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"face_model": "builtin-centerface", "plate_model": None}


def _save_model_config(cfg: dict) -> None:
    config._MODELS_PATH.mkdir(parents=True, exist_ok=True)
    _models_config_path().write_text(json.dumps(cfg, indent=2))


def _get_catalog() -> dict:
    cached = config._MODELS_PATH / "catalog.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text())
        except Exception:
            pass
    return dict(config._BUILTIN_CATALOG)


def _refresh_catalog_sync() -> dict:
    if not config._MODEL_CATALOG_URL:
        cat = dict(config._BUILTIN_CATALOG)
        cat["source"] = "integriert"
    else:
        try:
            r = requests.get(config._MODEL_CATALOG_URL, timeout=15)
            r.raise_for_status()
            cat = r.json()
            cat["source"] = config._MODEL_CATALOG_URL
            state._log(f"Katalog aktualisiert: {len(cat.get('models', []))} Modelle")
        except Exception as exc:
            state._log(f"Katalog-Abruf fehlgeschlagen: {exc} – nutze integrierten Katalog")
            cat = dict(config._BUILTIN_CATALOG)
            cat["source"] = "integriert (Fehler beim Abruf)"
    config._MODELS_PATH.mkdir(parents=True, exist_ok=True)
    (config._MODELS_PATH / "catalog.json").write_text(json.dumps(cat, indent=2))
    return cat


def _get_installed_models() -> dict:
    result: dict = {"builtin-centerface": {"builtin": True, "size_mb": 5.3}}
    if config._MODELS_PATH.exists():
        for f in config._MODELS_PATH.glob("*.onnx"):
            result[f.stem] = {"file": str(f), "size_mb": round(f.stat().st_size / 1_048_576, 1)}
    return result


def _validate_model_id(model_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]{1,80}", model_id))


def _install_model_bg(model_id: str, url: str, hf_token: str = "") -> None:
    with _install_lock:
        _install_progress[model_id] = {"status": "downloading", "pct": 0, "error": ""}
    config._MODELS_PATH.mkdir(parents=True, exist_ok=True)
    target = config._MODELS_PATH / f"{model_id}.onnx"
    target_tmp = config._MODELS_PATH / f"{model_id}.onnx.tmp"
    try:
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        r = requests.get(url, stream=True, timeout=120, allow_redirects=True, headers=headers)
        if r.status_code == 401:
            raise RuntimeError(
                "401 Unauthorized – HuggingFace-Token erforderlich. Token im Einstellungen-Panel eingeben."
            )
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "text/html" in content_type or "text/plain" in content_type:
            raise RuntimeError(f"Unerwarteter Content-Type: {content_type} – kein Modell?")
        total = int(r.headers.get("content-length", 0))
        if total and total > _MAX_MODEL_BYTES:
            raise RuntimeError(f"Modell zu groß: {total // 1_048_576} MB (Limit: 500 MB)")
        downloaded = 0
        with open(target_tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > _MAX_MODEL_BYTES:
                        raise RuntimeError("Modell überschreitet 500 MB Limit – Download abgebrochen")
                    if total:
                        pct = min(99, int(downloaded / total * 100))
                        with _install_lock:
                            _install_progress[model_id]["pct"] = pct
        os.replace(target_tmp, target)
        with _install_lock:
            _install_progress[model_id] = {"status": "done", "pct": 100, "error": ""}
        state._log(f"Modell installiert: {model_id} ({round(target.stat().st_size / 1_048_576, 1)} MB)")
    except Exception as exc:
        if target_tmp.exists():
            try:
                target_tmp.unlink()
            except Exception:
                pass
        with _install_lock:
            _install_progress[model_id] = {"status": "error", "pct": 0, "error": str(exc)[:300]}
        state._log(f"Modell-Installation fehlgeschlagen ({model_id}): {exc}")
