import logging
import subprocess
from datetime import datetime
from threading import Lock

log = logging.getLogger(__name__)

_lock = Lock()
_render_lock = Lock()

_cancel_flag: bool = False
_current_render_process: "subprocess.Popen | None" = None

_status: dict = {
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
    "hw_nvdec": False,
    "hw_nvenc": False,
    "hw_trt": False,
    "face_count": 0,
    "plate_count": 0,
    "logs": [],
}


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    log.info(msg)
    with _lock:
        _status["logs"].append(entry)
        if len(_status["logs"]) > 200:
            _status["logs"] = _status["logs"][-200:]


def _set(**kwargs) -> None:
    with _lock:
        _status.update(kwargs)


def _format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "–"
    if seconds < 60:
        return f"{seconds} Sek"
    m, s = divmod(seconds, 60)
    return f"{m} Min {s} Sek" if s else f"{m} Min"


def request_cancel() -> None:
    global _cancel_flag
    with _lock:
        _cancel_flag = True


def is_cancel_requested() -> bool:
    with _lock:
        return _cancel_flag


def check_and_clear_cancel() -> bool:
    global _cancel_flag
    with _lock:
        val = _cancel_flag
        if val:
            _cancel_flag = False
    return val


def clear_cancel() -> None:
    global _cancel_flag
    with _lock:
        _cancel_flag = False


def set_render_process(proc: "subprocess.Popen | None") -> None:
    global _current_render_process
    with _render_lock:
        _current_render_process = proc


def get_render_process() -> "subprocess.Popen | None":
    with _render_lock:
        return _current_render_process
