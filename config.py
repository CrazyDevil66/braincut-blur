import os
from pathlib import Path

_n8n_ip = os.getenv("N8N_SERVER_IP", "")
_n8n_port = os.getenv("N8N_SERVER_PORT", "5678")
COMPLETION_WEBHOOK = f"http://{_n8n_ip}:{_n8n_port}/webhook/blur-done" if _n8n_ip else ""

_media_host_path = os.getenv("MEDIA_HOST_PATH", "/mnt/user/n8n_automation/BrainCut").rstrip("/")
_CONTAINER_ROOT = "/data"

_MODELS_PATH = Path(os.getenv("MODELS_PATH", "/app/.cache/models"))
_MODEL_CATALOG_URL = os.getenv("MODEL_CATALOG_URL", "")

_DETECTION_INTERVAL = 4
_PLATE_CONF_THRESH = 0.45
_PLATE_GRID = 20
_FRAME_BUFFER = 32

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
