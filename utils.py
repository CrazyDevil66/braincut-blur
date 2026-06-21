import os

import requests

import state


def post_status(status_url: str, data: dict) -> None:
    if not status_url:
        return
    try:
        requests.post(status_url, json=data, timeout=5)
    except Exception as exc:
        state._log(f"Status-Update fehlgeschlagen: {exc}")


def format_bytes(b: int) -> str:
    if b > 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GB"
    if b > 1_048_576:
        return f"{b / 1_048_576:.1f} MB"
    return f"{b / 1024:.0f} KB"


def wakeup_disk(path: str) -> None:
    try:
        with open(path, "rb") as f:
            f.read(4096)
        state._log(f"Datenträger aktiv: {os.path.basename(path)}")
    except Exception as exc:
        state._log(f"Disk-Wakeup fehlgeschlagen: {exc}")
