from pathlib import Path

import config

DATA_ROOT = Path("/data").resolve()


def _remap(path: str) -> str:
    if path == config._media_host_path or path.startswith(config._media_host_path + "/"):
        return config._CONTAINER_ROOT + path[len(config._media_host_path):]
    return path


def validate_data_path(p: str) -> str:
    if not p:
        raise RuntimeError("Leerer Pfad nicht erlaubt")
    resolved = Path(p).resolve()
    if resolved != DATA_ROOT and DATA_ROOT not in resolved.parents:
        raise RuntimeError(f"Pfad außerhalb von /data nicht erlaubt: {p}")
    return str(resolved)


def _fix_status_url(url: str) -> str:
    if config._n8n_ip and "localhost" in url:
        return url.replace("localhost", config._n8n_ip)
    return url
