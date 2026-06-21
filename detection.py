import subprocess

import numpy as np

import state


def _yolov8_detect(sess, frame, conf_thresh: float = 0.5, aspect_filter: bool = False) -> list:
    import cv2
    h_orig, w_orig = frame.shape[:2]
    inp = sess.get_inputs()[0]

    raw_h, raw_w = inp.shape[2], inp.shape[3]
    in_h = raw_h if isinstance(raw_h, int) and raw_h > 0 else 640
    in_w = raw_w if isinstance(raw_w, int) and raw_w > 0 else 640

    img = cv2.resize(frame, (in_w, in_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = img.transpose(2, 0, 1)[np.newaxis]

    out = sess.run(None, {inp.name: blob})[0]

    if out.ndim == 3:
        if out.shape[1] <= out.shape[2]:
            out = out.transpose(0, 2, 1)
        out = out[0]

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
                area = (x2 - x1) * (y2 - y1) + (kb[2] - kb[0]) * (kb[3] - kb[1]) - inter
                if area > 0 and inter / area > 0.45:
                    discard = True
                    break
        if not discard:
            kept.append(b)

    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in kept]


def _load_centerface(in_shape: tuple, providers: list):
    from deface.centerface import CenterFace
    try:
        cf = CenterFace(in_shape=in_shape)
        state._log(f"CenterFace geladen mit in_shape={in_shape}")
    except Exception as e1:
        state._log(f"CenterFace(in_shape) fehlgeschlagen ({e1}), versuche CenterFace() ohne Argumente")
        try:
            cf = CenterFace()
            state._log("CenterFace ohne in_shape geladen")
        except Exception as e2:
            state._log(f"CenterFace komplett fehlgeschlagen: {e2}")
            raise RuntimeError(f"CenterFace konnte nicht geladen werden: {e2}") from e2
    for attr in ("centerface", "sess", "session", "ort_session"):
        sess = getattr(cf, attr, None)
        if hasattr(sess, "set_providers"):
            try:
                sess.set_providers(providers)
                state._log(f"CenterFace GPU-Provider gesetzt über cf.{attr}")
            except Exception as exc:
                state._log(f"CenterFace Provider-Setzen fehlgeschlagen (cf.{attr}): {exc}")
            break
    else:
        state._log("CenterFace: kein Session-Attribut gefunden – läuft auf Standard-Provider")
    return cf


def _check_nvdec() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        return "cuda" in r.stdout
    except Exception:
        return False


def _check_nvenc() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False
