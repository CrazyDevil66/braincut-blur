import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction as _Frac

import requests

import config
import state
from detection import _load_centerface, _yolov8_detect
from models import _load_model_config
from paths import _fix_status_url, _remap, validate_data_path
from utils import post_status, wakeup_disk


def process_jobs(jobs: list, resume_url: str, status_url: str = "", full_job: dict = None) -> None:
    state.clear_cancel()
    status_url = _fix_status_url(status_url)
    total = len(jobs)
    errors = []
    was_cancelled = False

    state._set(
        state="blur", current=0, total=total, error="",
        frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0,
        started_at=datetime.now().strftime("%H:%M:%S"),
        started_at_ts=time.time(),
    )
    state._log(f"Blur gestartet: {total} Video(s)")
    post_status(status_url, {"event": "start", "total": total})

    try:
        for i, job in enumerate(jobs, 1):
            try:
                input_path = validate_data_path(_remap(job.get("input_path", "")))
                output_path = validate_data_path(_remap(job.get("output_path", "")))
            except RuntimeError as exc:
                err = str(exc)
                state._log(f"[{i}/{total}] Pfadfehler: {err}")
                errors.append({"input": job.get("input_path", ""), "error": err})
                post_status(status_url, {"event": "error", "current": i, "total": total, "name": "", "error": err})
                continue

            blur_faces = job.get("blur_faces", False)
            blur_plates = job.get("blur_plates", False)
            name = os.path.basename(input_path)

            state._set(current=i, name=name, frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0)
            state._log(f"[{i}/{total}] Starte: {name} (faces={blur_faces}, plates={blur_plates})")
            post_status(status_url, {"event": "progress_start", "current": i, "total": total, "name": name})

            wakeup_disk(input_path)
            detection_resolution = job.get("detection_resolution", "720p")

            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if blur_faces and blur_plates:
                    run_deface(input_path, output_path, mode="both",
                               status_url=status_url, job_name=name,
                               detection_resolution=detection_resolution)
                elif blur_faces:
                    run_deface(input_path, output_path, mode="faces",
                               status_url=status_url, job_name=name,
                               detection_resolution=detection_resolution)
                elif blur_plates:
                    run_deface(input_path, output_path, mode="plates",
                               status_url=status_url, job_name=name,
                               detection_resolution=detection_resolution)
                else:
                    subprocess.run(["cp", "--", input_path, output_path], check=True)

                state._log(f"[{i}/{total}] Fertig: {name}")
                post_status(status_url, {"event": "progress_done", "current": i, "total": total, "name": name})
            except RuntimeError as exc:
                if "cancelled" in str(exc).lower():
                    state.clear_cancel()
                    state._log(f"[{i}/{total}] Job abgebrochen.")
                    errors.append({"input": input_path, "error": "Abgebrochen"})
                    post_status(status_url, {"event": "cancelled", "current": i, "total": total, "name": name})
                    was_cancelled = True
                    break
                err = str(exc)
                state._log(f"[{i}/{total}] FEHLER: {err[:300]}")
                errors.append({"input": input_path, "error": err})
                post_status(status_url, {"event": "error", "current": i, "total": total, "name": name, "error": err[:500]})
            except Exception as exc:
                err = str(exc)
                state._log(f"[{i}/{total}] FEHLER: {err[:300]}")
                errors.append({"input": input_path, "error": err})
                post_status(status_url, {"event": "error", "current": i, "total": total, "name": name, "error": err[:500]})
    finally:
        state._log(f"Blur abgeschlossen. Fehler: {len(errors)}")
        final_state = "cancelled" if was_cancelled else ("error" if errors else "idle")
        state._set(
            state=final_state,
            error=errors[0]["error"][:200] if errors else "",
            frame_current=0, frame_total=0, frame_pct=0, eta_seconds=0,
            started_at_ts=0.0,
        )
        post_status(status_url, {"event": "done", "total": total, "errors": errors})

        if resume_url:
            try:
                requests.post(resume_url, json={"status": "done", "errors": errors}, timeout=15)
                state._log("Blur-Callback gesendet")
            except Exception as exc:
                state._log(f"Blur-Callback fehlgeschlagen: {exc}")

        if config.COMPLETION_WEBHOOK and not was_cancelled:
            try:
                payload = {"status": "done", "errors": errors}
                if full_job:
                    payload["fullJob"] = full_job
                requests.post(config.COMPLETION_WEBHOOK, json=payload, timeout=10)
                state._log(f"Completion-Webhook gesendet (fullJob={'ja' if full_job else 'nein'})")
            except Exception as exc:
                state._log(f"Completion-Webhook fehlgeschlagen: {exc}")


def run_deface(
    input_path: str,
    output_path: str,
    mode: str = "faces",
    status_url: str = "",
    job_name: str = "",
    detection_resolution: str = "720p",
) -> None:
    cfg = _load_model_config()
    _det_interval = int(cfg.get("detection_interval", config._DETECTION_INTERVAL))
    _conf_thresh = float(cfg.get("plate_conf_thresh", config._PLATE_CONF_THRESH))

    plate_model_id = None
    plate_model_file = None
    if mode in ("plates", "both"):
        plate_model_id = cfg.get("plate_model")
        if not plate_model_id:
            if mode == "plates":
                raise RuntimeError(
                    "Kennzeichen-Blur angefordert, aber kein Kennzeichen-Modell aktiv. Job abgebrochen."
                )
            else:
                state._log("Kein Kennzeichen-Modell aktiv – nur Gesichter werden verarbeitet.")
        else:
            plate_model_file = config._MODELS_PATH / f"{plate_model_id}.onnx"
            if not plate_model_file.exists():
                if mode == "plates":
                    raise RuntimeError(
                        f"Kennzeichen-Modell '{plate_model_id}' nicht gefunden. Job abgebrochen."
                    )
                else:
                    state._log(f"Kennzeichen-Modell nicht gefunden ({plate_model_id}) – nur Gesichter.")
                    plate_model_file = None

    state._log(f"deface [{mode}] startet: {os.path.basename(input_path)}")

    if not os.path.exists(input_path):
        raise RuntimeError(
            f"Datei nicht gefunden: {input_path}\n"
            "Prüfe ob das Volume im Docker-Container korrekt gemountet ist."
        )

    import numpy as _np

    # Video-Metadaten komplett via ffprobe
    _pdata: dict = {}
    try:
        _probe_r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path],
            capture_output=True, text=True, timeout=30,
        )
        _pdata = json.loads(_probe_r.stdout)
    except Exception as _exc:
        state._log(f"ffprobe fehlgeschlagen: {_exc}")

    w = h = total_frames = 0
    fps = 30.0
    fps_str = "30/1"
    _rotation = 0
    for _ps in _pdata.get('streams', []):
        if _ps.get('codec_type') == 'video':
            w = _ps.get('width', 0)
            h = _ps.get('height', 0)
            _rfr = _ps.get('r_frame_rate', '30/1')
            if '/' in _rfr:
                _rn, _rd = _rfr.split('/', 1)
                fps = float(_rn) / float(_rd) if float(_rd) > 0 else 30.0
            else:
                fps = float(_rfr or 30)
            fps_str = _rfr
            _nb = str(_ps.get('nb_frames', '') or '')
            total_frames = int(_nb) if _nb.isdigit() else 0
            if not total_frames and _ps.get('duration'):
                total_frames = int(float(_ps['duration']) * fps)
            try:
                _rotation = int(str(_ps.get('tags', {}).get('rotate', '0') or '0'))
            except Exception:
                pass
            for _sd in _ps.get('side_data_list', []):
                if _sd.get('side_data_type') == 'Display Matrix':
                    try:
                        _rotation = (-int(_sd['rotation'])) % 360
                    except Exception:
                        pass
            break

    state._log(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} Frames")
    if _rotation not in (0, 90, 180, 270):
        _rotation = 0
    if _rotation:
        state._log(f"Video-Rotation erkannt: {_rotation}° – FFmpeg autorotiert")
    if _rotation in (90, 270):
        w, h = h, w  # angezeigte Auflösung (FFmpeg dreht automatisch)

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            providers_yolo = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers_face = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            state._log("deface: YOLO→TensorRT, CenterFace→CUDA (TRT unterstützt keine dyn. Shapes)")
            state._set(hw_trt=True)
        elif "CUDAExecutionProvider" in available:
            providers_yolo = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers_face = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            state._log("deface: nutze GPU (CUDA)")
        else:
            providers_yolo = ["CPUExecutionProvider"]
            providers_face = ["CPUExecutionProvider"]
            state._log("deface: CUDA nicht verfügbar – nutze CPU")
    except ImportError:
        providers_yolo = ["CPUExecutionProvider"]
        providers_face = ["CPUExecutionProvider"]
        state._log("deface: onnxruntime nicht gefunden – nutze CPU")

    import cv2

    cf = None
    face_yolo_sess = None
    if mode in ("faces", "both"):
        face_model_id = cfg.get("face_model", "builtin-centerface")
        if face_model_id == "builtin-centerface":
            res_map = {"720p": (720, 1280), "1080p": (1080, 1920)}
            in_shape = (h, w) if detection_resolution == "native" else res_map.get(detection_resolution, (720, 1280))
            state._log(f"deface: CenterFace, in_shape={in_shape}")
            cf = _load_centerface(in_shape, providers_face)
        else:
            model_path = str(config._MODELS_PATH / f"{face_model_id}.onnx")
            if not os.path.exists(model_path):
                state._log(f"Gesichts-Modell nicht gefunden ({face_model_id}) – Fallback auf CenterFace")
                in_shape = {"720p": (720, 1280), "1080p": (1080, 1920)}.get(detection_resolution, (720, 1280))
                cf = _load_centerface(in_shape, providers_face)
            else:
                try:
                    face_yolo_sess = ort.InferenceSession(model_path, providers=providers_yolo)
                    state._log(f"deface: YOLOv8 Gesichts-Modell geladen: {face_model_id}")
                except Exception as exc:
                    state._log(f"Gesichts-Modell Ladefehler: {exc} – Fallback auf CenterFace")
                    in_shape = {"720p": (720, 1280), "1080p": (1080, 1920)}.get(detection_resolution, (720, 1280))
                    cf = _load_centerface(in_shape, providers_face)

    plate_yolo_sess = None
    if mode in ("plates", "both") and plate_model_file:
        try:
            plate_yolo_sess = ort.InferenceSession(str(plate_model_file), providers=providers_yolo)
            state._log(f"deface: Kennzeichen-Modell geladen: {plate_model_id}")
        except Exception as exc:
            state._log(f"Kennzeichen-Modell Ladefehler: {exc}")
            if mode == "plates":
                raise RuntimeError(f"Kennzeichen-Modell konnte nicht geladen werden: {exc}") from exc

    wakeup_disk(input_path)
    tmp_output = output_path + ".enc.tmp.mp4"

    from detection import _check_nvenc
    use_nvenc = _check_nvenc()

    # System-FFmpeg: -hwaccel cuda → GPU-Decode wenn verfügbar, sonst CPU-Fallback
    dec_cmd = [
        'ffmpeg', '-loglevel', 'error',
        '-hwaccel', 'cuda',
        '-i', input_path,
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        'pipe:1',
    ]
    _enc_codec = (
        ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '18']
        if use_nvenc else
        ['-c:v', 'libx264', '-crf', '18', '-preset', 'fast']
    )
    enc_cmd = [
        'ffmpeg', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}', '-r', fps_str,
        '-i', 'pipe:0',
    ] + _enc_codec + ['-pix_fmt', 'yuv420p', '-an', tmp_output]

    state._log(f"Hardware: FFmpeg+CUDA (auto), NVENC={'ja (h264_nvenc)' if use_nvenc else 'nein (libx264)'}")
    state._set(hw_nvdec=True, hw_nvenc=use_nvenc)

    proc_dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_bytes = w * h * 3

    frame_idx = 0
    start_time = time.time()
    last_log_time = 0.0
    last_milestone = 0
    cancelled = False
    _PLATE_TTL = 20
    plate_buffer: dict = {}
    total_face_detections = 0
    total_plate_detections = 0
    face_bbox_sample: list = []
    face_bbox_areas: list = []
    last_face_dets: list = []

    _executor = ThreadPoolExecutor(max_workers=2)

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
                        min(_fw - 1, int(dx2) + ex), min(_fh - 1, int(dy2) + ey),
                    ))
                return expanded

    _plate_callable = None
    if mode in ("plates", "both") and plate_yolo_sess is not None:
        def _plate_callable(f, _s=plate_yolo_sess, _ct=_conf_thresh):
            return _yolov8_detect(_s, f, conf_thresh=_ct, aspect_filter=True)

    try:
        while True:
            raw = proc_dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            if state.is_cancel_requested():
                cancelled = True
                break

            frame = _np.frombuffer(raw, dtype=_np.uint8).reshape((h, w, 3)).copy()

            frame_idx += 1
            should_detect = (frame_idx % _det_interval == 1)

            face_future = None
            plate_future = None
            if should_detect:
                if _face_callable is not None:
                    face_future = _executor.submit(_face_callable, frame)
                if _plate_callable is not None:
                    plate_future = _executor.submit(_plate_callable, frame)

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
                    state._log(f"CenterFace Fehler Frame {frame_idx}: {exc}")
                last_face_dets = face_dets
            else:
                face_dets = last_face_dets

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
                        box[0] // config._PLATE_GRID, box[1] // config._PLATE_GRID,
                        box[2] // config._PLATE_GRID, box[3] // config._PLATE_GRID,
                    )
                    plate_buffer[key] = (box, _PLATE_TTL)
            plate_dets: list = [box for box, _ in plate_buffer.values()]

            total_face_detections += len(face_dets)
            total_plate_detections += len(plate_dets)

            for x, y, x2, y2 in face_dets:
                rw, rh = x2 - x, y2 - y
                if rw < 50 or rh < 50:
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
                    frame[y:y2, x:x2] = cv2.resize(cv2.resize(roi, (bw, bh)), (x2 - x, y2 - y))

            proc_enc.stdin.write(frame.tobytes())

            now = time.time()
            elapsed = now - start_time
            state._set(frame_current=frame_idx, frame_total=total_frames)
            if total_frames > 0 and elapsed > 0:
                pct = int(frame_idx / total_frames * 100)
                fps_actual = frame_idx / elapsed
                remaining = max(0, total_frames - frame_idx)
                eta = int(remaining / fps_actual)
                state._set(
                    frame_pct=pct, eta_seconds=eta,
                    face_count=total_face_detections, plate_count=total_plate_detections,
                )

                if now - last_log_time >= 10:
                    state._log(
                        f"  {pct}% | {frame_idx:,}/{total_frames:,} Frames "
                        f"| ~{state._format_eta(eta)} verbleibend"
                    )
                    last_log_time = now

                milestone = (pct // 10) * 10
                if milestone > 0 and milestone > last_milestone:
                    last_milestone = milestone
                    post_status(status_url, {
                        "event": "frame_progress", "name": job_name, "mode": mode,
                        "pct": milestone, "frame_current": frame_idx,
                        "frame_total": total_frames, "eta_seconds": eta,
                        "eta_human": state._format_eta(eta),
                    })
            elif now - last_log_time >= 30:
                state._log(f"  {frame_idx:,} Frames verarbeitet (Länge unbekannt)")
                last_log_time = now

    finally:
        _executor.shutdown(wait=False)
        state._log(
            f"Detektion-Zusammenfassung: {total_face_detections} Gesichts-Erkennungen, "
            f"{total_plate_detections} Kennzeichen-Erkennungen über {frame_idx} Frames "
            f"(Modus: {mode})"
        )
        if mode in ("faces", "both"):
            if total_face_detections == 0:
                state._log("WARNUNG: Kein Gesicht erkannt! CenterFace evtl. fehlerhaft geladen.")
                post_status(status_url, {"event": "warning_no_detections", "name": job_name, "mode": mode})
            else:
                if face_bbox_areas:
                    avg_a = sum(face_bbox_areas) / len(face_bbox_areas)
                    min_a = min(face_bbox_areas)
                    max_a = max(face_bbox_areas)
                    state._log(
                        f"Gesicht BBox-Fläche: min={min_a}px² avg={avg_a:.0f}px² max={max_a}px² "
                        f"(Framegröße: {w}x{h}={w * h}px²)"
                    )
                for s in face_bbox_sample:
                    fi, fx, fy, fx2, fy2 = s
                    state._log(
                        f"Gesicht Beispiel Frame {fi}: ({fx},{fy})-({fx2},{fy2}) → {fx2 - fx}x{fy2 - fy}px"
                    )

        state._set(state="render", out_name=os.path.basename(output_path))
        try:
            proc_enc.stdin.close()
        except Exception:
            pass
        try:
            proc_enc.wait(timeout=300)
        except Exception:
            proc_enc.kill()
        try:
            proc_dec.terminate()
            proc_dec.wait(timeout=10)
        except Exception:
            proc_dec.kill()
        state._set(state="blur")

    if cancelled:
        state.clear_cancel()
        state._log(f"  ⚠️ Abbruch – {frame_idx}/{total_frames} Frames verarbeitet")
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise RuntimeError("cancelled")

    if not os.path.exists(tmp_output) or os.path.getsize(tmp_output) < 1024:
        raise RuntimeError(f"Encoder fehlgeschlagen: Ausgabedatei fehlt oder leer ({tmp_output})")

    # Audio aus Originaldatei in die Ausgabe muxen (Encode hatte -an)
    final_tmp = output_path + ".final.tmp.mp4"
    try:
        subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-y',
             '-i', tmp_output, '-i', input_path,
             '-map', '0:v:0', '-map', '1:a?', '-c', 'copy',
             final_tmp],
            timeout=120, check=False,
        )
    except Exception as _mux_err:
        state._log(f"Audio-Mux fehlgeschlagen ({_mux_err}) – Video ohne Audio")
        final_tmp = tmp_output

    if os.path.exists(tmp_output) and final_tmp != tmp_output:
        os.remove(tmp_output)
    if not os.path.exists(final_tmp) or os.path.getsize(final_tmp) < 1024:
        raise RuntimeError("Ausgabedatei nach Audio-Mux fehlt oder leer")
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(final_tmp, output_path)
