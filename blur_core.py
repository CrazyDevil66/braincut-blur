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

    import av as _av

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
    fps_rate = in_vs.average_rate
    fps = float(fps_rate)
    _out_tb = _Frac(1, max(1, int(round(fps))))
    total_frames = in_vs.frames or 0
    if not total_frames and in_vs.duration and fps > 0:
        total_frames = int(float(in_vs.duration) * float(in_vs.time_base) * fps)
    state._log(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} Frames")

    _rotation = 0
    _rot_raw = in_vs.metadata.get('rotate', '') or in_container.metadata.get('rotate', '')
    state._log(f"PyAV Metadaten: stream={dict(in_vs.metadata)} container={dict(in_container.metadata)}")
    if _rot_raw:
        try:
            _rotation = int(_rot_raw)
        except (ValueError, TypeError):
            _rotation = 0

    if _rotation == 0:
        try:
            _probe = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path],
                capture_output=True, text=True, timeout=10,
            )
            _pdata = json.loads(_probe.stdout)
            for _ps in _pdata.get('streams', []):
                if _ps.get('codec_type') == 'video':
                    _rot_str = str(_ps.get('tags', {}).get('rotate', '0') or '0')
                    _rotation = int(_rot_str)
                    if _rotation:
                        state._log(f"Rotation via ffprobe erkannt: {_rotation}°")
                    break
        except Exception as _exc:
            state._log(f"ffprobe Rotations-Check fehlgeschlagen: {_exc}")

    if _rotation not in (0, 90, 180, 270):
        _rotation = 0
    if _rotation != 0:
        state._log(f"Video-Rotation erkannt: {_rotation}° – Frames werden korrigiert")
        if _rotation in (90, 270):
            w, h = h, w

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
    state._log(
        f"Hardware: PyAV-Decoder ({'NVDEC/h264_cuvid' if use_nvdec else 'CPU multithreaded'}), "
        f"NVENC={'ja' if use_nvenc else 'nein'}"
    )
    state._set(hw_nvdec=use_nvdec, hw_nvenc=use_nvenc)

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
            if state.is_cancel_requested():
                cancelled = True
                break

            if _rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif _rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif _rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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

            _out_frame = _av.VideoFrame.from_ndarray(frame, format='bgr24')
            _out_frame.pts = frame_idx - 1
            _out_frame.time_base = _out_tb
            for _enc_pkt in out_video.encode(_out_frame):
                out_container.mux(_enc_pkt)

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
        state._set(state="blur")

    if cancelled:
        state.clear_cancel()
        state._log(f"  ⚠️ Abbruch – {frame_idx}/{total_frames} Frames verarbeitet")
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise RuntimeError("cancelled")

    if not os.path.exists(tmp_output) or os.path.getsize(tmp_output) < 1024:
        raise RuntimeError(f"Encoder fehlgeschlagen: Ausgabedatei fehlt oder leer ({tmp_output})")

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_output, output_path)
