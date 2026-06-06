import logging
import os
import re
import subprocess
import tempfile

import requests
from fastapi import BackgroundTasks, FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="BrainCut Blur Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/blur")
def blur(data: dict, bg: BackgroundTasks):
    jobs = data.get("jobs", [])
    resume_url = data.get("resumeUrl", "")
    status_url = data.get("statusUrl", "")
    log.info(f"Blur-Auftrag empfangen: {len(jobs)} Job(s)")
    bg.add_task(process_jobs, jobs, resume_url, status_url)
    return {"status": "queued", "count": len(jobs)}


@app.post("/render")
def render(data: dict, bg: BackgroundTasks):
    cmd = data.get("cmd", "")
    resume_url = data.get("resumeUrl", "")
    status_url = data.get("statusUrl", "")
    out_name = data.get("outName", "")
    out_path = data.get("outPath", "")
    log.info(f"Render-Auftrag empfangen: {out_name}")
    bg.add_task(run_render, cmd, resume_url, status_url, out_name, out_path)
    return {"status": "queued"}


def post_status(status_url: str, data: dict):
    if not status_url:
        return
    try:
        requests.post(status_url, json=data, timeout=5)
    except Exception as exc:
        log.warning(f"Status-Update fehlgeschlagen: {exc}")


def format_bytes(b: int) -> str:
    if b > 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GB"
    if b > 1_048_576:
        return f"{b / 1_048_576:.1f} MB"
    return f"{b / 1024:.0f} KB"


def process_jobs(jobs: list, resume_url: str, status_url: str = ""):
    total = len(jobs)
    errors = []

    post_status(status_url, {"event": "start", "total": total})

    for i, job in enumerate(jobs, 1):
        input_path = job.get("input_path", "")
        output_path = job.get("output_path", "")
        blur_faces = job.get("blur_faces", False)
        blur_plates = job.get("blur_plates", False)
        name = os.path.basename(input_path)
        log.info(f"{input_path} → {output_path} (faces={blur_faces}, plates={blur_plates})")

        post_status(status_url, {"event": "progress_start", "current": i, "total": total, "name": name})

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if blur_faces and blur_plates:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    run_deface(input_path, tmp_path, mode="faces")
                    run_deface(tmp_path, output_path, mode="plates")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            elif blur_faces:
                run_deface(input_path, output_path, mode="faces")
            elif blur_plates:
                run_deface(input_path, output_path, mode="plates")
            else:
                subprocess.run(["cp", "--", input_path, output_path], check=True)
            log.info(f"Fertig: {output_path}")
            post_status(status_url, {"event": "progress_done", "current": i, "total": total, "name": name})
        except Exception as exc:
            log.error(f"Fehler bei {input_path}: {exc}")
            errors.append({"input": input_path, "error": str(exc)})
            post_status(status_url, {"event": "error", "current": i, "total": total, "name": name, "error": str(exc)[:500]})

    post_status(status_url, {"event": "done", "total": total, "errors": errors})

    if resume_url:
        try:
            requests.post(resume_url, json={"status": "done", "errors": errors}, timeout=15)
            log.info(f"Blur-Callback an {resume_url}")
        except Exception as exc:
            log.error(f"Blur-Callback fehlgeschlagen: {exc}")


def run_render(cmd: str, resume_url: str, status_url: str, out_name: str, out_path: str):
    log.info(f"FFmpeg startet: {out_name}")
    post_status(status_url, {"event": "render_start", "outName": out_name})
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

        size = "unbekannt"
        m = re.search(r"__BRAINCUT_SIZE__=(\d+)", stdout)
        if m:
            size = format_bytes(int(m.group(1)))

        success = exit_code == 0 and bool(m)

        if success:
            log.info(f"FFmpeg fertig: {out_name} ({size})")
            post_status(status_url, {"event": "render_done", "outName": out_name, "size": size})
        else:
            err = stderr[-800:] if stderr else stdout[-800:]
            log.error(f"FFmpeg Fehler (exit {exit_code}): {err[:200]}")
            post_status(status_url, {"event": "render_error", "error": err[:400]})

        if resume_url:
            requests.post(resume_url, json={
                "success": success,
                "exitCode": exit_code,
                "stdout": stdout[-2000:],
                "stderr": stderr[-1200:] if not success else "",
                "size": size,
                "outName": out_name,
                "outPath": out_path,
            }, timeout=15)
            log.info(f"Render-Callback an {resume_url}")

    except subprocess.TimeoutExpired:
        err = "FFmpeg-Timeout nach 2 Stunden."
        log.error(err)
        post_status(status_url, {"event": "render_error", "error": err})
        if resume_url:
            requests.post(resume_url, json={"success": False, "error": err, "outName": out_name}, timeout=15)
    except Exception as exc:
        err = str(exc)[:500]
        log.error(f"Render-Fehler: {err}")
        post_status(status_url, {"event": "render_error", "error": err})
        if resume_url:
            requests.post(resume_url, json={"success": False, "error": err, "outName": out_name}, timeout=15)


def run_deface(input_path: str, output_path: str, mode: str = "faces"):
    cmd = ["deface", "-i", input_path, "-o", output_path]
    if mode == "plates":
        weights = os.environ.get("PLATE_MODEL_PATH", "")
        if weights and os.path.exists(weights):
            cmd += ["--weights", weights]
        else:
            log.warning("Kein Kennzeichen-Modell (PLATE_MODEL_PATH nicht gesetzt). Datei wird kopiert.")
            subprocess.run(["cp", "--", input_path, output_path], check=True)
            return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"deface Fehler: {result.stderr[-800:]}")
