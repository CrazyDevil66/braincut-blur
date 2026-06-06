import logging
import os
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
    log.info(f"Auftrag empfangen: {len(jobs)} Job(s)")
    bg.add_task(process_jobs, jobs, resume_url)
    return {"status": "queued", "count": len(jobs)}


def process_jobs(jobs: list, resume_url: str):
    errors = []
    for job in jobs:
        input_path = job.get("input_path", "")
        output_path = job.get("output_path", "")
        blur_faces = job.get("blur_faces", False)
        blur_plates = job.get("blur_plates", False)
        log.info(f"{input_path} → {output_path} (faces={blur_faces}, plates={blur_plates})")
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
        except Exception as exc:
            log.error(f"Fehler bei {input_path}: {exc}")
            errors.append({"input": input_path, "error": str(exc)})

    if resume_url:
        try:
            requests.post(resume_url, json={"status": "done", "errors": errors}, timeout=15)
            log.info(f"Callback an {resume_url}")
        except Exception as exc:
            log.error(f"Callback fehlgeschlagen: {exc}")


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
