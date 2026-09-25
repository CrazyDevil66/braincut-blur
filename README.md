# BrainCut Blur Service

Automatische Gesichts- und Kennzeichenunschärfe für N8N-Workflows.  
Läuft als Docker-Container auf Unraid mit GPU-Beschleunigung (NVIDIA CUDA).

Seit dem Rust-Rewrite ersetzt dieser Service die frühere Python-Version (FastAPI).
Die Python-Version ist im Tag `python-letzter-stand` erhalten.

---

## Aufbau

```
NVDEC (FFmpeg-Subprocess) → Rohframes (BGR)
  → Erkennung alle N Frames (ONNX Runtime, CUDA/TensorRT)
      Gesicht: CenterFace (integriert), YOLOv8-Face oder SCRFD
      Kennzeichen: YOLOv8
  → Blur (Gesichter) / Verpixelung (Kennzeichen) direkt im Frame-Puffer
NVENC (FFmpeg-Subprocess) → Audio-Mux aus dem Original → Ausgabedatei
```

| Pfad | Inhalt |
|---|---|
| `src/main.rs` | HTTP-Server (axum), Routen |
| `src/handlers/` | Endpunkte für Jobs, Modelle, Status |
| `src/blur/` | Pipeline: Probe, Decode/Encode, Frame-Loop, Fortschritt, Mux |
| `src/detection/` | ONNX-Inferenz: CenterFace, SCRFD, YOLO, NMS |
| `src/image_ops.rs` | Blur und Verpixelung auf Frame-Ausschnitten |
| `src/models/` | Modell-Katalog und -Installation |
| `ui/` | Web-Oberfläche; `build.rs` setzt daraus `ui/index.html` zusammen |

---

## Build

`cargo` wird lokal nicht benötigt – gebaut wird ausschließlich per Docker:

```bash
docker build --platform linux/amd64 -t crazydevil35/braincut-blur:latest .
docker push crazydevil35/braincut-blur:latest
```

Danach in Unraid den Container neu starten – er zieht `latest` beim Start.

---

## Ordnerstruktur auf Unraid

```
/mnt/user/n8n_automation/BrainCut/
├── 01_upload/       ← Videos hier ablegen
├── 02_processing/   ← Blur-Zwischendateien (automatisch)
├── 03_output/       ← Fertige Ausgabedateien
├── 04_done/         ← Quelldateien nach Verarbeitung
├── 05_audio/        ← Musikdateien für Audio-Modus
└── sessions/        ← Session-Status-Dateien (JSON)
```

---

## Docker-Container installieren

1. Unraid → Apps → Container-XML manuell importieren: `braincut-blur.xml`
2. Oder: Unraid → Docker → Container hinzufügen (manuell):
   - Image: `crazydevil35/braincut-blur:latest`
   - Port: `8080`
   - Pfad `/data` → `/mnt/user/n8n_automation/BrainCut`
   - Pfad `/app/.cache` → `/mnt/user/appdata/braincut-blur`
   - Extra Params: `--gpus all`

### Umgebungsvariablen

| Variable | Beispiel | Beschreibung |
|---|---|---|
| `N8N_SERVER_IP` | `192.168.1.100` | Nur IP – kein Port. Leer = kein Abschluss-Webhook |
| `N8N_SERVER_PORT` | `5678` | Port des N8N-Servers |
| `MEDIA_HOST_PATH` | `/mnt/user/n8n_automation/BrainCut` | Host-Pfad, den N8N in Aufträgen verwendet; wird auf `/data` umgesetzt |
| `MODEL_CATALOG_URL` | *(leer)* | Optional: URL zu einer externen Modell-Katalog-JSON |
| `ORT_ENABLE_TRT` | `1` | Optional: TensorRT für YOLO-Modelle aktivieren |

---

## GPU-Anforderungen

- NVIDIA-GPU mit CUDA-Support
- Unraid: Nvidia-Driver-Plugin installiert
- Container wird mit `--gpus all` gestartet

---

## HTTP-API

| Methode | Pfad | Zweck |
|---|---|---|
| GET | `/` | Web-Oberfläche |
| GET | `/status` | Status, Fortschritt, Log |
| GET | `/health` | Healthcheck |
| POST | `/blur` | Blur-Aufträge starten (`jobs`, `statusUrl`, `fullJob`) |
| POST | `/cancel` | Laufenden Auftrag abbrechen |
| POST | `/job-control` | `{ "action": "status" \| "cancel" }` |
| POST | `/render` | Clips zusammenschneiden (Geschwindigkeit, Ton, Musik) – antwortet erst nach Fertigstellung |
| GET | `/api/models` | Katalog, installierte Modelle, Konfiguration |
| POST | `/api/models/install` | Modell von URL installieren |
| POST | `/api/models/activate` | Gesichts- oder Kennzeichen-Modell aktivieren |
| DELETE | `/api/models/:id` | Modell löschen |
| GET/POST | `/api/config` | Erkennungsintervall und Schwellwerte |
| GET | `/api/frame` | Vorschaubild des aktuellen Frames (JPEG) |

Beispiel `/blur`:

```json
{
  "jobs": [{
    "input_path": "/mnt/user/n8n_automation/BrainCut/01_upload/video.mp4",
    "output_path": "/mnt/user/n8n_automation/BrainCut/02_processing/blurred_1_video.mp4",
    "blur_faces": true,
    "blur_plates": true,
    "detection_resolution": "720p"
  }],
  "statusUrl": "http://<n8n>:5678/webhook/blur-status"
}
```

Zu jeder Ausgabedatei wird `<ausgabe>.detections.csv` mit allen Erkennungen geschrieben.
Nach Abschluss meldet der Service sich bei `http://<N8N_SERVER_IP>:<N8N_SERVER_PORT>/webhook/blur-done`.

Beispiel `/render` (so sendet es der Processing-Workflow):

```json
{
  "clips": [{ "path": "/mnt/.../02_processing/blurred_1_video.mp4", "speed_factor": 4, "mute_audio": false, "volume_factor": 1.0 }],
  "audio": { "mode": "mix_music", "musicFile": { "path": "/mnt/.../05_audio/musik.mp3" }, "musicVolume": 0.35, "loop": true },
  "output_path": "/mnt/.../03_output/fertig.mp4",
  "overwrite": false,
  "move_sources_to": "/mnt/.../04_done",
  "cleanup_paths": []
}
```

Audio-Modi: `original`, `mute` (über `mute_audio` je Clip), `replace_with_music`, `mix_music`.
Antwort: `{ "success": true, "out_path", "out_name", "size", "size_bytes" }` bzw. `{ "success": false, "error" }`.
Ausgabe: H.264 (CRF 22), 1920 px breit, 30 fps, AAC 192 kbit/s.

---

## Bekannte Einschränkungen

- Nur ein Auftrag gleichzeitig (weitere Anfragen → `409 Conflict`).
- Der Service hat keine Authentifizierung – nur im vertrauenswürdigen LAN betreiben.

---

## Fehlerbehebung

| Problem | Ursache | Lösung |
|---|---|---|
| `409 Conflict` beim Blur-Start | Auftrag läuft bereits | `/status` prüfen, warten oder `/cancel` |
| Keine Gesichter erkannt (WARNUNG im Log) | Schwellwert oder Modell passt nicht | Web-GUI → Einstellungen, Container-Log (`[CF-DIAG]`) prüfen |
| Kennzeichen-Blur schlägt fehl | Kein Kennzeichen-Modell aktiv | Web-GUI → Modelle → Kennzeichen-Modell installieren und aktivieren |
| Container startet nicht | GPU nicht verfügbar | `--gpus all` in Extra Params, Nvidia-Plugin prüfen |
