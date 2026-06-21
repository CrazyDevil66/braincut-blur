# BrainCut Blur Service

Automatische Gesichts- und Kennzeichenunschärfe für N8N-Workflows.  
Läuft als Docker-Container auf Unraid mit GPU-Beschleunigung (NVIDIA CUDA).

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

1. Unraid → Apps → Container XML manuell importieren: `blur-service/braincut-blur.xml`
2. Oder: Unraid → Docker → Container hinzufügen (manuell):
   - Image: `crazydevil35/braincut-blur:latest`
   - Port: `8080`
   - Pfad `/data` → `/mnt/user/n8n_automation/BrainCut`
   - Pfad `/app/.cache` → `/mnt/user/appdata/braincut-blur`
   - Extra Params: `--gpus all`

### Umgebungsvariablen

| Variable | Beispiel | Beschreibung |
|---|---|---|
| `N8N_SERVER_IP` | `192.168.1.100` | Nur IP – kein Port |
| `N8N_SERVER_PORT` | `5678` | Port des N8N-Servers |
| `MEDIA_HOST_PATH` | `/mnt/user/n8n_automation/BrainCut` | Muss mit dem gemounteten Pfad übereinstimmen |
| `MODEL_CATALOG_URL` | *(leer)* | Optional: URL zu externer Modell-Katalog-JSON |

---

## GPU-Anforderungen

- NVIDIA GPU mit CUDA-Support
- Unraid: GPU-Passthrough aktiviert
- Nvidia-Driver-Plugin installiert
- Container wird mit `--gpus all` gestartet

---

## N8N-Workflows importieren

1. N8N öffnen → Workflows → Importieren
2. Folgende Dateien importieren (Reihenfolge egal):
   - `N8N-Nodes/BrainCut – Eingang.json`
   - `N8N-Nodes/BrainCut – Processing.json`
   - `N8N-Nodes/BrainCut – Wizard.json`
3. Workflows aktivieren

---

## Testvideo ablegen und verarbeiten

1. Video in `/mnt/user/n8n_automation/BrainCut/01_upload/` kopieren
2. Telegram: `scan` schicken
3. Wizard führt durch Optionen (Gesicht/Kennzeichen, Audio, Geschwindigkeit)
4. Job startet automatisch

---

## Modelle verwalten

Web-GUI: `http://<unraid-ip>:8080`

- **CenterFace** (integriert): frontale Gesichtserkennung
- **YOLOv11n/s Kennzeichen**: muss über GUI installiert werden (ca. 10–38 MB)

Test-Endpoint:
```
POST /api/models/test
{ "model_id": "yolov8n-plates-eu", "image": "/data/01_upload/testbild.jpg" }
```

---

## Render-Endpoint

Der Service übernimmt seit Version 2 das FFmpeg-Rendering direkt (kein /run-shell mehr).

```
POST /render
{
  "clips": [{ "path": "/mnt/.../video.mp4", "speed_factor": 1.0, "mute_audio": false, "volume_factor": 1.0 }],
  "audio": { "mode": "original" },
  "output_path": "/mnt/.../03_output/fertig.mp4",
  "overwrite": true,
  "move_sources_to": "/mnt/.../04_done",
  "cleanup_paths": []
}
```

---

## Bekannte Einschränkungen

- Nur ein Job gleichzeitig (weitere Anfragen → 409 Conflict)
- Hochkant-Videos werden erkannt und korrekt rotiert
- Videos ohne Tonspur erhalten automatisch eine stille Audiospur
- Unterschiedliche Auflösungen werden auf 1920px Breite normalisiert

---

## Fehlerbehebung

| Problem | Ursache | Lösung |
|---|---|---|
| `Scan nicht möglich. Workflow ist aktuell: processing` | Letzter Job fehlgeschlagen ohne Reset | Telegram: `reset` schicken |
| `409 Conflict` beim Blur-Start | Job läuft bereits | Status prüfen unter `/status`, warten oder abbrechen |
| Keine Gesichter erkannt (WARNUNG) | Falsches Modell oder sehr dunkle/unscharfe Videos | CenterFace-Modell prüfen, Test-Frame hochladen |
| Kennzeichen-Blur ohne Modell | Kein Kennzeichen-Modell aktiv | Web-GUI → Modelle → YOLOv11 installieren |
| Container startet nicht | GPU nicht verfügbar | `--gpus all` in ExtraParams, Nvidia-Plugin prüfen |
