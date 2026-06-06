FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV XDG_CACHE_HOME=/app/.cache
ENV QT_QPA_PLATFORM=offscreen

# Gesichts-Modell beim Build herunterladen (kein Delay beim ersten Job)
RUN python -c "from deface.centerface import CenterFace; CenterFace()"

COPY blur_service.py .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "blur_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
