FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# deface 0.3.0 nutzt onnx.utils.polish_model, das in onnx>=1.13 entfernt wurde.
# Ersetzen durch das äquivalente shape_inference.infer_shapes.
RUN python -c "import deface; import os; path = os.path.join(os.path.dirname(deface.__file__), 'centerface.py'); \
    src = open(path).read(); \
    src = src.replace( \
        'dyn_model = onnx.utils.polish_model(dyn_model)', \
        'dyn_model = onnx.shape_inference.infer_shapes(dyn_model)' \
    ); \
    open(path, 'w').write(src)"

ENV XDG_CACHE_HOME=/app/.cache
ENV QT_QPA_PLATFORM=offscreen
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Gesichts-Modell beim Build herunterladen (kein Delay beim ersten Job)
RUN python -c "from deface.centerface import CenterFace; CenterFace()"

COPY blur_service.py .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "blur_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
