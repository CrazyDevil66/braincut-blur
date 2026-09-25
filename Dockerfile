# ── Stage 1: Rust build ───────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        ca-certificates \
        python3-pip \
        libssl-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only onnxruntime + deface to extract CenterFace model at build time.
# We use CPU onnxruntime here because the builder has no GPU.
RUN pip install --no-cache-dir --break-system-packages onnxruntime==1.18.1 onnx deface==0.3.0

# Fix deface compatibility with onnx >= 1.13
RUN python3 -c "\
import deface, os; \
path = os.path.join(os.path.dirname(deface.__file__), 'centerface.py'); \
src = open(path).read(); \
src = src.replace('onnx.utils.polish_model(dyn_model)', 'onnx.shape_inference.infer_shapes(dyn_model)'); \
src = src.replace('onnxruntime.InferenceSession(dyn_model.SerializeToString())', \
    'onnxruntime.InferenceSession(dyn_model.SerializeToString(), providers=[\"CPUExecutionProvider\"])'); \
open(path, 'w').write(src)"

# Extract CenterFace ONNX from deface and patch input to dynamic shape [N,3,H,W].
# The bundled model has fixed dims (e.g. [10,3,32,32]); CenterFace is a fully-convolutional
# network so patching the declared input dims + clearing cached intermediate shapes is enough.
ENV XDG_CACHE_HOME=/tmp/deface_cache
RUN python3 -c "from deface.centerface import CenterFace; CenterFace()" && \
    python3 -c "\
import onnx, onnxruntime as ort, numpy as np, os, glob, deface; \
hits = glob.glob('/tmp/deface_cache/**/*.onnx', recursive=True) + \
       glob.glob(os.path.join(os.path.dirname(deface.__file__),'**/*.onnx'), recursive=True); \
found = [h for h in hits if 'centerface' in os.path.basename(h).lower()]; \
assert found, 'centerface.onnx not found in deface package'; \
print('CenterFace original:', found[0]); \
m = onnx.load(found[0]); \
inp = m.graph.input[0]; \
print('Input name:', inp.name, 'orig shape:', [d.dim_value for d in inp.type.tensor_type.shape.dim]); \
[setattr(inp.type.tensor_type.shape.dim[i], 'dim_param', p) for i, p in [(0,'N'),(2,'H'),(3,'W')]]; \
del m.graph.value_info[:]; \
os.makedirs('/tmp/models', exist_ok=True); \
onnx.save(m, '/tmp/models/centerface.onnx'); \
sess = ort.InferenceSession('/tmp/models/centerface.onnx', providers=['CPUExecutionProvider']); \
dummy = np.zeros((1,3,256,320), dtype=np.float32); \
out = sess.run(None, {inp.name: dummy}); \
print('Verification OK – outputs:', [o.shape for o in out]); \
print('Saved dynamic CenterFace ONNX [N,3,H,W]')"

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Cache Cargo registry + download ORT binary (ort-sys download-binaries feature)
# in a separate layer — source-only changes skip this expensive step.
COPY Cargo.toml Cargo.lock* build.rs ./
RUN mkdir -p src ui && echo 'fn main(){}' > src/main.rs && \
    touch ui/style.css ui/body.html ui/app.js && \
    cargo build --release

# Build the real binary (reuses cached ORT binary from above)
COPY src/ ./src/
COPY ui/style.css ui/body.html ui/app.js ./ui/
RUN find src -name "*.rs" -exec touch {} + && touch build.rs && cargo build --release

# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        tzdata \
        ca-certificates \
        wget \
    && ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Rust binary
COPY --from=builder /build/target/release/blur-service /app/blur-service

# ORT shared libraries (placed next to binary by the copy-dylibs feature)
COPY --from=builder /build/target/release/libonnxruntime*.so* /app/

# CenterFace ONNX model (extracted in builder stage)
COPY --from=builder /tmp/models/centerface.onnx /app/models/centerface.onnx

# GPU / container runtime settings
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Service configuration
ENV MODELS_PATH=/app/.cache/models
ENV CENTERFACE_MODEL=/app/models/centerface.onnx
ENV CONTAINER_ROOT=/data
ENV MEDIA_HOST_PATH=/mnt/user/n8n_automation/BrainCut

# libonnxruntime.so lives next to the binary
ENV LD_LIBRARY_PATH=/app

VOLUME /app/.cache

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD wget -qO- http://localhost:8080/health | grep -q ok

CMD ["/app/blur-service"]
