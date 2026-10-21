# CPU-capable server: SAM 2 performs object segmentation; local recoloring preserves
# original texture/lighting. No Flux/diffusers or paid image-generation API is needed.
FROM python:3.11-slim

ARG TORCH_NUM_THREADS=2
ARG SAM2_MODEL_NAME=facebook/sam2.1-hiera-base-plus
ARG SAM2_LOCAL_FILES_ONLY=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=${SAM2_LOCAL_FILES_ONLY} \
    TORCH_NUM_THREADS=${TORCH_NUM_THREADS} \
    SAM2_MODEL_NAME=${SAM2_MODEL_NAME} \
    SAM2_LOCAL_FILES_ONLY=${SAM2_LOCAL_FILES_ONLY} \
    MAX_IMAGE_SIZE_BYTES=20971520 \
    MAX_IMAGE_DIMENSION=8192 \
    MAX_INFERENCE_SIZE=1536

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY requirements.txt requirements.sam2.txt ./

# 1. Install CPU-only PyTorch first (avoids CUDA wheels pulling in GPU libs).
#    MUST precede requirements.sam2.txt so pip sees the CPU torch already present.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
       "torch>=2.6.0" "torchvision>=0.21.0"

# 2. Install base service requirements (fastapi, opencv, etc.).
RUN python -m pip install -r requirements.txt

# 3. Install SAM 2 + its non-torch deps.
#    --no-deps on sam2 prevents pip from reinstalling/upgrading torch from PyPI.
#    Pinned PyPI release (not git+https) avoids cloning the full repo each build
#    and the slow "Installing build dependencies" isolation step.
RUN python -m pip install "huggingface-hub>=0.25.0" "hydra-core>=1.3.2" "iopath>=0.1.10" \
    && pip install --no-deps "sam2==1.1.0"

COPY app.py config.py logger.py ./
COPY recolor ./recolor

RUN mkdir -p /app/debug_output /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=8s --start-period=180s --retries=4 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=5)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers"]
