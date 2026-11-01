# Runs Zone Watch inference on CPU. The model, videos and outputs are mounted at run time.
FROM python:3.11.9-slim-bookworm

# Shared system libraries that OpenCV needs but the slim image leaves out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies come before the code, so that editing the code does not reinstall them.
# The CPU-only torch wheel is far smaller than the default one, which bundles CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

ENTRYPOINT ["python", "-m", "zone_watch"]
