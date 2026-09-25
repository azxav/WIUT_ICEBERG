# Offline evaluation image: CUDA 12.4 + PyTorch 2.6 runtime, weights baked in at build time.
#   docker build -t iceberg .
#   docker run --gpus all --rm -v /data/test:/data/test:ro -v "$PWD/out":/out iceberg
#   docker run --gpus all --rm -v ... iceberg python evaluate.py --pred /out/predictions.json --gt /data/ground_truth.json
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_OFFLINE=true \
    YOLO_VERBOSE=false

# libGL / glib: needed by the opencv-python wheel that ultralytics pulls in next to the headless one
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# idempotent: keeps weights already present in the build context, fetches the rest
RUN bash weights/download.sh

CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/out/predictions.json", "--team", "iceberg"]
