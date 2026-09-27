FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
COPY demo_api/requirements.txt /app/demo_api-requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && pip install --no-cache-dir -r /app/demo_api-requirements.txt
COPY solution.py README.md /app/
COPY src /app/src
COPY weights /app/weights
COPY demo_api /app/demo_api

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "demo_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
