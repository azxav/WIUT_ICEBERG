#!/usr/bin/env bash
# Fetch the model weights once, before the offline run.
#   bash weights/download.sh
# Idempotent: files that already exist (non-empty) are kept. Downloads go to a
# .part file first, so an interrupted run never leaves a truncated model.
set -euo pipefail
cd "$(dirname "$0")"

fetch() {
  local name="$1" url="$2"
  if [ -s "$name" ]; then
    echo "ok   $name"
    return
  fi
  echo "get  $name <- $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$name.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$name.part" "$url"
  else
    python - "$url" "$name.part" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  fi
  mv "$name.part" "$name"
}

# COCO-pretrained YOLO11-m detector (Ultralytics, AGPL-3.0), ~40 MB
fetch yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# YOLO11-n, ~5 MB: CPU-friendly detector used only by the web demo (demo/app.py)
fetch yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# Optional fire/smoke detector (YOLO format, classes fire/smoke). When
# weights/fire_smoke.pt is absent the pipeline simply skips fire detection.
# Add its release URL here once we have trained/selected one, e.g.:
# fetch fire_smoke.pt https://github.com/<org>/<repo>/releases/download/<tag>/fire_smoke.pt

echo "weights ready in $(pwd)"
