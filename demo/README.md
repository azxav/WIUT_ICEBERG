---
title: Iceberg Traffic Events
emoji: 🚦
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Detect traffic events and accident risk in road-camera clips
---

# Iceberg: live demo

Upload a fixed road-camera clip (.mp4, up to 2 minutes / 200 MB). The Space runs team
Iceberg's WIUT Hackathon 2026 pipeline on CPU and returns:

- an events table and an interactive timeline (one row per class),
- the annotated video (boxes, tracks, active-event banner, risk gauge; H.264),
- the causal accident-risk curve (`P(accident starts within 5 s)`),
- a `predictions.json` in the organizers' harness format.

CPU settings (in `app.py`, `DEMO_OVERRIDE`): YOLO11n at 640 px on every 4th frame. The GPU
submission uses the larger detector from `configs/params.json`, so results here are a little
less accurate. Set the Space variable `DEMO_DETECTOR` to use other weights from `weights/`.

## Files

| file | purpose |
|---|---|
| `app.py` | Gradio UI, input checks, progress, pipeline call, one render + risk pass |
| `visualize.py` | frame drawing, H.264 writer (ffmpeg), Plotly timeline and risk chart |
| `requirements.txt` / `packages.txt` | Python deps (CPU torch) / apt `ffmpeg` |
| `examples/*.mp4` | optional: clips shown as one-click examples |

## Run locally

```bash
# from the repo root
pip install -r demo/requirements.txt
python demo/app.py            # http://127.0.0.1:7860
```

`app.py` finds the pipeline either next to itself or one directory up, so it runs from the
repo as-is.

## Deploy to Hugging Face Spaces

The Space needs the demo files at its root plus the pipeline code:

```bash
# 1. create the Space once (web UI: New Space -> SDK Gradio -> CPU basic), then:
git clone https://huggingface.co/spaces/<user>/iceberg-traffic-demo space
# 2. copy the demo and the pipeline into it (from the repo root)
cp demo/app.py demo/visualize.py demo/requirements.txt demo/packages.txt demo/README.md space/
cp -r src configs solution.py space/
mkdir -p space/weights && cp weights/yolo11n.pt space/weights/   # plus fire_smoke.pt if used
# optional one-click examples (<= 2 min each)
mkdir -p space/examples && cp samples/sample_001.mp4 space/examples/
# 3. weights and videos go through Git LFS on the Hub
cd space && git lfs install && git lfs track "*.pt" "*.mp4"
git add . && git commit -m "Deploy demo" && git push
```

If `weights/yolo11n.pt` is missing, the app downloads the official checkpoint on first run.
Custom weights (for example `fire_smoke.pt`) must be copied in: the pipeline only uses the
fire/smoke detector when its file exists.

After the push, the Space builds (about 5 minutes) and is served at
`https://<user>-iceberg-traffic-demo.hf.space`. Put `<user>/iceberg-traffic-demo` into
`SPACE_ID` in `site/assets/config.js` so the website embeds it.

## Limits and errors

Clips over 2 minutes or 200 MB, and files that do not decode, are rejected with a message.
Any failure inside the pipeline is caught and reported in the status box; the app keeps
running. One job runs at a time; up to 8 wait in the queue.
