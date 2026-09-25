# Team Iceberg: traffic events from a fixed road camera

WIUT Hackathon 2026, Computer Vision track. From CCTV clips of a fixed road
camera we

- **Part A:** detect 14 kinds of traffic events as time segments
  `[start_sec, end_sec, label]` (accidents, red-light running, jaywalking,
  congestion and more), scored by macro F1 at tIoU 0.3 / 0.5 / 0.7;
- **Part B:** estimate, causally and frame by frame, the probability that an
  accident starts within the next 5 s.

Everything runs offline on one GPU (target: T4 16 GB) within the budget of
3x the video length for Parts A and B together.

## Quick start

```bash
# 1. environment (Python 3.11)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. weights (once, needs internet; ~40 MB)
bash weights/download.sh

# 3. the official commands
python run_submission.py --videos /data/test --out predictions.json --team iceberg
python evaluate.py --pred predictions.json --gt ground_truth.json
```

Or with Docker. The image bakes the weights in at build time, so the run needs no network:

```bash
docker build -t iceberg .
docker run --gpus all --rm -v /data/test:/data/test:ro -v "$PWD/out":/out iceberg
docker run --gpus all --rm -v "$PWD/out":/out -v /data:/data iceberg \
    python evaluate.py --pred /out/predictions.json --gt /data/ground_truth.json
```

`solution.py` loads and warms up the models at import time, which the
harness does before it starts timing videos. Optional environment variables:

| variable | effect |
|---|---|
| `ICEBERG_PARAMS` | JSON merged over `configs/params.json`, e.g. `configs/params.tuned.json` |
| `ICEBERG_SCENE` | scene layout file (default `configs/scene.json`) |
| `ICEBERG_CACHE_DIR` | cache perception output per video (dev only; leave unset for the official run) |
| `ICEBERG_NO_WARMUP` | skip model warm-up at import (tests, tooling) |

## Approach

```
video ──► one decode pass (every 2nd frame) ──► Observations ──► rules ──► segments ──► events
            YOLO11m detector + ByteTrack           tracks, light-ROI colours,        per-class thresholds,
            light-ROI colour features               fire detections, thumbnails        offsets, merging
```

1. **Perception (learned).** A COCO-pretrained YOLO11m (`imgsz` 960, FP16,
   batched) detects people, bicycles, vehicles, animals and traffic lights on
   every 2nd frame. ByteTrack (`supervision`) links detections into tracks.
   The same pass samples colour features inside the traffic-light boxes, grey
   1 fps thumbnails and, when `weights/fire_smoke.pt` is present, a
   fire/smoke detector at 1 fps (`src/observe.py`).
2. **Trajectories.** Tracks broken by ID switches are stitched back together.
   Positions are smoothed foot points (bottom centre of the box). Speeds are
   in box heights per second, which roughly corrects for perspective without
   camera calibration. Riders are separated from pedestrians (`src/tracks.py`).
3. **Scene knowledge.** Two sources. First, a hand-drawn layout of the fixed
   camera (`configs/scene.json`): carriageway, intersection box, crosswalks,
   lanes with legal directions, stop lines tied to lights, solid lines,
   entry/exit zones with allowed movements (`src/scene.py`). Second, a flow
   field learned from traffic itself: the dominant heading per grid cell
   (`src/flowfield.py`, with a prior from the sample clips). The light state
   (red, yellow or green) per frame comes from the ROI colour features
   (`src/lights.py`).
4. **Rules, one per class** (`src/rules/`). These are pure functions of the
   trajectories, the scene and the light states, so they re-run from cached
   observations in seconds. Examples: `red_light` is a crossing of a stop
   line in its legal direction while its light is red; `stopped_vehicle` is
   stationary for 10 s or more on the carriageway and not queued at a signal;
   `accident` is ground footprints touching together with abrupt braking or
   a heading jerk, followed by the vehicles coming to rest.
5. **Segments.** Per class: a score threshold, start/end offsets fitted on
   our labels, merging of fragments and a minimum length. Same-class
   overlaps are never emitted (`src/segments.py`, `src/pipeline.py`).
6. **Part B, the causal risk score** (`src/risk.py`, `src/ttc.py`). It
   detects and tracks every 3rd frame at `imgsz` 640, estimating causal
   velocities over short track histories. The main cue is the 2D
   time-to-collision between ground footprints of approaching road users,
   boosted by hard braking. It is mapped to a probability, then shaped into
   one sharp alarm per risky episode for the alarm-F1 part of the metric.

**Learned and rule-based parts.** Only perception is learned: YOLO11m is
used as released (COCO) and we did no fine-tuning. The traffic-direction
field is estimated from the clips' own tracks without labels. Everything
else is explicit, auditable rules with a small set of thresholds. Those
thresholds live in `configs/params.json` and per-module `DEFAULTS`. Some are
fitted on our hand labels with `scripts/tune.py`.

**Determinism.** `src/config.py: SEED = 0`. `seed_everything()` seeds
Python, NumPy and PyTorch and sets cuDNN to deterministic mode. Tracking and
rules are deterministic. FP16 inference can differ in the last bits between
GPU models, so boxes on a T4 may differ very slightly from our RTX 3050
runs. `scripts/tune.py` uses a seeded random search.

## Models, data and licences

| component | use | licence |
|---|---|---|
| [Ultralytics YOLO11m](https://github.com/ultralytics/ultralytics), COCO-pretrained weights `yolo11m.pt` | object detector (Part A and B) | AGPL-3.0 |
| [MS COCO 2017](https://cocodataset.org) | pre-training data of the detector (not used by us directly) | annotations CC BY 4.0; images under Flickr terms |
| [supervision](https://github.com/roboflow/supervision) (ByteTrack) | multi-object tracking | MIT |
| ByteTrack algorithm (Zhang et al., ECCV 2022) | tracking method | MIT (reference implementation) |
| 2D time-to-collision idea (Y. Jiao, TU Delft) | Part B risk cue, re-implemented for image-plane boxes | method only, no code copied |
| PyTorch, torchvision | inference runtime | BSD-3-Clause |
| OpenCV | video decoding, image ops | Apache-2.0 |
| NumPy, SciPy | numerics | BSD-3-Clause |
| matplotlib (dev only) | EDA plots | PSF-based (matplotlib licence) |
| optional `fire_smoke.pt` | fire/smoke detector, if added | to be listed with its source |
| sample clips from the organizers | our dev labels (`my_labels.json`) and tuning | organizers' terms; not redistributed |

Because the detector (Ultralytics) is AGPL-3.0, this repository has to stay
public under an AGPL-compatible licence.

## Repository layout

```
solution.py            entry point for the harness: CLASSES, detect_events, RiskEstimator
run_submission.py      organizers' harness (unchanged)
evaluate.py            organizers' metric (unchanged)
src/
  config.py            paths, COCO class groups, params loading (ICEBERG_PARAMS)
  video.py             probing, strided decoding
  perception.py        YOLO detector, ByteTrack tracker, seeding
  observe.py           one pass over a video -> Observations (+ optional cache)
  tracks.py            track building, stitching, kinematics
  scene.py             scene layout + geometry helpers
  flowfield.py         learned traffic direction per grid cell
  lights.py            traffic-light state from ROI colours
  rules/               one rule per event class (+ registry)
  segments.py          intervals, hysteresis, merging, per-class finalisation
  pipeline.py          Part A end to end
  ttc.py, risk.py      Part B
configs/
  params.json          all thresholds (perception, tracks, per-class post-processing, risk)
  scene.json           our camera's layout (scripts/annotate_scene.py); scene.example.json shows the schema
  params.tuned.json    tuned overrides (scripts/tune.py), used via ICEBERG_PARAMS
  flow_prior.npz       direction prior from the sample clips (scripts/build_flow_prior.py)
scripts/               dev tooling (below); never imported by the submission
tests/                 pytest on synthetic trajectories (tests/synth.py)
weights/download.sh    fetches the model weights (weights themselves are git-ignored)
site/, demo/           project website and demo
```

## Dev workflow

```bash
bash weights/download.sh                      # once
pip install -r requirements-dev.txt           # matplotlib, pytest

python scripts/annotate_scene.py --video samples/<clip>.mp4 --t 10   # draw configs/scene.json (S saves)
python scripts/annotate_scene.py --show                               # overlay PNG -> renders/scene_overlay.png
python scripts/extract.py                     # perception once per clip -> .cache/ (prints runtime vs 3x budget)
python scripts/build_flow_prior.py            # -> configs/flow_prior.npz
python scripts/eda.py                         # plots + summary -> renders/eda/

python run_submission.py --videos samples --out predictions_samples.json --team iceberg --no-risk
python scripts/label_events.py --candidates predictions_samples.json  # label / verify -> my_labels.json
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video

python scripts/tune.py                        # per-class fit -> configs/params.tuned.json
python scripts/render.py --labels my_labels.json --params configs/params.tuned.json  # annotated MP4 + JSON
python scripts/benchmark.py --videos samples  # the official harness, timed against the budget

ICEBERG_NO_WARMUP=1 python -m pytest -q tests
```

| script | what it does |
|---|---|
| `annotate_scene.py` | OpenCV tool to draw the scene layout on a reference frame. Keys pick the element type: `c` carriageway, `i` intersection, `w` crosswalk, `l` lane + direction, `s` stop line + direction + light, `d` solid line, `t` light ROI, `z` zone. Writes normalised coordinates. `--show` renders an overlay PNG. |
| `extract.py` | runs perception on every sample and caches it; prints per-video runtime, fps and share of the 3x budget |
| `label_events.py` | video player for labelling in `evaluate.py` ground-truth format. Seek with the arrow keys, `s` / `e` mark start and end, digits pick the class. It saves after every change; `--candidates` jumps between predicted events and `y` accepts one. |
| `tune.py` | per-class random search plus coordinate descent over score threshold, offsets, merge gap, minimum length and a few rule keys. It maximises that class's mean F1 from `evaluate.py`. A change is kept only when it gains at least `--min-gain`. |
| `render.py` | annotated H.264 MP4: boxes, ids, trails, scene, light states, active-event banner and a timeline against the labels. Also a poster, event snapshots and a per-video JSON for the website. |
| `eda.py` | brightness, object counts, foot-point heatmap, trajectories, flow field and lane density, plus `summary.json` |
| `build_flow_prior.py` | averages the learned flow field over the sample clips and warns if a clip looks like a different camera |
| `benchmark.py` | runs `run_submission.py` exactly as the organizers do and reports Part A, Part B and total time against the budget |

## Organizers' starter kit

This repository started from the organizers' kit. `run_submission.py` and
`evaluate.py` are theirs and unchanged, and `examples/` holds their format
examples. The interface we implement in `solution.py`:

```python
CLASSES = [...]                                   # subset of the 14 official ids
def detect_events(video_path: str) -> list[list]  # Part A: [[start, end, label], ...]
class RiskEstimator:                              # Part B, called on every frame in order
    def reset(self, meta: dict) -> None: ...
    def step(self, frame, t_sec: float) -> float: ...
```

Scoring: Model = 0.7 x Score A + 0.3 x Score B, or Score A alone when the
test set has no accidents. A video over the time budget, or one that
crashes, scores as empty. See the docstring of `evaluate.py` for the exact
metric.

## Team

| name | role |
|---|---|
| _TBD_ | _TBD_ |
