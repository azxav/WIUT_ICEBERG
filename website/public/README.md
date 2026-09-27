# ICEBERG — Traffic Event Detection

Offline fixed-camera traffic event detection and causal accident-risk baseline.

Public source repository: https://github.com/azxav/WIUT_ICEBERG

## Run

Python 3.10+ is required. The supplied YOLO11s weights are in `weights/yolo11s.pt`; no download step is needed.

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python run_submission.py --videos samples --out predictions_samples.json --team ICEBERG
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only
```

The organizer harness and evaluator are unchanged. `detect_events()` accepts one video path; `RiskEstimator.step()` processes frames in order and reads no video files. Sample and hidden-test runtime budget is 3× video duration for both parts together.

## Method

1. Register the hand-labelled reference scene to each video's first frame.
2. Run YOLO11s + ByteTrack every third frame at 960 px inference size; downscale 4K frames to 1920 px width, then restore source coordinates.
3. Extract smoothed motion, lane, road, queue, crosswalk and stop-line features.
4. Apply temporal rules for congestion, yielding conflicts and red-light running. Classify the mapped vehicle signal by lamp position and color.
5. Merge short gaps and discard fragments below the event minimum.
6. For Part B, sample a causal tracker every 10 frames and combine TTC, hard braking and red-signal movement into a decaying risk score.

**Current class gate (decision C11, minimum dev precision 0.7):** `congestion`, `failure_to_yield`, and `red_light` are emitted. `jaywalking` and `stop_line` remain in the official `CLASSES` list but are suppressed in the scored output. The current run emits no candidates for either, so their reported zero precision is not a candidate-rule precision estimate; the labels include 13 jaywalking and 2 stop-line events. Other official classes stay available in the interface and are disabled until a reviewed positive example supports a reliable rule.

## Sample validation

`my_labels.json` contains 118 reviewed intervals from four clips of the same fixed camera: 16 congestion, 83 failure-to-yield, 13 jaywalking, 4 red-light, and 2 stop-line labels. At tIoU 0.5, the emitted classes have development precision of 0.73, 0.85, and 1.00 respectively. The gated output has Score A 0.5084 across all five represented classes. This is a small rule-tuning set, not a held-out benchmark.

The red-signal ablation uses the same detections and labels: changing from a red-priority vote to lamp-position plus dominance keeps all 4 red-light matches, removes 2 green-phase false alarms, and raises Score A from 0.4684 to 0.5084.

`predictions_samples.json` contains 100 events and passes the official format validator. It is the last completed sample output with two green-phase red-light false alarms removed according to the corrected lamp classifier. The last complete four-video harness pass used the earlier signal rule: 102 events, no format errors, and 625.3 s / 340.3 s (C3896), 539.2 s / 317.8 s (C3897), 555.1 s / 317.8 s (C3902), and 228.2 s / 127.6 s (C3905). That run passed the hard 3× budget at 1.70–1.84×, but missed the preferred 1.5× target. A later all-video repeat with the final seeded code exceeded budget on C3896 and was stopped; full-run timing for the exact current version remains unverified. Two 10-second clean-environment smoke runs produced identical events and risk arrays (29.3 s and 20.7 s on this laptop).

There are no accident labels in the development set, so Part B is not scored or calibrated. Treat the risk curve as an experimental causal baseline.

## Data, weights, and licences

- `weights/yolo11s.pt`: pretrained Ultralytics YOLO11s COCO weights, approximately 19 MB. The model weights and Ultralytics code are under Ultralytics' AGPL-3.0 terms by default; this repository is licensed under AGPL-3.0. See the [Ultralytics licence](https://www.ultralytics.com/license).
- Pretraining data: COCO 2017 only; no external traffic-event training set or task-specific fine-tuning was used. COCO annotations are CC BY 4.0; source-image copyright and terms remain with each image's original owner. See [COCO terms of use](https://cocodataset.org/#termsofuse).
- No sample video is used for model training. `my_labels.json` is a manually reviewed development set for rule validation and tuning.

## Reproducibility

The detector and tracker use fixed inference settings in `src/tracking.py`; Python, NumPy, and PyTorch RNGs are seeded to 0 before each video part. There is no model training or stochastic augmentation in this repository. The expected output can vary across Ultralytics, PyTorch, CUDA, or GPU versions. `requirements.txt` pins the runtime packages.

## Development files

- `src/scene.json`: reviewed lanes, crossings, lights, queue zone, and road geometry.
- `my_labels.json`: reviewed sample-video event intervals.
- `labels/dev_labels_notes.md` and `labels/evidence/`: event rationales and start/middle/end image evidence.
- `cache/`: local track caches and rule evaluation artifacts; not required to run submission.
- `tests/`: unit tests for tracking, signal classification, rules, and event segmentation.

## Website and upload demo

The React/Vite site is in `website/`. For local development, start the API from the repository root with `python -m uvicorn demo_api.app:app --host 127.0.0.1 --port 8000`, then run `cd website`, `npm ci`, and `npm run dev`; Vite proxies `/api` to the local API. The site uses reviewed labels, cached tracks, sample predictions, and the registered junction map to build its charts. To rebuild the annotated sample clips, install `tools/requirements_render.txt` and run `python tools/render_site_videos.py`. Then run `python tools/build_site_data.py --pred predictions_samples.json --metrics cache/site_eval.json --ablation-metrics cache/site_eval_before_signal_fix.json` to regenerate `website/public/site-data.json`, copy the report/prediction JSON, and place the model weights in the static download area.

Deploy the complete site and upload API together on one cloud VM with Docker Compose. Caddy provides HTTPS; the web container serves the built site and proxies API requests internally. Only ports 80 and 443 are published. Uploads are limited to 2 minutes and 200 MB, resized to at most 1280×720, analyzed in a single-worker queue, and deleted after processing. See [DEPLOY_VM.md](DEPLOY_VM.md). No Vercel, GitHub deployment integration, or Hugging Face service is required; transfer the source directly to the VM. VM/domain provisioning and deploy approval remain owner tasks.

## Team and report

Team **ICEBERG**

| Member | Assigned responsibility | Profiles |
|---|---|---|
| Azizbek Xasanov | Scene-map and CVAT review, event-label QA, release documentation | [LinkedIn](https://www.linkedin.com/in/azizbek-xasanov/) · [GitHub](https://github.com/azxav) |
| Dilyorbek Muhammadjonov | Detection and tracking; Part A rules, runtime and reproducibility review | [LinkedIn](https://www.linkedin.com/in/dilyor/) · [GitHub](https://github.com/dilyorm) |
| Davlat Mahmudov | Evaluation and risk review; website/demo and delivery | Profile links not supplied |

These are assigned workstreams for this submission. Individual historical contributions, past projects, and Davlat's profile links were not supplied.
