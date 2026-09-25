"""Paths, COCO class groups and tunable parameters.

All thresholds live in ``configs/params.json`` so that ``scripts/tune.py`` can
override them without touching code. ``load_params`` deep-merges an optional
override file on top of the defaults.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
WEIGHTS_DIR = ROOT / "weights"

SEED = 0

# COCO ids used by the default detector.
PERSON = 0
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
TRAFFIC_LIGHT = 9
ANIMALS = {14, 15, 16, 17, 18, 19}  # bird, cat, dog, horse, sheep, cow

VEHICLES = {CAR, MOTORCYCLE, BUS, TRUCK}
TWO_WHEELERS = {BICYCLE, MOTORCYCLE}
DETECT_CLASSES = sorted({PERSON, BICYCLE, TRAFFIC_LIGHT} | VEHICLES | ANIMALS)

CATEGORY_OF = {PERSON: "person", BICYCLE: "bicycle", TRAFFIC_LIGHT: "traffic_light"}
CATEGORY_OF.update({c: "vehicle" for c in VEHICLES})
CATEGORY_OF.update({c: "animal" for c in ANIMALS})


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


TUNED_PARAMS = CONFIG_DIR / "params.tuned.json"


def load_params(override: str | Path | dict | None = None) -> dict:
    """Defaults from configs/params.json, optionally merged with an override.

    Without an explicit override, ``ICEBERG_PARAMS`` (a JSON path) is used if
    set, else ``configs/params.tuned.json`` if it exists, so the official run
    picks up the output of ``scripts/tune.py`` with no extra flags.
    """
    params = json.loads((CONFIG_DIR / "params.json").read_text())
    if override is None:
        if os.environ.get("ICEBERG_PARAMS"):
            override = os.environ["ICEBERG_PARAMS"]
        elif TUNED_PARAMS.exists():
            override = TUNED_PARAMS
    if isinstance(override, (str, Path)):
        override = json.loads(Path(override).read_text())
    return _deep_merge(params, override) if override else params


def scene_path() -> Path:
    return Path(os.environ.get("ICEBERG_SCENE", CONFIG_DIR / "scene.json"))
