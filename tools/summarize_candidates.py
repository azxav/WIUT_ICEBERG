"""Compact manual-review shortlist from cached candidates and signal refinements."""
import json
from pathlib import Path

import pandas as pd


def phase(name):
    data = json.loads(Path(f"cache/{name}_phase_refinement.json").read_text())
    changes = []
    for item in data:
        samples = item["median_samples"]
        target = item["left_state"]
        if target == "R":
            hit = next(((t, s) for t, s in samples if s == "R" and t >= item["left_change"]), None)
        else:
            hit = next(((t, s) for t, s in samples if s == "G" and t >= item["left_change"] - .3), None)
        if hit:
            changes.append((round(hit[0], 2), hit[1]))
    return changes


def color(changes, t):
    return next((s for a, s in reversed(changes) if t >= a), "?")


for name in ["C3896", "C3897", "C3902", "C3905"]:
    c = pd.read_csv(f"cache/{name}_candidates.csv")
    p = phase(name)
    print("\n", name, "phases", p)
    stops = c[c.kind == "stop_crossing"].copy()
    stops["phase"] = stops.start.map(lambda t: color(p, t))
    print("red line crossing", [(round(r.start, 1), int(r.track)) for r in stops.itertuples() if r.phase == "R"])
    for kind in ["stopped_or_queue", "jaywalking", "wrong_way", "turn_or_swerve"]:
        a = c[c.kind == kind].copy()
        a["dur"] = a.end - a.start
        print(kind, len(a), [(round(r.start,1), round(r.end,1), int(r.track),r.detail) for r in a.sort_values("dur", ascending=False).head(20).itertuples()])
