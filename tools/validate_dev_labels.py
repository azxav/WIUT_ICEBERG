"""Check raw GT constraints and run the official prediction-format validator."""
import json
import subprocess
import sys
from pathlib import Path

gt = json.loads(Path("labels/dev_labels.json").read_text())
allowed = {"accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"}
assert set(gt) == {f"{x}.MP4" for x in ("C3896", "C3897", "C3902", "C3905")}
total = 0
for name, video in gt.items():
    assert set(video) == {"duration", "fps", "events"}, name
    assert video["duration"] > 0 and video["fps"] > 0, name
    for i, (start, end, label) in enumerate(video["events"]):
        assert 0 <= start < end <= video["duration"], (name, i, start, end)
        assert label in allowed, (name, i, label)
        for j, (other_start, other_end, other_label) in enumerate(video["events"][:i]):
            assert label != other_label or (end <= other_start or start >= other_end), (name, i, j, label)
        total += 1
    assert [x[0] for x in video["events"]] == sorted(x[0] for x in video["events"]), name
wrapper = {"videos": {name: {"events": video["events"]} for name, video in gt.items()}}
dest = Path("cache/dev_labels_pred_format.json")
dest.write_text(json.dumps(wrapper))
subprocess.run([sys.executable, "evaluate.py", "--pred", str(dest), "--validate-only"], check=True)
evidence = list(Path("labels/evidence").glob("*.jpg"))
assert len(evidence) == total, (len(evidence), total)
print(f"GT constraints: {len(gt)} videos, {total} events, {len(evidence)} evidence strips; no same-class overlap")
