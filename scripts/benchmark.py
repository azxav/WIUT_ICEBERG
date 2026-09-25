"""Time the official harness end to end against the 3x-duration budget.

    python scripts/benchmark.py --videos samples/clip.mp4
    python scripts/benchmark.py --videos samples --no-risk

Runs ``run_submission.py`` in a subprocess exactly as the organizers will
(no observation cache, Part B on every frame unless told otherwise) and
reports Part A, Part B and total time per video from its log, with the
headroom left in the budget. Model loading happens at import, outside the
per-video clock, and is reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _common import DEFAULT_VIDEOS, ROOT, list_videos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="folder of .mp4 or one file")
    ap.add_argument("--solution", type=Path, default=ROOT / "solution.py")
    ap.add_argument("--no-risk", action="store_true", help="skip Part B")
    ap.add_argument("--risk-stride", type=int, default=1, help="official: 1")
    ap.add_argument("--time-factor", type=float, default=3.0, help="official: 3.0")
    ap.add_argument("--params", type=Path, help="ICEBERG_PARAMS override for the run")
    ap.add_argument("--keep", type=Path, help="also keep the predictions JSON here")
    args = ap.parse_args()

    videos = list_videos(args.videos)
    if not videos:
        print(f"no .mp4 files in {args.videos}", file=sys.stderr)
        return 2
    out = args.keep or Path(tempfile.gettempdir()) / "iceberg_benchmark_predictions.json"
    env = {k: v for k, v in os.environ.items() if k not in ("ICEBERG_CACHE_DIR", "ICEBERG_NO_WARMUP")}
    if args.params:
        env["ICEBERG_PARAMS"] = str(args.params.resolve())
    cmd = [sys.executable, "run_submission.py", "--videos", str(args.videos.resolve()), "--out", str(out),
           "--team", "benchmark", "--solution", str(args.solution.resolve()),
           "--risk-stride", str(args.risk_stride), "--time-factor", str(args.time_factor)]
    if args.no_risk:
        cmd.append("--no-risk")

    print(" ".join(cmd), "\n")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    wall = time.perf_counter() - t0
    if proc.returncode != 0 or not out.exists():
        print(f"run_submission failed (exit {proc.returncode})", file=sys.stderr)
        return 1

    log = json.loads(out.read_text())["log"]
    print(f"\n{'video':<28}{'dur':>7}{'budget':>8}{'part A':>8}{'part B':>8}{'total':>8}{'% used':>8}{'A x rt':>8}")
    tot_video = tot_total = 0.0
    for name, lg in log.items():
        dur, budget, total = lg["duration"], lg["budget_sec"], lg.get("total_sec", 0.0)
        a, b = lg.get("part_a_sec", 0.0), lg.get("part_b_sec", 0.0)
        tot_video += dur
        tot_total += total
        flag = "  OVER BUDGET" if total > budget else ""
        print(f"{name[:27]:<28}{dur:>7.1f}{budget:>8.0f}{a:>8.1f}{b:>8.1f}{total:>8.1f}"
              f"{100 * total / max(budget, 1e-9):>7.0f}%{a / max(dur, 1e-9):>8.2f}{flag}")
        for err in lg.get("errors", []):
            print("   !", err.splitlines()[0])
    print(f"\nvideo {tot_video:.0f}s, per-video time {tot_total:.0f}s "
          f"({tot_total / max(tot_video, 1e-9):.2f}x real time, budget {args.time_factor}x); "
          f"import + model warm-up ~{wall - tot_total:.0f}s (outside the budget)")
    if not args.keep:
        out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
