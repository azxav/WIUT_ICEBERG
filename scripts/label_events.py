"""Label events on the sample videos -> my_labels.json (evaluate.py ground-truth format).

    python scripts/label_events.py                                   # samples/*.mp4 -> my_labels.json
    python scripts/label_events.py --candidates predictions_samples.json   # verify predictions fast

Output: ``{video: {"duration": s, "fps": f, "events": [[start, end, label], ...]}}``,
saved after every change. Same-class overlapping labels are merged (one
segment per simultaneous same-class event, as in the task conventions).

Keys
    space        play / pause              + / -    playback speed
    Left/Right   -/+ 1 s                   Up/Down  -/+ 5 frames
    , / .        -/+ 1 frame               [ / ]    -/+ 10 s
    g            go to time (console)      click on the timeline to seek
    1-9, 0       class 1-10                ! @ # $  class 11-14 (shift+1..4)
    c / C        next / previous class
    s            mark start                e        mark end -> save [start, end, class]
    Esc          cancel the pending start  x        delete the label under the playhead
    z            undo the last added label
    j / k        next / previous candidate (jumps 1 s before it, selects its class)
    y            accept the candidate under the playhead as a label
    v            mark this video fully labelled (lists it even with no events)
    n / b        next / previous video     h        help      q  quit

A video enters the labels file with its first label (or ``v``): evaluate.py
counts every prediction on a listed video, so only list finished videos.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import DEFAULT_VIDEOS, ROOT, list_videos, write_json
from _viz import CLASS_COLORS, draw_playhead, text, timeline, timeline_time
from evaluate import OFFICIAL_CLASSES
from src.segments import merge_intervals

# waitKeyEx codes: Windows, then GTK (Linux)
LEFT, RIGHT, UP, DOWN = (2424832, 65361), (2555904, 65363), (2490368, 65362), (2621440, 65364)
SHIFT_DIGITS = "!@#$"
PRE_ROLL = 1.0


class Player:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.n / self.fps
        self.idx = -1
        self.frame: np.ndarray | None = None
        self.seek(0)

    @property
    def t(self) -> float:
        return self.idx / self.fps

    def seek(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, max(0, self.n - 1)))
        if idx != self.idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.idx, self.frame = idx, frame

    def step(self) -> bool:
        if self.idx + 1 >= self.n:
            return False
        self.seek(self.idx + 1)
        return True

    def close(self) -> None:
        self.cap.release()


class Labeller:
    def __init__(self, videos: list[Path], labels_path: Path, candidates: dict, width: int):
        self.videos, self.labels_path, self.cands_all, self.width = videos, labels_path, candidates, width
        self.labels: dict = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        self.cls = 0
        self.speed = 1.0
        self.playing = False
        self.pending: float | None = None
        self.history: list[list] = []
        self.vi = 0
        self.player: Player | None = None
        self.strip_cache: np.ndarray | None = None
        self.strip_y = 0
        self.seek_to: float | None = None

    # ------------------------------------------------------------ labels
    @property
    def name(self) -> str:
        return self.videos[self.vi].name

    def entry(self) -> dict:
        """This video's ground-truth entry, created on first use.

        A video only enters the file once it has a label or is marked done
        (``v``): evaluate.py treats every listed video as fully labelled.
        """
        p = self.player
        e = self.labels.setdefault(self.name, {"duration": round(p.duration, 3), "fps": p.fps, "events": []})
        e["duration"], e["fps"] = round(p.duration, 3), p.fps
        return e

    @property
    def events(self) -> list[list]:
        return self.labels.get(self.name, {}).get("events", [])

    def save(self) -> None:
        self.events.sort(key=lambda ev: (ev[0], ev[2]))
        write_json(self.labels_path, self.labels, indent=1)
        self.strip_cache = None

    def add(self, s: float, e: float, label: str) -> None:
        s, e = sorted((round(s, 2), round(e, 2)))
        if e - s < 0.1:
            print("  too short, ignored")
            return
        evs = self.entry()["events"]
        same = [ev for ev in evs if ev[2] == label and ev[0] <= e and ev[1] >= s]
        if same:
            print(f"  merged with {len(same)} overlapping {label} label(s)")
            for ev in same:
                evs.remove(ev)
        (s, e), = merge_intervals([(s, e)] + [(ev[0], ev[1]) for ev in same])
        ev = [s, e, label]
        evs.append(ev)
        self.history.append(ev)
        print(f"  + [{s:.2f}, {e:.2f}, {label}]")
        self.save()

    def delete_at(self, t: float) -> None:
        hits = [ev for ev in self.events if ev[0] <= t <= ev[1]]
        if not hits:
            print("  no label under the playhead")
            return
        ev = max(hits, key=lambda x: x[0])
        self.events.remove(ev)
        print(f"  - {ev}")
        self.save()

    def undo(self) -> None:
        while self.history:
            ev = self.history.pop()
            if ev in self.events:
                self.events.remove(ev)
                print(f"  undo {ev}")
                self.save()
                return
        print("  nothing to undo")

    # -------------------------------------------------------- candidates
    def candidates(self) -> list[list]:
        c = self.cands_all.get(self.name, {})
        return sorted(c.get("events", []) if isinstance(c, dict) else [], key=lambda ev: ev[0])

    def jump(self, forward: bool) -> None:
        t = self.player.t + PRE_ROLL
        cands = self.candidates()
        nxt = [c for c in cands if c[0] > t + 0.05] if forward else [c for c in cands if c[0] < t - 0.05]
        if not nxt:
            print("  no more candidates")
            return
        c = nxt[0] if forward else nxt[-1]
        if c[2] in OFFICIAL_CLASSES:
            self.cls = OFFICIAL_CLASSES.index(c[2])
        self.player.seek(int((c[0] - PRE_ROLL) * self.player.fps))
        print(f"  candidate [{c[0]:.2f}, {c[1]:.2f}, {c[2]}]")

    def accept(self) -> None:
        t = self.player.t
        hits = [c for c in self.candidates() if c[0] - PRE_ROLL - 0.1 <= t <= c[1]]
        if not hits:
            print("  no candidate under the playhead")
            return
        c = min(hits, key=lambda c: abs(c[0] - t))
        self.add(c[0], c[1], c[2])

    # ------------------------------------------------------------ render
    def strip(self) -> np.ndarray:
        if self.strip_cache is None:
            p = self.player
            cands = self.candidates()
            used = sorted({ev[2] for ev in self.events} | {c[2] for c in cands} | {OFFICIAL_CLASSES[self.cls]},
                          key=lambda c: OFFICIAL_CLASSES.index(c) if c in OFFICIAL_CLASSES else 99)
            rows = [{"name": c, "color": CLASS_COLORS.get(c, (200, 200, 200)),
                     "pred": [(ev[0], ev[1]) for ev in self.events if ev[2] == c],
                     "gt": [(ev[0], ev[1]) for ev in cands if ev[2] == c] if self.cands_all else None}
                    for c in used]
            self.strip_cache = timeline(self.width, p.duration, rows, row_h=16)
        return self.strip_cache

    def render(self) -> np.ndarray:
        p = self.player
        h = int(round(p.frame.shape[0] * self.width / p.frame.shape[1]))
        img = cv2.resize(p.frame, (self.width, h), interpolation=cv2.INTER_AREA)
        cur = OFFICIAL_CLASSES[self.cls]
        text(img, f"{self.name}  [{self.vi + 1}/{len(self.videos)}]   {p.t:7.2f} / {p.duration:.1f}s   "
                  f"frame {p.idx}   x{self.speed:g}{'' if self.playing else '  (paused)'}", (8, 20))
        text(img, f"class [{self.cls + 1}] {cur}", (8, 44), (255, 255, 255), 0.6, 1, CLASS_COLORS[cur])
        if self.pending is not None:
            text(img, f"start marked at {self.pending:.2f}s - press e at the end", (8, 70), (0, 255, 255))
        under = [ev for ev in self.events if ev[0] <= p.t <= ev[1]]
        for k, ev in enumerate(under):
            text(img, f"label {ev[2]} {ev[0]:.2f}-{ev[1]:.2f}", (8, h - 12 - 22 * k), (255, 255, 255), 0.5, 1,
                 CLASS_COLORS.get(ev[2], (90, 90, 90)))
        bar = self.strip().copy()
        if self.pending is not None:
            draw_playhead(bar, self.pending, p.duration, (0, 255, 255))
        draw_playhead(bar, p.t, p.duration)
        self.strip_y = h
        legend = np.full((18, self.width, 3), 24, np.uint8)
        text(legend, "timeline: filled = your labels" + (", white = candidates" if self.cands_all else "")
             + "   (h: keys)", (6, 13), (180, 180, 180), 0.4, 1, None)
        return np.vstack([img, bar, legend])

    # ------------------------------------------------------------- input
    def on_mouse(self, event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.strip_y <= y < self.strip_y + self.strip().shape[0]:
            self.seek_to = timeline_time(x, self.width, self.player.duration)

    def handle(self, k: int) -> bool:
        """Returns False to quit."""
        p = self.player
        fps = p.fps
        ch = chr(k) if 32 <= k < 127 else ""
        if k in LEFT:
            p.seek(p.idx - int(fps))
        elif k in RIGHT:
            p.seek(p.idx + int(fps))
        elif k in UP:
            p.seek(p.idx - 5)
        elif k in DOWN:
            p.seek(p.idx + 5)
        elif ch == ",":
            p.seek(p.idx - 1)
        elif ch == ".":
            p.seek(p.idx + 1)
        elif ch == "[":
            p.seek(p.idx - int(10 * fps))
        elif ch == "]":
            p.seek(p.idx + int(10 * fps))
        elif ch == " ":
            self.playing = not self.playing
        elif ch == "+" or ch == "=":
            self.speed = min(8.0, self.speed * 2)
        elif ch == "-":
            self.speed = max(0.125, self.speed / 2)
        elif ch.isdigit():
            self.cls = (int(ch) - 1) % 10
            self.strip_cache = None
        elif ch and ch in SHIFT_DIGITS:
            self.cls = 10 + SHIFT_DIGITS.index(ch)
            self.strip_cache = None
        elif ch in ("c", "C"):
            self.cls = (self.cls + (1 if ch == "c" else -1)) % len(OFFICIAL_CLASSES)
            self.strip_cache = None
        elif ch == "s":
            self.pending = p.t
        elif ch == "e":
            if self.pending is None:
                print("  mark a start first (s)")
            else:
                self.add(self.pending, p.t, OFFICIAL_CLASSES[self.cls])
                self.pending = None
        elif k == 27:
            self.pending = None
        elif ch == "x":
            self.delete_at(p.t)
        elif ch == "z":
            self.undo()
        elif ch == "j":
            self.jump(True)
        elif ch == "k":
            self.jump(False)
        elif ch == "y":
            self.accept()
        elif ch == "g":
            try:
                p.seek(int(float(input("  go to second: ")) * fps))
            except (ValueError, EOFError):
                pass
        elif ch in ("n", "b"):
            self.open((self.vi + (1 if ch == "n" else -1)) % len(self.videos))
        elif ch == "v":
            self.entry()
            self.save()
            print(f"  {self.name} marked as fully labelled ({len(self.events)} label(s))")
        elif ch == "h":
            print(__doc__)
        elif ch == "q":
            return False
        return True

    def open(self, vi: int) -> None:
        if self.player:
            self.player.close()
        self.vi, self.pending, self.playing, self.strip_cache = vi, None, False, None
        self.player = Player(self.videos[vi])
        print(f"[{self.name}] {self.player.duration:.1f}s, {len(self.events)} label(s), "
              f"{len(self.candidates())} candidate(s)")

    def run(self) -> None:
        win = "label_events"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        self.open(0)
        print("keys: h for help")
        while True:
            if self.seek_to is not None:
                self.player.seek(int(self.seek_to * self.player.fps))
                self.seek_to = None
            if self.playing:
                # at >1x skip frames instead of trying to decode faster than real time
                for _ in range(max(1, int(self.speed))):
                    if not self.player.step():
                        self.playing = False
                        break
            cv2.imshow(win, self.render())
            delay = max(1, int(1000 / (self.player.fps * min(self.speed, 1.0)))) if self.playing else 30
            k = cv2.waitKeyEx(delay)
            if k != -1 and not self.handle(k):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        self.player.close()
        cv2.destroyAllWindows()
        print(f"labels in {self.labels_path}")


def load_candidates(path: Path | None) -> dict:
    """predictions.json ({"videos": {...}}) or a ground-truth-shaped file ({video: {"events"}})."""
    if path is None:
        return {}
    d = json.loads(path.read_text())
    return d.get("videos", d) if isinstance(d, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="folder of .mp4 or one file")
    ap.add_argument("--labels", type=Path, default=ROOT / "my_labels.json", help="labels file (read + written)")
    ap.add_argument("--candidates", type=Path, help="predictions JSON to jump between / accept")
    ap.add_argument("--width", type=int, default=1280, help="display width")
    args = ap.parse_args()

    videos = list_videos(args.videos)
    if not videos:
        print(f"no .mp4 files in {args.videos}", file=sys.stderr)
        return 2
    Labeller(videos, args.labels, load_candidates(args.candidates), args.width).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
