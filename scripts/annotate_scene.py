"""Draw the static scene layout (configs/scene.json) on a reference frame.

    python scripts/annotate_scene.py --video samples/clip.mp4 --t 12
    python scripts/annotate_scene.py --show --out renders/scene_overlay.png

Choose an element type with a key, click points, press Enter to finish:

    c  carriageway polygon          i  intersection polygon
    w  crosswalk polygon            z  zone polygon (id asked in the console)
    l  lane polygon, Enter, then 2 clicks: travel direction (from -> to)
    s  stop line: 2 clicks, then 1 click on the side traffic goes to;
       id / light id / approach asked in the console
    d  solid line polyline          t  traffic-light ROI box (2 corner clicks;
                                       light id and roi name head|red|yellow|green asked)
    m  allowed movements (console, e.g. "A>C, B>D")   U  toggle u_turn_allowed
    Enter finish shape   right click / Backspace  drop last point   Esc  cancel shape
    u  undo last element   S  save   h  help   q  quit

Everything is stored in normalised coordinates (see src/scene.py), so the file
works for any resolution of the same camera. Saving also writes the full-size
reference frame (configs/scene_ref.jpg) for documentation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from _common import CONFIG_DIR, DEFAULT_VIDEOS, RENDERS, list_videos, read_frame
from _viz import draw_scene, text
from src.scene import Scene

MODES = {"c": "carriageway", "i": "intersection", "w": "crosswalk", "l": "lane", "s": "stop line",
         "d": "solid line", "t": "light roi", "z": "zone"}
KEY_ORDER = ["carriageway", "intersection", "crosswalks", "lanes", "stop_lines", "solid_lines",
             "lights", "zones", "allowed_movements", "u_turn_allowed"]
ENTER, ESC, BACKSPACE = (10, 13), 27, 8


def ask(prompt: str, default: str = "") -> str:
    """Console prompt (the OpenCV window freezes while it waits)."""
    try:
        ans = input(f"  {prompt} [{default}]: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def dumps_scene(d: dict) -> str:
    """JSON with number-only lists kept on one line."""
    ordered = {k: d[k] for k in KEY_ORDER if k in d}
    ordered.update({k: v for k, v in d.items() if k not in ordered})
    s = json.dumps(ordered, indent=2)
    return re.sub(r"\[[\d\s.,eE+-]*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(0)[1:-1].split(",") if x.strip()) + "]", s)


class Annotator:
    def __init__(self, frame: np.ndarray, scene: dict, out: Path, ref_out: Path, max_w: int, max_h: int):
        self.frame, self.out, self.ref_out = frame, out, ref_out
        H, W = frame.shape[:2]
        self.s = min(1.0, max_w / W, max_h / H)
        self.disp = cv2.resize(frame, (int(W * self.s), int(H * self.s)), interpolation=cv2.INTER_AREA)
        self.dw, self.dh = self.disp.shape[1], self.disp.shape[0]
        self.d = scene
        self.mode = "c"
        self.pts: list[list[float]] = []   # current shape, normalised
        self.dir_pts: list[list[float]] = []
        self.phase = "shape"               # shape | direction
        self.mouse = (0, 0)
        self.undo: list[Callable[[], None]] = []
        self.dirty = False
        self._base: np.ndarray | None = None

    # ------------------------------------------------------------ helpers
    def norm(self, x: int, y: int) -> list[float]:
        return [round(x / self.dw, 5), round(y / self.dh, 5)]

    def px(self, p) -> tuple[int, int]:
        return int(round(p[0] * self.dw)), int(round(p[1] * self.dh))

    def add(self, key: str, item) -> None:
        lst = self.d.setdefault(key, [])
        lst.append(item)
        self.undo.append(lambda: lst.remove(item))
        self.changed(f"added {key[:-1] if key.endswith('s') else key}")

    def changed(self, msg: str) -> None:
        self.dirty, self._base = True, None
        print(f"  {msg}")

    def next_id(self, key: str, prefix: str) -> str:
        return f"{prefix}{len(self.d.get(key, [])) + 1}"

    def reset_shape(self) -> None:
        self.pts, self.dir_pts, self.phase = [], [], "shape"

    # -------------------------------------------------------------- input
    def on_mouse(self, event, x, y, flags, _param) -> None:
        self.mouse = (x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.drop_point()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.click(self.norm(x, y))

    def drop_point(self) -> None:
        if self.dir_pts:
            self.dir_pts.pop()
        elif self.pts:
            self.pts.pop()
            self.phase = "shape"

    def click(self, p: list[float]) -> None:
        if self.phase == "direction":
            self.dir_pts.append(p)
            if self.mode == "l" and len(self.dir_pts) == 2:
                self.finish_lane()
            elif self.mode == "s" and len(self.dir_pts) == 1:
                self.finish_stop_line()
            return
        self.pts.append(p)
        if self.mode == "s" and len(self.pts) == 2:
            self.phase = "direction"
        elif self.mode == "t" and len(self.pts) == 2:
            self.finish_light()

    def finish(self) -> None:
        """Enter: close the current shape."""
        m, n = self.mode, len(self.pts)
        if m in "ciwzl" and n < 3:
            print("  a polygon needs at least 3 points")
            return
        if m == "d" and n < 2:
            print("  a polyline needs at least 2 points")
            return
        if m == "c":
            self.add("carriageway", self.pts)
        elif m == "i":
            self.add("intersection", self.pts)
        elif m == "w":
            self.add("crosswalks", {"id": ask("crosswalk id", self.next_id("crosswalks", "cw_")),
                                    "polygon": self.pts})
        elif m == "z":
            zid = ask("zone id", chr(ord("A") + len(self.d.get("zones", []))))
            self.add("zones", {"id": zid, "polygon": self.pts})
        elif m == "d":
            self.add("solid_lines", {"id": self.next_id("solid_lines", "solid_"), "polyline": self.pts})
        elif m == "l":
            self.phase = "direction"
            print("  lane: click two points along the legal travel direction (from -> to)")
            return
        else:
            return
        self.reset_shape()

    def finish_lane(self) -> None:
        (x0, y0), (x1, y1) = self.dir_pts
        lane = {"id": ask("lane id", self.next_id("lanes", "lane_")), "polygon": self.pts,
                "direction": [round(x1 - x0, 5), round(y1 - y0, 5)]}
        approach = ask("approach (optional, e.g. north)", "")
        if approach:
            lane["approach"] = approach
        self.add("lanes", lane)
        self.reset_shape()

    def finish_stop_line(self) -> None:
        # keep only the component normal to the line (in pixels): the legal crossing direction
        px = np.array([self.dw, self.dh], float)
        a, b = np.array(self.pts) * px
        v = np.array(self.dir_pts[0]) * px - (a + b) / 2
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        v = (v - np.dot(v, t) * t) / px
        lights = [lt["id"] for lt in self.d.get("lights", [])]
        sl = {"id": ask("stop line id", self.next_id("stop_lines", "sl_")), "line": self.pts,
              "direction": [round(float(v[0]), 5), round(float(v[1]), 5)]}
        light = ask(f"light id (known: {', '.join(lights) or 'none'}; '-' for none)", lights[0] if lights else "-")
        if light != "-":
            sl["light"] = light
        approach = ask("approach (optional)", "")
        if approach:
            sl["approach"] = approach
        self.add("stop_lines", sl)
        self.reset_shape()

    def finish_light(self) -> None:
        (x0, y0), (x1, y1) = self.pts
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        lights = self.d.setdefault("lights", [])
        lid = ask("light id", lights[-1]["id"] if lights else "tl_1")
        name = ask("roi name (head | red | yellow | green)", "head")
        light = next((lt for lt in lights if lt["id"] == lid), None)
        is_new = light is None
        if is_new:
            light = {"id": lid, "rois": {}}
            lights.append(light)
        prev = light["rois"].get(name)
        light["rois"][name] = box

        def undo() -> None:
            if is_new:
                lights.remove(light)
            elif prev is None:
                light["rois"].pop(name, None)
            else:
                light["rois"][name] = prev
        self.undo.append(undo)
        self.changed(f"light {lid}:{name}")
        self.reset_shape()

    def ask_movements(self) -> None:
        cur = ", ".join(f"{a}>{b}" for a, b in self.d.get("allowed_movements", []))
        ans = ask("allowed movements 'A>C, B>D' ('-' = unrestricted)", cur or "-")
        prev = self.d.get("allowed_movements")
        if ans == "-":
            self.d.pop("allowed_movements", None)
        else:
            self.d["allowed_movements"] = [[p.split(">")[0].strip(), p.split(">")[1].strip()]
                                           for p in ans.split(",") if ">" in p]

        def undo() -> None:
            if prev is None:
                self.d.pop("allowed_movements", None)
            else:
                self.d["allowed_movements"] = prev
        self.undo.append(undo)
        self.changed(f"allowed_movements = {self.d.get('allowed_movements')}")

    def save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(dumps_scene(self.d) + "\n")
        cv2.imwrite(str(self.ref_out), self.frame)
        self.dirty = False
        print(f"saved {self.out} and {self.ref_out}")

    # ------------------------------------------------------------- render
    def render(self) -> np.ndarray:
        if self._base is None:
            try:
                scene = Scene.from_dict(self.d, self.dw, self.dh)
                self._base = draw_scene(self.disp, scene)
            except (KeyError, ValueError, TypeError) as exc:
                print(f"  cannot draw scene: {exc}")
                self._base = self.disp.copy()
        img = self._base.copy()
        pts = [self.px(p) for p in self.pts]
        for p in pts:
            cv2.circle(img, p, 4, (0, 255, 255), -1)
        if pts:
            closed = self.mode in "ciwzl" and self.phase == "direction"
            cv2.polylines(img, [np.array(pts, np.int32)], closed, (0, 255, 255), 1, cv2.LINE_AA)
            if self.phase == "shape" and not (self.mode == "t" and len(pts) == 2):
                if self.mode == "t":
                    cv2.rectangle(img, pts[0], self.mouse, (0, 255, 0), 1)
                else:
                    cv2.line(img, pts[-1], self.mouse, (0, 200, 200), 1, cv2.LINE_AA)
        if self.phase == "direction":
            start = self.px(self.dir_pts[0]) if self.dir_pts else (
                self.px(np.mean(self.pts, axis=0)) if self.mode == "s" else None)
            if start is not None:
                cv2.arrowedLine(img, start, self.mouse, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)
        hint = "click direction" if self.phase == "direction" else "click points, Enter to finish"
        status = (f"mode [{self.mode}] {MODES[self.mode]} - {hint}   |   c i w l s d t z  m U  u undo  "
                  f"S save  h help  q quit{'   *unsaved*' if self.dirty else ''}")
        text(img, status, (8, 20), (255, 255, 255), 0.5)
        return img

    def run(self) -> None:
        win = "annotate_scene"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        print(__doc__.split("Everything")[0])
        while True:
            cv2.imshow(win, self.render())
            k = cv2.waitKey(20)
            if k < 0:
                continue
            k &= 0xFF
            ch = chr(k) if 32 <= k < 127 else ""
            if k in ENTER:
                self.finish()
            elif k == ESC:
                self.reset_shape()
            elif k == BACKSPACE:
                self.drop_point()
            elif ch in MODES:
                self.mode = ch
                self.reset_shape()
            elif ch == "u":
                if self.pts or self.dir_pts:
                    self.reset_shape()
                elif self.undo:
                    self.undo.pop()()
                    self.changed("undone")
            elif ch == "m":
                self.ask_movements()
            elif ch == "U":
                self.d["u_turn_allowed"] = not self.d.get("u_turn_allowed", False)
                self.changed(f"u_turn_allowed = {self.d['u_turn_allowed']}")
            elif ch == "S":
                self.save()
            elif ch == "h":
                print(__doc__)
            elif ch == "q":
                if self.dirty and ask("unsaved changes - save? (y/n)", "y").lower().startswith("y"):
                    self.save()
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def reference_frame(args) -> np.ndarray:
    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            sys.exit(f"cannot read image {args.image}")
        return img
    video = args.video or next(iter(list_videos(DEFAULT_VIDEOS)), None)
    if video is None:
        if (CONFIG_DIR / "scene_ref.jpg").exists():
            return cv2.imread(str(CONFIG_DIR / "scene_ref.jpg"))
        sys.exit("give --video or --image (no samples/*.mp4 and no configs/scene_ref.jpg found)")
    return read_frame(video, args.t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, help="video to take the reference frame from (default: first in samples/)")
    ap.add_argument("--t", type=float, default=0.0, help="time of the reference frame, seconds")
    ap.add_argument("--image", type=Path, help="use an image instead of a video frame")
    ap.add_argument("--scene", type=Path, default=CONFIG_DIR / "scene.json", help="scene file to edit / show")
    ap.add_argument("--ref-out", type=Path, default=CONFIG_DIR / "scene_ref.jpg", help="reference frame output")
    ap.add_argument("--max-width", type=int, default=1600)
    ap.add_argument("--max-height", type=int, default=900)
    ap.add_argument("--show", action="store_true", help="overlay the existing scene and write a PNG, no GUI")
    ap.add_argument("--out", type=Path, default=RENDERS / "scene_overlay.png", help="PNG written by --show")
    args = ap.parse_args()

    frame = reference_frame(args)
    scene = json.loads(args.scene.read_text()) if args.scene.exists() else {}
    if args.show:
        img = draw_scene(frame, Scene.from_dict(scene, frame.shape[1], frame.shape[0]))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), img)
        print(f"wrote {args.out}")
        return 0
    Annotator(frame, scene, args.scene, args.ref_out, args.max_width, args.max_height).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
