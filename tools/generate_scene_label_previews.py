"""Create one preview per CVAT scene label from src/scene.json geometry."""
from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "src" / "scene_ref.jpg"
SCENE_PATH = ROOT / "src" / "scene.json"
OUTPUT_DIR = ROOT / "eda" / "scene_label_previews"
FOOTER_HEIGHT = 140

COLORS = [
    (255, 90, 150), (170, 70, 240), (40, 220, 230), (40, 105, 250),
    (235, 130, 170), (30, 45, 220), (180, 135, 10), (210, 30, 140),
    (130, 40, 180), (50, 195, 70), (30, 150, 250), (255, 100, 30),
    (30, 235, 150), (40, 210, 240), (220, 180, 30), (35, 235, 230),
    (255, 45, 190), (235, 80, 60), (70, 175, 245), (20, 155, 220),
    (240, 75, 80), (70, 240, 60), (210, 120, 245), (40, 220, 250),
    (255, 165, 40), (160, 230, 60), (100, 100, 255), (220, 205, 35),
    (35, 210, 150), (205, 80, 235), (80, 205, 245),
    (0, 166, 255), (212, 188, 0), (238, 64, 115), (116, 210, 87),
    (178, 94, 230),
]


def collect_specs(scene: dict) -> list[tuple[str, str, list, str]]:
    specs: list[tuple[str, str, list, str]] = []

    def add(name: str, geometry: str, points: list, description: str) -> None:
        specs.append((name, geometry, points, description))

    for key in ("near", "far"):
        item = scene["carriageways"][key]
        add(f"carriageway_{key}", "polygon", item["polygon"], item["comment"])

    add("median_edge", "polyline", scene["median"]["polyline"], scene["median"]["comment"])
    stop_line = scene["stop_lines"]["near"]
    add("stop_line_near", "polyline", stop_line["line"], stop_line["comment"])

    crosswalk_descriptions = {
        "A_near": "Pedestrian crossing across the near approach.",
        "B_far": "Pedestrian crossing across the far side branch road.",
        "C_lower": "Pedestrian crossing on the lower-left side of the junction.",
    }
    for key, description in crosswalk_descriptions.items():
        name = "crosswalk_B_branch" if key == "B_far" else f"crosswalk_{key}"
        add(name, "polygon", scene["crosswalks"][key]["polygon"], description)

    for key, item in scene["signals"].items():
        name = "signal_ped_A_left" if key == "left_red_green" else f"signal_{key}"
        add(name, "rectangle", item["roi"], item.get("comment", "Traffic signal region of interest."))

    for key, item in scene["zones"].items():
        name = "zone_bus_berth_far" if key == "bus_stop_far" else f"zone_{key}"
        add(name, "polygon", item["polygon"], item.get("comment", f"Mapped {key.replace('_', ' ')} area."))

    for key, item in scene["solid_lines"].items():
        add(f"solid_line_{key}", "polyline", item["polyline"], item["comment"])
    for key, item in scene["dashed_lines"].items():
        add(f"dashed_line_{key}", "polyline", item["polyline"], item["comment"])
    for key, item in scene["curbs"].items():
        add(f"curb_{key}", "polyline", item["polyline"], item["comment"])

    for key, item in scene["no_stopping"].items():
        description = item.get("comment") or item.get("basis", "Mapped no-stopping area.")
        add(f"no_stopping_{key}", "polygon", item["polygon"], description)

    lane_names = {
        "near_curb": "lane_near_right_turn",
        "near_through_1": "lane_near_through_1",
        "near_through_2": "lane_near_through_2",
        "near_through_3": "lane_near_through_3",
        "near_through_4": "lane_near_through_4",
    }
    for key, item in scene["lanes"].items():
        description = item.get("comment") or f"Near-approach lane; permitted manoeuvres: {item.get('allowed_manoeuvres', 'unknown')}."
        add(lane_names[key], "polygon", item["polygon"], description)

    turn_path = scene["turn_paths"]["near_right"]
    add("turn_path_near_right", "polyline", turn_path["polyline"], turn_path["comment"])

    return specs


def draw_preview(base: np.ndarray, name: str, geometry: str, points: list, description: str, color: tuple[int, int, int]) -> np.ndarray:
    height, width = base.shape[:2]
    canvas = np.full((height + FOOTER_HEIGHT, width, 3), (22, 27, 35), dtype=np.uint8)
    canvas[:height, :width] = base.copy()
    scene_canvas = canvas[:height, :width]
    pts = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32) if points else np.empty((0, 2), dtype=np.int32)

    if not points:
        cv2.putText(scene_canvas, "NOT YET TRACED - DRAW IN CVAT", (300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    elif geometry == "polygon":
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        tint = np.zeros_like(scene_canvas)
        tint[:] = color
        blended = cv2.addWeighted(scene_canvas, 0.76, tint, 0.24, 0)
        scene_canvas[mask > 0] = blended[mask > 0]
        cv2.polylines(scene_canvas, [pts], True, color, 5, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(scene_canvas, (int(x), int(y)), 5, (245, 245, 245), -1, cv2.LINE_AA)
    elif geometry == "polyline":
        cv2.polylines(scene_canvas, [pts], False, color, 7, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(scene_canvas, (int(x), int(y)), 7, color, -1, cv2.LINE_AA)
            cv2.circle(scene_canvas, (int(x), int(y)), 3, (250, 250, 250), -1, cv2.LINE_AA)
    elif geometry == "rectangle":
        x1, y1, x2, y2 = np.rint(np.asarray(points, dtype=np.float32)).astype(int).tolist()
        roi = scene_canvas[y1:y2 + 1, x1:x2 + 1]
        fill = np.zeros_like(roi)
        fill[:] = color
        cv2.addWeighted(fill, 0.28, roi, 0.72, 0, roi)
        cv2.rectangle(scene_canvas, (x1, y1), (x2, y2), color, 5, cv2.LINE_AA)
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    footer_y = height
    cv2.rectangle(canvas, (0, footer_y), (width, height + FOOTER_HEIGHT), (22, 27, 35), -1)
    cv2.rectangle(canvas, (32, footer_y + 30), (60, footer_y + 58), color, -1)
    cv2.putText(canvas, name, (78, footer_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (250, 250, 250), 2, cv2.LINE_AA)
    geometry_label = {"polygon": "AREA  /  POLYGON", "polyline": "LINE  /  POLYLINE", "rectangle": "SIGNAL ROI  /  RECTANGLE"}[geometry]
    cv2.putText(canvas, geometry_label, (width - 475, footer_y + 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, color, 2, cv2.LINE_AA)
    text = description.encode("ascii", errors="replace").decode("ascii")
    max_chars = 145
    lines = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [""]
    for idx, line in enumerate(lines[:2]):
        cv2.putText(canvas, line, (32, footer_y + 94 + idx * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (205, 210, 218), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    base = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(IMAGE_PATH)
    if [base.shape[1], base.shape[0]] != scene["ref_size"]:
        raise ValueError(f"Reference image size {base.shape[1]}x{base.shape[0]} != {scene['ref_size']}")

    specs = collect_specs(scene)
    if len(specs) > len(COLORS):
        raise ValueError(f"Need {len(specs)} colors, but only {len(COLORS)} are defined")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for index, (name, geometry, points, description) in enumerate(specs):
        image = draw_preview(base, name, geometry, points, description, COLORS[index])
        output = OUTPUT_DIR / f"{name}.jpg"
        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Could not write {output}")
        cards.append(
            f'<article><h2>{html.escape(name)}</h2><p>{html.escape(description)}</p>'
            f'<a href="{html.escape(output.name)}"><img src="{html.escape(output.name)}" alt="{html.escape(name)} only"></a></article>'
        )

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WIUT Scene Labels — Single-label previews</title>
<style>
body{margin:0;background:#11151b;color:#edf1f5;font:16px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:#1b222b;padding:18px 24px;z-index:2;border-bottom:1px solid #36404c}
h1{margin:0 0 4px;font-size:22px}header p{margin:0;color:#b7c0cb}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px;padding:20px}
article{background:#1b222b;border:1px solid #36404c;border-radius:10px;padding:14px;overflow:hidden}
h2{font-size:17px;margin:0 0 4px}article p{font-size:13px;color:#b7c0cb;min-height:36px;margin:0 0 10px}
img{display:block;width:100%;height:auto;border-radius:5px;background:#080a0d}
</style></head><body><header><h1>WIUT scene labels — one label per image</h1>
<p>Geometry synced from CVAT task #2623023; coordinates use the 1920×1080 reference image. Click an image to open it full size.</p></header>
<main>""" + "\n".join(cards) + "</main></body></html>"
    (OUTPUT_DIR / "index.html").write_text(page, encoding="utf-8")
    print(f"Created {len(specs)} single-label previews in {OUTPUT_DIR}")
    print(f"Gallery: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
