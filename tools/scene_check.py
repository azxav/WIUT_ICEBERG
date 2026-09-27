"""Scene map checker: zoom crops + metric table for every element on every video.

Usage:
    python tools/scene_check.py [--videos C3896,C3897] [--elements crosswalk:A_near,...]
                                [--scene src/scene.json] [--out eda/scene_check] [--quiet]

Writes eda/scene_check/<video>_<element>.jpg  (x3 zoom, element drawn 1 px)
       eda/scene_check/<video>_sheet.jpg      (contact sheet)
       eda/scene_check/report.md              (metrics + PASS/FAIL)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene import load_scene  # noqa: E402
from tools.scene_lib import VIDEOS, load_bg, paint_mask  # noqa: E402

ZOOM = 3.0
CROP_MARGIN = 60          # px around the element bbox in the original frame
CROSSWALK_BAND = 40       # px band around the polygon for the paint test
LINE_BAND = 25            # px half-width for line paint search
EDGE_SEARCH = 14          # px half-width for the curb edge search


# ----------------------------------------------------------------------------- helpers
def element_kind(key, name):
    if key == "crosswalks":
        return "polygon"
    if key == "signals":
        return "roi"
    if key in ("zones", "no_stopping", "carriageways", "lanes"):
        return "polygon"
    if key in ("stop_lines", "solid_lines", "dashed_lines", "curbs", "turn_paths"):
        return "polyline"
    if key == "median":
        return "polyline"
    return None


def element_points(item, kind):
    if kind == "polygon":
        return np.float32(item["polygon"])
    if kind == "polyline":
        return np.float32(item.get("line", item.get("polyline")))
    if kind == "roi":
        x1, y1, x2, y2 = item["roi"]
        return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    raise ValueError(kind)


def iter_elements(scene):
    """Yield (group, name, item, kind) for every element that carries geometry."""
    for key in ("crosswalks", "stop_lines", "solid_lines", "dashed_lines", "median",
                "curbs", "carriageways", "lanes", "zones", "signals", "no_stopping", "turn_paths"):
        group = scene.get(key)
        if not group:
            continue
        if key == "median":
            yield key, "median", group, "polyline"
            continue
        for name, item in group.items():
            kind = element_kind(key, name)
            if kind is None:
                continue
            if kind == "polyline" and not ({"line", "polyline"} & set(item)):
                continue
            yield key, name, item, kind


def mask_from_poly(shape, poly):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(poly))], 255)
    return m


def band_from_poly(shape, poly, width):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(poly))], 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1,) * 2)
    return cv2.dilate(m, k)


def polyline_samples(pts, step=2.0):
    """Densely sample points along a polyline (Nx2) with arc-length step."""
    pts = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total <= 0:
        return pts[:1], np.zeros(1), np.zeros((1, 2))
    n = max(int(total / step) + 1, 2)
    t = np.linspace(0, total, n)
    cum = np.concatenate([[0], np.cumsum(seg)])
    out = np.zeros((n, 2))
    tang = np.zeros((n, 2))
    for i, ti in enumerate(t):
        j = np.searchsorted(cum, ti, side="right") - 1
        j = min(max(j, 0), len(seg) - 1)
        f = (ti - cum[j]) / max(seg[j], 1e-9)
        out[i] = pts[j] + f * (pts[j + 1] - pts[j])
        d = pts[j + 1] - pts[j]
        tang[i] = d / max(np.linalg.norm(d), 1e-9)
    return out, t, tang


def stripe_paint(paint, band, min_area=30, min_elong=1.8):
    """Keep only elongated (stripe-like) components of the paint mask inside the band."""
    m = cv2.bitwise_and(paint, band)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(lab == i)
        pts = np.float32(np.c_[xs, ys])
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        if max(w, h) / max(min(w, h), 1e-6) >= min_elong:
            out[lab == i] = 255
    return out


def dist_to_mask(mask, pts):
    """Distance from each point to the nearest set pixel of mask."""
    inv = (mask == 0).astype(np.uint8)
    d = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    h, w = mask.shape[:2]
    out = np.full(len(pts), np.nan)
    for i, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            out[i] = d[yi, xi]
    return out


def boundary_samples(poly, step=3.0):
    closed = np.vstack([poly, poly[:1]])
    pts, _, _ = polyline_samples(closed, step)
    return pts


# ----------------------------------------------------------------------------- metrics
def metric_crosswalk(scene, paint, name, item, kind):
    poly = element_points(item, kind)
    h, w = paint.shape[:2]
    inside = mask_from_poly((h, w), poly)
    band = band_from_poly((h, w), poly, CROSSWALK_BAND)
    band = cv2.bitwise_and(band, cv2.bitwise_not(cv2.erode(inside, np.ones((3, 3), np.uint8))))
    sp = stripe_paint(paint, band)
    total = int((sp > 0).sum())
    if total == 0:
        return dict(status="PASS", note="no stripe paint found near polygon", n_paint=0), sp, poly
    covered = int(((sp > 0) & (inside > 0)).sum())
    pct = 100.0 * covered / total
    # how tight is the polygon: distance from boundary samples to the nearest stripe pixel
    d = dist_to_mask(sp, boundary_samples(poly, 2.0))
    d = d[~np.isnan(d)]
    mean_edge = float(d.mean()) if len(d) else 0.0
    # also: distance from stripe pixels to the polygon (how far outside do stripes stick?)
    out_px = (sp > 0) & (inside == 0)
    d_out = dist_to_mask(inside, np.argwhere(out_px)[:, ::-1].astype(np.float64)) if out_px.any() else np.array([0.0])
    ok = pct >= 98.0 and mean_edge <= 3.0
    res = dict(status="PASS" if ok else "FAIL", pct_inside=pct, mean_edge_to_paint=mean_edge,
               n_paint=total, n_outside=int(out_px.sum()),
               max_outside=float(np.nanmax(d_out)) if len(d_out) else 0.0)
    return res, sp, poly


def metric_line(scene, gray, paint, name, item, kind):
    pts = element_points(item, kind)
    h, w = paint.shape[:2]
    band = band_from_poly((h, w), pts, LINE_BAND) if len(pts) > 2 else None
    if band is None:
        band = np.zeros((h, w), np.uint8)
        cv2.polylines(band, [np.int32(pts)], False, 255, 2 * LINE_BAND)
    samples, t, tang = polyline_samples(pts, 2.0)
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    ys, xs = np.where(cv2.bitwise_and(paint, band) > 0)
    if len(xs) == 0:
        return dict(status="FAIL", note="no paint along line"), None, pts
    P = np.c_[xs, ys].astype(np.float64)
    # for each sample: perpendicular offset of nearby paint (window +-8 px along, +-LINE_BAND across)
    dists = []
    for i, (p, nvec) in enumerate(zip(samples, nrm)):
        rel = P - p
        along = rel @ tang[i]
        across = rel @ nvec
        m = (np.abs(along) <= 8.0) & (np.abs(across) <= LINE_BAND)
        if m.sum() >= 3:
            dists.append(float(np.median(across[m])))
    if not dists:
        return dict(status="FAIL", note="no paint samples along line"), None, pts
    dists = np.array(dists)
    mean_d = float(np.abs(dists).mean())
    max_d = float(np.abs(dists).max())
    # coverage: how much of the painted extent the polyline spans
    proj = (P - samples[0]) @ tang[0]
    proj -= proj.min()
    L = float(t[-1])
    span_lo, span_hi = float(proj.min()), float(proj.max())
    cover = 100.0 * (min(span_hi, L) - max(span_lo, 0.0)) / max(span_hi - span_lo, 1e-6)
    cover = max(0.0, min(100.0, cover))
    ok = mean_d <= 2.0 and max_d <= 4.0 and cover >= 95.0
    return dict(status="PASS" if ok else "FAIL", mean_dist=mean_d, max_dist=max_d,
                coverage=cover, n_paint=int(len(P))), None, pts


def metric_curb(scene, gray, name, item, kind):
    pts = element_points(item, kind)
    samples, _, tang = polyline_samples(pts, 2.0)
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    h, w = gray.shape[:2]
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    dists = []
    for p, nvec in zip(samples, nrm):
        best, bestd = 0.0, None
        for d in np.arange(-EDGE_SEARCH, EDGE_SEARCH + 0.5, 0.5):
            q = p + d * nvec
            xi, yi = int(round(q[0])), int(round(q[1]))
            if not (1 <= xi < w - 1 and 1 <= yi < h - 1):
                continue
            gx = g[yi, xi + 1] - g[yi, xi - 1]
            gy = g[yi + 1, xi] - g[yi - 1, xi]
            mag = abs(gx * nvec[0] + gy * nvec[1])
            if mag > best:
                best, bestd = mag, d
        if bestd is not None and best > 6.0:
            dists.append(bestd)
    if not dists:
        return dict(status="FAIL", note="no edge found"), None
    dists = np.array(dists)
    mean_d = float(np.abs(dists).mean())
    ok = mean_d <= 3.0
    return dict(status="PASS" if ok else "FAIL", mean_dist=mean_edge if False else mean_d,
                n_samples=int(len(dists)), signed_mean=float(dists.mean())), None


# ----------------------------------------------------------------------------- drawing
def draw_element(canvas, item, kind, offset, zoom, color=(0, 255, 0)):
    def T(p):
        return (int(round((p[0] - offset[0]) * zoom)), int(round((p[1] - offset[1]) * zoom)))
    if kind == "roi":
        pts = element_points(item, kind)
        cv2.rectangle(canvas, T(pts[0]), T(pts[2]), color, 1)
        return
    pts = element_points(item, kind)
    closed = kind == "polygon"
    xy = np.array([T(p) for p in pts], np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [xy], closed, color, 1, cv2.LINE_AA)
    if kind == "polyline":
        for p in pts:
            cv2.circle(canvas, T(p), 2, (0, 200, 255), 1)


def element_bbox(item, kind, shape):
    pts = element_points(item, kind)
    h, w = shape[:2]
    x1 = int(max(0, np.floor(pts[:, 0].min() - CROP_MARGIN)))
    y1 = int(max(0, np.floor(pts[:, 1].min() - CROP_MARGIN)))
    x2 = int(min(w, np.ceil(pts[:, 0].max() + CROP_MARGIN)))
    y2 = int(min(h, np.ceil(pts[:, 1].max() + CROP_MARGIN)))
    return x1, y1, x2, y2


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(VIDEOS))
    ap.add_argument("--elements", default="")
    ap.add_argument("--scene", default=str(ROOT / "src" / "scene.json"))
    ap.add_argument("--out", default=str(ROOT / "eda" / "scene_check"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scene_file = Path(args.scene)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    want = set()
    if args.elements:
        for e in args.elements.split(","):
            grp, _, nm = e.strip().partition(":")
            want.add((grp, nm))

    rows = []
    for video in videos:
        bg = load_bg(video)
        scene, n_inl = load_scene(bg)
        if scene is None:
            print(f"{video}: REGISTRATION FAILED ({n_inl} inliers)")
            continue
        gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        paint = paint_mask(bg)
        tiles = []
        for group, name, item, kind in iter_elements(scene):
            if want and (group, name) not in want:
                continue
            tag = f"{group}_{name}"
            if group == "crosswalks":
                res, extra, pts = metric_crosswalk(scene, paint, name, item, kind)
            elif group in ("stop_lines", "solid_lines", "dashed_lines"):
                res, extra, pts = metric_line(scene, gray, paint, name, item, kind)
            elif group == "curbs" or group == "median":
                res, extra = metric_curb(scene, gray, name, item, kind)
                pts = element_points(item, kind)
            else:
                res, extra, pts = dict(status="INFO", note="visual only"), None, element_points(item, kind)
            res["video"], res["element"], res["group"], res["name"] = video, tag, group, name
            rows.append(res)

            x1, y1, x2, y2 = element_bbox(item, kind, bg.shape)
            crop = bg[y1:y2, x1:x2].copy()
            crop = cv2.convertScaleAbs(crop, alpha=1.35, beta=10)
            big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_LANCZOS4)
            draw_element(big, item, kind, (x1, y1), ZOOM)
            if extra is not None:
                vis = np.zeros_like(big)
                sub = extra[y1:y2, x1:x2]
                sub = cv2.resize(sub, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
                vis[sub > 0] = (0, 0, 255)
                big = cv2.addWeighted(big, 0.75, vis, 0.55, 0)
            cv2.putText(big, f"{video} {tag} {res['status']}", (6, 20), 0, 0.55, (255, 255, 255), 2)
            cv2.putText(big, f"{video} {tag} {res['status']}", (6, 20), 0, 0.55, (0, 0, 0), 1)
            p = out_dir / f"{video}_{tag}.jpg"
            cv2.imwrite(str(p), big, [cv2.IMWRITE_JPEG_QUALITY, 92])
            tiles.append((f"{tag} {res['status']}", big))
        # contact sheet
        if tiles:
            scale = 420.0 / max(t[1].shape[1] for t in tiles)
            hh = int(420.0 * max(t[1].shape[0] / t[1].shape[1] for t in tiles))
            rows_img = []
            for i in range(0, len(tiles), 3):
                chunk = tiles[i:i + 3]
                ims = []
                for label, im in chunk:
                    im2 = cv2.resize(im, (420, hh), interpolation=cv2.INTER_AREA)
                    cv2.putText(im2, label, (6, im2.shape[0] - 8), 0, 0.5, (255, 255, 255), 2)
                    cv2.putText(im2, label, (6, im2.shape[0] - 8), 0, 0.5, (0, 0, 0), 1)
                    ims.append(im2)
                while len(ims) < 3:
                    ims.append(np.zeros_like(ims[0]))
                rows_img.append(np.hstack(ims))
            sheet = np.vstack(rows_img)
            cv2.imwrite(str(out_dir / f"{video}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not args.quiet:
            npass = sum(1 for r in rows if r["status"] == "PASS")
            print(f"{video}: {npass}/{len(rows)} PASS so far")

    # ---- report
    lines = ["# Scene check report", "",
             f"scene: `{scene_file}`", ""]
    videos_seen = sorted({r["video"] for r in rows})
    groups = ["crosswalks", "stop_lines", "solid_lines", "dashed_lines", "curbs", "median",
              "carriageways", "lanes", "zones", "signals", "no_stopping", "turn_paths"]
    for group in groups:
        sub = [r for r in rows if r["group"] == group]
        if not sub:
            continue
        lines += [f"## {group}", "",
                  "| element | " + " | ".join(videos_seen) + " | metrics |",
                  "|" + "---|" * (len(videos_seen) + 2)]
        names = sorted({r["name"] for r in sub})
        for name in names:
            cells, metrics = [], []
            for v in videos_seen:
                r = next((x for x in sub if x["name"] == name and x["video"] == v), None)
                cells.append(r["status"] if r else "-")
                if r and r["video"] == videos_seen[0]:
                    metrics = [f"{k}={v2:.2f}" if isinstance(v2, float) else f"{k}={v2}"
                               for k, v2 in r.items() if k not in ("status", "video", "element", "group", "name")]
            lines.append(f"| {name} | " + " | ".join(cells) + " | " + ", ".join(metrics) + " |")
        lines.append("")
    lines += ["## All element/video results", "",
              "| video | element | status | details |", "|---|---|---|---|"]
    for r in rows:
        det = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in r.items()
                        if k not in ("status", "video", "element", "group", "name"))
        lines.append(f"| {r['video']} | {r['element']} | {r['status']} | {det} |")
    fails = [r for r in rows if r["status"] == "FAIL"]
    lines += ["", f"**{len(rows) - len(fails)}/{len(rows)} PASS, {len(fails)} FAIL**", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"report: {out_dir / 'report.md'}  ({len(rows) - len(fails)}/{len(rows)} PASS)")
    for r in fails:
        print(f"  FAIL {r['video']} {r['element']}: " +
              ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in r.items() if k not in ("status", "video", "element", "group", "name")))


if __name__ == "__main__":
    main()
