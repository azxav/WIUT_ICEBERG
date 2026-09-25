"""Drawing helpers for the demo: annotated frames, H.264 output, Plotly charts.

Self-contained on purpose (only numpy / cv2 / plotly): the Space must not
depend on the offline render script. Colours match the website palette.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# One colour per event class (hex), shared with site/assets/app.js.
CLASS_COLORS: dict[str, str] = {
    "accident": "#E5484D",
    "near_miss": "#F76B15",
    "red_light": "#D6409F",
    "wrong_way": "#8E4EC6",
    "illegal_u_turn": "#3E63DD",
    "illegal_turn": "#0090FF",
    "solid_line_crossing": "#12A594",
    "stop_line": "#46A758",
    "jaywalking": "#FFC53D",
    "failure_to_yield": "#BDEE63",
    "stopped_vehicle": "#8D8D8D",
    "congestion": "#978365",
    "road_obstacle": "#A18072",
    "fire_smoke": "#FF977D",
}
CATEGORY_COLORS: dict[str, str] = {
    "vehicle": "#F2C230", "person": "#7CE2FE", "bicycle": "#B1A9FF",
    "animal": "#FFA057", "other": "#B4B4B4",
}


def pretty(label: str) -> str:
    return label.replace("_", " ")


def bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# ---------------------------------------------------------------- track lookup
@dataclass
class TrackView:
    id: int
    category: str
    box: np.ndarray    # (4,) xyxy at the requested time
    trail: np.ndarray  # (K, 2) recent foot points


class TrackIndex:
    """Fast "which tracks are visible at time t" lookup over pipeline Track objects.

    ``tracks`` are ``src.tracks.Track`` (t, box, category, foot); duck-typed so
    this module does not import the pipeline.
    """

    def __init__(self, tracks: list, tolerance: float, trail_sec: float = 2.0):
        self.tracks = [tr for tr in tracks if len(tr.t)]
        self.starts = np.array([tr.t[0] for tr in self.tracks]) if self.tracks else np.zeros(0)
        self.ends = np.array([tr.t[-1] for tr in self.tracks]) if self.tracks else np.zeros(0)
        self.tol = tolerance
        self.trail_sec = trail_sec

    def at(self, t: float) -> list[TrackView]:
        out: list[TrackView] = []
        live = np.nonzero((self.starts - self.tol <= t) & (self.ends + self.tol >= t))[0]
        for k in live:
            tr = self.tracks[k]
            i = int(np.clip(np.searchsorted(tr.t, t), 0, len(tr.t) - 1))
            if i > 0 and abs(tr.t[i - 1] - t) < abs(tr.t[i] - t):
                i -= 1
            if abs(tr.t[i] - t) > self.tol:
                continue
            foot = tr.foot if getattr(tr, "foot", None) is not None else np.column_stack(
                [(tr.box[:, 0] + tr.box[:, 2]) / 2, tr.box[:, 3]])
            j0 = int(np.searchsorted(tr.t, t - self.trail_sec))
            out.append(TrackView(int(tr.id), str(getattr(tr, "category", "other")), tr.box[i],
                                 foot[j0:i + 1]))
        return out


# ---------------------------------------------------------------- frame drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int],
           scale: float, fg: tuple[int, int, int] = (20, 20, 20)) -> None:
    th = max(1, int(round(scale * 2)))
    (w, h), base = cv2.getTextSize(text, FONT, scale, th)
    x, y = org
    y = max(y, h + base + 2)
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - base + 1), FONT, scale, fg, th, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, t: float, tracks: list[TrackView], active: list,
               risk: float | None = None, scale: float = 1.0) -> np.ndarray:
    """Draw boxes, trails, active-event banner and risk gauge on a (resized) frame.

    ``active`` are events (start, end, label, track_ids) overlapping ``t``;
    ``scale`` maps original pixel coordinates to the frame being drawn.
    """
    img = frame
    H, W = img.shape[:2]
    fs = max(0.4, W / 1600)
    involved: dict[int, str] = {}
    for ev in active:
        for tid in ev.track_ids:
            involved.setdefault(int(tid), ev.label)

    for tv in tracks:
        hot = involved.get(tv.id)
        color = bgr(CLASS_COLORS[hot]) if hot else bgr(CATEGORY_COLORS.get(tv.category, "#B4B4B4"))
        x1, y1, x2, y2 = (tv.box * scale).astype(int)
        if len(tv.trail) >= 2:
            pts = (tv.trail * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, color, max(1, int(fs * 2)), cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(fs * 3)) if hot else max(1, int(fs * 1.5)))
        tag = f"{tv.id} {pretty(hot)}" if hot else str(tv.id)
        _label(img, tag, (x1, y1 - 2), color, fs * (0.55 if hot else 0.45))

    # banner with active events
    if active:
        bar_h = int(34 * fs / 0.6)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (W, bar_h), (32, 28, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        x = 10
        for ev in sorted(active, key=lambda e: e.start):
            text = f"{pretty(ev.label)}  {ev.start:.1f}-{ev.end:.1f}s"
            (w, _), _ = cv2.getTextSize(text, FONT, fs * 0.7, 2)
            if x + w + 20 > W:
                break
            _label(img, text, (x, int(bar_h * 0.78)), bgr(CLASS_COLORS.get(ev.label, "#FFFFFF")), fs * 0.7)
            x += w + 20

    # clock + risk gauge, bottom left
    y0 = H - int(12 * fs / 0.6)
    _label(img, f"t={t:6.1f}s", (10, y0), (40, 40, 40), fs * 0.6, fg=(240, 240, 240))
    if risk is not None:
        gx, gw, gh = int(160 * fs / 0.6), int(160 * fs / 0.6), int(14 * fs / 0.6)
        cv2.rectangle(img, (gx, y0 - gh), (gx + gw, y0), (40, 40, 40), -1)
        r = float(np.clip(risk, 0, 1))
        col = (80, 200, 90) if r < 0.3 else (40, 190, 240) if r < 0.5 else (70, 70, 230)
        cv2.rectangle(img, (gx, y0 - gh), (gx + int(gw * r), y0), col, -1)
        cv2.putText(img, f"risk {r:.2f}", (gx + gw + 8, y0 - 2), FONT, fs * 0.55, (240, 240, 240),
                    max(1, int(fs * 2)), cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- video output
def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # pip wheel shipping a static ffmpeg, if installed
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoSink:
    """Write BGR frames to a browser-playable MP4 (H.264 via ffmpeg, mp4v fallback)."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = str(path)
        self.size = (width - width % 2, height - height % 2)  # yuv420p needs even dims
        exe = ffmpeg_exe()
        self.proc = self.writer = None
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps:.3f}", "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", self.path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)

    @property
    def h264(self) -> bool:
        return self.proc is not None

    def write(self, frame: np.ndarray) -> None:
        frame = frame[: self.size[1], : self.size[0]]
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode(errors="ignore")
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
        elif self.writer is not None:
            self.writer.release()


# ---------------------------------------------------------------- charts
def timeline_figure(events: list, duration: float, classes: list[str]):
    """Plotly Gantt-style timeline: one row per class, one bar per event."""
    import plotly.graph_objects as go

    fig = go.Figure()
    rows = [pretty(c) for c in classes]
    for c in classes:
        evs = [e for e in events if e.label == c]
        if not evs:
            continue
        fig.add_trace(go.Bar(
            y=[pretty(c)] * len(evs), x=[e.end - e.start for e in evs], base=[e.start for e in evs],
            orientation="h", marker=dict(color=CLASS_COLORS.get(c, "#888"), line=dict(width=0)),
            name=pretty(c), width=0.6,
            customdata=[[e.start, e.end, ", ".join(map(str, e.track_ids)) or "-"] for e in evs],
            hovertemplate="<b>%{y}</b><br>%{customdata[0]:.1f}-%{customdata[1]:.1f} s"
                          "<br>tracks: %{customdata[2]}<extra></extra>",
        ))
    fig.update_layout(
        barmode="overlay", showlegend=False, height=60 + 26 * len(rows),
        margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="time (s)", range=[0, max(duration, 1.0)], gridcolor="rgba(128,128,128,.2)"),
        yaxis=dict(categoryorder="array", categoryarray=rows[::-1], type="category",
                   gridcolor="rgba(128,128,128,.25)", griddash="dash"),
    )
    fig.update_yaxes(tickvals=rows, ticktext=rows)
    return fig


def risk_figure(times: np.ndarray, scores: np.ndarray, events: list, theta: float = 0.5):
    """Risk curve with the alarm threshold and detected accidents / near misses shaded."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for e in events:
        if e.label in ("accident", "near_miss"):
            fig.add_vrect(x0=e.start, x1=e.end, fillcolor=CLASS_COLORS[e.label], opacity=0.18,
                          line_width=0, annotation_text=pretty(e.label), annotation_position="top left")
    fig.add_trace(go.Scatter(x=times, y=scores, mode="lines", line=dict(color="#F2C230", width=2),
                             fill="tozeroy", fillcolor="rgba(242,194,48,.12)", name="risk",
                             hovertemplate="%{x:.1f} s: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=theta, line=dict(color="#E5484D", dash="dash", width=1),
                  annotation_text="alarm threshold", annotation_position="bottom right")
    fig.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=40),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="time (s)", gridcolor="rgba(128,128,128,.2)"),
                      yaxis=dict(title="P(accident within 5 s)", range=[0, 1.02],
                                 gridcolor="rgba(128,128,128,.2)"))
    return fig
