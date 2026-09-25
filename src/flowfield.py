"""Learned traffic-direction field: a heading histogram per image grid cell.

Built from vehicle tracks (of the video itself, optionally plus a prior saved
from the sample videos). It gives two things without any manual annotation:
the dominant legal direction at a point (for ``wrong_way``) and a road mask
(cells vehicles actually drive through).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_BINS = 8


class FlowField:
    def __init__(self, width: int, height: int, grid: tuple[int, int] = (48, 27)):
        self.width, self.height = width, height
        self.gx, self.gy = int(grid[0]), int(grid[1])
        self.hist = np.zeros((self.gy, self.gx, N_BINS), float)

    # ---------------------------------------------------------------- build
    def cells(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(pts, float).reshape(-1, 2)
        cx = np.clip((pts[:, 0] / self.width * self.gx).astype(int), 0, self.gx - 1)
        cy = np.clip((pts[:, 1] / self.height * self.gy).astype(int), 0, self.gy - 1)
        return cy, cx

    def add(self, pts: np.ndarray, headings: np.ndarray, weight: float = 1.0) -> None:
        ok = ~np.isnan(headings)
        if not ok.any():
            return
        cy, cx = self.cells(pts[ok])
        b = np.round((headings[ok] + np.pi) / (2 * np.pi) * N_BINS).astype(int) % N_BINS  # bins centred on multiples of 45 deg
        np.add.at(self.hist, (cy, cx, b), weight)

    @classmethod
    def from_tracks(cls, tracks, width: int, height: int, grid=(48, 27),
                    prior: "FlowField | None" = None) -> "FlowField":
        ff = cls(width, height, grid)
        if prior is not None and prior.hist.shape == ff.hist.shape:
            ff.hist += prior.hist
        for t in tracks:
            if t.category == "vehicle":
                ff.add(t.foot, t.heading)
        return ff

    # -------------------------------------------------------------- queries
    def count(self, pts: np.ndarray) -> np.ndarray:
        cy, cx = self.cells(pts)
        return self.hist[cy, cx].sum(axis=1)

    def dominant(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(angle of dominant bin centre, share of that bin incl. neighbours, count)."""
        cy, cx = self.cells(pts)
        h = self.hist[cy, cx]
        smoothed = h + 0.5 * (np.roll(h, 1, axis=1) + np.roll(h, -1, axis=1))
        b = smoothed.argmax(axis=1)
        total = h.sum(axis=1)
        idx = np.arange(len(b))
        share = (h[idx, b] + h[idx, (b + 1) % N_BINS] + h[idx, (b - 1) % N_BINS]) / np.maximum(total, 1e-9)
        angle = b / N_BINS * 2 * np.pi - np.pi
        return angle, share, total

    def opposite(self, pts: np.ndarray, headings: np.ndarray, min_count: float,
                 dominance: float, min_angle_deg: float = 135.0) -> np.ndarray:
        """True where a moving object heads against a clearly dominant flow."""
        angle, share, total = self.dominant(pts)
        diff = np.abs((headings - angle + np.pi) % (2 * np.pi) - np.pi)
        ok = (~np.isnan(headings)) & (total >= min_count) & (share >= dominance)
        return ok & (diff >= np.deg2rad(min_angle_deg))

    def road_mask(self, pts: np.ndarray, min_count: float) -> np.ndarray:
        return self.count(pts) >= min_count

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, hist=self.hist, size=np.array([self.width, self.height]))

    @classmethod
    def load(cls, path: str | Path, width: int, height: int) -> "FlowField | None":
        path = Path(path)
        if not path.exists():
            return None
        z = np.load(path)
        ff = cls(width, height, (z["hist"].shape[1], z["hist"].shape[0]))
        ff.hist = z["hist"].astype(float)
        return ff
