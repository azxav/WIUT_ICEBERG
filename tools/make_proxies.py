"""1080p browser-playable proxies (H.264 yuv420p, keyframe every 15 frames) with the same frame timing."""
import sys
import time
from fractions import Fraction
from pathlib import Path

import av

OUT = Path("samples/proxy")
OUT.mkdir(parents=True, exist_ok=True)
RATE = Fraction(30000, 1001)
TB = 1 / RATE

for src in sorted(Path("samples").glob("*.MP4")) if len(sys.argv) < 2 else [Path(p) for p in sys.argv[1:]]:
    dst = OUT / f"{src.stem}_1080p.mp4"
    t0 = time.time()
    with av.open(str(src)) as inp, av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
        vin = inp.streams.video[0]
        vin.thread_type = "AUTO"
        vout = out.add_stream("libx264", rate=RATE)
        vout.width, vout.height, vout.pix_fmt = 1920, 1080, "yuv420p"
        vout.codec_context.time_base = TB
        vout.options = {"preset": "veryfast", "crf": "23", "g": "15", "bf": "0"}
        for i, frame in enumerate(inp.decode(vin)):
            f = frame.reformat(width=1920, height=1080, format="yuv420p")
            f.pts, f.time_base = i, TB
            for pkt in vout.encode(f):
                out.mux(pkt)
        for pkt in vout.encode():
            out.mux(pkt)
    print(f"{dst}: {i + 1} frames, {time.time() - t0:.0f} s, {dst.stat().st_size / 2**20:.0f} MiB", flush=True)
