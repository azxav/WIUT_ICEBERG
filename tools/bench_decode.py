"""Decode speed test for 4K samples: frames per second for each method."""
import sys
import time

import cv2

PATH = sys.argv[1] if len(sys.argv) > 1 else "samples/C3905.MP4"
N = 600


def bench(name, fn):
    t = time.perf_counter()
    n = fn()
    dt = time.perf_counter() - t
    print(f"{name:32s} {n / dt:7.1f} fps  ({n} frames, {dt:.1f} s)")


def cv_read(hw=None):
    def run():
        args = [] if hw is None else [cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, hw]]
        cap = cv2.VideoCapture(PATH, *args)
        if hw is not None:
            print("  hw accel prop:", cap.get(cv2.CAP_PROP_HW_ACCELERATION))
        n = 0
        while n < N and cap.read()[0]:
            n += 1
        return n
    return run


def cv_grab(stride):
    def run():
        cap = cv2.VideoCapture(PATH)
        n = 0
        while n < N:
            if not cap.grab():
                break
            if n % stride == 0:
                cap.retrieve()
            n += 1
        return n
    return run


def pyav(stride):
    def run():
        import av
        with av.open(PATH) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            n = 0
            for f in c.decode(s):
                if n % stride == 0:
                    f.to_ndarray(format="bgr24")
                n += 1
                if n >= N:
                    break
        return n
    return run


if __name__ == "__main__":
    print("cv2 threads:", cv2.getNumThreads())
    bench("cv2 read (harness Part B)", cv_read())
    bench("cv2 grab, retrieve 1/3", cv_grab(3))
    bench("cv2 read, hw accel ANY", cv_read(cv2.VIDEO_ACCELERATION_ANY))
    try:
        bench("pyav threaded, bgr 1/3", pyav(3))
        bench("pyav threaded, bgr all", pyav(1))
    except ImportError:
        print("pyav not installed")
