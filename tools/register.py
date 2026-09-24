"""Homography from each video background to the reference background (SIFT + RANSAC)."""
import sys

import cv2
import numpy as np

REF = sys.argv[1] if len(sys.argv) > 1 else "C3905"


def gray(v):
    return cv2.cvtColor(cv2.imread(f"eda/frames/{v}_background.jpg"), cv2.COLOR_BGR2GRAY)


def homography(src, dst):
    sift = cv2.SIFT_create(4000)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    k1, d1 = sift.detectAndCompute(clahe.apply(src), None)
    k2, d2 = sift.detectAndCompute(clahe.apply(dst), None)
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    p1 = np.float32([k1[a.queryIdx].pt for a in good])
    p2 = np.float32([k2[a.trainIdx].pt for a in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    return H, int(inl.sum()), len(good)


if __name__ == "__main__":
    ref = gray(REF)
    corners = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080], [960, 700]]).reshape(-1, 1, 2)
    for v in ["C3896", "C3897", "C3902", "C3905"]:
        H, ninl, ngood = homography(gray(v), ref)
        shift = np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2).ravel()
        print(f"{v} -> {REF}: inliers {ninl}/{ngood}, point shift px (1080p) {np.round(shift, 1)}")
