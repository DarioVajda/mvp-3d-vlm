"""Contact sheet of 8 uniformly spaced frames for a VSI-Bench video, to look at a suspicious scene by eye (CPU only, a few frames)."""
import sys
import cv2
import numpy as np
from PIL import Image

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
OUT = f"{ROOT}/.claude/worktrees/je-xattn-proposal/reports/frames"
import os
os.makedirs(OUT, exist_ok=True)
for scene in sys.argv[1:]:
    cap = cv2.VideoCapture(f"{ROOT}/data/raw/VSI-Bench/arkitscenes/{scene}.mp4")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); tiles = []
    for i in np.linspace(0, n - 1, 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if not ok:
            continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); h, w = f.shape[:2]; s = 360 / max(h, w)
        f = cv2.resize(f, (int(w * s), int(h * s))); pad = np.zeros((360, 360, 3), np.uint8); pad[:f.shape[0], :f.shape[1]] = f
        cv2.putText(pad, str(int(i)), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2); tiles.append(pad)
    sheet = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:8], 1)], 0)
    Image.fromarray(sheet).save(f"{OUT}/{scene}.jpg", quality=85)
    print(scene, "frames", n, "size", tiles[0].shape, "->", f"{OUT}/{scene}.jpg")
