"""Load SAM 3 masks saved earlier, so tracking and fusion can reuse them instead of running SAM 3.

Reads a folder of <stem>__masks.npz files, matched to frames by file stem. It accepts the output of
tasks/openvocab_adapter/sam3_infer.py (finished runs have manifest.json) and of milestone 2 (report.json).
Anything that would silently change the result is an error: a frame or prompt that is missing, a different image
than the one segmented, masks of another size, or masks saved with a stricter score threshold.
"""

import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image


class SavedMasks:
    def __init__(self, masks_dir, prompts, image_paths, min_score):
        self.dir = Path(masks_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"[masks] no such folder: {self.dir}")
        self.prompts = list(prompts)
        self.images = {Path(p).stem: Path(p) for p in image_paths}
        self.has_manifest = (self.dir / "manifest.json").is_file()
        finished = self._finished_frames(min_score)

        digest = hashlib.sha256()
        for stem in sorted(self.images):
            if stem not in finished:
                raise ValueError(f"[masks] frame {stem} was not segmented in {self.dir}")
            expected_sha = finished[stem]
            if expected_sha and hashlib.sha256(self.images[stem].read_bytes()).hexdigest() != expected_sha:
                raise ValueError(f"[masks] {self.images[stem]} is not the image that was segmented")
            path = self._file(stem)
            if path.is_file():
                with np.load(path) as z:
                    saved = {str(p) for p in z["prompts"]}
                missing = [p for p in self.prompts if p not in saved]
                if missing:
                    raise ValueError(f"[masks] {path.name} has no prompt {missing}; it has {sorted(saved)}")
                digest.update(f"{stem}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
            elif self.has_manifest:
                raise FileNotFoundError(f"[masks] {path} is listed in manifest.json but missing")
            else:  # milestone 2 writes no file for a frame without detections
                digest.update(f"{stem}:empty\n".encode())
        self.fingerprint = digest.hexdigest()
        self._cached = (None, None)

    def _file(self, stem):
        return self.dir / f"{stem}__masks.npz"

    def _finished_frames(self, min_score):
        """Frames the saved run completed: stem -> image SHA-256, or None if it recorded none."""
        if self.has_manifest:
            manifest = json.loads((self.dir / "manifest.json").read_text())
            if manifest["min_score"] > min_score:
                raise ValueError(f"[masks] saved with min_score {manifest['min_score']}, "
                                 f"but this run needs detections from {min_score}")
            return {f["stem"]: f["image_sha256"] for f in manifest["frames"]}
        report = self.dir / "report.json"
        if report.is_file():
            results = json.loads(report.read_text())["results"]
            return {PureWindowsPath(r["image"]).stem: None for r in results}
        raise FileNotFoundError(f"[masks] {self.dir} has no manifest.json or report.json: the run did not finish")

    def _frame(self, stem):
        if self._cached[0] != stem:
            frame = None
            if self._file(stem).is_file():
                with np.load(self._file(stem)) as z:
                    frame = {"masks": z["masks"].astype(bool), "scores": z["scores"], "prompt_ids": z["prompt_ids"],
                             "prompts": [str(p) for p in z["prompts"]]}
                with Image.open(self.images[stem]) as image:
                    width, height = image.size
                if frame["masks"].shape[1:] != (height, width):
                    raise ValueError(f"[masks] {stem}: masks are {frame['masks'].shape[1:]}, "
                                     f"the image is {(height, width)}")
            self._cached = (stem, frame)
        return self._cached[1]

    def get(self, image_path, prompt):
        """(masks [K,H,W] bool, scores [K]) of one prompt in one frame, at the image's resolution."""
        if prompt not in self.prompts:
            raise ValueError(f"[masks] prompt {prompt!r} was not requested")
        stem = Path(image_path).stem
        frame = self._frame(stem)
        if frame is None:
            with Image.open(self.images[stem]) as image:
                width, height = image.size
            return np.zeros((0, height, width), bool), np.zeros(0)
        keep = frame["prompt_ids"] == frame["prompts"].index(prompt)
        return frame["masks"][keep], frame["scores"][keep]

    def describe(self):
        """What to record with the results, so that two runs can prove they used identical masks."""
        return {"dir": str(self.dir), "fingerprint": self.fingerprint, "frames": len(self.images),
                "prompts": self.prompts}
