"""Unpack VSI-590K's video archives and check that every video the training loader may use decodes.

  python -m mvp.train.prep_vsi590k extract scannetppv2 procthor      # data/raw/VSI-590K/media_tar/<src>.tar.gz
  python -m mvp.train.prep_vsi590k validate --workers 16             # writes data/train_media/vsi590k/manifest.json

extract: each archive holds one top-level <src>/ directory; it is unpacked into data/train_media/vsi590k/, keeping
files that are already there (the appearance-order subsets of ScanNet++ v2 and ProcTHOR were unpacked for v4).
validate: every video of the VSI-590K annotations (video rows only) is opened with decord and 8 uniform frames are
decoded, as the training loader does. The manifest maps each media path (relative to data/train_media, as the samples
store it) to {"ok", "frames", "hw"} or {"ok": false, "error"}; data.load_vsi590k trains only on ok videos. ScanNet
videos are the VLM-3R copies under data/train_media/scannet/videos.
"""
import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mvp.paths import DATA, TRAIN_MEDIA

TARS = DATA / "raw" / "VSI-590K" / "media_tar"
OUT = TRAIN_MEDIA / "vsi590k"
MANIFEST = OUT / "manifest.json"
ANN = DATA / "raw" / "VSI-590K" / "vsi_590k.jsonl"


def media_path(video):
    """VSI-590K annotation path -> path under data/train_media (the samples' "video" field)."""
    src, rest = video.split("/", 1)
    return f"scannet/videos/{rest}" if src == "scannet" else f"vsi590k/{video}"


def check(path):
    from mvp.fusion.inputs import uniform_frames
    try:
        frames, idx = uniform_frames(TRAIN_MEDIA / path, 8)
        from decord import VideoReader, cpu
        n = len(VideoReader(str(TRAIN_MEDIA / path), ctx=cpu(0)))
        return path, {"ok": True, "frames": n, "hw": [frames[0].height, frames[0].width]}
    except Exception as e:  # missing file, broken container, decord EOF errors
        return path, {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("sources", nargs="+")
    va = sub.add_parser("validate")
    va.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.cmd == "extract":
        OUT.mkdir(parents=True, exist_ok=True)
        for src in args.sources:
            tar = TARS / f"{src}.tar.gz"
            print(f"extracting {tar} -> {OUT}", flush=True)
            subprocess.run(["tar", "xzf", str(tar), "-C", str(OUT), "--skip-old-files"], check=True)
            print(f"{src}: {sum(1 for _ in (OUT / src).rglob('*.mp4'))} mp4 files", flush=True)
        return
    videos = sorted({media_path(r["video"]) for r in map(json.loads, open(ANN)) if "video" in r})
    print(f"validating {len(videos)} videos with {args.workers} workers", flush=True)
    res = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (p, r) in enumerate(pool.map(check, videos, chunksize=8)):
            res[p] = r
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(videos)} checked, {sum(not v['ok'] for v in res.values())} failed", flush=True)
    MANIFEST.write_text(json.dumps(res, indent=0))
    by_src = {}
    for p, r in res.items():
        s = p.split("/")[1] if p.startswith("vsi590k/") else "scannet"
        by_src.setdefault(s, [0, 0])[0 if r["ok"] else 1] += 1
    print("ok / failed per source:", by_src, flush=True)
    bad = [(p, r["error"]) for p, r in res.items() if not r["ok"]]
    print(f"{len(bad)} failed; first 10:", *bad[:10], sep="\n  ")


if __name__ == "__main__":
    main()
