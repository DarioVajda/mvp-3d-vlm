"""Transcode the VSI-Bench videos linked from README.md to VP8 .webm (data/previews/<dataset>/<scene>.webm).

The original videos are H.264, which VS Code's built-in media preview cannot play; VP8 webm plays there.

  slurm/submit.sh -J previews -t 00:30:00 -- python src/make_previews.py
  ... python src/make_previews.py --videos scannetpp/09c1414f1b        # specific videos instead
"""
import argparse
import re
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[1]


def transcode(src, dst, bitrate=1_500_000, max_frames=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        istream = inp.streams.video[0]
        ostream = out.add_stream("libvpx", rate=istream.average_rate)
        ostream.width, ostream.height, ostream.pix_fmt = istream.width, istream.height, "yuv420p"
        ostream.bit_rate = bitrate
        ostream.options = {"deadline": "realtime", "cpu-used": "8"}
        ostream.codec_context.time_base = 1 / istream.average_rate
        for i, frame in enumerate(inp.decode(istream)):
            if max_frames and i >= max_frames:
                break
            frame = frame.reformat(format="yuv420p")
            frame.pts, frame.time_base = i, ostream.codec_context.time_base
            for packet in ostream.encode(frame):
                out.mux(packet)
        for packet in ostream.encode():
            out.mux(packet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", help="<dataset>/<scene> (default: all videos linked from README.md)")
    args = ap.parse_args()
    readme = (ROOT / "README.md").read_text()
    videos = args.videos or sorted(set(re.findall(r"\(data/raw/VSI-Bench/(\w+/\w+)\.mp4\)", readme)))
    for v in videos:
        dst = ROOT / "data" / "previews" / f"{v}.webm"
        src = ROOT / "data" / "raw" / "VSI-Bench" / f"{v}.mp4"
        if not dst.exists():
            transcode(src, dst)
        with av.open(str(src)) as a, av.open(str(dst)) as b:
            print(f"{v}: {dst.stat().st_size / 1e6:.1f} MB, {b.duration / 1e6:.1f}s (original {a.duration / 1e6:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
