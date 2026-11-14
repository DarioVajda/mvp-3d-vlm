"""Read a zip archive that was split into byte-level parts (`cat part_* > all.zip` style) without joining them.

  python -m mvp.train.concat_zip list data/raw/PhysGame-40K/PhysInstruct_part_*
  python -m mvp.train.concat_zip extract --to data/train_media/physgame data/raw/PhysGame-40K/PhysInstruct_part_*

The parts are opened as one seekable file, so zipfile reads the central directory at the end and each member in place.
"""
import argparse
import bisect
import io
import os
import zipfile
from pathlib import Path


class ConcatFile(io.RawIOBase):
    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        sizes = [os.path.getsize(p) for p in paths]
        self.starts = [sum(sizes[:i]) for i in range(len(sizes))]
        self.size, self.pos = sum(sizes), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        i = bisect.bisect_right(self.starts, self.pos) - 1
        f = self.files[i]
        f.seek(self.pos - self.starts[i])
        n = f.readinto(memoryview(b)[:min(len(b), (self.starts[i + 1] if i + 1 < len(self.files) else self.size) - self.pos)])
        self.pos += n
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "extract"])
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--to")
    args = ap.parse_args()
    z = zipfile.ZipFile(io.BufferedReader(ConcatFile(sorted(args.parts)), buffer_size=1 << 24))
    infos = z.infolist()
    if args.cmd == "list":
        print(len(infos), "members,", f"{sum(i.file_size for i in infos) / 1e9:.1f} GB uncompressed")
        for i in infos[:10]:
            print(i.filename, i.file_size, i.compress_type)
        return
    out = Path(args.to)
    done = 0
    for i in infos:
        p = out / i.filename
        if i.is_dir() or (p.exists() and p.stat().st_size == i.file_size):
            continue
        z.extract(i, out)
        done += 1
        if done % 2000 == 0:
            print(f"{done} extracted", flush=True)
    print(f"done: {done} extracted, {len(infos)} members")


if __name__ == "__main__":
    main()
