"""Paths, presets and small helpers shared by the OpenVocab-4D adapter scripts (numpy only)."""
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRACK_A = REPO / "openvocab-4D-main" / "trackA"
DATA_ROOT = REPO / "tasks" / "vggt_data" / "out" / "vggt_vsibench_75"
OUT_ROOT = HERE / "out"
# Folders that may hold an export's scenes/<dataset>/<scene>/, in the order they are tried (as in hpc/run_all.sh)
DATA_ROOTS = [REPO / "tasks" / "vggt_data", DATA_ROOT, REPO / "tasks" / "vggt_data" / "out", OUT_ROOT / "export"]
DENSE_ROOT = REPO / "data" / "vggt"  # <dataset>/<scene>/depth.npy and depth_conf.npy, on the cluster only

DEFAULT_SCENE = "scannet/scene0025_01"
# Objects named in the VSI-Bench questions of scannet/scene0025_01 (questions.jsonl)
SCENE0025_PROMPTS = ["chair", "door", "fan", "keyboard", "monitor", "pillow", "sofa", "table", "telephone", "window"]

# smoke5: five nearby frames, so objects can reach the 3 frames milestone 4 needs to keep a track.
# eval32: the uniform32 subset the group's experiments already use (meta.json).
# half: every other frame (0, 2, ..., 126), half the cost of all 128.
PRESETS = {"smoke5": "window:118:5:2", "eval32": "subset:uniform32", "half": "window:0:64:2", "all": "all"}


def frames_for(presets, num_frames, meta=None):
    """The frames a set of presets needs together, as a list:... spec (what SAM 3 has to segment)."""
    needed = sorted({i for name in presets for i in select_frames(PRESETS[name], num_frames, meta)})
    return "list:" + ",".join(map(str, needed))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}  # same as the pipeline


def select_frames(spec, num_frames, meta=None):
    """Resolve a frame-selection spec to a strictly increasing list of frame indices.

    all                         every frame
    uniform:K                   np.linspace(0, T-1, K, dtype=int), the rule behind meta.json's uniform<K>
    subset:NAME                 meta.json's frame_subsets[NAME]
    window:START:COUNT[:STRIDE] START, START+STRIDE, ... (COUNT frames, default STRIDE 1)
    list:3,7,20                 explicit indices

    Frames must be increasing because the pipeline lists a folder in sorted order.
    """
    kind, _, rest = spec.partition(":")
    if kind == "all" and not rest:
        indices = list(range(num_frames))
    elif kind == "uniform":
        indices = np.linspace(0, num_frames - 1, int(rest), dtype=int).tolist()
    elif kind == "subset":
        subsets = (meta or {}).get("frame_subsets", {})
        if rest not in subsets:
            raise ValueError(f"unknown subset {rest!r}; available: {sorted(subsets)}")
        indices = [int(i) for i in subsets[rest]]
    elif kind == "window":
        parts = [int(x) for x in rest.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"expected window:START:COUNT[:STRIDE], got {spec!r}")
        start, count, stride = parts[0], parts[1], parts[2] if len(parts) == 3 else 1
        indices = [start + i * stride for i in range(count)]
    elif kind == "list":
        indices = [int(x) for x in rest.split(",") if x.strip()]
    else:
        raise ValueError(f"unknown frame selection {spec!r}")

    if not indices:
        raise ValueError(f"{spec!r} selects no frames")
    if min(indices) < 0 or max(indices) >= num_frames:
        raise ValueError(f"{spec!r} selects frames outside 0..{num_frames - 1}")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"{spec!r} must select strictly increasing frames: {indices}")
    return indices


def find_scene(scene, data_root=None):
    """(data root, "<dataset>/<scene>") of an exported scene, given "<dataset>/<scene>" or only the scene's name.

    Looks in data_root if given, otherwise in each of DATA_ROOTS. A scene name alone must be in one dataset only.
    """
    scene = str(scene).replace("\\", "/").strip("/")
    roots = [Path(data_root)] if data_root else DATA_ROOTS
    for root in roots:
        if "/" in scene:
            found = [scene] if (root / "scenes" / scene / "meta.json").is_file() else []
        else:
            found = sorted(f"{p.parent.parent.name}/{scene}" for p in (root / "scenes").glob(f"*/{scene}/meta.json"))
        if len(found) > 1:
            raise ValueError(f"scene {scene!r} is in several datasets, name one of them: {found}")
        if found:
            return root, found[0]
    raise FileNotFoundError(f"no exported scene {scene!r} (scenes/<dataset>/<scene>/meta.json) in: "
                            + ", ".join(str(r) for r in roots))


def list_images(folder):
    """Image files of a folder, in the sorted order the pipeline uses."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unproject_frame(depth, extrinsic, intrinsic):
    """World point of every pixel of one depth map: x_world = R^T (x_cam - t), pixels at integer coordinates."""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = np.asarray(depth, dtype=np.float32)
    cam = np.stack([(u - intrinsic[0, 2]) * z / intrinsic[0, 0], (v - intrinsic[1, 2]) * z / intrinsic[1, 1], z], -1)
    return (cam - extrinsic[:3, 3]) @ extrinsic[:3, :3]


def project(points, extrinsic, intrinsic):
    """Pixel coordinates [N,2] and camera depth [N] of world points [N,3]."""
    cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    uvw = cam @ intrinsic.T
    return uvw[:, :2] / uvw[:, 2:], cam[:, 2]
