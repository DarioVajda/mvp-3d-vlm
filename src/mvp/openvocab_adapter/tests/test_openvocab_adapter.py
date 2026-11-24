"""CPU-only tests of the adapter, the saved-mask path and the comparison. No model is run or downloaded.

The milestone scripts are run for real, on a small synthetic scene. Three things they import are replaced by small
stand-ins written to a temporary folder: cv2, rerun, and vggt.utils.geometry (a numpy version of the same formulas).
SAM 3 is replaced by a fake that returns fixed rectangles.

  bash src/mvp/openvocab_adapter/run_tests.sh
  (or, from the repository root, with the venv's Python: python -m unittest discover -s src/mvp/openvocab_adapter/tests -v)
"""
import ast
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADAPTER))
import adapt_vggt  # noqa: E402
import camera_sheet_format_converter  # noqa: E402
import common  # noqa: E402
import compare  # noqa: E402
import question_scene_sheet  # noqa: E402
import reconstruct_pipeline  # noqa: E402
import scene_sheet_creator  # noqa: E402
import scene_sheet_report  # noqa: E402
import sheet_index  # noqa: E402
import upright  # noqa: E402

sys.path.insert(0, str(common.TRACK_A))
from saved_masks import SavedMasks  # noqa: E402

M3 = common.TRACK_A / "milestone3_fuse.py"
M4 = common.TRACK_A / "milestone4_video.py"
SAM3_INFER = ADAPTER / "sam3_infer.py"
PROMPTS = "chair,table"
T, GH, GW, P = 8, 4, 6, 14
H, W = GH * P, GW * P

CV2_STUB = """
import numpy as np
INTER_NEAREST_EXACT = 6
def resize(a, size, interpolation=None):
    w, h = size
    ys = ((np.arange(h) + 0.5) * a.shape[0] / h).astype(int)
    xs = ((np.arange(w) + 0.5) * a.shape[1] / w).astype(int)
    return a[ys][:, xs]
"""
RERUN_STUB = """
from pathlib import Path
class _Any:
    def __init__(self, *a, **k): pass
def init(*a, **k): pass
def save(path): Path(path).write_bytes(b"stub")
def log(*a, **k): pass
def disconnect(): pass
Points3D = LineStrips3D = Transform3D = Pinhole = Image = _Any
class ViewCoordinates: RDF = "RDF"
"""
GEOMETRY_STUB = """
import numpy as np
def depth_to_world_coords_points(d, ext, intr):
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    cam = np.stack([(u - intr[0, 2]) * d / intr[0, 0], (v - intr[1, 2]) * d / intr[1, 1], d], -1)
    return (cam - ext[:3, 3]) @ ext[:3, :3], cam, d > 1e-8
def unproject_depth_map_to_point_map(depth, extrinsic, intrinsic):
    return np.stack([depth_to_world_coords_points(depth[i].squeeze(-1), extrinsic[i], intrinsic[i])[0]
                     for i in range(len(depth))])
"""
SAM3_FAKE_PROCESSOR = """
import numpy as np
class Tensor:
    def __init__(self, a): self.a = np.asarray(a)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.a
    def __array__(self, dtype=None, copy=None): return self.a
RECTS = {"chair": [(8, 8, 38, 38, 0.9)], "table": [(44, 10, 80, 46, 0.8), (0, 0, 6, 6, 0.52)]}
class Sam3Processor:
    def __init__(self, model): pass
    def set_image(self, image): return image.size
    def set_text_prompt(self, prompt, state):
        w, h = state
        rects = RECTS.get(prompt, [])
        masks = np.zeros((len(rects), 1, h, w), np.float32)
        for i, (x0, y0, x1, y1, _) in enumerate(rects):
            masks[i, 0, y0:y1, x0:x1] = 1
        return {"masks": Tensor(masks), "boxes": Tensor(np.array([r[:4] for r in rects], np.float32).reshape(-1, 4)),
                "scores": Tensor(np.array([r[4] for r in rects], np.float32))}
"""
TRANSFORMERS_STUB = """
import numpy as np
RECTS = {"chair": [(8, 8, 38, 38, 0.9)], "table": [(44, 10, 80, 46, 0.8), (0, 0, 6, 6, 0.52)]}
class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def bool(self): return _T(self.a.astype(bool))
    def float(self): return _T(self.a.astype(np.float32))
    def cpu(self): return self
    def numpy(self): return self.a
class _Batch(dict):
    __getattr__ = dict.__getitem__
    def to(self, device): return self
class Sam3Processor:
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def __call__(self, images=None, text=None, return_tensors=None):
        return _Batch(pixel_values=images.size) if images is not None else _Batch(input_ids=text, attention_mask=None)
    def post_process_instance_segmentation(self, output, threshold, target_sizes):
        (h, w), rects = target_sizes[0], [r for r in RECTS.get(output[1], []) if r[4] >= threshold]
        masks = np.zeros((len(rects), h, w), bool)
        for i, (x0, y0, x1, y1, _) in enumerate(rects):
            masks[i, y0:y1, x0:x1] = True
        return [{"masks": _T(masks), "boxes": _T(np.array([r[:4] for r in rects], np.float32).reshape(-1, 4)),
                 "scores": _T(np.array([r[4] for r in rects], np.float32))}]
class Sam3Model:
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def to(self, device): return self
    def eval(self): return self
    def get_vision_features(self, pixel_values): return pixel_values
    def __call__(self, vision_embeds, input_ids, attention_mask): return (vision_embeds, input_ids)
"""
FAKE_TORCH = """
class _Context:
    def __enter__(self): return self
    def __exit__(self, *a): return False
def no_grad(): return _Context()
def autocast(*a, **k): return _Context()
bfloat16 = "bf16"
def is_tensor(x): return False
class cuda:
    is_available = staticmethod(lambda: True)
    get_device_name = staticmethod(lambda i=0: "fake gpu")
"""


def write_stubs(root):
    files = {
        "stubs/cv2.py": CV2_STUB, "stubs/rerun.py": RERUN_STUB,
        "stubs/vggt/__init__.py": "", "stubs/vggt/utils/__init__.py": "", "stubs/vggt/utils/geometry.py": GEOMETRY_STUB,
        "stubs/sam3/__init__.py": "", "stubs/sam3/model/__init__.py": "",
        "stubs/sam3/model/sam3_image_processor.py": SAM3_FAKE_PROCESSOR,
        "stubs/sam3/model_builder.py": "def build_sam3_image_model(checkpoint_path=None):\n    return object()\n",
        "gpu/torch.py": FAKE_TORCH,
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root / "stubs", root / "gpu"


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_scene(root, dataset="scannet", name="synthetic"):
    """A scene in the exact layout of the VGGT export. Returns (scene_dir, data_root, dense_dir)."""
    data_root = root / "data"
    scene = data_root / "scenes" / dataset / name
    (scene / "frames").mkdir(parents=True)
    extrinsic, intrinsic = np.zeros((T, 3, 4), np.float32), np.zeros((T, 3, 3), np.float32)
    for t in range(T):
        R, centre = rotation_y(0.01 * t), np.array([0.01 * t, 0.0, 0.0])
        extrinsic[t, :, :3], extrinsic[t, :, 3] = R, -R @ centre
        intrinsic[t] = [[80, 0, W / 2], [0, 80, H / 2], [0, 0, 1]]
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    depth = np.stack([2 + 0.05 * np.sin(x / 10) + 0.02 * np.cos(y / 7) + 0.01 * t for t in range(T)]).astype(np.float16)
    conf = (1 + rng.uniform(0, 5, (T, H, W))).astype(np.float16)

    cloud_xyz, cloud_rgb, cloud_frame, patch_xyz, patch_conf = [], [], [], [], []
    for t in range(T):
        rgb = np.stack([30 + 20 * t + x, 60 + 2 * y, np.full_like(x, 200 - 20 * t)], -1).astype(np.uint8)
        Image.fromarray(rgb).save(scene / "frames" / f"{t:03d}.jpg", quality=95)
        decoded = np.asarray(Image.open(scene / "frames" / f"{t:03d}.jpg").convert("RGB"))
        world = common.unproject_frame(depth[t].astype(np.float32), extrinsic[t], intrinsic[t])
        patch_xyz.append(np.median(world.reshape(GH, P, GW, P, 3), axis=(1, 3)))
        patch_conf.append(conf[t].astype(np.float32).reshape(GH, P, GW, P).mean(axis=(1, 3)))
        vs, us = rng.integers(0, H, 200), rng.integers(0, W, 200)
        cloud_xyz.append(world[vs, us])
        cloud_rgb.append(decoded[vs, us])
        cloud_frame.append(np.full(200, t, np.int16))

    R, t_vec = extrinsic[:, :, :3], extrinsic[:, :, 3]
    np.savez(scene / "cameras.npz", extrinsic=extrinsic, intrinsic=intrinsic,
             camera_center=np.einsum("tji,tj->ti", R, -t_vec).astype(np.float32),
             camera_forward=R[:, 2, :].astype(np.float32), frame_indices=np.arange(T, dtype=np.int32) * 10,
             timestamps_s=(np.arange(T) * 0.4).astype(np.float32))
    np.savez(scene / "patch_points.npz", xyz=np.stack(patch_xyz).astype(np.float32),
             conf=np.stack(patch_conf).astype(np.float32))
    np.savez(scene / "point_cloud.npz", xyz=np.concatenate(cloud_xyz).astype(np.float32),
             rgb=np.concatenate(cloud_rgb), frame=np.concatenate(cloud_frame))
    (scene / "meta.json").write_text(json.dumps({
        "dataset": dataset, "scene": name, "num_frames": T, "image_hw": [H, W], "patch_size": P,
        "patch_grid_hw": [GH, GW], "frame_subsets": {"uniform4": [0, 2, 4, 7]}}))

    dense = root / "dense"
    dense.mkdir()
    np.save(dense / "depth.npy", depth)
    np.save(dense / "depth_conf.npy", conf)
    np.save(dense / "extrinsic.npy", extrinsic)
    np.save(dense / "intrinsic.npy", intrinsic)
    np.save(dense / "frame_indices.npy", np.arange(T, dtype=np.int32) * 10)
    return scene, data_root, dense


def tree_digest(folder):
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        digest.update(path.as_posix().encode() + path.read_bytes())
    return digest.hexdigest()


def adapt(data_root, out, *extra):
    return adapt_vggt.main(["--scene", "scannet/synthetic", "--data-root", str(data_root), "--out", str(out), *extra])


class FrameSelection(unittest.TestCase):
    def test_uniform_matches_the_rule_in_metas(self):
        self.assertEqual(common.select_frames("uniform:32", 128), np.linspace(0, 127, 32, dtype=int).tolist())

    def test_presets_on_128_frames(self):
        self.assertEqual(common.select_frames(common.PRESETS["smoke5"], 128), [118, 120, 122, 124, 126])
        meta = {"frame_subsets": {"uniform32": [0, 4, 127]}}
        self.assertEqual(common.select_frames(common.PRESETS["eval32"], 128, meta), [0, 4, 127])
        self.assertEqual(len(common.select_frames("all", 128)), 128)
        self.assertEqual(common.select_frames(common.PRESETS["half"], 128), list(range(0, 128, 2)))

    def test_frames_sam3_needs_for_several_presets(self):
        meta = {"frame_subsets": {"uniform32": [0, 4, 127]}}
        self.assertEqual(common.frames_for(["half"], 128), "list:" + ",".join(map(str, range(0, 128, 2))))
        spec = common.frames_for(["half", "eval32"], 128, meta)
        self.assertEqual(common.select_frames(spec, 128), sorted(set(range(0, 128, 2)) | {127}))

    def test_list_and_window(self):
        self.assertEqual(common.select_frames("list:1,5,9", 10), [1, 5, 9])
        self.assertEqual(common.select_frames("window:2:3", 10), [2, 3, 4])

    def test_rejects_bad_selections(self):
        for spec in ("list:5,1", "list:1,1", "window:8:5", "list:-1,2", "nonsense", "subset:missing"):
            with self.subTest(spec), self.assertRaises(ValueError):
                common.select_frames(spec, 10, {"frame_subsets": {}})


class Adapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scene, self.data, self.dense = make_scene(self.root)
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def test_cameras_only_writes_no_depth_and_says_so(self):
        self.assertEqual(adapt(self.data, self.out, "--frames", "list:0,2,5"), 0)
        self.assertFalse((self.out / "depth.npz").exists())
        report = json.loads((self.out / "adapter_report.json").read_text())
        self.assertTrue(report["depth_npz"].startswith("missing"))
        self.assertEqual(report["source_video_frames"], [0, 20, 50])

    def test_cameras_npz_has_the_exact_pipeline_format(self):
        adapt(self.data, self.out, "--frames", "list:0,2,5")
        with np.load(self.out / "cameras.npz") as z, np.load(self.scene / "cameras.npz") as source:
            self.assertEqual(sorted(z.files), ["extrinsic", "image_paths", "intrinsic"])
            self.assertEqual((z["extrinsic"].shape, z["extrinsic"].dtype), ((3, 3, 4), np.float32))
            self.assertEqual((z["intrinsic"].shape, z["intrinsic"].dtype), ((3, 3, 3), np.float32))
            self.assertEqual(z["image_paths"].dtype.kind, "U")
            np.testing.assert_array_equal(z["extrinsic"], source["extrinsic"][[0, 2, 5]])
            self.assertEqual([Path(str(p)).name for p in z["image_paths"]], ["000.jpg", "002.jpg", "005.jpg"])

    def test_frames_are_byte_identical_and_only_the_selected_ones(self):
        adapt(self.data, self.out, "--frames", "list:1,3")
        self.assertEqual([p.name for p in common.list_images(self.out / "frames")], ["001.jpg", "003.jpg"])
        self.assertEqual(common.sha256_file(self.out / "frames/003.jpg"), common.sha256_file(self.scene / "frames/003.jpg"))

    def test_dense_depth_is_sliced_exactly(self):
        self.assertEqual(adapt(self.data, self.out, "--frames", "list:0,2,5", "--dense-dir", str(self.dense)), 0)
        source = np.load(self.dense / "depth.npy")
        with np.load(self.out / "depth.npz") as z:
            self.assertEqual(sorted(z.files), ["conf", "depth"])
            self.assertEqual((z["depth"].shape, z["depth"].dtype), ((3, H, W), np.float16))
            self.assertEqual(z["conf"].dtype, np.float16)
            np.testing.assert_array_equal(z["depth"], source[[0, 2, 5]])

    def test_patch_depth_is_the_patch_points_depth_in_every_pixel_of_the_patch(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(adapt(self.data, self.out, "--frames", "list:0,2,5", "--patch-depth"), 0)
        self.assertIn("WARNING: depth.npz is APPROXIMATED", err.getvalue())
        report = json.loads((self.out / "adapter_report.json").read_text())
        self.assertEqual((report["depth_source"], report["depth_npz"]),
                         ("patch_points", "approximated from the patch points (--patch-depth)"))
        with np.load(self.out / "depth.npz") as z, np.load(self.scene / "patch_points.npz") as patches:
            self.assertEqual((z["depth"].shape, z["depth"].dtype, z["conf"].dtype), ((3, H, W), np.float16, np.float16))
            depth = z["depth"].astype(np.float32).reshape(3, GH, P, GW, P)
            conf = z["conf"].astype(np.float32).reshape(3, GH, P, GW, P)
            self.assertTrue((depth == depth[:, :, :1, :, :1]).all() and (conf == conf[:, :, :1, :, :1]).all())
            np.testing.assert_allclose(conf[:, :, 0, :, 0], patches["conf"][[0, 2, 5]], rtol=1e-3)
        dense = np.load(self.dense / "depth.npy")[[0, 2, 5]].astype(np.float32).reshape(3, GH, P, GW, P)
        np.testing.assert_allclose(depth[:, :, 0, :, 0], np.median(dense, axis=(2, 4)), rtol=2e-3)

    def test_patch_depth_and_dense_depth_exclude_each_other(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            adapt(self.data, self.out, "--frames", "all", "--patch-depth", "--dense-dir", str(self.dense))

    def test_patch_depth_rejects_a_patch_point_behind_its_camera(self):
        with np.load(self.scene / "patch_points.npz") as z:
            patches = {k: z[k] for k in z.files}
        patches["xyz"][2, 1, 1] = [0.0, 0.0, -1.0]
        np.savez(self.scene / "patch_points.npz", **patches)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(adapt(self.data, self.out, "--frames", "list:0,2", "--patch-depth"), 1)
        self.assertIn("FAIL patch depth: every patch point is in front of its camera", log.getvalue())
        self.assertFalse(self.out.exists())

    def test_all_frames(self):
        self.assertEqual(adapt(self.data, self.out, "--frames", "all", "--dense-dir", str(self.dense)), 0)
        with np.load(self.out / "depth.npz") as z:
            self.assertEqual(z["depth"].shape, (T, H, W))

    def test_source_data_is_not_modified(self):
        before = tree_digest(self.data), tree_digest(self.dense)
        adapt(self.data, self.out, "--frames", "all", "--dense-dir", str(self.dense))
        self.assertEqual((tree_digest(self.data), tree_digest(self.dense)), before)

    def test_refuses_to_write_into_the_source(self):
        with self.assertRaises(SystemExit):
            adapt(self.data, self.scene / "adapted", "--frames", "all")

    def test_refuses_to_replace_a_run_without_overwrite(self):
        adapt(self.data, self.out, "--frames", "all")
        with self.assertRaises(SystemExit):
            adapt(self.data, self.out, "--frames", "all")
        self.assertEqual(adapt(self.data, self.out, "--frames", "list:0", "--overwrite"), 0)
        self.assertEqual(len(common.list_images(self.out / "frames")), 1)

    def assert_rejected(self, *extra):
        self.assertEqual(adapt(self.data, self.out, *extra), 1)
        self.assertFalse((self.out / "cameras.npz").exists())

    def rewrite_cameras(self, **changes):
        with np.load(self.scene / "cameras.npz") as z:
            arrays = {k: z[k] for k in z.files}
        arrays.update(changes)
        np.savez(self.scene / "cameras.npz", **arrays)

    def test_rejects_world_from_camera_poses(self):
        with np.load(self.scene / "cameras.npz") as z:
            inverted = z["extrinsic"].copy()
        R, t = inverted[:, :, :3].copy(), inverted[:, :, 3].copy()
        inverted[:, :, :3] = R.transpose(0, 2, 1)
        inverted[:, :, 3] = -np.einsum("tji,tj->ti", R, t)
        self.rewrite_cameras(extrinsic=inverted)
        self.assert_rejected("--frames", "all")

    def test_rejects_non_rotation(self):
        with np.load(self.scene / "cameras.npz") as z:
            scaled = z["extrinsic"].copy()
        scaled[3, :, :3] *= 1.2
        self.rewrite_cameras(extrinsic=scaled)
        self.assert_rejected("--frames", "all")

    def test_rejects_wrong_camera_shape(self):
        with np.load(self.scene / "cameras.npz") as z:
            self.rewrite_cameras(extrinsic=z["extrinsic"][:-1])
        self.assert_rejected("--frames", "all")

    def test_rejects_swapped_frame_images(self):
        a, b = self.scene / "frames/001.jpg", self.scene / "frames/006.jpg"
        swap = self.root / "swap.jpg"
        shutil.move(a, swap), shutil.move(b, a), shutil.move(swap, b)
        self.assert_rejected("--frames", "list:1,6")

    def test_rejects_a_missing_frame(self):
        (self.scene / "frames/004.jpg").unlink()
        self.assert_rejected("--frames", "list:0,1")

    def test_rejects_dense_frames_in_the_wrong_order(self):
        depth = np.load(self.dense / "depth.npy")
        np.save(self.dense / "depth.npy", depth[::-1])
        self.assert_rejected("--frames", "all", "--dense-dir", str(self.dense))

    def test_rejects_dense_depth_of_the_wrong_shape(self):
        np.save(self.dense / "depth.npy", np.load(self.dense / "depth.npy")[:, :-1])
        self.assert_rejected("--frames", "all", "--dense-dir", str(self.dense))

    def test_rejects_dense_depth_in_another_precision(self):
        np.save(self.dense / "depth.npy", np.load(self.dense / "depth.npy").astype(np.float32))
        self.assert_rejected("--frames", "all", "--dense-dir", str(self.dense))

    def test_rejects_a_missing_dense_file(self):
        (self.dense / "depth_conf.npy").unlink()
        self.assert_rejected("--frames", "all", "--dense-dir", str(self.dense))


@unittest.skipUnless((common.DATA_ROOT / "scenes" / common.DEFAULT_SCENE / "meta.json").is_file(), "the VGGT export is absent")
class RealScene(unittest.TestCase):
    """Validates the real export of scannet/scene0025_01, written only to a temporary folder."""

    def test_presets(self):
        scene = common.DATA_ROOT / "scenes" / common.DEFAULT_SCENE
        before = common.sha256_file(scene / "cameras.npz")
        for preset, count in (("smoke5", 5), ("eval32", 32), ("all", 128)):
            with self.subTest(preset), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "adapted"
                self.assertEqual(adapt_vggt.main(["--preset", preset, "--out", str(out)]), 0)
                with np.load(out / "cameras.npz") as z:
                    self.assertEqual(z["extrinsic"].shape, (count, 3, 4))
                self.assertEqual(len(common.list_images(out / "frames")), count)
                self.assertFalse((out / "depth.npz").exists())
        self.assertEqual(common.sha256_file(scene / "cameras.npz"), before)


class Sam3Script(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scene, _, _ = make_scene(self.root)
        self.stubs, self.gpu = write_stubs(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *extra, gpu=True):
        paths = [str(self.gpu), str(self.stubs)] if gpu else []
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
        if "--backend" not in extra:  # the stubs fake Meta's sam3 package
            extra = ("--backend", "sam3", *extra)
        return subprocess.run([sys.executable, str(SAM3_INFER), "--images", str(self.scene / "frames"), "--prompts", PROMPTS,
                               "--out", str(self.root / "masks"), *extra], env=env, capture_output=True, text=True)

    def test_transformers_backend_gives_the_same_format(self):
        # a stub transformers whose Sam3Model finds the same rectangles as the sam3 stub; masks above min_score only
        (self.stubs / "transformers.py").write_text(TRANSFORMERS_STUB)
        done = self.run_script("--backend", "transformers", "--frames", "list:0,3")
        self.assertEqual(done.returncode, 0, done.stderr)
        manifest = json.loads((self.root / "masks/manifest.json").read_text())
        self.assertEqual(manifest["backend"], "transformers")
        self.assertIn("transformers", manifest["model"])
        with np.load(self.root / "masks/003__masks.npz") as z:
            self.assertEqual(z["masks"].dtype, bool)
            self.assertEqual(z["masks"].shape[0], 3)  # chair 0.9, table 0.8 and 0.52, all above 0.5
            self.assertEqual(sorted(z["prompt_ids"].tolist()), [0, 1, 1])

    def test_dry_run_needs_no_torch_or_sam3(self):
        # a torch module that raises proves the dry run never imports it
        (self.root / "blocked").mkdir()
        (self.root / "blocked/torch.py").write_text("raise RuntimeError('torch was imported')\n")
        env = {**os.environ, "PYTHONPATH": str(self.root / "blocked")}
        done = subprocess.run([sys.executable, str(SAM3_INFER), "--images", str(self.scene / "frames"), "--prompts", PROMPTS,
                               "--out", str(self.root / "masks"), "--dry-run", "--frames", "subset:uniform4"],
                              env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("4 frames", done.stdout)
        self.assertFalse((self.root / "masks").exists())

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_stops_before_sam3_without_a_gpu(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("this machine has a GPU")
        done = self.run_script(gpu=False)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no CUDA GPU", done.stderr)
        self.assertNotIn("sam3", done.stderr.replace("[sam3]", "").replace("sam3_infer", "").replace("src/mvp/openvocab_adapter/README", ""))

    def test_output_is_the_milestone_2_format_plus_a_manifest(self):
        done = self.run_script("--frames", "list:0,3")
        self.assertEqual(done.returncode, 0, done.stderr)
        with np.load(self.root / "masks/003__masks.npz") as z:
            self.assertEqual(sorted(z.files), ["boxes", "masks", "prompt_ids", "prompts", "scores", "source"])
            self.assertEqual((z["masks"].shape, z["masks"].dtype), ((3, H, W), np.bool_))
            self.assertEqual(z["boxes"].shape, (3, 4))
            self.assertEqual([str(p) for p in z["prompts"]], ["chair", "table"])
            self.assertEqual(z["prompt_ids"].tolist(), [0, 1, 1])
            self.assertAlmostEqual(float(z["scores"][2]), 0.52, places=4)
        manifest = json.loads((self.root / "masks/manifest.json").read_text())
        self.assertEqual([f["stem"] for f in manifest["frames"]], ["000", "003"])
        self.assertEqual(manifest["frames"][1]["image_sha256"], common.sha256_file(self.scene / "frames/003.jpg"))

    def test_no_detections_still_writes_a_file(self):
        done = self.run_script("--frames", "list:0", "--prompts", "unknown thing")
        self.assertEqual(done.returncode, 0, done.stderr)
        with np.load(self.root / "masks/000__masks.npz") as z:
            self.assertEqual(z["masks"].shape, (0, H, W))

    def test_refuses_to_replace_a_finished_run(self):
        self.assertEqual(self.run_script("--frames", "list:0").returncode, 0)
        self.assertNotEqual(self.run_script("--frames", "list:0").returncode, 0)
        self.assertEqual(self.run_script("--frames", "list:0", "--overwrite").returncode, 0)


class Loader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frames = []
        for i in range(2):
            path = self.root / f"{i:03d}.png"
            Image.fromarray(np.full((10, 20, 3), i * 50, np.uint8)).save(path)
            self.frames.append(path)
        self.masks = self.root / "masks"
        self.masks.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write_frame(self, stem, prompts=("chair", "table"), hw=(10, 20)):
        masks = np.zeros((2, *hw), bool)
        masks[0, 0, 0] = True
        masks[1, 1, 1] = True
        np.savez_compressed(self.masks / f"{stem}__masks.npz", masks=masks, boxes=np.zeros((2, 4)),
                            scores=np.array([0.9, 0.7]), prompt_ids=np.array([0, 1]), prompts=np.array(prompts))

    def write_manifest(self, min_score=0.5):
        frames = [{"stem": p.stem, "image_sha256": common.sha256_file(p)} for p in self.frames]
        (self.masks / "manifest.json").write_text(json.dumps({"min_score": min_score, "frames": frames}))

    def write_both(self, **kwargs):
        for p in self.frames:
            self.write_frame(p.stem, **kwargs)
        self.write_manifest()

    def load(self, prompts=("chair", "table"), min_score=0.55):
        return SavedMasks(self.masks, list(prompts), self.frames, min_score)

    def test_prompts_are_matched_by_name_not_position(self):
        self.write_both()
        saved = self.load(prompts=("table", "chair"))
        masks, scores = saved.get(self.frames[0], "table")
        self.assertEqual(scores.tolist(), [0.7])
        self.assertTrue(masks[0, 1, 1])

    def test_missing_prompt_is_an_error(self):
        self.write_both()
        with self.assertRaisesRegex(ValueError, "no prompt"):
            self.load(prompts=("chair", "sofa"))

    def test_missing_frame_is_an_error(self):
        self.write_frame("000")
        (self.masks / "manifest.json").write_text(json.dumps({"min_score": 0.5, "frames": [
            {"stem": "000", "image_sha256": common.sha256_file(self.frames[0])}]}))
        with self.assertRaisesRegex(ValueError, "not segmented"):
            self.load()

    def test_a_different_image_is_an_error(self):
        self.write_both()
        Image.fromarray(np.full((10, 20, 3), 200, np.uint8)).save(self.frames[1])
        with self.assertRaisesRegex(ValueError, "not the image"):
            self.load()

    def test_wrong_mask_size_is_an_error(self):
        self.write_both(hw=(8, 20))
        with self.assertRaisesRegex(ValueError, "masks are"):
            self.load().get(self.frames[0], "chair")

    def test_stricter_saved_threshold_is_an_error(self):
        self.write_both()
        self.write_manifest(min_score=0.7)
        with self.assertRaisesRegex(ValueError, "min_score"):
            self.load()

    def test_unfinished_run_is_an_error(self):
        for p in self.frames:
            self.write_frame(p.stem)
        with self.assertRaisesRegex(FileNotFoundError, "did not finish"):
            self.load()

    def test_milestone_2_output_with_an_empty_frame(self):
        self.write_frame("000")
        (self.masks / "report.json").write_text(json.dumps({"results": [
            {"image": "C:\\frames\\000.png"}, {"image": "C:\\frames\\001.png"}]}))
        saved = self.load()
        self.assertEqual(saved.get(self.frames[1], "chair")[0].shape, (0, 10, 20))
        self.assertEqual(len(saved.get(self.frames[0], "chair")[1]), 1)


def sim3_copy(adapted, native, scale, rotation, translation):
    """A second reconstruction of the same scene in another world frame and scale."""
    (native).mkdir(parents=True)
    with np.load(adapted / "cameras.npz") as z:
        extrinsic, intrinsic, paths = z["extrinsic"].copy(), z["intrinsic"], z["image_paths"]
    with np.load(adapted / "depth.npz") as z:
        depth, conf = z["depth"], z["conf"]
    for k in range(len(extrinsic)):
        R, t = extrinsic[k, :, :3].astype(np.float64), extrinsic[k, :, 3].astype(np.float64)
        Rg = R @ rotation.T
        extrinsic[k, :, :3], extrinsic[k, :, 3] = Rg, scale * t - Rg @ translation
    np.savez(native / "cameras.npz", extrinsic=extrinsic, intrinsic=intrinsic, image_paths=paths)
    np.savez_compressed(native / "depth.npz", depth=(depth.astype(np.float32) * scale).astype(np.float16), conf=conf)


class Integration(unittest.TestCase):
    """The real milestone scripts, on synthetic geometry and masks from the fake SAM 3."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        _, data, dense = make_scene(cls.root)
        cls.stubs, cls.gpu = write_stubs(cls.root)
        cls.adapted, cls.masks = cls.root / "run/adapted", cls.root / "masks"
        assert adapt(data, cls.adapted, "--frames", "all", "--dense-dir", str(dense)) == 0
        cls.frames = cls.adapted / "frames"
        done = subprocess.run([sys.executable, str(SAM3_INFER), "--images", str(cls.frames), "--prompts", PROMPTS,
                               "--out", str(cls.masks), "--backend", "sam3"],  # the stubs fake Meta's sam3 package
                              env=cls.env(gpu=True), capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        cls.saved_run = cls.milestone4(cls.adapted, "--masks-dir", str(cls.masks))
        assert cls.saved_run.returncode == 0, cls.saved_run.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def env(cls, gpu=False):
        paths = [str(cls.gpu)] if gpu else []
        return {**os.environ, "PYTHONPATH": os.pathsep.join([*paths, str(cls.stubs)])}

    @classmethod
    def milestone4(cls, out, *extra, no_reid=True):
        flags = ["--no-reid"] if no_reid else []
        return subprocess.run([sys.executable, "-u", str(M4), "--images", str(cls.frames), "--out", str(out),
                               "--prompts", PROMPTS, "--step", "1", "--stages", "track,fuse", "--min-points", "20",
                               "--conf-percentile", "10", *flags, *extra],
                              env=cls.env(), capture_output=True, text=True)

    @classmethod
    def milestone3(cls, out, *extra):
        return subprocess.run([sys.executable, "-u", str(M3), "--geometry", str(cls.adapted), "--out", str(out),
                               "--prompts", PROMPTS, "--min-points", "20", "--conf-percentile", "10", *extra],
                              env=cls.env(), capture_output=True, text=True)

    def copy_geometry(self, target):
        target.mkdir(parents=True)
        for name in ("cameras.npz", "depth.npz"):
            shutil.copyfile(self.adapted / name, target / name)

    def test_milestone4_writes_every_output_from_saved_masks(self):
        for name in ("objects.ply", "objects.rrd", "labels.npz", "report.json", "tracks/chair.npz", "tracks/table.npz"):
            self.assertTrue((self.adapted / name).exists(), name)
        report = json.loads((self.adapted / "report.json").read_text())
        self.assertEqual(report["objects_per_concept"], {"chair": 1, "table": 1})  # the 0.52 detection is dropped

    def test_milestone4_records_which_masks_it_used(self):
        summary = json.loads((self.adapted / "tracks/summary.json").read_text())
        self.assertEqual(len(summary["saved_masks"]["fingerprint"]), 64)

    def test_min_track_frames_scales_linearly_with_the_frame_count(self):
        tree = ast.parse(M4.read_text(encoding="utf-8"))  # the script parses its arguments on import
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "min_track_frames")
        namespace = {}
        exec(compile(ast.Module([function], type_ignores=[]), str(M4), "exec"), namespace)
        rule = namespace["min_track_frames"]
        self.assertEqual({n: rule(n) for n in (1, 5, 8, 16, 32, 64, 128, 240)},
                         {1: 1, 5: 1, 8: 1, 16: 1, 32: 1, 64: 2, 128: 4, 240: 7})
        values = [rule(n) for n in range(1, 300)]
        self.assertEqual(values, sorted(values))

    def test_milestone4_minimum_track_frames(self):
        summary = json.loads((self.adapted / "tracks/summary.json").read_text())
        self.assertEqual(summary["min_track_frames"], 1)  # the default rule, on the synthetic scene's 8 frames
        strict = self.root / "strict"
        self.copy_geometry(strict)
        done = self.milestone4(strict, "--masks-dir", str(self.masks), "--min-track-frames", str(T + 1))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads((strict / "tracks/summary.json").read_text())["min_track_frames"], T + 1)
        self.assertEqual(json.loads((strict / "tracks/summary.json").read_text())["chair"]["objects"], 0)

    def test_saved_masks_give_the_same_tracks_as_live_sam3(self):
        live = self.root / "live"
        self.copy_geometry(live)
        done = self.milestone4(live)  # no --masks-dir: goes through the (fake) SAM 3 code path
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("SAM 3 image model loaded", done.stdout)
        for prompt in ("chair", "table"):
            with np.load(live / f"tracks/{prompt}.npz") as a, np.load(self.adapted / f"tracks/{prompt}.npz") as b:
                np.testing.assert_array_equal(a["labels"], b["labels"])
        with np.load(live / "labels.npz") as a, np.load(self.adapted / "labels.npz") as b:
            np.testing.assert_array_equal(a["xyz"], b["xyz"])
            np.testing.assert_array_equal(a["object_id"], b["object_id"])

    def test_milestone3_saved_masks_give_the_same_result_as_live_sam3(self):
        saved, live = self.root / "m3_saved", self.root / "m3_live"
        done = self.milestone3(saved, "--masks-dir", str(self.masks))
        self.assertEqual(done.returncode, 0, done.stderr)
        done = self.milestone3(live)
        self.assertEqual(done.returncode, 0, done.stderr)
        with np.load(saved / "labels.npz") as a, np.load(live / "labels.npz") as b:
            self.assertGreater(len(a["xyz"]), 0)
            np.testing.assert_array_equal(a["xyz"], b["xyz"])
            np.testing.assert_array_equal(a["concept_id"], b["concept_id"])

    def test_reid_without_a_gpu_is_refused_rather_than_changed_silently(self):
        target = self.root / "reid"
        self.copy_geometry(target)
        import torch

        if torch.cuda.is_available():
            self.skipTest("this machine has a GPU")
        done = self.milestone4(target, "--masks-dir", str(self.masks), no_reid=False)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--no-reid", done.stderr)
        self.assertFalse((target / "tracks").exists())

    def test_scene_sheet_works_on_the_adapted_result(self):
        out = self.root / "sheet"
        objects_csv, cameras_csv = scene_sheet_creator.build_scene_sheet(
            self.adapted, ["chair", "table"], include_size=True, out_dir=out, conf_percentile=10, min_points=20)
        with open(objects_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["concept"] for r in rows], ["chair", "table"])
        self.assertTrue(all(int(r["num_frames"]) == T for r in rows))
        with open(cameras_csv, newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), T)

    def test_scene_sheet_in_the_upright_frame(self):
        """The upright sheet is the camera-0 sheet moved into the scene's upright frame: the same cameras, turned and
        moved, and objects at the same distances from them. The synthetic scene is one wall facing the camera, so no
        floor plane is found: up is the cameras', with a warning, kept in upright.json."""
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            objects_csv, cameras_csv = scene_sheet_creator.build_scene_sheet(
                self.adapted, ["chair", "table"], include_size=True, out_dir=self.root / "upright_sheet",
                conf_percentile=10, min_points=20)
        self.assertIn("WARNING: no floor plane found", log.getvalue())
        frame = upright.scene_upright(self.adapted, log=lambda text: None)
        self.assertEqual(frame["method"], "cameras")
        self.assertTrue((self.adapted / "upright.json").is_file())
        R = frame["rotation"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
        base_objects, base_cameras = scene_sheet_creator.build_scene_sheet(
            self.adapted, ["chair", "table"], include_size=True, out_dir=self.root / "camera0_sheet",
            conf_percentile=10, min_points=20, axes="camera0")
        q = lambda r: camera_sheet_format_converter.quaternion_to_rotation(*(float(r[k]) for k in ("qw", "qx", "qy", "qz")))
        pos = lambda r: np.array([float(r[k]) for k in "xyz"])
        cams, base = read_csv(cameras_csv), read_csv(base_cameras)
        for a, b in zip(cams, base):
            np.testing.assert_allclose(pos(a), R @ (pos(b) - frame["origin"]), atol=1e-4)
            np.testing.assert_allclose(q(a), R @ q(b), atol=1e-4)
        self.assertAlmostEqual(float(cams[0]["x"]), 0.0, places=4)  # the origin is below the first camera
        self.assertAlmostEqual(float(cams[0]["y"]), 0.0, places=4)
        self.assertGreater(float(cams[0]["z"]), 0.0)
        forward = np.array([float(cams[0][f"forward_{k}"]) for k in "xyz"])
        self.assertAlmostEqual(forward[0], 0.0, places=4)  # the first camera faces +y, along the floor
        self.assertGreater(forward[1], 0.0)
        for a, b in zip(read_csv(objects_csv), read_csv(base_objects)):  # medians: close, not exact, after a turn
            self.assertEqual((a["concept"], a["num_frames"]), (b["concept"], b["num_frames"]))
            near = np.linalg.norm(pos(a) - pos(cams[0])), np.linalg.norm(pos(b) - pos(base[0]))
            self.assertLess(abs(near[0] - near[1]), 0.05 * near[1])

    def test_scene_sheet_without_frames_or_cameras(self):
        out = self.root / "sheet_global"
        objects_csv, cameras_csv = scene_sheet_creator.build_scene_sheet(
            self.adapted, ["chair", "table"], include_size=True, out_dir=out, conf_percentile=10, min_points=20,
            frame_columns=False, cameras=False)
        self.assertIsNone(cameras_csv)
        self.assertFalse((out / "cameras.csv").exists())
        with open(objects_csv, newline="") as f:
            self.assertEqual(next(csv.reader(f)), ["object_id", "concept", "track_id", "x", "y", "z",
                                                   "size_x", "size_y", "size_z", "size"])

    def run_native(self, name, scale, rotation, translation):
        native = self.root / name / "native"
        sim3_copy(self.adapted, native, scale, rotation, translation)
        done = self.milestone4(native, "--masks-dir", str(self.masks))
        self.assertEqual(done.returncode, 0, done.stderr)
        return native

    @staticmethod
    def build_sheets(*runs):
        for run in runs:  # compare.py aligns the two runs in the first camera's axes
            scene_sheet_creator.build_scene_sheet(run, ["chair", "table"], include_size=True, conf_percentile=10,
                                                  min_points=20, axes="camera0")

    def test_comparison_undoes_a_different_frame_and_scale(self):
        rotation, translation = rotation_y(0.7), np.array([1.0, -2.0, 0.5])
        native = self.run_native("cmp", 1.7, rotation, translation)
        self.build_sheets(self.adapted, native)
        report = compare.compare(self.adapted, native, self.masks)
        rows = {r["comparison"]: r for r in report.rows}
        self.assertFalse(report.has_fail(), [r for r in report.rows if r["status"] == "FAIL"])
        self.assertEqual(report.counts()["BLOCKED"], 0)
        self.assertEqual(rows["masks: both runs used identical masks"]["status"], "PASS")
        self.assertAlmostEqual(rows["Sim(3) alignment of the reconstructions"]["values"]["scale_adapted_to_native"], 1.7, 2)
        self.assertEqual(rows["tracking consistency"]["values"]["mean_matched_iou"], 1.0)
        self.assertLess(rows["3D object positions after alignment"]["values"]["max_error_of_scene"], 0.01)
        self.assertLess(rows["object point clouds after alignment"]["values"]["all_objects_chamfer_of_scene"], 0.01)
        self.assertLess(rows["cameras after alignment"]["values"]["rotation_error_max_deg"], 0.1)
        sheet = rows["scene sheet: objects"]["values"]
        self.assertEqual((sheet["paired_rows"], sheet["matched_share"]), (2, 1.0))
        self.assertEqual((sheet["adapted_only"], sheet["native_only"]), ([], []))
        self.assertLess(sheet["max_position_error_of_scene"], 0.01)
        self.assertEqual(sheet["same_num_frames_share"], 1.0)
        self.assertEqual(compare.verdict(report)["verdict"], "MATCHES")

    def test_scene_sheet_comparison_sees_a_moved_object(self):
        native = self.root / "moved" / "native"
        sim3_copy(self.adapted, native, 1.0, np.eye(3), np.zeros(3))
        self.assertEqual(self.milestone4(native, "--masks-dir", str(self.masks)).returncode, 0)
        self.build_sheets(self.adapted, native)
        path = native / compare.SHEET
        with open(path, newline="") as f:
            table = list(csv.reader(f))
        x = table[0].index("x")
        table[1][x] = str(float(table[1][x]) + 1.0)  # the chair moves by a third of this small scene
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(table)
        report = compare.compare(self.adapted, native, self.masks)
        sheet = {r["comparison"]: r for r in report.rows}["scene sheet: objects"]["values"]
        self.assertGreater(sheet["max_position_error_of_scene"], 0.1)
        self.assertEqual(compare.verdict(report)["verdict"], "DIFFERS")

    def test_scene_sheet_comparison_counts_objects_in_one_sheet_only(self):
        native = self.root / "extra" / "native"
        sim3_copy(self.adapted, native, 1.0, np.eye(3), np.zeros(3))
        self.assertEqual(self.milestone4(native, "--masks-dir", str(self.masks)).returncode, 0)
        self.build_sheets(self.adapted, native)
        with open(native / compare.SHEET, newline="") as f:
            table = list(csv.reader(f))
        table.append([table[1][0], "table", "99", *table[1][3:]])
        with open(native / compare.SHEET, "w", newline="") as f:
            csv.writer(f).writerows(table)
        sheet = {r["comparison"]: r for r in compare.compare(self.adapted, native, self.masks).rows}[
            "scene sheet: objects"]["values"]
        self.assertEqual(sheet["native_only"], ["table#99"])
        self.assertAlmostEqual(sheet["matched_share"], 0.8)

    def test_comparison_detects_a_real_geometric_difference(self):
        native = self.root / "bent" / "native"
        sim3_copy(self.adapted, native, 1.0, np.eye(3), np.zeros(3))
        with np.load(native / "depth.npz") as z:
            depth, conf = z["depth"].astype(np.float32), z["conf"]
        depth[:, :, 40:] *= 1.15  # one part of every frame is further away, which no Sim(3) can explain
        np.savez_compressed(native / "depth.npz", depth=depth.astype(np.float16), conf=conf)
        self.assertEqual(self.milestone4(native, "--masks-dir", str(self.masks)).returncode, 0)
        rows = {r["comparison"]: r for r in compare.compare(self.adapted, native, self.masks).rows}
        self.assertGreater(rows["3D object positions after alignment"]["values"]["max_error_of_scene"], 0.01)

    def test_comparison_fails_if_runs_used_different_masks(self):
        native = self.run_native("other", 1.0, np.eye(3), np.zeros(3))
        path = native / "tracks/summary.json"
        summary = json.loads(path.read_text())
        summary["saved_masks"]["fingerprint"] = "0" * 64
        path.write_text(json.dumps(summary))
        rows = {r["comparison"]: r for r in compare.compare(self.adapted, native, self.masks).rows}
        self.assertEqual(rows["masks: both runs used identical masks"]["status"], "FAIL")

    def test_comparison_reports_depth_maps_of_different_sizes(self):
        native = self.root / "cropped" / "native"
        sim3_copy(self.adapted, native, 1.0, np.eye(3), np.zeros(3))
        with np.load(native / "depth.npz") as z:
            np.savez_compressed(native / "depth.npz", depth=z["depth"][:, :-14], conf=z["conf"][:, :-14])
        rows = {r["comparison"]: r for r in compare.compare(self.adapted, native, self.masks).rows}
        self.assertEqual(rows["Sim(3) alignment of the reconstructions"]["status"], "FAIL")

    def test_comparison_names_what_is_missing(self):
        adapted = self.root / "partial/adapted"
        adapted.mkdir(parents=True)
        shutil.copyfile(self.adapted / "cameras.npz", adapted / "cameras.npz")
        report = compare.compare(adapted, self.root / "partial/native", None)
        rows = {r["comparison"]: r for r in report.rows}
        self.assertFalse(report.has_fail())
        self.assertEqual(rows["format: adapted/cameras.npz"]["status"], "PASS")
        positions = rows["3D object positions after alignment"]
        self.assertEqual(positions["status"], "BLOCKED")
        self.assertTrue(any("dense depth" in n for n in positions["needs"]))
        self.assertTrue(any("SAM 3 masks" in n for n in rows["masks: cover the selected frames"]["needs"]))
        self.assertTrue(any("native VGGT" in n for n in rows["Sim(3) alignment of the reconstructions"]["needs"]))
        self.assertTrue(any("scene sheets" in n for n in rows["scene sheet: objects"]["needs"]))
        self.assertEqual(compare.verdict(report)["verdict"], "INCOMPLETE")


LEGACY_COMMIT = "9be239b"  # the last commit before --association and --depth-rule existed
EXPERIMENT = common.REPO / "experiments" / "2026-11-18-scene-sheet-formats"


def file_at(commit, path):
    """A file's text at a commit of this repository, or None (no git, a copy without history)."""
    try:
        done = subprocess.run(["git", "-C", str(common.REPO), "show", f"{commit}:{path}"], capture_output=True,
                              text=True)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def write_masks(folder, frames_dir, detections, prompts):
    """Saved SAM 3 masks in sam3_infer.py's format, for the frames of frames_dir. detections: {frame index:
    [(prompt, (x0, y0, x1, y1), score)]}, each a rectangle."""
    folder.mkdir(parents=True)
    records = []
    for t, path in enumerate(common.list_images(frames_dir)):
        dets = detections.get(t, [])
        masks = np.zeros((len(dets), H, W), bool)
        for i, (_, (x0, y0, x1, y1), _) in enumerate(dets):
            masks[i, y0:y1, x0:x1] = True
        np.savez_compressed(folder / f"{path.stem}__masks.npz", masks=masks,
                            boxes=np.array([d[1] for d in dets], np.float32).reshape(-1, 4),
                            scores=np.array([d[2] for d in dets], np.float64),
                            prompt_ids=np.array([prompts.index(d[0]) for d in dets], np.int64),
                            prompts=np.array(prompts), source=str(path))
        records.append({"stem": path.stem, "file": f"{path.stem}__masks.npz", "image_sha256": common.sha256_file(path),
                        "image_hw": [H, W], "detections": {}})
    (folder / "manifest.json").write_text(json.dumps({
        "format": "openvocab4d-sam3-masks/v1", "model": "test", "backend": "sam3", "prompts": prompts,
        "min_score": 0.5, "frame_selection": "all", "frames": records}))
    return folder


# One chair seen in every frame through a window that slides 8 pixels (0.2 scene units) a frame: its visible centre
# moves farther than the centroid rule's 5% of the first frame's diagonal (about 0.13), its points overlap.
SLIDING = {t: [("chair", (2 + 8 * t, 10, 26 + 8 * t, 46), 0.9)] for t in range(T)}
TWO_CHAIRS = {t: [("chair", (2, 10, 22, 46), 0.9), ("chair", (60, 10, 80, 46), 0.85)] for t in range(T)}
# one chair segmented twice in every frame, as a whole and as a part of it
DUPLICATES = {t: [("chair", (8, 8, 38, 46), 0.9), ("chair", (14, 14, 32, 40), 0.7)] for t in range(T)}
LOW_CONF_BOX = (8, 8, 38, 46)  # where low_confidence_geometry sets the depth confidence to its minimum


class OverlapTracking(unittest.TestCase):
    """milestone4_video.py's --association overlap and --depth-rule mask against the original rules (its defaults), on
    saved masks of the synthetic scene, without re-identification (no GPU)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        _, data, dense = make_scene(cls.root)
        cls.stubs, _ = write_stubs(cls.root)
        cls.adapted = cls.root / "adapted"
        assert adapt(data, cls.adapted, "--frames", "all", "--dense-dir", str(dense)) == 0
        cls.frames = cls.adapted / "frames"
        cls.masks = {name: write_masks(cls.root / f"masks_{name}", cls.frames, dets, ["chair"])
                     for name, dets in (("sliding", SLIDING), ("two", TWO_CHAIRS), ("duplicates", DUPLICATES))}
        cls.low_conf = cls.root / "low_conf"
        cls.low_conf.mkdir()
        shutil.copyfile(cls.adapted / "cameras.npz", cls.low_conf / "cameras.npz")
        with np.load(cls.adapted / "depth.npz") as z:
            depth, conf = z["depth"], z["conf"].copy()
        x0, y0, x1, y1 = LOW_CONF_BOX
        conf[:, y0:y1, x0:x1] = 1.0  # VGGT's lowest confidence: the original rule drops every pixel of the chair
        np.savez_compressed(cls.low_conf / "depth.npz", depth=depth, conf=conf)
        cls.runs = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def track(self, masks, *extra, geometry=None, script=M4, conf_percentile=10, min_points=20):
        type(self).runs += 1
        out = self.root / f"run{self.runs}"
        out.mkdir()
        for name in ("cameras.npz", "depth.npz"):
            shutil.copyfile((geometry or self.adapted) / name, out / name)
        points = [] if min_points is None else ["--min-points", str(min_points)]
        done = subprocess.run([sys.executable, "-u", str(script), "--images", str(self.frames), "--out", str(out),
                               "--prompts", "chair", "--step", "1", "--stages", "track", "--no-reid",
                               "--masks-dir", str(masks), "--conf-percentile", str(conf_percentile), *points, *extra],
                              env={**os.environ, "PYTHONPATH": str(self.stubs)}, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return out

    @staticmethod
    def labels(out):
        with np.load(out / "tracks" / "chair.npz") as z:
            return z["labels"]

    def objects(self, out):
        return sorted(set(np.unique(self.labels(out)).tolist()) - {0})

    NEW = ("--association", "overlap", "--depth-rule", "mask")

    def test_overlap_follows_an_object_whose_visible_part_moves(self):
        self.assertGreater(len(self.objects(self.track(self.masks["sliding"]))), 2)  # the original rule splits it
        out = self.track(self.masks["sliding"], *self.NEW)
        self.assertEqual(self.objects(out), [1])
        self.assertEqual(json.loads((out / "tracks/summary.json").read_text())["chair"]["max_detections_per_track"], T)

    def test_overlap_keeps_two_objects_apart(self):
        for extra in ((), self.NEW):
            self.assertEqual(len(self.objects(self.track(self.masks["two"], *extra))), 2, extra)

    def test_two_masks_of_one_object_are_joined(self):
        self.assertEqual(len(self.objects(self.track(self.masks["duplicates"]))), 2)  # the original rule, no re-id
        out = self.track(self.masks["duplicates"], *self.NEW)
        self.assertEqual(self.objects(out), [1])
        self.assertTrue((self.labels(out)[:, 14:40, 14:32] == 1).all())  # the part's pixels now belong to the whole

    def test_a_low_confidence_mask_counts_under_the_mask_rule_only(self):
        original = self.track(self.masks["duplicates"], geometry=self.low_conf, conf_percentile=55, min_points=None)
        self.assertEqual(self.objects(original), [])  # 300 pixels of top-45% confidence: none
        out = self.track(self.masks["duplicates"], *self.NEW, geometry=self.low_conf, conf_percentile=55,
                         min_points=None)
        self.assertEqual(self.objects(out), [1])
        summary = json.loads((out / "tracks/summary.json").read_text())
        self.assertEqual(summary["depth_rule"]["min_points"], 50)  # depth_rule.MIN_POINTS["mask"]
        self.assertEqual(summary["association"]["rule"], "overlap")
        rows = read_csv(scene_sheet_creator.build_scene_sheet(
            out, ["chair"], include_size=True, out_dir=self.root / "low_conf_sheet", conf_percentile=55)[0])
        self.assertEqual([int(r["num_frames"]) for r in rows], [T])  # the sheet counts the frames by the same rule
        rows = read_csv(scene_sheet_creator.build_scene_sheet(
            out, ["chair"], include_size=True, out_dir=self.root / "low_conf_global", conf_percentile=55,
            depth_rule="global")[0])
        self.assertEqual(rows, [])

    def test_the_original_rules_give_exactly_the_tracks_of_the_code_before_them(self):
        """The defaults, and --association centroid --depth-rule global, track as milestone4_video.py did before
        either option existed (LEGACY_COMMIT), on every test scene; the sheets match the scene sheet code of then."""
        old = file_at(LEGACY_COMMIT, "openvocab-4D-main/trackA/milestone4_video.py")
        old_sheet = file_at(LEGACY_COMMIT, "src/mvp/openvocab_adapter/scene_sheet_creator.py")
        if old is None or old_sheet is None:
            self.skipTest(f"the code at {LEGACY_COMMIT} is not available (no git history)")
        folder = self.root / "legacy"
        folder.mkdir()
        (folder / "milestone4_video.py").write_text(old)
        shutil.copyfile(common.TRACK_A / "saved_masks.py", folder / "saved_masks.py")
        (folder / "scene_sheet_creator_legacy.py").write_text(old_sheet)
        spec = importlib.util.spec_from_file_location("scene_sheet_creator_legacy", folder / "scene_sheet_creator_legacy.py")
        legacy_sheet = importlib.util.module_from_spec(spec)
        path = list(sys.path)
        spec.loader.exec_module(legacy_sheet)  # it puts its own folder on sys.path, as scene_sheet_creator.py does
        sys.path[:] = path
        ignore = {"seconds", "peak_vram_gib"}
        for name, masks in self.masks.items():
            before = self.track(masks, script=folder / "milestone4_video.py")
            for extra in ((), ("--association", "centroid", "--depth-rule", "global")):
                now = self.track(masks, *extra)
                np.testing.assert_array_equal(self.labels(now), self.labels(before), err_msg=f"{name} {extra}")
                a, b = (json.loads((o / "tracks/summary.json").read_text()) for o in (now, before))
                self.assertEqual({k: v for k, v in a.items() if k not in ignore},
                                 {k: v for k, v in b.items() if k not in ignore}, f"{name} {extra}")
            sheets = []
            for build, extra in ((legacy_sheet.build_scene_sheet, {}),
                                 (scene_sheet_creator.build_scene_sheet, {"depth_rule": "global", "axes": "camera0"})):
                out = self.root / f"sheet_{name}_{len(sheets)}"
                objects_csv, cameras_csv = build(before, ["chair"], include_size=True, out_dir=out, conf_percentile=10,
                                                 min_points=20, size_trim=0.02, **extra)
                sheets.append((objects_csv.read_text(), cameras_csv.read_text()))
            self.assertEqual(sheets[0], sheets[1], name)

    def test_rerun_detection_can_still_patch_the_tracker(self):
        """The frozen pipeline_v3 runs (rerun_detection.patch_tracker) edit milestone4_video.py by exact strings."""
        sys.path.insert(0, str(EXPERIMENT))
        try:
            import rerun_detection
        finally:
            sys.path.remove(str(EXPERIMENT))
        code_root = self.root / "patch"
        (code_root / rerun_detection.TRACKER.parent).mkdir(parents=True)
        shutil.copyfile(M4, code_root / rerun_detection.TRACKER)
        patched = rerun_detection.patch_tracker(code_root)
        compile(patched.read_text(), str(patched), "exec")
        shutil.copyfile(common.TRACK_A / "saved_masks.py", patched.parent / "saved_masks.py")
        shutil.copyfile(common.TRACK_A / "depth_rule.py", patched.parent / "depth_rule.py")
        out = self.track(self.masks["two"], "--det-score", "0.55", "--near-frac", "0.05", script=patched)
        np.testing.assert_array_equal(self.labels(out), self.labels(self.track(self.masks["two"])))
        self.assertTrue((out / "tracks/chair.scores.json").is_file())


class DepthRule(unittest.TestCase):
    """The depth rule is written twice, in the tracker (trackA/depth_rule.py) and the scene sheet
    (scene_sheet_creator.usable_pixels): they must agree."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("depth_rule_tracker", common.TRACK_A / "depth_rule.py")
        cls.tracker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tracker)

    def test_tracker_and_sheet_have_the_same_rules(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            mask = rng.uniform(size=(20, 30)) < rng.uniform(0, 1)
            depth = rng.uniform(-0.2, 3, (20, 30)).astype(np.float32)
            conf = (1 + rng.uniform(0, 5, (20, 30))).astype(np.float16).astype(np.float32)
            threshold, quantile = float(rng.uniform(1, 6)), float(rng.uniform(10, 90))
            for rule in ("global", "mask"):
                np.testing.assert_array_equal(
                    self.tracker.usable_pixels(mask, depth, conf, threshold, rule, quantile),
                    scene_sheet_creator.usable_pixels(mask, depth, conf, threshold, rule, quantile))
        self.assertEqual(self.tracker.MIN_POINTS, scene_sheet_creator.MIN_POINTS)
        self.assertEqual(self.tracker.DEFAULT_QUANTILE, scene_sheet_creator.MASK_CONF_QUANTILE)
        self.assertEqual(set(self.tracker.RULES), set(scene_sheet_creator.DEPTH_RULES))

    def test_the_rules(self):
        mask = np.zeros((4, 4), bool)
        mask[:2] = True
        depth = np.ones((4, 4), np.float32)
        depth[0, 0] = 0
        conf = np.array([[1, 1, 2, 4]] * 4, np.float32)
        usable = scene_sheet_creator.usable_pixels
        np.testing.assert_array_equal(usable(mask, depth, conf, 3.0, "global"), mask & (conf >= 3) & (depth > 0))
        # the mask's own median (1.5 over its seven pixels with depth: 1, 2, 4, 1, 1, 2, 4 -> 2) is the lower threshold
        np.testing.assert_array_equal(usable(mask, depth, conf, 3.0, "mask"), mask & (conf >= 2) & (depth > 0))
        np.testing.assert_array_equal(usable(mask, depth, conf, 1.0, "mask"), mask & (depth > 0))  # global is lower
        self.assertFalse(usable(np.zeros((4, 4), bool), depth, conf, 3.0, "mask").any())
        with self.assertRaises(ValueError):
            usable(mask, depth, conf, 3.0, "nearest")
        self.assertEqual((scene_sheet_creator.min_points_for("global"), scene_sheet_creator.min_points_for("mask"),
                          scene_sheet_creator.min_points_for("mask", 20)), (300, 50, 20))


def tracker_defaults():
    """milestone4_video.py's option defaults, {name with underscores: value}, read from its parse_args (the script
    parses its arguments when imported)."""
    tree = ast.parse(M4.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            default = next((k.value for k in node.keywords if k.arg == "default"), None)
            out[node.args[0].value.lstrip("-").replace("-", "_")] = (ast.literal_eval(default) if default is not None
                                                                     else None)
    return out


def heading_rotation(heading, pitch, roll=0.0):
    """Camera-to-world rotation (columns: the camera's x right, y down, z forward) of a camera in a z-up world that
    faces `heading` degrees clockwise from +y, `pitch` degrees up (negative: down), rolled `roll` degrees."""
    h, p, r = np.radians([heading, pitch, roll])
    flat = np.array([np.sin(h), np.cos(h), 0.0])
    forward = np.cos(p) * flat + np.sin(p) * np.array([0.0, 0.0, 1.0])
    right = np.cross(flat, [0.0, 0.0, 1.0])
    down = np.cross(forward, right)
    c, s = np.cos(r), np.sin(r)  # roll about the forward axis
    right, down = c * right + s * down, c * down - s * right
    return np.stack([right, down, forward], axis=1)


def synthetic_room(rng, pitch0=-25.0, roll_deg=3.0, floor_points=6000):
    """A room in a z-up world (floor z = 0, a 5 x 4 floor, two walls, a dense table top at 0.75, noise), seen by 16
    cameras 1.4 above the floor that turn around it, slightly rolled; then expressed, as VGGT's are, in the first
    camera's frame. Returns (points, extrinsic, true up, the first camera's heading along the floor), the last two in
    that frame."""
    noise = lambda n: rng.normal(0, 0.005, (n, 3))
    floor = np.c_[rng.uniform(-2, 3, floor_points), rng.uniform(0, 4, floor_points), np.zeros(floor_points)]
    wall_x = np.c_[np.full(3000, 3.0), rng.uniform(0, 4, 3000), rng.uniform(0, 2.5, 3000)]
    wall_y = np.c_[rng.uniform(-2, 3, 3000), np.full(3000, 4.0), rng.uniform(0, 2.5, 3000)]
    table = np.c_[rng.uniform(0, 1.2, 4000), rng.uniform(1, 2, 4000), np.full(4000, 0.75)]
    points = np.concatenate([floor, wall_x, wall_y, table])
    points = points + noise(len(points))
    headings = np.linspace(20, 320, 16)
    centres = np.c_[np.linspace(-1, 2, 16), np.linspace(0.5, 3, 16), np.full(16, 1.4)]
    rotations = [heading_rotation(h, pitch0 if i == 0 else -25 + 10 * np.sin(i), rng.uniform(-roll_deg, roll_deg))
                 for i, h in enumerate(headings)]
    R0, c0 = rotations[0], centres[0]  # VGGT's world: the first camera's frame
    to0 = lambda p: (p - c0) @ R0
    extrinsic = np.zeros((16, 3, 4))
    for i, (R, c) in enumerate(zip(rotations, centres)):
        Rw = R0.T @ R  # camera-to-world in the first camera's frame
        extrinsic[i, :, :3], extrinsic[i, :, 3] = Rw.T, -Rw.T @ to0(c)
    up = R0.T @ np.array([0.0, 0.0, 1.0])
    heading = R0.T @ np.array([np.sin(np.radians(headings[0])), np.cos(np.radians(headings[0])), 0.0])
    return to0(points), extrinsic, up, heading


class UprightFrame(unittest.TestCase):
    """upright.py: the floor fit and the upright frame, on synthetic rooms (no depth maps), and its cache."""

    def fit(self, points, extrinsic):
        centres, rotations = upright.camera_poses(extrinsic)
        up0 = camera_sheet_format_converter.estimate_up(rotations)
        return up0, upright.fit_floor(points, up0, 1.5, centres)

    def test_the_floor_is_found_and_not_the_table(self):
        points, extrinsic, up, heading = synthetic_room(np.random.default_rng(1))
        up0, (n, offset, diagnostics) = self.fit(points, extrinsic)
        self.assertIsNotNone(n, diagnostics)
        self.assertLess(upright.angle_deg(n, up), 0.3)
        # the floor's normal, not the cameras' up: from an up 6 degrees off, the floor comes back as well
        axis = np.cross(up0, [1.0, 0.0, 0.0])
        axis /= np.linalg.norm(axis)
        off = np.cos(np.radians(6)) * up0 + np.sin(np.radians(6)) * np.cross(axis, up0)
        centres, _ = upright.camera_poses(extrinsic)
        n_off, _, found = upright.fit_floor(points, off, 1.5, centres)
        self.assertLess(upright.angle_deg(n_off, up), 0.3, found)
        frame = upright.upright_frame(extrinsic, n, offset)
        R = frame["rotation"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
        moved = upright.to_upright(points, frame)
        z = moved[:, 2]
        self.assertLess(abs(float(np.median(z[:6000]))), 0.005)  # the floor's points at z = 0, the table's at 0.75
        self.assertAlmostEqual(float(np.median(z[-4000:])), 0.75, delta=0.01)
        centre0 = upright.to_upright(upright.camera_poses(extrinsic)[0][:1], frame)[0]
        np.testing.assert_allclose(centre0, [0, 0, 1.4], atol=0.01)  # the origin is on the floor below camera 0
        self.assertLess(upright.angle_deg(R[1], heading), 0.5)  # y: the way camera 0 faces, along the floor
        self.assertEqual(frame["forward"], "camera")
        self.assertGreater(diagnostics["inlier_share"], 0.2)

    def test_a_first_camera_looking_straight_down_faces_its_image_up(self):
        points, extrinsic, up, heading = synthetic_room(np.random.default_rng(2), pitch0=-89.5, roll_deg=0.0)
        _, (n, offset, _) = self.fit(points, extrinsic)
        frame = upright.upright_frame(extrinsic, n, offset)
        self.assertEqual(frame["forward"], "image up")
        self.assertLess(upright.angle_deg(frame["rotation"][1], heading), 1.0)

    def test_no_floor_is_reported(self):
        points, extrinsic, _, _ = synthetic_room(np.random.default_rng(3), floor_points=0)
        keep = points[:, 2] > -10  # walls and the table only: the table top has the walls below it
        _, (n, _, diagnostics) = self.fit(points[keep], extrinsic)
        self.assertIsNone(n)
        self.assertIn("reason", diagnostics)

    def test_the_frame_is_kept_per_scene_and_made_again_for_other_cameras(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        _, data, dense = make_scene(root)
        scene = root / "adapted"
        self.assertEqual(adapt(data, scene, "--frames", "all", "--dense-dir", str(dense)), 0)
        warnings = []
        first = upright.scene_upright(scene, log=warnings.append)
        self.assertEqual(first["method"], "cameras")  # a wall, no floor
        self.assertEqual(len(warnings), 1)
        self.assertIn("no floor plane found", warnings[0])
        with mock.patch.object(upright, "compute_upright", side_effect=AssertionError("computed again")):
            again = upright.scene_upright(scene, log=warnings.append)  # from upright.json
        np.testing.assert_array_equal(again["rotation"], first["rotation"])
        self.assertEqual(len(warnings), 2)  # said again each time the frame is used
        with np.load(scene / "cameras.npz") as z:
            cams = dict(z)
        cams["extrinsic"] = cams["extrinsic"].copy()
        cams["extrinsic"][:, :, 3] *= 2
        np.savez(scene / "cameras.npz", **cams)
        with mock.patch.object(upright, "compute_upright", wraps=upright.compute_upright) as compute:
            upright.scene_upright(scene, log=lambda text: None)
        self.assertEqual(compute.call_count, 1)

    def test_upright_boxes_turn_only_about_z(self):
        rng = np.random.default_rng(4)
        box = rng.uniform(-0.5, 0.5, (20000, 3)) * [2.0, 1.0, 0.8]
        turn = np.radians(30)
        Rz = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]])
        _, sides = scene_sheet_creator.fit_upright_box(box @ Rz.T + [3, 1, 0.4])
        np.testing.assert_allclose(sides, [2.0, 1.0, 0.8], rtol=0.02)
        _, sides = scene_sheet_creator.fit_upright_box(box)  # unturned: the axis-aligned box
        np.testing.assert_allclose(sides, [2.0, 1.0, 0.8], rtol=0.02)
        tilted = box @ np.array([[1, 0, 0], [0, np.cos(0.5), -np.sin(0.5)], [0, np.sin(0.5), np.cos(0.5)]]).T
        _, upright_sides = scene_sheet_creator.fit_upright_box(tilted)  # never turned about x: it stays upright
        self.assertGreater(upright_sides[2], 0.8 * 1.1)


class TrackingSettings(unittest.TestCase):
    """What question_scene_sheet.py records and passes to the tracker, with the fixed rules (the default) and with
    the original ones; and how a rebuild reads a sheet's depth rule back."""

    ORIGINAL = {"reid": True, "backend": "transformers", "min_score": 0.5, "conf_percentile": 55.0, "min_points": 300,
                "min_track_frames": question_scene_sheet.MIN_TRACK_FRAMES_RULE}

    def test_original_rules_record_exactly_the_settings_of_every_earlier_sheet(self):
        legacy = question_scene_sheet.track_settings(True, "transformers", 0.5, 55.0, None, "centroid", "global")
        self.assertEqual(legacy, self.ORIGINAL)  # so their saved tracks are reused, not made again
        self.assertEqual(question_scene_sheet.track_settings(True, "transformers", 0.5, 55.0, 300, "centroid",
                                                             "global"), self.ORIGINAL)
        self.assertEqual(question_scene_sheet.tracker_options(legacy),
                         {"association": None, "depth_rule": None, "mask_conf_quantile": None,
                          "tracker_options": None})

    def test_defaults_are_the_fixed_rules(self):
        default = question_scene_sheet.track_settings(True, "transformers", 0.5, 55.0)
        self.assertEqual(default, {**self.ORIGINAL, "min_points": 50, "association": "overlap", "depth_rule": "mask",
                                   "mask_conf_quantile": 50.0, **question_scene_sheet.OVERLAP_OPTIONS})
        args = question_scene_sheet.parse_args(["synthetic", "Where is the chair?"])
        self.assertEqual((args.frames, args.association, args.depth_rule, args.min_points, args.voxel_frac),
                         ("all", "overlap", "mask", None, question_scene_sheet.OVERLAP_OPTIONS["voxel_frac"]))
        for function in (question_scene_sheet.create_scene_sheet, question_scene_sheet.prepare_scene):
            parameters = inspect.signature(function).parameters
            self.assertEqual((parameters["frames"].default, parameters["association"].default,
                              parameters["depth_rule"].default), ("all", "overlap", "mask"))
        self.assertEqual(question_scene_sheet.frame_selection(question_scene_sheet.DEFAULT_FRAMES), ("all", "all"))
        # the milestone 4 defaults these settings stand in for, so that tuning one of them makes the tracks again
        defaults = tracker_defaults()
        for name, value in question_scene_sheet.OVERLAP_OPTIONS.items():
            self.assertEqual(defaults[name], value, name)
        # milestone 4's own defaults stay the original rules, which the frozen runs rely on
        self.assertEqual((defaults["association"], defaults["depth_rule"], defaults["min_points"]),
                         ("centroid", "global", None))

    def test_tracker_command_line(self):
        commands = []

        def run(cmd, **kwargs):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        legacy = question_scene_sheet.track_settings(True, "transformers", 0.5, 55.0, None, "centroid", "global")
        default = question_scene_sheet.track_settings(False, "transformers", 0.5, 55.0)
        with mock.patch.object(subprocess, "run", run), contextlib.redirect_stdout(io.StringIO()):
            for settings in (legacy, default):
                scene_sheet_creator.run_pipeline("frames", "scene", ["chair"], "track", 1,
                                                 no_reid=not settings["reid"], masks_dir="masks",
                                                 conf_percentile=settings["conf_percentile"],
                                                 min_points=settings["min_points"],
                                                 **question_scene_sheet.tracker_options(settings))
        tail = lambda cmd: cmd[cmd.index("--overlap") + 2:]  # after the VGGT chunk options
        masks = str(Path("masks").resolve())
        self.assertEqual(tail(commands[0]), ["--masks-dir", masks, "--conf-percentile", "55.0", "--min-points", "300"])
        self.assertEqual(tail(commands[1]), ["--no-reid", "--masks-dir", masks, "--conf-percentile", "55.0",
                                             "--min-points", "50", "--association", "overlap", "--depth-rule", "mask",
                                             "--mask-conf-quantile", "50.0", "--voxel-frac", "0.065",
                                             "--overlap-min", "0.1", "--reid-radius-depth", "0.5",
                                             "--merge-overlap", "0.2", "--duplicate-overlap", "0.6"])

    def test_a_rebuild_reads_the_depth_rule_back(self):
        rebuild = reconstruct_pipeline.rebuild
        self.assertEqual(rebuild.depth_options({}), {"depth_rule": "global", "mask_conf_quantile": 50.0})
        self.assertEqual(rebuild.depth_options({"depth_rule": "mask", "mask_conf_quantile": 40.0}),
                         {"depth_rule": "mask", "mask_conf_quantile": 40.0})

        def before_the_rule(scene, prompts, relative_positions=False, include_size=False, out_dir=None,
                            conf_percentile=55.0, min_points=300, size_trim=0.0, max_frame_points=5000,
                            frame_columns=True, cameras=True):
            """build_scene_sheet's signature before the depth rule (as pipeline_v3.py runs it, from 50fe543)."""

        with mock.patch.object(rebuild.sheet, "build_scene_sheet", before_the_rule):
            self.assertEqual(rebuild.depth_options({}), {})
            self.assertIsNone(rebuild.depth_options({"depth_rule": "mask"}))
            self.assertEqual(rebuild.axes_options({}), {})
            self.assertIsNone(rebuild.axes_options({"axes": "upright"}))
        self.assertEqual(rebuild.axes_options({}), {"axes": "camera0"})  # every sheet before the option
        self.assertEqual(rebuild.axes_options({"axes": "upright"}), {"axes": "upright"})

    def test_frames_folders_of_a_run(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        scene = root / "scannet__s1"
        for label in ("all", "subset-uniform16", "gold"):
            (scene / label / "scene").mkdir(parents=True)
            (scene / label / "scene" / "adapter_report.json").write_text("{}")
        (scene / "notes").mkdir()
        rebuild = reconstruct_pipeline.rebuild
        self.assertEqual(rebuild.pipeline_labels(scene), ["all", "subset-uniform16"])
        self.assertEqual(rebuild.pipeline_labels(scene, "all"), ["all"])
        with self.assertRaises(ValueError):  # two frames folders and no index.csv: the run must say which
            reconstruct_pipeline.run_label(root)
        self.assertEqual(reconstruct_pipeline.run_label(root, "all"), "all")
        with open(root / "index.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, ["id", "pipeline_sheet", "gold_sheet"])
            writer.writeheader()
            writer.writerow({"id": 1, "pipeline_sheet": "scannet__s1/all/sheets/chair", "gold_sheet": ""})
        self.assertEqual(reconstruct_pipeline.run_label(root), "all")


class ComparisonMaths(unittest.TestCase):
    def test_track_matching_ignores_id_numbers(self):
        a, b = np.zeros((2, 4, 4), np.uint16), np.zeros((2, 4, 4), np.uint16)
        a[:, :2, :2], a[:, 2:, 2:] = 1, 2
        b[:, :2, :2], b[:, 2:, 2:] = 7, 3
        matches, _, _ = compare.match_tracks(a, b)
        self.assertEqual(sorted((m[0], m[1]) for m in matches), [(1, 7), (2, 3)])
        self.assertTrue(all(m[2] == 1.0 for m in matches))

    def test_chamfer(self):
        points = np.random.default_rng(0).normal(size=(50, 3))
        rng = np.random.default_rng(1)
        self.assertAlmostEqual(compare.chamfer(points, points.copy(), 100, rng), 0.0)
        self.assertGreater(compare.chamfer(points, points + 10, 100, rng), 5)

    def test_a_fail_makes_the_verdict_differs_even_if_numbers_are_missing(self):
        report = compare.Report()
        report.add("frames: same frames in the same order", "FAIL", "different frames")
        self.assertEqual(compare.verdict(report)["verdict"], "DIFFERS")

    def test_verdict_applies_the_limits(self):
        report = compare.Report()
        report.add("tracking consistency", "MEASURED", values={"pixels_with_matching_identity": 0.95})
        report.add("3D object positions after alignment", "MEASURED", values={"median_error_of_scene": 0.02})
        report.add("scene sheet: objects", "MEASURED",
                   values={"matched_share": 0.9, "median_position_error_of_scene": 0.02})
        self.assertEqual(compare.verdict(report)["verdict"], "MATCHES")
        report.rows[1]["values"]["median_error_of_scene"] = 0.2
        self.assertEqual(compare.verdict(report)["verdict"], "DIFFERS")

    def test_format_check_catches_a_wrong_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.npz"
            np.savez(path, depth=np.zeros((1, 2, 3), np.float32), conf=np.zeros((1, 2, 3), np.float16))
            self.assertTrue(any("dtype" in p for p in compare.array_problems(path, compare.ARRAY_SPECS["depth.npz"], {})))


class ObjectSize(unittest.TestCase):
    """scene_sheet_creator.fit_box: the smallest box around an object's points, turned to fit it."""

    SIDES = np.array([2.0, 1.0, 0.5])

    def box_points(self, rotation, n=20000, seed=0):
        """Points filling a box with sides SIDES, turned by rotation and moved away from the origin."""
        local = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(n, 3)) * self.SIDES
        return local @ rotation.T + np.array([3.0, -1.0, 7.0])

    def test_an_unturned_box_is_the_axis_aligned_box(self):
        points = self.box_points(np.eye(3))
        axes, sides = scene_sheet_creator.fit_box(points)
        np.testing.assert_allclose(sides, points.max(axis=0) - points.min(axis=0), rtol=1e-6)
        np.testing.assert_allclose(np.abs(axes), np.eye(3), atol=1e-6)

    def test_a_box_turned_about_the_vertical_keeps_its_sides(self):
        points = self.box_points(rotation_y(np.deg2rad(30)))
        aligned = points.max(axis=0) - points.min(axis=0)
        self.assertGreater(np.prod(aligned) / np.prod(self.SIDES), 2.5)  # what the axis-aligned box made of it
        axes, sides = scene_sheet_creator.fit_box(points)
        np.testing.assert_allclose(sides, self.SIDES, rtol=0.01)  # 30 degrees: each side is still nearest its axis
        np.testing.assert_allclose(axes, rotation_y(np.deg2rad(30)), atol=0.01)

    def test_any_rotation_is_undone_and_the_box_is_never_bigger(self):
        rng = np.random.default_rng(3)
        for trim in (0.0, 0.02):
            for _ in range(3):
                q, r = np.linalg.qr(rng.normal(size=(3, 3)))
                rotation = q * np.sign(np.diag(r)) * np.sign(np.linalg.det(q))
                points = self.box_points(rotation)
                axes, sides = scene_sheet_creator.fit_box(points, trim)
                np.testing.assert_allclose(np.sort(sides), np.sort(self.SIDES), rtol=0.01)
                np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-9)
                self.assertLessEqual(np.prod(sides), np.prod(points.max(axis=0) - points.min(axis=0)))

    def test_trim_keeps_outliers_from_turning_or_growing_the_box(self):
        points = self.box_points(rotation_y(np.deg2rad(20)))
        points[:2] += [[30.0, 30.0, 0.0], [-30.0, 0.0, 30.0]]  # 0.01% of the points, far away
        _, sides = scene_sheet_creator.fit_box(points, trim=0.02)
        np.testing.assert_allclose(sides, self.SIDES, rtol=0.01)

    def test_a_flat_object_gets_a_flat_box(self):
        local = np.random.default_rng(0).uniform(-0.5, 0.5, size=(5000, 3)) * [1.5, 0.0, 1.0]
        _, sides = scene_sheet_creator.fit_box(local @ rotation_y(0.4).T)
        np.testing.assert_allclose(sides, [1.5, 0.0, 1.0], atol=0.01)

    def test_a_single_point_has_no_size(self):
        for points in (np.ones((1, 3)), np.ones((10, 3))):
            axes, sides = scene_sheet_creator.fit_box(points)
            np.testing.assert_array_equal(axes, np.eye(3))
            np.testing.assert_array_equal(sides, np.zeros(3))

    def test_sides_are_matched_to_the_nearest_world_axes(self):
        axes = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # box axes: y, z, -x
        matched, sides = scene_sheet_creator.match_world_axes(axes, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(matched, np.eye(3))
        np.testing.assert_array_equal(sides, [3.0, 1.0, 2.0])


class SceneList(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ADAPTER / "hpc"))
        import make_scene_list

        self.objects_in = make_scene_list.objects_in

    def check(self, kind, question, expected):
        self.assertEqual(self.objects_in({"question_type": kind, "question": question}), expected)

    def test_every_template(self):
        self.check("object_counting", "How many table(s) are in this room?", ["table"])
        self.check("object_size_estimation", "What is the length of the longest dimension (length, width, or height) "
                   "of the trash bin, measured in centimeters?", ["trash bin"])
        self.check("object_abs_distance", "Measuring from the closest point of each object, what is the distance "
                   "between the sofa and the TV (in meters)?", ["sofa", "tv"])
        self.check("object_rel_distance", "Measuring from the closest point of each object, which of these objects "
                   "(chair, stool, stove, sofa) is the closest to the tv?", ["chair", "stool", "stove", "sofa", "tv"])
        for kind, tail in (("easy", "to the left or the right of the sofa?"), ("medium", "to my left, right, or back?"),
                           ("hard", "to my front-left, front-right, back-left, or back-right?")):
            self.check(f"object_rel_direction_{kind}", f"If I am standing by the stove and facing the sofa, is the "
                       f"tv {tail}", ["stove", "sofa", "tv"])
        self.check("obj_appearance_order", "What will be the first-time appearance order of the following categories "
                   "in the video: door, lamp, bed, mirror?", ["door", "lamp", "bed", "mirror"])

    def test_skipped_and_unknown_types(self):
        self.check("route_planning", "You are a robot beginning at the sofa", [])
        self.check("object_counting", "A question in another wording", None)


class Summary(unittest.TestCase):
    def test_table_has_every_scene_including_failed_ones(self):
        sys.path.insert(0, str(ADAPTER))
        import summarize

        report = compare.Report()
        report.add("tracking consistency", "MEASURED", values={"pixels_with_matching_identity": 0.95})
        report.add("object labels: objects per concept", "MEASURED",
                   values={"adapted": {"chair": 2, "table": 1}, "native": {"chair": 2, "table": 2}})
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "scannet__a/eval32").mkdir(parents=True)
            (results / "scannet__a/eval32/comparison.json").write_text(
                json.dumps({"counts": report.counts(), **compare.verdict(report), "comparisons": report.rows}))
            (results / "scannet__b").mkdir()
            (results / "scannet__b/run.log").write_text("...\nno dense depth in data/vggt/scannet/b\n")
            summarize.main(["--results", str(results)])
            text = (results / "SUMMARY.md").read_text()
        self.assertIn("| scannet/a | eval32 | **INCOMPLETE** | 3 / 4 | 95.0% |", text)
        self.assertIn("| scannet/b | - | no result |", text)
        self.assertIn("no dense depth", text)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class QuestionSceneSheet(unittest.TestCase):
    """question_scene_sheet.py on the synthetic scene: the real adapter, sam3_infer.py with the fake SAM 3, the real
    track stage and the real scene sheet. Only the prompt extractor is replaced, so that spaCy is not needed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _, self.data, self.dense = make_scene(self.root)
        self.stubs, self.gpu = write_stubs(self.root)
        self.work = self.root / "work"
        self.calls = []  # (script, prompts) of every step that was run as a process
        real_run = subprocess.run

        def run(cmd, **kwargs):  # the steps' own processes, with the stand-ins on their path
            script = Path(cmd[2]).name
            self.calls.append((script, cmd[cmd.index("--prompts") + 1]))
            paths = [str(self.gpu), str(self.stubs)] if script == "sam3_infer.py" else [str(self.stubs)]
            done = real_run(cmd, env={**os.environ, "PYTHONPATH": os.pathsep.join(paths)}, capture_output=True, text=True)
            if done.returncode:
                sys.__stderr__.write(done.stderr)
            return done

        patcher = mock.patch.object(subprocess, "run", run)
        patcher.start()
        self.addCleanup(patcher.stop)

    def sheet(self, prompts, work=None, **kwargs):
        options = {"data_root": self.data, "dense_dir": self.dense, "work_dir": work or self.work, "reid": False,
                   "backend": "sam3", "conf_percentile": 10, "min_points": 20, "frames": "all", **kwargs}
        self.log = io.StringIO()
        with mock.patch.object(question_scene_sheet, "extract_prompts", lambda question, syntactic=False: (prompts, "full")), \
                contextlib.redirect_stderr(self.log):
            return question_scene_sheet.create_scene_sheet("synthetic", "Is the chair near the table?", **options)

    def test_sheet_lists_the_objects_the_question_names(self):
        objects_csv, cameras_csv = self.sheet(["chair", "table"])
        self.assertEqual(objects_csv, (self.work / "sheets/chair+table/objects.csv").resolve())
        self.assertEqual(cameras_csv, (self.work / "sheets/chair+table/cameras.csv").resolve())
        rows = read_csv(objects_csv)
        self.assertEqual([r["concept"] for r in rows], ["chair", "table"])
        self.assertTrue(all(int(r["num_frames"]) == T for r in rows))
        self.assertTrue({"size_x", "size", "frame_000", f"frame_{T - 1:03d}"} <= set(rows[0]))
        self.assertEqual(len(read_csv(cameras_csv)), T)
        self.assertEqual(self.calls, [("sam3_infer.py", "chair,table"), ("milestone4_video.py", "chair,table")])
        record = json.loads((objects_csv.parent / "question.json").read_text())
        self.assertEqual((record["scene"], record["question"], record["prompts"]),
                         ("scannet/synthetic", "Is the chair near the table?", ["chair", "table"]))
        self.assertEqual(record["axes"], "upright")  # the default; the synthetic wall has no floor
        self.assertEqual(record["floor"]["method"], "cameras")
        self.assertIn("no floor plane found", record["floor"]["warning"])
        self.assertIn("no floor plane found", self.log.getvalue())
        self.assertTrue((self.work / "scene" / upright.FILE).is_file())
        camera0, _ = self.sheet(["chair", "table"], axes="camera0", out_dir=self.root / "camera0")
        self.assertEqual(json.loads((camera0.parent / "question.json").read_text())["axes"], "camera0")
        self.assertNotIn("floor", json.loads((camera0.parent / "question.json").read_text()))
        self.assertFalse((self.work / "scene/labels.npz").exists())  # the fuse stage is not run

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "the track stage needs torch")
    def test_report_makes_a_correct_sheet_for_every_question(self):
        """scene_sheet_report.py through the real steps: every question gets a sheet that passes its checks, and SAM 3
        runs once, for every prompt of the scene's questions."""
        questions = {"Is the chair near the table?": ["chair", "table"], "Where is the chair?": ["chair"],
                     "How many lamps are there?": ["chair", "lamp"]}
        with open(self.data / "scenes/scannet/synthetic/questions.jsonl", "w") as f:
            for i, question in enumerate(questions):
                f.write(json.dumps({"id": i, "question_type": "test", "question": question}) + "\n")
        out = io.StringIO()
        with mock.patch.object(question_scene_sheet, "extract_prompts", lambda q, syntactic=False: (questions[q], "full")), \
                contextlib.redirect_stderr(io.StringIO()):
            results = scene_sheet_report.report(
                "synthetic", data_root=self.data, work_dir=self.work, out=out, dense_dir=self.dense, reid=False,
                backend="sam3", conf_percentile=10, min_points=20, frames="all")
        self.assertEqual([r["problems"] for r in results], [[], [], []], out.getvalue())
        self.assertEqual([r["counts"] for r in results],
                         [{"chair": 1, "table": 1}, {"chair": 1}, {"chair": 1, "lamp": 0}])
        self.assertIn("no rows:  lamp (SAM 3 found nothing)", out.getvalue())
        lines = out.getvalue().strip().splitlines()  # the synthetic wall has no floor: said once, before the summary
        self.assertEqual(sum(scene_sheet_report.NO_FLOOR in line for line in lines), 1)
        self.assertIn(scene_sheet_report.NO_FLOOR, "\n".join(lines[-3:-1]))
        self.assertTrue(lines[-1].startswith("3 questions:"))
        self.assertEqual([c for c in self.calls if c[0] == "sam3_infer.py"],
                         [("sam3_infer.py", "chair,table,lamp")])

    def test_a_later_question_only_processes_new_prompts(self):
        self.sheet(["chair"])
        chair = common.sha256_file(self.work / "scene/tracks/chair.npz")
        self.calls.clear()
        objects_csv, _ = self.sheet(["table", "chair"])
        self.assertEqual(self.calls, [("sam3_infer.py", "table"), ("milestone4_video.py", "table")])
        self.assertEqual(common.sha256_file(self.work / "scene/tracks/chair.npz"), chair)
        self.assertEqual([r["concept"] for r in read_csv(objects_csv)], ["table", "chair"])
        self.assertIn("reusing the converted geometry", self.log.getvalue())
        self.calls.clear()
        self.sheet(["chair", "table"])
        self.assertEqual(self.calls, [])

    def test_reused_tracks_equal_tracking_every_prompt_at_once(self):
        self.sheet(["chair"])
        stepwise, _ = self.sheet(["chair", "table"])
        at_once, _ = self.sheet(["chair", "table"], work=self.root / "other")
        for prompt in ("chair", "table"):
            with np.load(self.work / f"scene/tracks/{prompt}.npz") as a, \
                    np.load(self.root / f"other/scene/tracks/{prompt}.npz") as b:
                np.testing.assert_array_equal(a["labels"], b["labels"])
        self.assertEqual(stepwise.read_text(), at_once.read_text())

    def test_other_tracking_settings_track_again_from_the_saved_masks(self):
        self.sheet(["chair"])
        self.calls.clear()
        self.sheet(["chair"], min_points=25)
        self.assertEqual(self.calls, [("milestone4_video.py", "chair")])
        settings = json.loads((self.work / "scene/tracks/chair.settings.json").read_text())["settings"]
        self.assertEqual((settings["min_points"], settings["reid"]), (25, False))

    def test_tracks_made_under_the_old_fixed_minimum_are_made_again(self):
        self.sheet(["chair"])
        path = self.work / "scene/tracks/chair.settings.json"
        record = json.loads(path.read_text())
        self.assertEqual(record["settings"]["min_track_frames"], question_scene_sheet.MIN_TRACK_FRAMES_RULE)
        del record["settings"]["min_track_frames"]  # as written before the rule scaled with the frame count
        path.write_text(json.dumps(record))
        self.calls.clear()
        self.sheet(["chair"])
        self.assertEqual(self.calls, [("milestone4_video.py", "chair")])  # masks reused, tracking redone

    def test_fresh_starts_the_scene_over(self):
        self.sheet(["chair"])
        self.calls.clear()
        self.sheet(["chair"], fresh=True)
        self.assertEqual(self.calls, [("sam3_infer.py", "chair"), ("milestone4_video.py", "chair")])

    def test_a_prompt_sam3_does_not_find_gives_no_rows(self):
        objects_csv, _ = self.sheet(["chair", "piano"])  # the fake SAM 3 knows no piano
        self.assertEqual([r["concept"] for r in read_csv(objects_csv)], ["chair"])

    def test_no_prompts_gives_a_sheet_without_objects(self):
        objects_csv, cameras_csv = self.sheet([])
        self.assertEqual(objects_csv.parent.name, "no_prompts")
        self.assertEqual(read_csv(objects_csv), [])
        self.assertIn("object_id,concept,track_id,x,y,z", objects_csv.read_text())
        self.assertEqual(len(read_csv(cameras_csv)), T)
        self.assertEqual(self.calls, [])

    def test_stops_without_dense_depth(self):
        with self.assertRaises(SystemExit) as stop:
            self.sheet(["chair"], dense_dir=self.root / "nowhere")
        self.assertIn("no dense depth", str(stop.exception))
        self.assertIn("--patch-depth-fallback", str(stop.exception))
        self.assertFalse(self.work.exists())
        self.assertEqual(self.calls, [])

    def test_patch_depth_fallback_warns_and_gives_way_to_the_dense_depth_once_found(self):
        nowhere = self.root / "nowhere"
        objects_csv, _ = self.sheet(["chair", "table"], dense_dir=nowhere, patch_depth_fallback=True)
        self.assertIn("WARNING: no dense depth", self.log.getvalue())
        self.assertIn("made with depth approximated from the patch points", self.log.getvalue())
        self.assertEqual(json.loads((objects_csv.parent / "question.json").read_text())["depth"], "patch_points")
        def main_tracks(path):  # each concept's longest track: with 8 frames a 1-frame fragment is also kept
            return {r["concept"]: r for r in sorted(read_csv(path), key=lambda r: int(r["num_frames"]))}

        approximated = main_tracks(objects_csv)
        self.assertEqual(sorted(approximated), ["chair", "table"])

        self.calls.clear()  # a second question reuses the approximated geometry and tracks, and warns again
        self.sheet(["chair"], dense_dir=nowhere, patch_depth_fallback=True)
        self.assertEqual(self.calls, [])
        self.assertIn("reusing the converted geometry", self.log.getvalue())
        self.assertIn("WARNING: no dense depth", self.log.getvalue())
        with self.assertRaises(SystemExit) as stop:  # without the option, approximated geometry is not used
            self.sheet(["chair"], dense_dir=nowhere)
        self.assertIn("no dense depth", str(stop.exception))

        objects_csv, _ = self.sheet(["chair", "table"])  # the dense depth: converted and tracked again, masks reused
        self.assertEqual(self.calls, [("milestone4_video.py", "chair,table")])
        self.assertEqual(json.loads((objects_csv.parent / "question.json").read_text())["depth"], "dense")
        self.assertEqual(json.loads((self.work / "scene/adapter_report.json").read_text())["depth_source"], "dense")
        for concept, row in main_tracks(objects_csv).items():  # close on this smooth depth
            moved = np.linalg.norm([float(approximated[concept][a]) - float(row[a]) for a in ("x", "y", "z")])
            self.assertLess(moved, 0.1 * max(float(row[a]) for a in ("size_x", "size_y", "size_z")), concept)

    def test_unknown_scene_and_empty_question(self):
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(io.StringIO()):
            question_scene_sheet.create_scene_sheet("scannet/absent", "Where is the chair?", data_root=self.data)
        self.assertIn("no exported scene", str(stop.exception))
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(io.StringIO()):
            question_scene_sheet.create_scene_sheet("synthetic", "  ", data_root=self.data)
        self.assertIn("empty", str(stop.exception))

    def test_folder_names(self):
        self.assertEqual(question_scene_sheet.folder_name(["chair", "trash can"]), "chair+trash_can")
        many = [f"object {i}" for i in range(40)]
        name = question_scene_sheet.folder_name(many)
        self.assertLess(len(name), 100)
        self.assertNotEqual(name, question_scene_sheet.folder_name(many[:-1]))

    def test_frame_selection_names(self):
        self.assertEqual(question_scene_sheet.frame_selection("all"), ("all", "all"))
        self.assertEqual(question_scene_sheet.frame_selection("half"), (common.PRESETS["half"], "half"))
        self.assertEqual(question_scene_sheet.frame_selection("list:0,4"), ("list:0,4", "list-0_4"))

    def test_default_frames_are_all_128_and_the_vlms_16_can_be_chosen(self):
        """By default, from Python and from the command line, the sheet uses all 128 VGGT frames; subset:uniform16, the
        default before 2026-11-24, is the VLM's 16 uniform frames of the 128."""
        default = question_scene_sheet.DEFAULT_FRAMES
        self.assertEqual(default, "all")
        self.assertEqual(inspect.signature(question_scene_sheet.create_scene_sheet).parameters["frames"].default, default)
        self.assertEqual(question_scene_sheet.parse_args(["synthetic", "Where is the chair?"]).frames, default)
        self.assertEqual(question_scene_sheet.frame_selection(default), ("all", "all"))
        spec, label = question_scene_sheet.frame_selection("subset:uniform16")
        self.assertEqual(label, "subset-uniform16")
        uniform16 = np.linspace(0, 127, 16, dtype=int).tolist()
        self.assertEqual(common.select_frames(spec, 128, {"frame_subsets": {"uniform16": uniform16}}), uniform16)
        self.assertEqual(common.select_frames(spec, 128, {"frame_subsets": {"uniform16": uniform16}}),
                         common.select_frames("uniform:16", 128))


class FindScene(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for scene in ("scannet/a", "scannet/b", "arkitscenes/b"):
            (self.root / "scenes" / scene).mkdir(parents=True)
            (self.root / "scenes" / scene / "meta.json").write_text("{}")

    def test_full_name_and_scene_name_alone(self):
        self.assertEqual(common.find_scene("scannet/b", self.root), (self.root, "scannet/b"))
        self.assertEqual(common.find_scene("scannet\\b", self.root), (self.root, "scannet/b"))
        self.assertEqual(common.find_scene("a", self.root), (self.root, "scannet/a"))

    def test_a_name_in_two_datasets_must_say_which(self):
        with self.assertRaises(ValueError):
            common.find_scene("b", self.root)

    def test_missing_scene(self):
        with self.assertRaises(FileNotFoundError):
            common.find_scene("scannet/c", self.root)


@unittest.skipUnless(importlib.util.find_spec("spacy") and importlib.util.find_spec("nltk"), "spaCy and NLTK are absent")
class QuestionPrompts(unittest.TestCase):
    """The real prompt extractor, and the command line. No SAM 3: these questions name nothing it could segment."""

    def quiet(self, question, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return question_scene_sheet.extract_prompts(question, **kwargs)

    def test_physical_objects_first_then_every_noun_phrase(self):
        self.assertEqual(self.quiet("Is the chair closer to the table or to the door?"),
                         (["chair", "table", "door"], "full"))
        self.assertEqual(self.quiet("What is the total length?"), (["total length"], "syntactic fallback"))
        self.assertEqual(self.quiet("How big is it?"), ([], "syntactic fallback"))
        self.assertEqual(self.quiet("What is the size of this room?", syntactic=True)[1], "syntactic")

    def test_command_line_prints_only_the_two_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, data, dense = make_scene(root)
            done = subprocess.run(
                [sys.executable, str(ADAPTER / "question_scene_sheet.py"), "scannet/synthetic", "How big is it?",
                 "--data-root", str(data), "--dense-dir", str(dense), "--work-dir", str(root / "work"),
                 "--frames", "all"], capture_output=True, text=True)  # the synthetic scene has only 8 frames
            self.assertEqual(done.returncode, 0, done.stderr)
            lines = done.stdout.splitlines()
            self.assertEqual([Path(line).name for line in lines], ["objects.csv", "cameras.csv"])
            self.assertTrue(all(Path(line).is_file() for line in lines))
            self.assertIn("[adapter]", done.stderr)


def gold_box(centre, axes_lengths, axes=np.eye(3)):
    """A gold-label box in the metadata's format: axes has one box axis per row, as normalizedAxes does, and min/max
    are the axis-aligned box around it (so centre -+ half the lengths when it is not turned)."""
    centre, axes = np.array(centre, float), np.array(axes, float)
    half = np.abs(axes.T) @ np.array(axes_lengths, float) / 2
    return {"centroid": centre.tolist(), "axesLengths": list(axes_lengths),
            "normalizedAxes": axes.ravel().tolist(), "min": (centre - half).tolist(),
            "max": (centre + half).tolist()}


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class GoldSheet(unittest.TestCase):
    """scene_sheet_creator.py --source gold, on a small metadata folder (nothing is downloaded)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.gold = self.root / "meta_info"
        self.gold.mkdir()
        scenes = {"scannet": {"s1": {"object_bbox": {
            "chair": [gold_box([1, 2, 0.5], [0.5, 0.6, 1.0]), gold_box([3, 1, 0.5], [0.5, 0.5, 1.0])],
            "table": [gold_box([2, 2, 0.4], [2.0, 1.0, 0.8])],
            "coat hanger": [gold_box([0, 0, 1], [0.4, 0.4, 1.8])],
            # turned 30 degrees about z, with its axes listed long side, height, short side
            "sofa": [gold_box([4, 0, 0.4], [2.0, 0.8, 0.9], rotation_z(np.deg2rad(30)).T[[0, 2, 1]])],
        }}}, "scannetpp": {}, "arkitscenes": {}}
        for dataset, (name, _) in scene_sheet_creator.GOLD_FILES.items():
            (self.gold / name).write_text(json.dumps(scenes[dataset]))

    def sheet(self, prompts=None, **options):
        out = self.root / "sheet"
        with contextlib.redirect_stdout(io.StringIO()):
            objects_csv, cameras_csv = scene_sheet_creator.build_gold_scene_sheet(
                "scannet/s1", prompts, out_dir=out, gold_dir=self.gold, **options)
        self.assertIsNone(cameras_csv)
        return read_csv(objects_csv), json.loads((out / "sheet.json").read_text())

    def test_metres_sheet_has_the_boxes_and_no_per_frame_columns(self):
        rows, info = self.sheet(["chair", "table"], include_size=True)
        self.assertEqual(list(rows[0]), ["object_id", "concept", "track_id", "x", "y", "z",
                                         "size_x", "size_y", "size_z", "size"])
        self.assertEqual([(r["concept"], r["track_id"]) for r in rows], [("chair", "1"), ("chair", "2"), ("table", "1")])
        table = rows[2]
        self.assertEqual([float(table[a]) for a in ("x", "y", "z")], [2, 2, 0.4])
        self.assertEqual([float(table[a]) for a in ("size_x", "size_y", "size_z")], [2, 1, 0.8])
        self.assertAlmostEqual(float(table["size"]), 1.6)
        self.assertFalse((self.root / "sheet" / "cameras.csv").exists())
        self.assertEqual(info["source"], "gold")
        self.assertIn(scene_sheet_creator.GOLD_COMMIT, info["labels"])

    def test_a_turned_box_has_its_own_sides_matched_to_the_axes(self):
        rows, _ = self.sheet(["sofa"], include_size=True)
        sofa = {a: float(rows[0][a]) for a in ("size_x", "size_y", "size_z", "size")}
        box = gold_box([4, 0, 0.4], [2.0, 0.8, 0.9], rotation_z(np.deg2rad(30)).T[[0, 2, 1]])
        self.assertGreater(box["max"][1] - box["min"][1], 1.5)  # what max - min made of the 0.9 m side
        self.assertEqual([sofa[a] for a in ("size_x", "size_y", "size_z")], [2.0, 0.9, 0.8])
        self.assertAlmostEqual(sofa["size"], 1.44)

    def test_prompts_match_classes_by_case_plural_and_alias(self):
        rows, info = self.sheet(["Chairs", "coat rack", "chair", "lamp"])
        self.assertEqual([r["concept"] for r in rows], ["Chairs", "Chairs", "coat rack"])  # chair only once
        self.assertEqual(info["matched_classes"], {"Chairs": "chair", "coat rack": "coat hanger", "chair": "chair",
                                                   "lamp": None})

    def test_without_prompts_every_class_is_listed(self):
        rows, _ = self.sheet()
        self.assertEqual(len(rows), 5)

    def test_per_frame_columns_and_cameras_are_refused(self):
        for options in ({"relative_positions": True}, {"frame_columns": True}, {"cameras": True}):
            with self.assertRaises(ValueError, msg=options):
                self.sheet(["chair"], **options)
        with self.assertRaises(ValueError):
            scene_sheet_creator.load_gold_scene("scannet/absent", self.gold)

    def test_a_stale_cameras_csv_is_removed(self):
        out = self.root / "sheet"
        out.mkdir()
        (out / "cameras.csv").write_text("frame\n0\n")
        self.sheet(["chair"])
        self.assertFalse((out / "cameras.csv").exists())

    def test_command_line(self):
        out = self.root / "cli"
        base = ["--source", "gold", "--scene", "s1", "--gold-dir", str(self.gold), "--out-dir", str(out)]
        with contextlib.redirect_stdout(io.StringIO()):
            scene_sheet_creator.main(base + ["--prompts", "chair,table", "--size"])
        self.assertEqual(len(read_csv(out / "objects.csv")), 3)
        for bad in (base + ["--relative-positions"], base + ["--images", "frames"],
                    ["--scene", "x", "--prompts", "chair", "--relative-positions", "--no-frames"],
                    ["--scene", "x"]):
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                scene_sheet_creator.main(bad)
            self.assertEqual(caught.exception.code, 2, bad)

    def test_pipeline_sheet_refuses_relative_positions_without_frames(self):
        with self.assertRaises(ValueError):
            scene_sheet_creator.build_scene_sheet(self.root, ["chair"], relative_positions=True, frame_columns=False)


def fake_pipeline_sheet(out, prompts, frames=3, rows=None):
    """A pipeline sheet in the default layout: one object per prompt, seen in every frame (or the given rows)."""
    out.mkdir(parents=True, exist_ok=True)
    names = [f"frame_{t:03d}" for t in range(frames)]
    if rows is None:
        rows = [[i, p, 1, 0.1, 0.2, 0.3, 1, 1, 1, 1, frames] + [1] * frames for i, p in enumerate(prompts)]
    scene_sheet_creator.write_csv(out / "objects.csv", scene_sheet_report.GLOBAL_COLUMNS + ["num_frames"] + names, rows)
    scene_sheet_creator.write_csv(out / "cameras.csv", scene_sheet_report.CAMERA_COLUMNS,
                                  [[t, f"{t}.jpg", 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] for t in range(frames)])
    (out / "question.json").write_text(json.dumps({"prompts": prompts}))
    return out / "objects.csv", out / "cameras.csv"


class SceneSheetReport(unittest.TestCase):
    """scene_sheet_report.py: which questions it runs, with which settings, and what its checks catch. The prompt
    extractor is replaced, and so is the pipeline (its real run is QuestionSceneSheet.test_report_...)."""

    QUESTIONS = {"Is the chair near the table?": ["chair", "table"], "Where is the coat rack?": ["coat rack"],
                 "How far is the lamp from the chair?": ["lamp", "chair"]}

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.data = self.root / "data"
        scene = self.data / "scenes" / "scannet" / "s1"
        scene.mkdir(parents=True)
        (scene / "meta.json").write_text(json.dumps({"dataset": "scannet", "scene": "s1", "num_frames": 3}))
        self.write_questions(scene / "questions.jsonl", "s1")
        self.gold = self.root / "meta_info"
        self.gold.mkdir()
        labels = {"s1": {"object_bbox": {"chair": [gold_box([1, 2, 0.5], [0.5, 0.6, 1]), gold_box([3, 1, 0.5], [0.5, 0.5, 1])],
                                          "table": [gold_box([2, 2, 0.4], [2, 1, 0.8])],
                                          "coat hanger": [gold_box([0, 0, 1], [0.4, 0.4, 1.8])]}},
                  "s2": {"object_bbox": {"chair": [gold_box([0, 0, 0.5], [0.5, 0.5, 1])]}}}
        for dataset, (name, _) in scene_sheet_creator.GOLD_FILES.items():
            (self.gold / name).write_text(json.dumps(labels if dataset == "scannet" else {}))
        patcher = mock.patch.object(question_scene_sheet, "extract_prompts",
                                    lambda question, syntactic=False: (self.QUESTIONS[question], "full"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_questions(self, path, scene, extra=()):
        with open(path, "w") as f:
            for i, question in enumerate(self.QUESTIONS):
                f.write(json.dumps({"id": i, "dataset": "scannet", "scene_name": scene,
                                    "question_type": "object_counting" if i == 1 else "object_rel_distance",
                                    "question": question}) + "\n")
            for row in extra:
                f.write(json.dumps(row) + "\n")

    def report(self, **options):
        out = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            results = scene_sheet_report.report(options.pop("scene", "s1"), data_root=self.data,
                                                work_dir=self.root / "work", gold_dir=self.gold, out=out, **options)
        return results, out.getvalue()

    def test_gold_report_makes_a_correct_sheet_for_every_question(self):
        results, text = self.report(source="gold")
        self.assertEqual([r["question"] for r in results], list(self.QUESTIONS))
        self.assertTrue(all(r["ok"] for r in results), text)
        self.assertEqual([r["counts"] for r in results],
                         [{"chair": 2, "table": 1}, {"coat rack": 1}, {"lamp": 0, "chair": 2}])
        for question, prompts in self.QUESTIONS.items():
            self.assertIn(f"question: {question}", text)
            self.assertIn(f"prompts:  {', '.join(prompts)}  (full)", text)
        self.assertIn("no rows:  lamp (no gold class)", text)
        for result in results:
            objects_csv, sheet_json = map(Path, result["files"])
            self.assertTrue(objects_csv.is_file() and sheet_json.is_file())
            self.assertIn(str(objects_csv), text)
            self.assertFalse((objects_csv.parent / "cameras.csv").exists())
            self.assertEqual(json.loads((objects_csv.parent / "question.json").read_text())["prompts"],
                             result["prompts"])
        self.assertEqual(read_csv(results[0]["files"][0])[0].keys(), set(scene_sheet_report.GLOBAL_COLUMNS))
        self.assertIn("3 questions: 3 sheets OK (1 with a prompt that has no rows), 0 failed", text)

    def test_types_and_limit_choose_the_questions(self):
        results, _ = self.report(source="gold", types=["object_rel_distance"])
        self.assertEqual([r["id"] for r in results], [0, 2])
        results, _ = self.report(source="gold", limit=1)
        self.assertEqual([r["id"] for r in results], [0])

    def test_a_prompts_file_replaces_the_extractor(self):
        given = self.root / "prompts.jsonl"
        given.write_text("".join(json.dumps({"id": i, "prompts": p, "prompt_mode": "given"}) + "\n"
                                 for i, p in [(0, ["table"]), (2, ["chair", "lamp"])]))
        with mock.patch.object(question_scene_sheet, "extract_prompts", side_effect=AssertionError("extracted")):
            results, text = self.report(source="gold", prompts=scene_sheet_report.read_prompts(given))
        self.assertEqual([r["prompts"] for r in results], [["table"], None, ["chair", "lamp"]])
        self.assertEqual([r["ok"] for r in results], [True, False, True])
        self.assertIn("prompts:  table  (given)", text)
        self.assertIn("no prompts for question 1 in the prompts file", text)
        self.assertEqual(results[2]["counts"], {"chair": 2, "lamp": 0})

    def test_gold_only_index_from_gold_reports_and_a_questions_file(self):
        out = self.root / "out"
        (out / "reports").mkdir(parents=True)
        questions = self.root / "test.jsonl"
        extra = [{"id": 7, "dataset": "scannet", "scene_name": "s2", "question_type": "object_counting",
                  "question": "How many chair(s) are in this room?", "ground_truth": "1"}]
        self.write_questions(questions, "s1", extra)
        rows = [json.loads(line) for line in questions.read_text().splitlines()]
        for row in rows:
            row.setdefault("ground_truth", "2")
        questions.write_text("".join(json.dumps(r) + "\n" for r in rows))
        prompts = {"0": (["chair", "table"], "full"), "1": (["coat rack"], "full"), "2": (["lamp", "chair"], "full"),
                   "7": (["chair"], "full")}
        for scene in ("s1", "s2"):
            with open(out / "reports" / f"scannet__{scene}_gold.txt", "w") as f, \
                    contextlib.redirect_stderr(io.StringIO()):
                scene_sheet_report.report(scene, source="gold", questions=questions, work_dir=out / f"scannet__{scene}" / "gold",
                                          gold_dir=self.gold, out=f, prompts=prompts)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            sheet_index.main(["--gold-only", "--questions", str(questions), "--out", str(out)])
        index = read_csv(out / "index.csv")
        self.assertEqual([(r["id"], r["scene"]) for r in index],
                         [("0", "scannet/s1"), ("1", "scannet/s1"), ("2", "scannet/s1"), ("7", "scannet/s2")])
        self.assertEqual([r["prompts"] for r in index], ["chair, table", "coat rack", "lamp, chair", "chair"])
        self.assertTrue(all(r["gold_ok"] == "True" and r["pipeline_ok"] == "False" and r["pipeline_sheet"] == ""
                            for r in index))
        self.assertEqual(index[0]["gold_sheet"], "scannet__s1/gold/sheets/chair+table")
        self.assertTrue((out / index[3]["gold_sheet"] / "objects.csv").is_file())
        self.assertIn("gold rows of the asked class vs the true count, 1 counting questions:\n  exact 1",
                      printed.getvalue())
        for bad in ([], ["--gold-only"], ["--questions", str(questions)],
                    ["--gold-only", "--questions", str(questions), "--data-root", str(self.data)]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                sheet_index.main(bad + ["--out", str(out)])

    def test_gold_check_catches_a_wrong_sheet(self):
        info = scene_sheet_creator.load_gold_scene("scannet/s1", self.gold)[1]
        (result,), _ = self.report(source="gold", limit=1)
        objects_csv = Path(result["files"][0])
        with open(objects_csv, newline="") as f:
            header, *rows = list(csv.reader(f))

        def problems(new_rows, new_header=header, prompts=("chair", "table")):
            scene_sheet_creator.write_csv(objects_csv, new_header, new_rows)
            return scene_sheet_report.check_gold_sheet(objects_csv, list(prompts), info)[0]

        self.assertEqual(problems(rows), [])
        self.assertIn("1 rows for 'chair', the gold labels have 2", problems(rows[1:]))
        self.assertTrue(any("not one of the prompts" in p for p in problems(rows + [[3, "sofa", 1, 0, 0, 0, 1, 1, 1, 1]])))
        self.assertTrue(any("size is not" in p for p in problems([r[:-1] + ["9"] for r in rows])))
        self.assertTrue(any("columns" in p for p in problems([r + ["1"] for r in rows], header + ["num_frames"])))
        self.assertTrue(any("other prompts" in p for p in problems(rows, prompts=("chair", "table", "lamp"))))
        (objects_csv.parent / "cameras.csv").write_text("frame\n")
        self.assertIn("a cameras.csv next to a gold sheet", problems(rows))

    def test_pipeline_report_uses_the_default_settings_and_goes_on_after_a_failure(self):
        calls = []

        def create(scene, question, **options):
            calls.append((scene, question, options))
            if question == "Where is the coat rack?":
                sys.exit("[question] no dense depth for scannet/s1")
            return fake_pipeline_sheet(self.root / "sheets" / str(len(calls)), self.QUESTIONS[question])

        with mock.patch.object(question_scene_sheet, "create_scene_sheet", create):
            results, text = self.report()
        self.assertEqual([(s, q) for s, q, _ in calls], [("scannet/s1", q) for q in self.QUESTIONS])
        self.assertEqual(calls[0][2], {"data_root": self.data, "work_dir": self.root / "work"})  # nothing else set
        self.assertEqual([r["ok"] for r in results], [True, False, True])
        self.assertIn("FAILED: not made: [question] no dense depth for scannet/s1", text)
        self.assertEqual(results[2]["counts"], {"lamp": 1, "chair": 1})
        self.assertIn(str(self.root / "sheets" / "1" / "cameras.csv"), text)
        self.assertIn("3 questions: 2 sheets OK (0 with a prompt that has no rows), 1 failed", text)

    def test_pipeline_check_catches_an_inconsistent_sheet(self):
        prompts = ["chair", "table"]
        check = scene_sheet_report.check_pipeline_sheet
        objects_csv, cameras_csv = fake_pipeline_sheet(self.root / "ok", prompts)
        self.assertEqual(check(objects_csv, cameras_csv, prompts), ([], {"chair": 1, "table": 1}))
        cases = {
            "num_frames 3, 2 frame flags set": [[0, "chair", 1, 0, 0, 0, 1, 1, 1, 1, 3, 1, 0, 1]],
            "not one of the prompts": [[0, "sofa", 1, 0, 0, 0, 1, 1, 1, 1, 3, 1, 1, 1]],
            "not a finite number": [[0, "chair", 1, "nan", 0, 0, 1, 1, 1, 1, 3, 1, 1, 1]],
            "num_frames 0, 0 frame flags set": [[0, "chair", 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]],
            "object_id 5": [[5, "chair", 1, 0, 0, 0, 1, 1, 1, 1, 3, 1, 1, 1]],
        }
        for i, (expected, rows) in enumerate(cases.items()):
            objects_csv, cameras_csv = fake_pipeline_sheet(self.root / f"bad{i}", prompts, rows=rows)
            self.assertTrue(any(expected in p for p in check(objects_csv, cameras_csv, prompts)[0]), expected)
        objects_csv, cameras_csv = fake_pipeline_sheet(self.root / "frames", prompts)
        scene_sheet_creator.write_csv(cameras_csv, scene_sheet_report.CAMERA_COLUMNS, [[0] * 12])
        self.assertIn("3 frame columns in objects.csv, 1 frames in cameras.csv", check(objects_csv, cameras_csv, prompts)[0])
        cameras_csv.unlink()
        self.assertTrue(check(objects_csv, cameras_csv, prompts)[0][0].startswith("no cameras.csv"))

    def test_questions_come_from_the_export_or_a_file(self):
        scene, rows = scene_sheet_report.load_questions("s1", data_root=self.data)
        self.assertEqual((scene, len(rows)), ("scannet/s1", 3))
        other = self.root / "test.jsonl"
        self.write_questions(other, "s1", [{"id": 9, "dataset": "scannet", "scene_name": "s2", "question": "x"}])
        scene, rows = scene_sheet_report.load_questions("scannet/s2", "gold", self.data, other, self.gold)
        self.assertEqual((scene, [r["id"] for r in rows]), ("scannet/s2", [9]))  # not exported: gold only
        with self.assertRaises(ValueError):
            scene_sheet_report.load_questions("scannet/s2", "pipeline", self.data, other)

    def test_command_line_exit_status(self):
        base = ["s1", "--source", "gold", "--data-root", str(self.data), "--gold-dir", str(self.gold),
                "--work-dir", str(self.root / "cli")]
        with mock.patch.object(importlib.util, "find_spec", lambda name, *a: object()), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(scene_sheet_report.main(base), 0)
            with mock.patch.object(scene_sheet_report, "check_gold_sheet", lambda *a: (["broken"], {})):
                self.assertEqual(scene_sheet_report.main(base), 1)
        with mock.patch.object(importlib.util, "find_spec", lambda name, *a: None), \
                contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            scene_sheet_report.main(base)
        self.assertIn("spacy", err.getvalue())


def make_sheet_run(root, questions):
    """A finished sheet run in the layout of data/scene_sheets_75/: index.csv and one scene, scannet/s1, with its
    saved geometry and tracks (chair and table, each seen in all 3 frames), SAM 3's masks, and a pipeline and a gold
    sheet folder per question, each holding its question.json and a stale file. Returns the scene folder."""
    scene_dir = root / "scannet__s1"
    work = scene_dir / "subset-uniform16"
    scene_out = work / "scene"
    (scene_out / "tracks").mkdir(parents=True)
    (work / "masks" / "chair").mkdir(parents=True)
    (work / "masks" / "chair" / "manifest.json").write_text("{}")
    frames, h, w = 3, 12, 16
    np.savez(scene_out / "cameras.npz", extrinsic=np.tile(np.eye(3, 4), (frames, 1, 1)),
             intrinsic=np.tile([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]], (frames, 1, 1)),
             image_paths=np.array([f"/cluster/frames/{t:03d}.jpg" for t in range(frames)]))
    depth = 2 + 0.1 * np.random.default_rng(0).random((frames, h, w))
    np.savez_compressed(scene_out / "depth.npz", depth=depth.astype(np.float16), conf=np.ones((frames, h, w), np.float16))
    (scene_out / "adapter_report.json").write_text(json.dumps(
        {"scene": "scannet/s1", "selected_frames": [0, 4, 7], "depth_source": "dense", "image_hw": [h, w]}))
    for prompt, pixels in {"chair": np.s_[:, :6, :6], "table": np.s_[:, 6:, 8:]}.items():
        labels = np.zeros((frames, h, w), np.int32)
        labels[pixels] = 1  # 36 and 48 pixels a frame: seen, under question.json's min_points of 20
        np.savez(scene_out / "tracks" / f"{prompt}.npz", labels=labels)
    settings = {"subset-uniform16": {"tracking": {"conf_percentile": 55.0, "min_points": 20}, "size": True,
                                     "relative_positions": False, "size_trim": 0.02, "depth": "dense"},
                "gold": {"source": "gold"}}
    with open(root / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "scene", "question_type", "question", "options", "ground_truth", "prompts",
                         "pipeline_sheet", "gold_sheet", "pipeline_ok", "gold_ok"])
        for i, (question, prompts) in enumerate(questions.items()):
            name = question_scene_sheet.folder_name(prompts)
            for label, extra in settings.items():
                out = scene_dir / label / "sheets" / name
                out.mkdir(parents=True)
                (out / "question.json").write_text(json.dumps(
                    {"scene": "scannet/s1", "question": question, "prompts": prompts, "prompt_mode": "full", **extra},
                    indent=1))
                (out / "stale.txt").write_text("a file of an older sheet format")
            writer.writerow([i, "scannet/s1", "test", question, "", "", ", ".join(prompts),
                             f"scannet__s1/subset-uniform16/sheets/{name}", f"scannet__s1/gold/sheets/{name}", True, True])
    return scene_dir


class ReconstructPipeline(unittest.TestCase):
    """reconstruct_pipeline.py on a small sheet run: every sheets/ folder made again from scratch, and kept as it was
    when a sheet fails."""

    QUESTIONS = {"Is the chair near the table?": ["chair", "table"], "How many chairs are there?": ["chair"]}

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.gold = self.root / "meta_info"
        self.gold.mkdir()
        labels = {"s1": {"object_bbox": {"chair": [gold_box([1, 2, 0.5], [0.5, 0.6, 1]), gold_box([3, 1, 0.5], [0.5, 0.5, 1])],
                                          "table": [gold_box([2, 2, 0.4], [2, 1, 0.8])]}}}
        for dataset, (name, _) in scene_sheet_creator.GOLD_FILES.items():
            (self.gold / name).write_text(json.dumps(labels if dataset == "scannet" else {}))
        self.run_dir = self.root / "scene_sheets_75"
        self.scene_dir = make_sheet_run(self.run_dir, self.QUESTIONS)
        self.pipeline = self.scene_dir / "subset-uniform16" / "sheets"
        self.gold_sheets = self.scene_dir / "gold" / "sheets"

    def reconstruct(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = reconstruct_pipeline.main([str(self.run_dir), "--workers", "1", "--gold-dir", str(self.gold), *extra])
        return code, out.getvalue()

    def assert_no_leftovers(self):
        self.assertEqual(sorted(p.name for p in self.pipeline.parent.iterdir()), ["masks", "scene", "sheets"])
        self.assertEqual(sorted(p.name for p in self.gold_sheets.parent.iterdir()), ["sheets"])

    def test_every_sheet_folder_is_made_again_from_scratch(self):
        records = {path: path.read_bytes() for path in self.scene_dir.rglob("question.json")}
        masks = tree_digest(self.pipeline.parent / "masks")
        code, out = self.reconstruct()
        self.assertEqual(code, 0, out)
        self.assertIn("pipeline 2/2 sheets OK, gold 2/2 sheets OK", out)
        self.assertEqual({path: path.read_bytes() for path in records}, records)  # copied as they were
        for name in ("chair", "chair+table"):  # the stale files are gone
            self.assertEqual(sorted(p.name for p in (self.pipeline / name).iterdir()),
                             ["cameras.csv", "objects.csv", "question.json"])
            self.assertEqual(sorted(p.name for p in (self.gold_sheets / name).iterdir()),
                             ["objects.csv", "question.json", "sheet.json"])
        # num_frames 3 for both: made with question.json's min_points (20), not the default 300
        self.assertEqual([(r["concept"], r["num_frames"]) for r in read_csv(self.pipeline / "chair+table/objects.csv")],
                         [("chair", "3"), ("table", "3")])
        self.assertEqual([r["concept"] for r in read_csv(self.gold_sheets / "chair+table/objects.csv")],
                         ["chair", "chair", "table"])
        self.assertEqual(tree_digest(self.pipeline.parent / "masks"), masks)
        self.assert_no_leftovers()

    def test_a_sheet_that_fails_keeps_the_old_sheets(self):
        (self.scene_dir / "subset-uniform16/scene/tracks/table.npz").unlink()
        before = tree_digest(self.pipeline)
        code, out = self.reconstruct()
        self.assertEqual(code, 1)
        self.assertIn("no saved tracks for ['table'] (needs SAM 3)", out)
        self.assertIn("old sheets kept for ['scannet__s1 pipeline']", out)
        self.assertEqual(tree_digest(self.pipeline), before)
        self.assertFalse((self.gold_sheets / "chair/stale.txt").exists())  # the gold sheets were still made again
        self.assert_no_leftovers()

    def test_an_interrupted_run_is_put_right(self):
        self.pipeline.rename(self.pipeline.with_name("sheets.old"))  # stopped between the swap's two renames
        (self.pipeline.with_name("sheets.new") / "chair").mkdir(parents=True)
        (self.gold_sheets.with_name("sheets.new") / "chair").mkdir(parents=True)  # stopped while making them
        code, out = self.reconstruct()
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(p.name for p in self.pipeline.iterdir()), ["chair", "chair+table"])
        self.assertFalse((self.pipeline / "chair/stale.txt").exists())
        self.assert_no_leftovers()

    def test_every_sheet_the_index_names_must_be_there(self):
        shutil.rmtree(self.gold_sheets / "chair")
        code, out = self.reconstruct()
        self.assertEqual(code, 1)
        self.assertIn("index.csv names gold sheets that", out)
        self.assertEqual(sorted(p.name for p in self.gold_sheets.iterdir()), ["chair+table"])
        self.assertTrue((self.gold_sheets / "chair+table/stale.txt").exists())  # left as it was

    def test_command_line_errors(self):
        with self.assertRaises(SystemExit) as caught:
            self.reconstruct("--scenes", "scannet__s9")
        self.assertEqual(caught.exception.code, 2)
        shutil.rmtree(self.scene_dir)  # a scene folder index.csv names
        with self.assertRaises(SystemExit) as caught:
            self.reconstruct()
        self.assertEqual(caught.exception.code, 2)


def camera_in_gravity(heading, pitch):
    """Camera-to-world rotation, in east-north-up axes, of a camera with no roll facing heading (degrees clockwise
    from north) at pitch (degrees, up positive). Its columns are the OpenCV axes: x right, y down, z forward."""
    h, p = np.radians(heading), np.radians(pitch)
    forward = np.array([np.cos(p) * np.sin(h), np.cos(p) * np.cos(h), np.sin(p)])
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    return np.column_stack([right, np.cross(forward, right), forward])


def write_camera_sheet(path, steps, views, scale=0.3):
    """cameras.csv of a trajectory with frame 0 at the origin, given in east-north-up axes: the step to each later
    frame, and every frame's (heading, pitch). It is written as the pipeline writes it: in frame 0's camera axes, at
    an arbitrary scale."""
    first = camera_in_gravity(*views[0])
    positions = np.cumsum([[0, 0, 0]] + list(steps), axis=0)
    rows = []
    for t, (position, view) in enumerate(zip(positions, views)):
        rotation = first.T @ camera_in_gravity(*view)
        rows.append([t, f"{t}.jpg", *(scale * first.T @ position), *scene_sheet_creator.rotation_to_quaternion(rotation),
                     *rotation[:, 2]])
    scene_sheet_creator.write_csv(path, scene_sheet_report.CAMERA_COLUMNS, rows)
    return path


class CameraSheetText(unittest.TestCase):
    """camera_sheet_format_converter.py, on synthetic trajectories whose words are known."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_moves_and_turns(self):
        s = np.sqrt(0.5)
        steps = [(0, 1, 0), (s, s, 0), (-0.3, 0, 0), (3 * s, -3 * s, 0), (0.05, 0, 0), (0, 0, 1), (0, s, s),
                 (0, 0, -0.4)]  # median length 1
        views = [(0, -30), (45, -30), (45, -20), (15, -20), (15, -25), (15, -25), (-150, -25), (-120, -25),
                 (-119, -24)]  # frame 0 looks down, so the sheet's world axes are tilted
        sheet = write_camera_sheet(self.root / "cameras.csv", steps, views)
        self.assertEqual(camera_sheet_format_converter.frame_instructions(sheet), [
            "Start facing north, looking 30° down.",
            "Go north and turn 45° to the right.",
            "Go north-east and turn 10° up.",
            "Go a little west and turn 30° to the left.",
            "Go far south-east and turn 5° down.",
            "Stay still.",  # 0.05 median steps, no turn
            "Go up and turn 165° to the left.",
            "Go north and up, and turn 30° to the right.",
            "Go a little down.",  # turns under 2° are left out
        ])

    def test_paragraph_folder_and_command_line(self):
        write_camera_sheet(self.root / "cameras.csv", [(1, 0, 0)], [(0, 0), (-90, 0)])
        expected = "Start facing north, looking straight ahead. Go east and turn 90° to the left."
        self.assertEqual(camera_sheet_format_converter.describe_cameras(self.root), expected)  # the sheet folder
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(camera_sheet_format_converter.main([str(self.root / "cameras.csv")]), 0)
        self.assertEqual(out.getvalue().strip(), expected)

    def test_camera_that_never_turns_or_looks_straight_down(self):
        sheet = write_camera_sheet(self.root / "a.csv", [(0, 1, 0), (0, 1, 0)], [(0, 0)] * 3)
        self.assertEqual(camera_sheet_format_converter.describe_cameras(sheet),
                         "Start facing north, looking straight ahead. Go north. Go north.")
        sheet = write_camera_sheet(self.root / "b.csv", [], [(0, 10)])  # no horizon to fit: its own up is taken
        self.assertEqual(camera_sheet_format_converter.frame_instructions(sheet),
                         ["Start facing north, looking straight ahead."])
        sheet = write_camera_sheet(self.root / "c.csv", [(0, 0, 0)], [(0, -90), (90, -30)])
        self.assertEqual(camera_sheet_format_converter.frame_instructions(sheet),
                         ["Start facing north, looking 90° down.", "Turn 90° to the right and 60° up."])

    def test_not_a_camera_sheet(self):
        bad = self.root / "objects.csv"
        bad.write_text("object_id,concept,x,y,z\n")
        with self.assertRaises(ValueError):
            camera_sheet_format_converter.describe_cameras(bad)


if __name__ == "__main__":
    unittest.main()
