"""CPU-only tests of the adapter, the saved-mask path and the comparison. No model is run or downloaded.

The milestone scripts are run for real, on a small synthetic scene. Three things they import are replaced by small
stand-ins written to a temporary folder: cv2, rerun, and vggt.utils.geometry (a numpy version of the same formulas).
SAM 3 is replaced by a fake that returns fixed rectangles.

  py -m unittest discover -s tasks/openvocab_adapter/tests -v
"""
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADAPTER))
import adapt_vggt  # noqa: E402
import common  # noqa: E402
import compare  # noqa: E402

sys.path.insert(0, str(common.TRACK_A))
sys.path.insert(0, str(common.REPO / "openvocab-4D-main" / "scene_sheet_creation"))
import scene_sheet_creator  # noqa: E402
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
    """A scene in the exact layout of Dario's export. Returns (scene_dir, data_root, dense_dir)."""
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


@unittest.skipUnless((common.DATA_ROOT / "scenes" / common.DEFAULT_SCENE / "meta.json").is_file(), "Dario's data is absent")
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
        return subprocess.run([sys.executable, str(SAM3_INFER), "--images", str(self.scene / "frames"), "--prompts", PROMPTS,
                               "--out", str(self.root / "masks"), *extra], env=env, capture_output=True, text=True)

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
        self.assertNotIn("sam3", done.stderr.replace("[sam3]", "").replace("sam3_infer", "").replace("tasks/openvocab_adapter/README", ""))

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
                               "--out", str(cls.masks)], env=cls.env(gpu=True), capture_output=True, text=True)
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

    def run_native(self, name, scale, rotation, translation):
        native = self.root / name / "native"
        sim3_copy(self.adapted, native, scale, rotation, translation)
        done = self.milestone4(native, "--masks-dir", str(self.masks))
        self.assertEqual(done.returncode, 0, done.stderr)
        return native

    @staticmethod
    def build_sheets(*runs):
        for run in runs:
            scene_sheet_creator.build_scene_sheet(run, ["chair", "table"], include_size=True, conf_percentile=10,
                                                  min_points=20)

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


if __name__ == "__main__":
    unittest.main()
