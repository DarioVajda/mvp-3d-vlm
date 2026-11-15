# OpenVocab-4D on Dario's VGGT reconstructions

This folder runs OpenVocab-4D's tracking and fusion on the saved VGGT reconstructions in
`tasks/vggt_data/out/vggt_vsibench_75/`, and compares the result with OpenVocab-4D on its own VGGT. Both runs use the
same frames and the same saved SAM 3 masks, so the only thing that differs is the geometry. Both runs also get a
scene sheet (`openvocab-4D-main/scene_sheet_creation/scene_sheet_creator.py`), and the two sheets are compared too.

| File | Purpose | Needs |
|---|---|---|
| `hpc/setup_env.sh`, `hpc/run_all.sh` | The whole comparison as one Slurm job ([HPC_RUNBOOK.md](HPC_RUNBOOK.md)) | the cluster |
| `adapt_vggt.py` | Writes `frames/`, `cameras.npz` and `depth.npz` in the pipeline's format, after validating them | numpy, Pillow |
| `sam3_infer.py` | Saves SAM 3 masks for a list of frames | NVIDIA GPU, SAM 3 |
| `compare.py` | Compares an adapted run with a native run, including their scene sheets, and gives a verdict | numpy |
| `tests/` | CPU-only tests | numpy, Pillow, CPU torch |
| `results/` | Small result files committed from the cluster | |

The saved-mask loader is `openvocab-4D-main/trackA/saved_masks.py`, used through `--masks-dir` on milestones 3 and 4.

## The usual route: one job on the cluster

SAM 3 and VGGT need a GPU, and the dense depth (`data/vggt/<dataset>/<scene>/depth.npy`) is only on the cluster. So
the comparison runs there as one job, described in [HPC_RUNBOOK.md](HPC_RUNBOOK.md):

```bash
slurm/submit.sh -J ov4d_compare -t 02:00:00 -- bash tasks/openvocab_adapter/hpc/run_all.sh
```

The job commits its small outputs to `results/<dataset>__<scene>/`:

```
SUMMARY.txt                  verdict per frame selection
run.log, environment.txt     the full log, the GPU, pip freeze and commit
masks_manifest.json          SAM 3 detections per frame and prompt
<selection>/                 smoke5, eval32, all
├── comparison.json          every comparison, the verdict and its criteria
├── adapted/                 adapter_report.json, report.json, tracks_summary.json, objects.csv, cameras.csv
└── native/                  align_report.json, report.json, tracks_summary.json, objects.csv, cameras.csv
```

`objects.csv` and `cameras.csv` are each run's scene sheet.

## Running it by hand

Commands are written for PowerShell, from the repository root. Generated files go to `out/`, which is gitignored.

1. **Inputs.** Get `depth.npy` and `depth_conf.npy` from `data/vggt/scannet/scene0025_01/` and pass their folder as
   `--dense-dir`. Unpack SAM 3 masks (from `sam3_infer.py`) to `out/scannet__scene0025_01/masks/`.
2. **Adapt.**
   `py tasks/openvocab_adapter/adapt_vggt.py --preset eval32 --overwrite --dense-dir <folder>`. Other selections
   work the same way, for example `--frames uniform:16 --label u16` or `--frames list:0,40,80 --label three`.
3. **Track and fuse** on `out/scannet__scene0025_01/eval32/adapted`, then build its scene sheet:
   ```powershell
   $scene = 'tasks/openvocab_adapter/out/scannet__scene0025_01'
   $prompts = 'chair,door,fan,keyboard,monitor,pillow,sofa,table,telephone,window'
   py openvocab-4D-main/trackA/milestone4_video.py --images "$scene/eval32/adapted/frames" `
      --out "$scene/eval32/adapted" --prompts $prompts --step 1 --stages track,fuse --masks-dir "$scene/masks" --no-reid
   py openvocab-4D-main/scene_sheet_creation/scene_sheet_creator.py --scene "$scene/eval32/adapted" --prompts $prompts --size
   ```
   This needs `opencv-python`, `rerun-sdk` and the VGGT repository (code only, no weights), in an environment with
   `numpy<2`. It needs no GPU.
4. **Native reference.** This is the same as step 3 into `eval32/native` with `--stages vggt,track,fuse`, and it
   needs a GPU.
5. **Compare.** `py tasks/openvocab_adapter/compare.py --run-dir "$scene/eval32" --masks "$scene/masks"`. You can run
   this at any stage: anything that cannot run yet is listed as `BLOCKED`, along with what it needs.

Some notes:

- `--no-reid` is the default in both runs. Re-identification needs a GPU, and the two runs must use the same settings.
- Milestone 4 discards objects seen in fewer than 3 frames, so `smoke5` can only keep objects in at least 3 of its 5
  frames.

## How the comparison works

- Adapted poses come from one joint 128-frame VGGT pass. Native poses come from VGGT on the selected frames only, in
  chunks of 40. They have different world frames and scales, and VGGT's output really does change with the frames it
  sees, so the two can never agree exactly.
- A Sim(3) is fitted on pixels that are confident in both depth maps, then applied before positions are compared. It
  uses no object information. Errors are given as fractions of the scene's size.
- Tracks are matched on their pixels, which needs no alignment. Scene-sheet rows are paired through the same track
  matching, because a sheet's `track_id` is the track's id.
- Scene-sheet sizes are not compared. They are extents along the world axes, and the two world frames are rotated
  relative to each other.
- `PASS` and `FAIL` are used only where there is a right answer: file formats, the same frames, identical masks.
  Geometric numbers are `MEASURED`.
- **Verdict.** The report ends in `MATCHES`, `DIFFERS` or `INCOMPLETE`, judged against `CRITERIA` in `compare.py`:

  | Criterion | Pass mark |
  |---|---|
  | Pixels whose tracks are matched to each other | at least 90% |
  | Median 3D object position error | at most 5% of the scene's size |
  | Scene-sheet rows that appear in both sheets | at least 80% |
  | Median scene-sheet position error | at most 5% of the scene's size |

  These pass marks are provisional and nobody has agreed on them yet. Revisit them once the first real results are in.

## What the adapter validates

The adapter writes nothing unless every check passes.

- **Camera convention:** rotations are orthonormal with determinant +1, the world is frame 0's camera, the saved
  centres and forward axes agree with `[R|t]`, and the intrinsics are pixel pinholes centred on the image.
- **Convention and alignment together:** every patch point must project into its own 14×14 patch on at least 99% of
  patches, in every frame.
- **Frame alignment:** there is one correctly sized JPEG per frame, and point-cloud colours must match the selected
  JPEGs.
- **Dense depth:** shape, float16 precision, finite values and confidence of at least 1. Unprojecting the dense depth
  must also reproduce Dario's exported patch points, which catches the wrong scene or frames in the wrong order.

## Tests

```powershell
py -m unittest discover -s tasks/openvocab_adapter/tests
```

The tests run the real milestone scripts and the scene sheet creator on a small synthetic scene. `cv2`, `rerun` and
`vggt.utils.geometry` are replaced by small stand-ins, and SAM 3 by a fake that returns fixed rectangles. They need no
GPU and no model weights. The cluster job runs them first.
