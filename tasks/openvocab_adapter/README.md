# OpenVocab-4D on Dario's VGGT reconstructions

This folder checks that OpenVocab-4D gives the same objects when it runs on Dario's saved VGGT reconstructions
(`data/vggt/`, exported in `tasks/vggt_data/`) as when it runs on its own VGGT.

For one scene (`scannet/scene0025_01` by default), it does the following:

1. It runs SAM 3 once on the scene's 128 frames and saves the masks.
2. It runs OpenVocab-4D's tracking and fusion twice, with those same masks:
   - **adapted**: on Dario's VGGT geometry, converted to OpenVocab-4D's format by `adapt_vggt.py`;
   - **native**: on OpenVocab-4D's own VGGT, run on the same frames.
3. It builds a scene sheet (`openvocab-4D-main/scene_sheet_creation/scene_sheet_creator.py`) for each run.
4. It compares the two runs and their scene sheets, and gives a verdict: `MATCHES`, `DIFFERS` or `INCOMPLETE`.

The frames and the masks are identical, so the geometry is the only difference. If the adapter is right, both runs
find the same objects in the same places.

Everything below is one Slurm job on the cluster. SAM 3 and VGGT need a GPU, and the dense depth that the adapted run
needs is only on the cluster.

Contents:

- [Before you start](#before-you-start)
- [Step 1: get the code](#step-1-get-the-code)
- [Step 2: set up the environment (once)](#step-2-set-up-the-environment-once)
- [Step 3: run the comparison](#step-3-run-the-comparison)
- [Step 4: send the results back](#step-4-send-the-results-back)
- [Step 5: read the results](#step-5-read-the-results)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)
- [Running the tests without a GPU](#running-the-tests-without-a-gpu)
- [Running the steps by hand](#running-the-steps-by-hand)
- [How the comparison works](#how-the-comparison-works)
- [Files](#files)

---

## Before you start

You need the following:

- **An account on the cluster** that can submit to the `frida` partition with the `povejmo` account. This is what
  `slurm/submit.sh` uses.
- **The project checkout on the cluster** that holds `data/vggt/`. The job reads
  `data/vggt/scannet/scene0025_01/depth.npy` and `depth_conf.npy` from it. To check:
  ```bash
  ls data/vggt/scannet/scene0025_01/depth.npy data/vggt/scannet/scene0025_01/depth_conf.npy
  ```
- **The project's `.venv`**, which already exists if you have run the project's other jobs. The job uses it only if it
  has to re-export the scene's frames.
- **Access to SAM 3's weights.** They are gated on Hugging Face:
  1. Accept the license at <https://huggingface.co/facebook/sam3> while logged in to your Hugging Face account. It
     can take a little while to be approved.
  2. Log in on the cluster, if you haven't already. The token is saved in your home folder, which the jobs can read.
     ```bash
     .venv/bin/hf auth whoami      # prints your user name if you are logged in
     .venv/bin/hf auth login       # otherwise: paste a token from https://huggingface.co/settings/tokens
     ```

The model weights (SAM 3 and VGGT-1B, a few GB in total) are downloaded the first time they are used. They go into
the group's shared Hugging Face cache, which `slurm/submit.sh` sets up.

## Step 1: get the code

From the project root on the cluster:

```bash
git fetch origin
git checkout vggt_output_adapter
git pull
```

## Step 2: set up the environment (once)

This makes a separate Python environment, `.venv-ov4d`, for OpenVocab-4D. The project's `.venv` is not changed.

**2a. Create the environment on the login node.** The container has no `ensurepip`, so this step can't run inside a
job:

```bash
/usr/bin/python3 -m venv .venv-ov4d
```

**2b. Install everything, as a job:**

```bash
slurm/submit.sh -J ov4d_setup -t 00:45:00 -- bash tasks/openvocab_adapter/hpc/setup_env.sh
```

`slurm/submit.sh` prints the job id. To follow the job:

```bash
squeue -u $USER                         # is it queued or running?
tail -f logs/ov4d_setup_<job id>.out    # the live log (Ctrl+C stops watching, not the job)
```

The setup does the following:

- clones SAM 3 and VGGT at the commits OpenVocab-4D pins, into `openvocab-4D-main/sam3` and
  `openvocab-4D-main/vggt` (both gitignored);
- installs PyTorch 2.14.1 for CUDA 13.0, the same build as the project's `.venv`. OpenVocab-4D's own `install.sh`
  uses CUDA 12.6, which does not run on the B200/B300 GPUs;
- installs `numpy<2`, which SAM 3 and VGGT need, and the pipeline's other packages;
- checks the result.

**2c. Check that it worked.** The log should end like this:

```
numpy 1.26.x | torch 2.14.1+cu130
GPU ok: NVIDIA B200            (or B300)
imports ok: sam3, vggt, cv2, rerun
Hugging Face: access to facebook/sam3 ok
setup done
```

If it doesn't, see [Troubleshooting](#troubleshooting). You can run the setup again safely, because it reuses what is
already there.

## Step 3: run the comparison

```bash
slurm/submit.sh -J ov4d_compare -t 02:00:00 -- bash tasks/openvocab_adapter/hpc/run_all.sh
```

To follow it, use `tail -f logs/ov4d_compare_<job id>.out`. The same log is also written to
`tasks/openvocab_adapter/results/scannet__scene0025_01/run.log`.

The job prints a `=== time step ===` line as each step starts:

| Step | What happens |
|---|---|
| `inputs` | Finds the scene's 128 frames (see below) and the dense depth. Records the GPU, the commit and `pip freeze` in `environment.txt` |
| `1/5 CPU tests` | The unit tests. They must end with `OK`, or the job stops |
| `2/5 SAM 3 masks` | SAM 3 on all 128 frames × 10 prompts. Skipped if a finished set of masks already exists |
| `3/5 <selection>: adapt Dario's VGGT` | Validates Dario's cameras and depth, then writes them in OpenVocab-4D's format. Every check line starts with `ok`. If one says `FAIL`, nothing is written and the job stops |
| `4/5 <selection>: OpenVocab-4D ...` | Tracking and fusion on the adapted geometry, then VGGT, tracking and fusion for the native run |
| `5/5 <selection>: scene sheets and comparison` | A scene sheet for each run, then `compare.py` |
| `summary` | The verdict for each selection |

Steps 3 to 5 are repeated for three frame selections:

| Selection | Frames | Why |
|---|---|---|
| `smoke5` | 5 nearby frames (118, 120, …, 126) | A quick check that everything runs |
| `eval32` | The `uniform32` subset from `meta.json` | The frames the group's experiments use |
| `all` | All 128 frames | The same frames Dario's reconstruction saw |

Where the frames come from: the job uses an existing export in `tasks/vggt_data/out/`. If there is none, it
re-exports this one scene from `data/vggt/` into `tasks/openvocab_adapter/out/export/`, using the project's `.venv`.
Your existing export is never changed.

It should take well under the 2-hour limit. Most of the time goes on SAM 3 and on the first download of the weights,
but this hasn't been timed on the cluster yet. If it times out, run one selection at a time (see [Settings](#settings)).
A rerun reuses the saved SAM 3 masks.

All large intermediate files (frames, depth, masks, tracks, point clouds) go to
`tasks/openvocab_adapter/out/`, which is gitignored.

## Step 4: send the results back

At the end, the job prints the command to commit the results:

```bash
git add tasks/openvocab_adapter/results/scannet__scene0025_01
git commit -m 'OpenVocab-4D comparison results for scannet/scene0025_01'
git push
```

This commits only small text files, a few hundred KB in total. Please don't commit anything from
`tasks/openvocab_adapter/out/`: it holds the ScanNet frames, and ScanNet's terms of use don't allow passing those on.

**If the job failed,** commit `results/.../run.log` anyway, because it shows where it stopped.

## Step 5: read the results

Get them with `git pull` on branch `vggt_output_adapter`. Everything is in
`tasks/openvocab_adapter/results/scannet__scene0025_01/`:

```
SUMMARY.txt                  start here: the verdict for each selection, and the numbers behind it
run.log                      the job's full output
environment.txt              date, machine, GPU, commit, settings, pip freeze
masks_manifest.json          how many objects SAM 3 found per frame and prompt
<selection>/                 smoke5, eval32 and all
├── comparison.json          every comparison with its status and values, plus the verdict
├── adapted/                 OpenVocab-4D on Dario's geometry
│   ├── objects.csv          the scene sheet: one row per object, with position, size and frames seen
│   ├── cameras.csv          the scene sheet: one row per frame, with camera position and orientation
│   ├── report.json          objects found per prompt
│   ├── tracks_summary.json  tracking statistics, and which masks were used
│   └── adapter_report.json  every validation check of Dario's data
└── native/                  OpenVocab-4D on its own VGGT (same files, plus align_report.json from VGGT)
```

### The verdict

`SUMMARY.txt` looks like this:

```
eval32: MATCHES (provisional pass marks)  {'PASS': 18, 'FAIL': 0, 'MEASURED': 7, 'BLOCKED': 0}
   ok   tracks: share of pixels with the same identity: 0.97 (needs >= 0.9)
   ok   objects: median 3D position error, share of scene size: 0.012 (needs <= 0.05)
   ok   scene sheet: share of rows found in both sheets: 0.91 (needs >= 0.8)
   ok   scene sheet: median position error, share of scene size: 0.015 (needs <= 0.05)
   ok   no FAIL in the comparison: [] (needs [])
```

The numbers above are only an illustration, not real results.

| Verdict | Meaning |
|---|---|
| `MATCHES` | Every pass mark below is met. OpenVocab-4D finds the same objects in the same places on Dario's geometry |
| `DIFFERS` | A pass mark is missed, or a hard check failed (see below) |
| `INCOMPLETE` | A step produced nothing to measure. The `needs` lines in `comparison.json` say what was missing |

| Criterion | Pass mark | What it measures |
|---|---|---|
| Pixels with the same track identity | at least 90% | The two runs group detections into the same objects |
| Median 3D object position error | at most 5% of the scene's size | Objects end up in the same place |
| Scene-sheet rows found in both sheets | at least 80% | The two scene sheets list the same objects |
| Median scene-sheet position error | at most 5% of the scene's size | The scene sheets give the same positions |

These pass marks are a first guess, and nobody has agreed on them yet. Revisit them with the group once real numbers
are in. They are `CRITERIA` at the top of `compare.py`.

### The status of each comparison

Each comparison in `comparison.json` has one of four statuses:

- `PASS` / `FAIL`: checks that have a right answer. Both runs used the same frames in the same order, the same masks
  (compared by fingerprint), and files in the pipeline's format. Any `FAIL` makes the verdict `DIFFERS`.
- `MEASURED`: a number with no right answer, such as how far apart the two reconstructions are.
- `BLOCKED`: could not run. `needs` names the missing input.

### Where to look when the verdict is DIFFERS

- `comparison.json` → `tracking consistency` → `per_prompt`: which prompts' tracks disagree.
- `3D object positions after alignment` → `per_object`: the error for each object.
- `scene sheet: objects` → `adapted_only` / `native_only`: rows that appear in only one sheet.
- Compare `adapted/objects.csv` with `native/objects.csv` directly. Their coordinates are in different world frames
  and scales, so compare the objects found and their `num_frames`, not the raw x, y, z.

## Settings

Set these in front of the submit command. `slurm/submit.sh` passes your environment on to the job.

```bash
RUNS=eval32 slurm/submit.sh -J ov4d_compare -t 01:00:00 -- bash tasks/openvocab_adapter/hpc/run_all.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `RUNS` | `smoke5 eval32 all` | Which frame selections to run |
| `SCENE` | `scannet/scene0025_01` | `<dataset>/<scene>`. It must have `data/vggt/<scene>/` |
| `PROMPTS` | `chair,door,fan,keyboard,monitor,pillow,sofa,table,telephone,window` | Comma-separated objects to look for. The default is the objects named in scene0025_01's VSI-Bench questions. Change it when you change `SCENE`. Delete `out/<scene>/masks/` when you change the prompts |
| `REID` | `0` | `1` turns on DINOv2 re-identification in both runs |
| `PY` | `.venv-ov4d/bin/python` | The Python to use |
| `DENSE` | `data/vggt/<scene>` | The folder with `depth.npy` and `depth_conf.npy` |
| `DATA` | found automatically | The export folder that holds `scenes/<scene>/` |
| `WORK`, `RESULTS` | `out/<name>`, `results/<name>` | Where the intermediate files and the results go |

Results for another scene go to their own folder, `results/<dataset>__<scene>/`.

## Troubleshooting

| You see | Do this |
|---|---|
| `no .venv-ov4d/bin/python` | Do step 2a first, on the login node |
| `no git in this container` | Run the two `git clone` commands it prints on the login node, then submit the setup again |
| `no CUDA GPU visible` / `no CUDA GPU here` | Run through `slurm/submit.sh`, not directly on the login node |
| `no kernel image is available for execution on the device` | This PyTorch build doesn't support the GPU. Check that `.venv-ov4d/bin/pip show torch` says `2.14.1+cu130` |
| numpy fails to install (building from source) | The container's Python is too new for numpy<2. Run the setup again with `NUMPY_SPEC=numpy slurm/submit.sh ...` |
| `no access to facebook/sam3`, `401`, `gated repo` | Accept the license on Hugging Face (see [Before you start](#before-you-start)) and check `hf auth whoami` |
| The tests in step 1/5 fail | The environment differs from the one the tests expect. Send `run.log` |
| `no dense depth in data/vggt/...` | Check the paths in [Before you start](#before-you-start), or set `DENSE=` |
| `[adapter] FAIL ...` and `nothing written` | Dario's export and `data/vggt/` disagree, for example frames from another run. The line names the check. Send `run.log` |
| `masks are (...) the image is (...)` | SAM 3 returned masks at another size. Send `run.log` |
| `already holds a finished run` | Old masks are in the way. Delete `tasks/openvocab_adapter/out/<scene>/masks/` to run SAM 3 again |
| The job hits the time limit | Run one selection at a time with `RUNS=...`. The masks are kept, so SAM 3 doesn't run again |
| CUDA out of memory in the VGGT stage | Unlikely on a B200, but if it happens, run `RUNS="smoke5 eval32"` first and send the log |

## Running the tests without a GPU

The tests need only numpy, Pillow and a CPU build of torch. They run the real milestone scripts and the scene sheet
creator on a small synthetic scene, with SAM 3, VGGT, `cv2` and `rerun` replaced by small fakes. On Windows, from the
repository root:

```powershell
py -m unittest discover -s tasks/openvocab_adapter/tests
```

On Linux, use `python -m unittest discover -s tasks/openvocab_adapter/tests`.

They should end with `OK`. A few tests are skipped where they don't apply, for example the tests for a machine without
a GPU when one is present.

## Running the steps by hand

You don't need this for the cluster job. It's useful for running one step on its own, or another frame selection.
The commands are for PowerShell, run from the repository root. On Linux, use `python` instead of `py`.

```powershell
$scene   = 'tasks/openvocab_adapter/out/scannet__scene0025_01'
$prompts = 'chair,door,fan,keyboard,monitor,pillow,sofa,table,telephone,window'
$frames  = 'tasks/vggt_data/out/vggt_vsibench_75/scenes/scannet/scene0025_01/frames'
```

1. **SAM 3 masks (GPU).** `--dry-run` checks the inputs without loading SAM 3.
   ```powershell
   py tasks/openvocab_adapter/sam3_infer.py --images $frames --frames all --prompts $prompts --out "$scene/masks" --overlays
   ```
2. **Adapt Dario's geometry (CPU).** `--dense-dir` is the folder with `depth.npy` and `depth_conf.npy`. Without it,
   only the cameras are written, and tracking can't run.
   ```powershell
   py tasks/openvocab_adapter/adapt_vggt.py --preset eval32 --dense-dir <dense folder> --out "$scene/eval32/adapted" --overwrite
   ```
   Other selections: `--frames uniform:16 --label u16`, `--frames list:0,40,80 --label three`,
   `--frames window:START:COUNT[:STRIDE]`, or `--frames subset:<name from meta.json>`.
3. **Track and fuse on the adapted geometry (CPU is enough with saved masks).**
   ```powershell
   py openvocab-4D-main/trackA/milestone4_video.py --images "$scene/eval32/adapted/frames" --out "$scene/eval32/adapted" `
      --prompts $prompts --step 1 --stages track,fuse --masks-dir "$scene/masks" --no-reid
   ```
4. **The native run (GPU, for VGGT).** These are the same frames, written to `native/`, with the `vggt` stage added:
   ```powershell
   py openvocab-4D-main/trackA/milestone4_video.py --images "$scene/eval32/adapted/frames" --out "$scene/eval32/native" `
      --prompts $prompts --step 1 --stages vggt,track,fuse --masks-dir "$scene/masks" --no-reid
   ```
5. **Scene sheets:**
   ```powershell
   foreach ($side in 'adapted','native') {
     py openvocab-4D-main/scene_sheet_creation/scene_sheet_creator.py --scene "$scene/eval32/$side" --prompts $prompts --size
   }
   ```
6. **Compare.** This writes `$scene/eval32/comparison.json`. It runs at any stage, and anything that can't run yet is
   reported as `BLOCKED`, along with what it needs.
   ```powershell
   py tasks/openvocab_adapter/compare.py --run-dir "$scene/eval32" --masks "$scene/masks"
   ```

Steps 3 to 5 need `opencv-python`, `rerun-sdk`, `numpy<2` and the VGGT repository code in `openvocab-4D-main/vggt`.
Use a separate environment, not your system Python. Use the same `--no-reid` setting in both runs.

## How the comparison works

- **The two runs can't agree exactly.** Dario's poses and depth come from one VGGT pass over all 128 frames. The native
  run's come from VGGT on only the selected frames, in chunks of 40. Each has its own world frame and scale, and VGGT's
  output really does change with the frames it sees.
- **Alignment.** Before any positions are compared, a similarity transform (rotation, translation and scale) is fitted
  between the two reconstructions. It uses pixels that are confident in both depth maps, and no object information, so
  the object errors that follow aren't circular. Errors are given as fractions of the scene's size.
- **Tracks** are matched on their pixels, because both runs use the same masks. This needs no alignment.
- **Scene-sheet rows** are paired through the same track matching, because a row's `track_id` is the track's id.
  Sizes are not compared: they are extents along the world axes, and the two world frames are rotated relative to each
  other.
- **Before writing anything, the adapter checks Dario's data:**
  - the camera convention (rotations are proper rotations, and the world is frame 0's camera);
  - that every patch point projects into its own 14×14 patch;
  - that the point-cloud colours match the JPEGs (the right frames, in the right order);
  - that the dense depth reproduces the exported patch points (the right scene and the right frames).

## Files

| File | Purpose |
|---|---|
| `hpc/setup_env.sh` | Step 2: installs `.venv-ov4d` and checks it |
| `hpc/run_all.sh` | Step 3: the whole comparison as one job |
| `adapt_vggt.py` | Converts Dario's cameras and dense depth to OpenVocab-4D's `frames/`, `cameras.npz` and `depth.npz`, after validating them |
| `sam3_infer.py` | Runs SAM 3 on a list of frames and saves the masks |
| `compare.py` | Compares an adapted run with a native run, including their scene sheets, and gives the verdict |
| `common.py` | Paths, frame-selection presets and small helpers |
| `tests/` | CPU-only tests |
| `results/` | The small result files committed from the cluster |
| `../../openvocab-4D-main/trackA/saved_masks.py` | Lets milestones 3 and 4 reuse saved masks (`--masks-dir`) instead of running SAM 3 |
