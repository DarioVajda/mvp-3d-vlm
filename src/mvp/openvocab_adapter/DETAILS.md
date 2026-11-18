# OpenVocab-4D comparison: details

This is the reference for the test described in [README.md](README.md). The README explains what is being tested and
how to run it. This file covers everything else.

The code is in `src/mvp/openvocab_adapter/`, and its outputs go to `tasks/openvocab_adapter/`. Below, `out/` and
`results/` are the folders there: `tasks/openvocab_adapter/out/` (large intermediate files, gitignored) and
`tasks/openvocab_adapter/results/` (the small files that are committed). Every command runs from the repository root.

Contents:

- [Before you start](#before-you-start)
- [What start.sh runs](#what-startsh-runs)
- [What each scene's job does](#what-each-scenes-job-does)
- [Frame selections](#frame-selections)
- [The scenes and prompts](#the-scenes-and-prompts)
- [Running one scene, or other settings](#running-one-scene-or-other-settings)
- [Reading the results](#reading-the-results)
- [How the comparison works](#how-the-comparison-works)
- [Troubleshooting](#troubleshooting)
- [Running the tests without a GPU](#running-the-tests-without-a-gpu)
- [Running the steps by hand](#running-the-steps-by-hand)
- [A scene sheet for one question](#a-scene-sheet-for-one-question)
- [Files](#files)

## Before you start

You need the following:

- **An account on the cluster** that can submit to the `frida` partition with the `povejmo` account. This is what
  `slurm/submit.sh` uses.
- **The project checkout on the cluster** that holds `data/vggt/`. The jobs read
  `data/vggt/<dataset>/<scene>/depth.npy` and `depth_conf.npy` from it.
- **The project's `.venv`.** It is used only to check the Hugging Face login, and to re-export a scene's frames if
  `tasks/vggt_data/out/` doesn't have them.
- **Access to SAM 3's weights.** They are gated: accept the license at <https://huggingface.co/facebook/sam3>, then
  check with `.venv/bin/hf auth whoami`, or log in with `.venv/bin/hf auth login`. The token is saved in your home
  folder, which the jobs can read.

The model weights (SAM 3 and VGGT-1B, a few GB in total) are downloaded the first time they are used. They go into
the group's shared Hugging Face cache, which `slurm/submit.sh` sets up.

## What start.sh runs

`hpc/start.sh` runs these steps on the login node. You can also run them one by one:

1. **Checks the inputs:** the depth maps of every scene in `hpc/scenes.tsv`, and the Hugging Face access (a warning
   only).
2. **Creates the environment**, if it is missing: `/usr/bin/python3 -m venv .venv-ov4d`. This has to happen on the
   login node, because the container has no `ensurepip`.
3. **Submits the setup job**, unless an earlier one finished (`SETUP=1` forces it):
   `slurm/submit.sh -J ov4d_setup -t 00:45:00 -- bash src/mvp/openvocab_adapter/hpc/setup_env.sh`. The setup:
   - clones SAM 3 and VGGT at the commits OpenVocab-4D pins, into `openvocab-4D-main/sam3` and
     `openvocab-4D-main/vggt` (both gitignored);
   - installs PyTorch 2.14.1 for CUDA 13.0, the same build as the project's `.venv`. OpenVocab-4D's own
     `install.sh` uses CUDA 12.6, which doesn't run on the B200/B300 GPUs;
   - installs `numpy<2`, which SAM 3 and VGGT need, and the pipeline's other packages;
   - checks the result. Its log, `logs/ov4d/ov4d_setup_<id>.out`, should end with `GPU ok`, `imports ok`,
     `access to facebook/sam3 ok` and `setup done`.
4. **Submits the scenes:** `bash src/mvp/openvocab_adapter/hpc/submit_scenes.sh`. This submits one `hpc/run_all.sh`
   job per scene, all running in parallel and each waiting for the setup job. It also submits an `ov4d_summary` job,
   which waits for every scene and then writes `results/SUMMARY.md`.

## What each scene's job does

`hpc/run_all.sh` prints a `=== time step ===` line as each step starts. The same log is saved to
`results/<dataset>__<scene>/run.log`.

| Step | What happens |
|---|---|
| `inputs` | Finds the scene's frames and depth maps. Records the GPU, the commit and `pip freeze` in `environment.txt` |
| `1/5 CPU tests` | The unit tests. They must end with `OK`, or the job stops |
| `2/5 SAM 3 masks` | SAM 3 on the frames the selections need × the scene's prompts. Skipped if saved masks already cover them |
| `3/5 <selection>: adapt the saved VGGT outputs` | Validates the saved cameras and depth, then writes them in OpenVocab-4D's format. If any check says `FAIL`, nothing is written and the job stops |
| `4/5 <selection>: OpenVocab-4D ...` | Tracking and fusion on the adapted geometry, then VGGT, tracking and fusion for the native run |
| `5/5 <selection>: scene sheets and comparison` | A scene sheet for each run (`scene_sheet_creator.py --size`), then `compare.py` |
| `summary` | The verdict for each selection |

The frames come from an existing export in `tasks/vggt_data/out/`. If there is none, the job re-exports that one
scene from `data/vggt/` into `tasks/openvocab_adapter/out/export/`, using `.venv`, and leaves your export unchanged.
All large intermediate files (frames, depth, masks, tracks, point clouds) go to `tasks/openvocab_adapter/out/`, which
is gitignored.

## Frame selections

| Selection | Frames | Why |
|---|---|---|
| `half` (default) | 64 frames: 0, 2, 4, …, 126 | Half the cost of all 128 |
| `smoke5` | 5 nearby frames (118, 120, …, 126) | A quick check that everything runs |
| `eval32` | The `uniform32` subset from `meta.json` | The frames the group's experiments use |
| `all` | All 128 frames | The same frames the saved VGGT outputs were made from |

**Using every other frame doesn't weaken the test.** Both runs always see exactly the same frames, and the adapter
converts the saved VGGT outputs the same way whichever frames are picked. Two things do change slightly:

- OpenVocab-4D's own VGGT sees 64 frames instead of 128, so it may disagree a little more with the saved 128-frame
  reconstruction. That makes the test slightly stricter.
- Tracking jumps about a second between frames instead of half a second, which it handles well.

For a spot check on all 128 frames, rerun one scene with `RUNS=all`.

SAM 3 segments only the frames the chosen selections need. Its masks are kept, and reused by later runs unless those
need frames or prompts that the saved masks lack.

## The scenes and prompts

`hpc/scenes.tsv` has one scene per line: `<dataset>/<scene>`, a tab, then comma-separated SAM 3 prompts.
`hpc/make_scene_list.py` writes it:

- The scenes are scannet/scene0025_01, then the first landscape scenes of each dataset in name order: 4 ScanNet,
  3 ScanNet++ and 3 ARKitScenes.
- Portrait ARKitScenes videos are left out. OpenVocab-4D's own VGGT loader resizes portrait frames to a different
  shape than the saved VGGT outputs, so the two runs' depth maps couldn't be compared pixel by pixel.
- The prompts are the objects named in the scene's VSI-Bench questions, read from the question templates. The most
  asked-about objects come first, and there are at most 12. Route-planning questions are skipped.

To change the scenes, edit `scenes.tsv` directly, or rerun
`python src/mvp/openvocab_adapter/hpc/make_scene_list.py --per-dataset scannet=2 scannetpp=2 arkitscenes=1`.

## Running one scene, or other settings

To run one scene, as a job:

```bash
slurm/submit.sh -J ov4d_compare -t 01:00:00 -- bash src/mvp/openvocab_adapter/hpc/run_all.sh
```

Settings go in front of the command, because `slurm/submit.sh` passes your environment on to the job. For example:

```bash
RUNS=all bash src/mvp/openvocab_adapter/hpc/start.sh
SCENE=scannet/scene0046_01 PROMPTS="lamp,door,bed" slurm/submit.sh -J ov4d_compare -t 01:00:00 -- bash src/mvp/openvocab_adapter/hpc/run_all.sh
```

For `start.sh` and `submit_scenes.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `RUNS` | `half` | The frame selections to run for every scene. Several run one after the other, e.g. `"half all"` |
| `TIME` | `01:00:00` | The time limit of each scene's job |
| `SCENES_FILE` | `hpc/scenes.tsv` | The scenes and their prompts |
| `SETUP` | `0` | `1` runs the setup job again (`start.sh` only) |

For `run_all.sh` (one scene):

| Variable | Default | Meaning |
|---|---|---|
| `RUNS` | `half` | Which frame selections to run |
| `SCENE` | `scannet/scene0025_01` | `<dataset>/<scene>`. It must have `data/vggt/<scene>/` |
| `PROMPTS` | scene0025_01's 10 objects | Comma-separated objects to look for. Change it when you change `SCENE` |
| `REID` | `0` | `1` turns on DINOv2 re-identification in both runs |
| `PY` | `.venv-ov4d/bin/python` | The Python to use |
| `DENSE` | `data/vggt/<scene>` | The folder with `depth.npy` and `depth_conf.npy` |
| `DATA` | found automatically | The export folder that holds `scenes/<scene>/` |
| `WORK`, `RESULTS` | `tasks/openvocab_adapter/out/<name>`, `tasks/openvocab_adapter/results/<name>` | Where the intermediate files and the results go |

## Reading the results

Start with `results/SUMMARY.md`, which has one row per scene. A scene whose job failed still gets a row, saying
`no result`, with the last line of its log. If the summary job didn't run, write the summary with
`.venv-ov4d/bin/python src/mvp/openvocab_adapter/summarize.py`.

Each scene has its own folder, `results/<dataset>__<scene>/`:

```
SUMMARY.txt                  the verdict for each selection, and the numbers behind it
run.log                      the job's full output
environment.txt              date, machine, GPU, commit, settings, pip freeze
masks_manifest.json          how many objects SAM 3 found per frame and prompt
<selection>/                 half (and any other selection that was run)
├── comparison.json          every comparison with its status and values, plus the verdict
├── adapted/                 OpenVocab-4D on the saved VGGT outputs
│   ├── objects.csv          the scene sheet: one row per object, with position, size and frames seen
│   ├── cameras.csv          the scene sheet: one row per frame, with camera position and orientation
│   ├── report.json          objects found per prompt
│   ├── tracks_summary.json  tracking statistics, and which masks were used
│   └── adapter_report.json  every validation check of the saved VGGT outputs
└── native/                  OpenVocab-4D on its own VGGT (same files, plus align_report.json from VGGT)
```

Each comparison in `comparison.json` has one of four statuses:

- `PASS` / `FAIL`: checks that have a right answer. Both runs used the same frames in the same order, the same masks
  (compared by fingerprint), and files in the pipeline's format. Any `FAIL` makes the verdict `DIFFERS`.
- `MEASURED`: a number with no right answer, such as how far apart the two reconstructions are. The four numbers
  behind the verdict are `MEASURED`.
- `BLOCKED`: could not run. `needs` names the missing input, and the verdict is then `INCOMPLETE`.

The pass marks are `CRITERIA` at the top of `compare.py`.

When a scene says `DIFFERS`, look here:

- `comparison.json` → `tracking consistency` → `per_prompt`: which prompts' tracks disagree.
- `3D object positions after alignment` → `per_object`: the error for each object.
- `scene sheet: objects` → `adapted_only` / `native_only`: rows that appear in only one sheet.
- Compare `adapted/objects.csv` with `native/objects.csv` directly. Their coordinates are in different world frames
  and scales, so compare the objects found and their `num_frames`, not the raw x, y, z.

## How the comparison works

- **Why the runs differ at all.** The saved poses and depth come from one VGGT pass over all 128 frames. The native
  run's come from VGGT on only the selected frames, in chunks of 40. Each has its own world frame and scale, and VGGT's
  output really does change with the frames it sees.
- **Alignment.** Before any positions are compared, a similarity transform (rotation, translation and scale) is fitted
  between the two reconstructions. It uses pixels that are confident in both depth maps, and no object information, so
  the object errors that follow aren't circular. Errors are given as fractions of the scene's size.
- **Tracks** are matched on their pixels, because both runs use the same masks. This needs no alignment.
- **Scene-sheet rows** are paired through the same track matching, because a row's `track_id` is the track's id.
  Sizes are not compared: they are extents along the world axes, and the two world frames are rotated relative to each
  other.
- **The adapter checks the saved VGGT outputs before writing anything:**
  - the camera convention (rotations are proper rotations, and the world is frame 0's camera);
  - that every patch point projects into its own 14×14 patch;
  - that the point-cloud colours match the JPEGs (the right frames, in the right order);
  - that the dense depth reproduces the exported patch points (the right scene and the right frames).

## Troubleshooting

| You see | Do this |
|---|---|
| `missing: data/vggt/...` from `start.sh` | Run it from the checkout that has `data/vggt/`, or remove those scenes from `scenes.tsv` |
| `no .venv-ov4d/bin/python` | Create it on the login node: `/usr/bin/python3 -m venv .venv-ov4d` (`start.sh` does this) |
| `no git in this container` | Run the two `git clone` commands it prints on the login node, then run `start.sh` again |
| `no CUDA GPU visible` / `no CUDA GPU here` | Run through `slurm/submit.sh`, not directly on the login node |
| `no kernel image is available for execution on the device` | This PyTorch build doesn't support the GPU. Check that `.venv-ov4d/bin/pip show torch` says `2.14.1+cu130` |
| numpy fails to install (building from source) | The container's Python is too new for numpy<2. Run `NUMPY_SPEC=numpy SETUP=1 bash src/mvp/openvocab_adapter/hpc/start.sh` |
| `no access to facebook/sam3`, `401`, `gated repo` | Accept the license on Hugging Face and check `.venv/bin/hf auth whoami` |
| The tests in step 1/5 fail | The environment differs from the one the tests expect. Send `run.log` |
| `[adapter] FAIL ...` and `nothing written` | The exported frames and `data/vggt/` disagree, for example frames from another run. The line names the check. Send `run.log` |
| `masks are (...) the image is (...)` | SAM 3 returned masks at another size. Send `run.log` |
| A job hits its time limit | Rerun with a longer limit, e.g. `TIME=02:00:00 bash src/mvp/openvocab_adapter/hpc/start.sh`. Saved masks are reused |
| A scene says `no result` in `SUMMARY.md` | Its job stopped before comparing. The row shows the last line of its log, and `results/<scene>/run.log` has the rest |
| CUDA out of memory in the VGGT stage | Unlikely on a B200, but if it happens, run with `RUNS=smoke5` and send the log |

## Running the tests without a GPU

The tests need only numpy, Pillow and torch (a CPU build is enough), plus spaCy, its `en_core_web_md` model and NLTK
for the prompt extractor's tests; all are in the repository's `requirements.txt`. They run the real milestone scripts
and the scene sheet creator on a small synthetic scene, with SAM 3, VGGT, `cv2` and `rerun` replaced by small fakes.
From any folder, on Linux or in Git Bash on Windows:

```bash
bash src/mvp/openvocab_adapter/run_tests.sh             # uses .venv; PYTHON=... for another; extra arguments go to unittest
```

It checks the packages first, then runs `python -m unittest discover -s src/mvp/openvocab_adapter/tests -v` from the
repository root with the venv's Python. The tests start the pipeline scripts with that same Python, so on Windows don't
run them with `py`, which is usually another Python. They should end with `OK`. A few tests are skipped where they don't apply, for example the tests for a machine without
a GPU when one is present.

## Running the steps by hand

You don't need this for the cluster job. It's useful for running one step on its own. The commands are for
PowerShell, run from the repository root. On Linux, use `python` instead of `py`.

```powershell
$scene   = 'tasks/openvocab_adapter/out/scannet__scene0025_01'
$prompts = 'chair,door,fan,keyboard,monitor,pillow,sofa,table,telephone,window'
$frames  = 'tasks/vggt_data/out/vggt_vsibench_75/scenes/scannet/scene0025_01/frames'
```

1. **SAM 3 masks (GPU).** `--dry-run` checks the inputs without loading SAM 3.
   ```powershell
   py src/mvp/openvocab_adapter/sam3_infer.py --images $frames --frames all --prompts $prompts --out "$scene/masks" --overlays
   ```
2. **Adapt the saved VGGT outputs (CPU).** `--dense-dir` is the folder with `depth.npy` and `depth_conf.npy`. Without it,
   only the cameras are written, and tracking can't run, unless `--patch-depth` approximates the depth from the patch
   points (not for the comparison).
   ```powershell
   py src/mvp/openvocab_adapter/adapt_vggt.py --preset half --dense-dir <dense folder> --out "$scene/half/adapted" --overwrite
   ```
   Other selections: `--frames uniform:16 --label u16`, `--frames list:0,40,80 --label three`,
   `--frames window:START:COUNT[:STRIDE]`, or `--frames subset:<name from meta.json>`.
3. **Track and fuse on the adapted geometry (CPU is enough with saved masks):**
   ```powershell
   py openvocab-4D-main/trackA/milestone4_video.py --images "$scene/half/adapted/frames" --out "$scene/half/adapted" `
      --prompts $prompts --step 1 --stages track,fuse --masks-dir "$scene/masks" --no-reid
   ```
4. **The native run (GPU, for VGGT).** These are the same frames, written to `native/`, with the `vggt` stage added:
   ```powershell
   py openvocab-4D-main/trackA/milestone4_video.py --images "$scene/half/adapted/frames" --out "$scene/half/native" `
      --prompts $prompts --step 1 --stages vggt,track,fuse --masks-dir "$scene/masks" --no-reid
   ```
5. **Scene sheets:**
   ```powershell
   foreach ($side in 'adapted','native') {
     py src/mvp/openvocab_adapter/scene_sheet_creator.py --scene "$scene/half/$side" --prompts $prompts --size
   }
   ```
6. **Compare.** This writes `$scene/half/comparison.json`. It runs at any stage, and anything that can't run yet is
   reported as `BLOCKED`, along with what it needs.
   ```powershell
   py src/mvp/openvocab_adapter/compare.py --run-dir "$scene/half" --masks "$scene/masks"
   ```

Steps 3 to 5 need `opencv-python`, `rerun-sdk`, `numpy<2` and the VGGT repository code in `openvocab-4D-main/vggt`.
Use a separate environment, not your system Python. Use the same `--no-reid` setting in both runs.

## A scene sheet for one question

`question_scene_sheet.py` takes one scene of the VGGT export and one question, and writes the scene sheet of the
objects the question names. It is separate from the comparison above and uses only the saved VGGT outputs.

```bash
slurm/submit.sh -J scene_sheet -t 00:30:00 -- .venv-ov4d/bin/python src/mvp/openvocab_adapter/question_scene_sheet.py \
    scannet/scene0025_01 "Which of these objects (table, chair, door, telephone) is the closest to the fan?"
```

It prints exactly two lines on stdout, the path of `objects.csv` and then of `cameras.csv`. Every log goes to stderr.
(A cluster job writes both to the same file, `logs/scene_sheet_<job id>.out`, so there the paths are its last two
lines.) From Python, `question_scene_sheet.create_scene_sheet(scene, question)` returns the two paths.

The question is one string and doesn't have to be one of the scene's `questions.jsonl`. The scene is
`<dataset>/<scene>`, or the scene's name alone.

| Step | What happens |
|---|---|
| Prompts | `prompt_object_extractor.py` picks the SAM 3 prompts: the physical things the question names. If it finds none, its syntactic mode is tried instead (every noun phrase). If that finds none either, `objects.csv` has only its header row, and `cameras.csv` is written as usual |
| Geometry | `adapt_vggt.py` validates the saved cameras and dense depth and writes them in OpenVocab-4D's format. The dense depth is read from `data/vggt/<dataset>/<scene>/` (or `--dense-dir`, or `$DENSE_ROOT`). Without it the script stops, unless `--patch-depth-fallback` is given (see below) |
| SAM 3 | `sam3_infer.py` segments every prompt in every selected frame |
| Tracking | OpenVocab-4D's track stage links the masks into objects, with DINOv2 re-identification. The fuse stage is not run, because the scene sheet only needs the tracks |
| Sheet | `scene_sheet_creator.py` writes the two CSVs, with the size columns and one 0/1 column per frame |

**Defaults that differ from the comparison:** 16 frames uniformly sampled from the 128 (`subset:uniform16` of
`meta.json`, the frames the VLM sees with `--sampling uniform128`; the comparison uses 64), and re-identification on
(the comparison ran with `--no-reid`). So a sheet from this script is not expected to equal the comparison's sheet for
the same scene. `--frames half --no-reid` gives the comparison's settings, and `--frames all` uses all 128 frames.

The track stage drops tracks seen in too few frames, with a minimum that scales linearly with the number of frames:
1 of 16, 2 of 64, 4 of 128 (`min_track_frames` in `milestone4_video.py`, which records the value it used in
`tracks/summary.json`; `--min-track-frames N` there fixes it). With the default 16 frames nothing is dropped for being
short, so an object only briefly in view still gets a row, and so can a one-frame false detection or a fragment of an
object that tracking failed to link. Tracks made before this rule are made again from the saved masks.

**What is kept and reused.** Everything goes to `tasks/openvocab_adapter/out/<dataset>__<scene>/subset-uniform16/`
(`subset-uniform16` is the frame selection):

```
scene/              the converted geometry (frames/, cameras.npz, depth.npz) and tracks/<prompt>.npz
masks/<prompts>/    the SAM 3 masks of the prompts that were segmented together
sheets/<prompts>/   objects.csv, cameras.csv and question.json (the question, its prompts and the settings)
```

Tracks are independent between prompts. A later question about the same scene therefore runs SAM 3 and tracking only
for prompts that have no tracks yet, and a question that names only known objects just rebuilds the sheet. Two
questions that name the same objects, in the same order, share one sheet folder. Tracks made with other settings
(re-identification, SAM 3 backend, thresholds) are made again from the saved masks. `--fresh` discards the scene's
geometry, masks and tracks first. With `--frames all`, the work goes to `out/<dataset>__<scene>/all/`, which `RUNS=all`
of the comparison deletes too, since it uses the same folder.

Options (`--help` lists all of them):

| Option | Default | Meaning |
|---|---|---|
| `--frames` | `subset:uniform16` | 16 uniformly sampled frames of the 128, the VLM's. A preset (`half`, `eval32`, `smoke5`, `all`) or a selection such as `uniform:32`. `frame_<t>` in the sheet counts the selected frames, and `cameras.csv` names each one's image |
| `--dense-dir` | `data/vggt/<dataset>/<scene>` | The folder with `depth.npy` and `depth_conf.npy` |
| `--patch-depth-fallback` | off | Without the dense depth, approximate it from the export's patch points instead of stopping (also in `scene_sheet_report.py`) |
| `--out-dir` | `<work>/sheets/<prompts>` | Where the two CSVs go |
| `--work-dir` | `out/<dataset>__<scene>/<frames>` | Where geometry, masks and tracks are kept |
| `--no-reid` | off | Track without DINOv2 re-identification |
| `--syntactic` | off | Use every noun phrase as a prompt from the start |
| `--no-size`, `--relative-positions` | off | The sheet's columns, as in `scene_sheet_creator.py` |
| `--backend` | `transformers` | How SAM 3 is run, as in `sam3_infer.py` |

Things to know:

- **It needs the environment from `hpc/setup_env.sh`**, which now also installs spaCy, its `en_core_web_md` model and
  NLTK for the prompt extractor. An environment set up before that needs the setup job once more:
  `slurm/submit.sh -J ov4d_setup -t 00:45:00 -- bash src/mvp/openvocab_adapter/hpc/setup_env.sh`.
- **Without `data/vggt/`, `--patch-depth-fallback` makes a sheet from the exported scene alone.** Each 14×14 patch gets
  the camera depth of its patch point (the median of VGGT's points in that patch) and its mean confidence
  (`adapt_vggt.py --patch-depth`). A warning is printed on every run, `question.json` records `"depth": "patch_points"`,
  and `adapter_report.json` records `"depth_source": "patch_points"`. Object outlines are blocky at that resolution, so
  positions are coarser and sizes can take in a strip of background; on the synthetic test scene, centres move by
  under 5% of the object's size and volumes are about 10% smaller. Once the dense depth is found, the scene is
  converted and tracked again with it (the SAM 3 masks are kept). The comparison jobs in `hpc/` always use the dense
  depth.
- **Re-identification loads DINOv2 through `torch.hub`**, which downloads it the first time. If the job has no
  network access and no cached copy, pass `--no-reid`.
- **A question with an apostrophe can't be passed through `slurm/submit.sh`**, because of how it quotes the command.
  The route-planning questions have them (`'turn left'`). Put the command in a small shell script and submit that.
- **The syntactic fallback gives prompts that are not objects**, such as `distance` or `total length`. SAM 3 usually
  finds nothing for them, so the sheet has no rows, but it can also return something unrelated.

To see which prompts a question gives, and why, without running anything else:

```bash
python src/mvp/openvocab_adapter/prompt_object_extractor.py --explain "How far is the red car from the stove?"
python src/mvp/openvocab_adapter/prompt_extractor_report.py --per-dataset 1    # every question of a few scenes
```

## A scene sheet from the gold labels

`scene_sheet_creator.py --source gold` writes the perfect sheet from VSI-Bench's ground-truth boxes
([`thinking-in-space`](https://github.com/vision-x-nyu/thinking-in-space) `data/meta_info/`, commit `ba7a6b2`,
downloaded to `data/raw/VSI-Bench/meta_info/` the first time). It needs no GPU:

```bash
python src/mvp/openvocab_adapter/scene_sheet_creator.py --source gold --scene scannet/scene0025_01     --prompts "table,chair,fan" --size --out-dir out/gold_sheet
```

The columns are the pipeline sheet's up to `size`, in metres and in the dataset's frame (z up). There are no per-frame
columns and no `cameras.csv`, because the saved VGGT cameras can't be aligned to the gold labels; asking for them is
an error. Pass `--no-frames --no-cameras` for a pipeline sheet with the same columns.

## Files

| File | Purpose |
|---|---|
| `question_scene_sheet.py` | One scene and one question in, the paths of its scene sheet out: prompts, SAM 3, the saved VGGT outputs, tracking, sheet |
| `scene_sheet_creator.py` | Writes the scene sheet (`objects.csv`, `cameras.csv`) of a scene that has tracks, or from VSI-Bench's ground truth (`--source gold`) |
| `camera_sheet_format_converter.py` | Turns a `cameras.csv` into one sentence per frame, with an up direction it estimates. `--mode global`: compass moves and absolute facings ("Go north-east and face 30° clockwise from north, looking 20° down."). `--mode relative`: moves and turns relative to the previous frame ("Move forward and to the left, and turn 23° to the right and 5° down.") |
| `scene_sheet_report.py` | Makes and checks the sheet of every question of a scene (pipeline or gold), and prints it all for checking by eye |
| `prompt_object_extractor.py` | Picks the SAM 3 prompts from a question |
| `prompt_extractor_report.py` | Prints the extractor's decisions on the questions of a few exported scenes, for checking it by eye |
| `hpc/start.sh` | The one command: checks the inputs, then submits the setup, every scene and the summary |
| `hpc/setup_env.sh` | Installs `.venv-ov4d` and checks it (a job) |
| `hpc/submit_scenes.sh` | Submits one `run_all.sh` job per scene in `scenes.tsv`, then the summary job |
| `hpc/scenes.tsv` | The 10 scenes and their SAM 3 prompts |
| `hpc/make_scene_list.py` | Writes `scenes.tsv`: picks the scenes and reads each one's objects from its questions |
| `hpc/run_all.sh` | The whole comparison for one scene, as one job |
| `summarize.py` | Writes `results/SUMMARY.md` from every scene's `comparison.json` |
| `adapt_vggt.py` | Converts the saved VGGT cameras and dense depth to OpenVocab-4D's `frames/`, `cameras.npz` and `depth.npz`, after validating them |
| `sam3_infer.py` | Runs SAM 3 on a list of frames and saves the masks |
| `compare.py` | Compares an adapted run with a native run, including their scene sheets, and gives the verdict |
| `common.py` | Paths, frame-selection presets and small helpers |
| `tests/` | CPU-only tests |
| `run_tests.sh` | Checks the packages the tests need, then runs them with the repository's `.venv` |
| `tasks/openvocab_adapter/results/` | The small result files committed from the cluster |
| `tasks/openvocab_adapter/out/` | The large intermediate files (gitignored) |
| `openvocab-4D-main/trackA/saved_masks.py` | Lets milestones 3 and 4 reuse saved masks (`--masks-dir`) instead of running SAM 3 |
