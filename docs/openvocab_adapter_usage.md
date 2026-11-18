# Running the scene-sheet pipeline (`src/mvp/openvocab_adapter`)

This guide is for running the code in `src/mvp/openvocab_adapter/`. That code turns a question about a VSI-Bench video
into a **scene sheet**: a table of the objects the question names, with their 3D positions and sizes. The design, and
how it differs from the first proposal, is in [scene_sheet_plan.md](scene_sheet_plan.md). The test that checks the
saved VGGT outputs work in OpenVocab-4D is described in more depth in the folder's
[README.md](../src/mvp/openvocab_adapter/README.md) and [DETAILS.md](../src/mvp/openvocab_adapter/DETAILS.md).

Every command runs from the repository root.

Contents:

- [What the pipeline does](#what-the-pipeline-does)
- [Before you start](#before-you-start)
- [Example: a scene sheet for one question](#example-a-scene-sheet-for-one-question)
- [Frame selections](#frame-selections)
- [The scripts and their settings](#the-scripts-and-their-settings)
- [Troubleshooting](#troubleshooting)

## What the pipeline does

The folder has two uses:

| Use | Entry point | Output |
|---|---|---|
| **A scene sheet for one question** (the main one) | `question_scene_sheet.py <scene> "<question>"` | `objects.csv` and `cameras.csv` for that question |
| **A test of the VGGT adapter** | `hpc/start.sh` | `tasks/openvocab_adapter/results/SUMMARY.md`: the saved VGGT outputs against OpenVocab-4D's own VGGT, on 10 scenes |

A scene sheet is made in five steps, each done by one script. `question_scene_sheet.py` runs them in order:

| Step | Script | What it does |
|---|---|---|
| 1. Prompts | `prompt_object_extractor.py` | Picks the SAM 3 prompts: the physical objects the question names |
| 2. Geometry | `adapt_vggt.py` | Checks the saved VGGT cameras and depth, and converts them to OpenVocab-4D's format |
| 3. Masks | `sam3_infer.py` | Runs SAM 3 on every frame for each prompt, and saves the masks |
| 4. Tracks | `openvocab-4D-main/trackA/milestone4_video.py` | Links the masks across frames into objects, in 3D |
| 5. Sheet | `scene_sheet_creator.py` | Writes `objects.csv` and `cameras.csv` from the tracks |

`scene_sheet_creator.py --source gold` skips all five steps. It writes the **perfect sheet** from VSI-Bench's
ground-truth boxes, which needs no GPU.

## Before you start

### What needs what

| Task | GPU | Dense depth (`data/vggt/`) | Python |
|---|---|---|---|
| Prompts only (`prompt_object_extractor.py`) | no | no | spaCy, `en_core_web_md`, NLTK |
| Gold sheet (`scene_sheet_creator.py --source gold`) | no | no | numpy |
| Pipeline sheet (`question_scene_sheet.py`) | yes, for SAM 3 and DINOv2 | yes, or `--patch-depth-fallback` | the OpenVocab-4D environment, below |
| Tracking from saved masks with `--no-reid` | no | yes, or `--patch-depth-fallback` | the OpenVocab-4D environment |
| The adapter test (`hpc/start.sh`) | yes, 5 by default | yes | `.venv-ov4d`, made by the script |
| Unit tests (`run_tests.sh`) | no | no | the project's `.venv` |

### The environment

- **On the cluster:** use `.venv-ov4d`. `bash src/mvp/openvocab_adapter/hpc/start.sh` creates it and submits its setup
  job. To run only the setup:
  ```bash
  /usr/bin/python3 -m venv .venv-ov4d          # on the login node
  slurm/submit.sh -J ov4d_setup -t 00:45:00 -- bash src/mvp/openvocab_adapter/hpc/setup_env.sh
  ```
  The setup clones SAM 3 and VGGT at OpenVocab-4D's pinned commits, into `openvocab-4D-main/sam3` and
  `openvocab-4D-main/vggt`. It installs PyTorch for CUDA 13.0, `numpy<2`, spaCy, `en_core_web_md` and NLTK. Its log
  should end with `setup done`.
- **Locally:** use the project's `.venv`. `requirements.txt` has spaCy, `en_core_web_md`, NLTK, `rerun-sdk` and OpenCV.
  The VGGT code comes from the clone, installed without its dependencies:
  ```bash
  uv pip install --python .venv -r requirements.txt
  uv pip install --python .venv --no-deps -e openvocab-4D-main/vggt
  ```
  WordNet is downloaded through NLTK the first time the prompt extractor runs.
- **SAM 3's weights:** on the cluster, `sam3_infer.py` loads them from the group's cached snapshot of
  `facebook/sam3`, which needs no login. Elsewhere the weights are gated: accept the license at
  <https://huggingface.co/facebook/sam3>, then run `hf auth login`.
- **DINOv2** (re-identification) is downloaded through `torch.hub` the first time it is used. A job with no network
  and no cached copy needs `--no-reid`.

### Inputs

- **The exported scene**, `tasks/vggt_data/scenes/<dataset>/<scene>/`: frames, cameras, patch points, `meta.json` and
  `questions.jsonl`. The scripts look for it in `tasks/vggt_data/`, then `tasks/vggt_data/out/vggt_vsibench_75/`, then
  `tasks/vggt_data/out/`, then `tasks/openvocab_adapter/out/export/`. `--data-root` gives another folder.
- **The dense depth**, `data/vggt/<dataset>/<scene>/depth.npy` and `depth_conf.npy`, is only on the cluster.
  `--dense-dir` or `$DENSE_ROOT` gives another place. Without it, `--patch-depth-fallback` approximates the depth from
  the exported patch points, at one depth per 14×14 patch. That is coarser: positions shift slightly, and sizes can
  take in some background.
- **The gold labels** are downloaded to `data/raw/VSI-Bench/meta_info/` the first time a gold sheet is made. They come
  from thinking-in-space at commit `ba7a6b2`, and are checked by SHA-256.

### Where outputs go

All of it goes under `tasks/openvocab_adapter/`:

```text
out/                                  large intermediate files (gitignored)
└── <dataset>__<scene>/
    ├── <frames>/                     one folder per frame selection, "subset-uniform16" by default
    │   ├── scene/                    frames/, cameras.npz, depth.npz, adapter_report.json, tracks/<prompt>.npz
    │   ├── masks/<prompts>/          SAM 3 masks of the prompts segmented together
    │   └── sheets/<prompts>/         objects.csv, cameras.csv, question.json   <- the scene sheet
    └── gold/sheets/<prompts>/        objects.csv, sheet.json, question.json (from scene_sheet_report.py --source gold)
results/                              the adapter test's small files (committed)
```

`<prompts>` is the prompts joined with `+`, for example `table+chair+door+telephone+fan`.

## Example: a scene sheet for one question

The scene is `scannet/scene0025_01`, and the question is number 3137 in its `questions.jsonl`. It is a
relative-distance question:

> Measuring from the closest point of each object, which of these objects (table, chair, door, telephone) is the
> closest to the fan?

### 1. Check the prompts (CPU, optional)

```bash
.venv/Scripts/python.exe src/mvp/openvocab_adapter/prompt_object_extractor.py --explain \
    "Measuring from the closest point of each object, which of these objects (table, chair, door, telephone) is the closest to the fan?"
```

(On Linux, use `.venv/bin/python`.)

```text
  drop  point                        main sense is cognition, physical senses rare (2% of uses)
  drop  object                       too generic
  drop  object                       too generic
  keep  table                        has a physical sense (artifact, food, object), kept to be safe
  keep  chair                        main sense is physical (artifact)
  keep  door                         main sense is physical (artifact)
  keep  telephone                    main sense is physical (artifact)
  keep  fan                          main sense is physical (artifact)
  -> table, chair, door, telephone, fan
```

### 2. Make the sheet

**On the cluster**, which has the dense depth:

```bash
slurm/submit.sh -J scene_sheet -t 00:30:00 -- .venv-ov4d/bin/python src/mvp/openvocab_adapter/question_scene_sheet.py \
    scannet/scene0025_01 "Measuring from the closest point of each object, which of these objects (table, chair, door, telephone) is the closest to the fan?"
```

**On a local GPU**, which has no `data/vggt/`, so the depth must be approximated:

```powershell
.venv\Scripts\python.exe src/mvp/openvocab_adapter/question_scene_sheet.py scannet/scene0025_01 `
    "Measuring from the closest point of each object, which of these objects (table, chair, door, telephone) is the closest to the fan?" `
    --patch-depth-fallback
```

The script prints exactly two lines on stdout, the paths of `objects.csv` and `cameras.csv`. Every log goes to stderr.
In a cluster job both go to `logs/scene_sheet_<job id>.out`, so the paths are its last two lines:

```text
tasks/openvocab_adapter/out/scannet__scene0025_01/subset-uniform16/sheets/table+chair+door+telephone+fan/objects.csv
tasks/openvocab_adapter/out/scannet__scene0025_01/subset-uniform16/sheets/table+chair+door+telephone+fan/cameras.csv
```

By default the sheet uses the 16 frames the VLM sees with `--sampling uniform128`: 16 frames uniformly sampled from the
128 (`subset:uniform16` of `meta.json`). To use all 128 frames, add `--frames all`. That selection gets its own work
folder, `out/scannet__scene0025_01/all/`. Tracking keeps an object seen in enough of the selected frames, a minimum
that scales linearly with their number: 1 of 16, 4 of 128. With 16 frames, a one-frame detection therefore gets a
row too, including a false one or a fragment of an object that tracking didn't link.

### 3. Read the sheet

`objects.csv` has one row per object. This is the file of a local run with `--frames all`, with its 128 `frame_<t>`
columns cut (the default run has 16):

```text
object_id,concept,track_id,x,y,z,size_x,size_y,size_z,size,num_frames
0,table,1,0.607989,0.165686,0.823529,0.326722,0.374568,0.252647,0.0309188,10
1,table,2,0.729091,0.222638,0.441652,0.705032,0.327505,0.713551,0.16476,7
...
11,door,1,-0.346897,-0.178234,0.783728,0.385473,1.04383,0.766152,0.308277,33
12,door,6,-0.498932,-0.397644,0.516914,0.34796,0.72016,0.656619,0.16454,6
13,telephone,1,1.0656,0.492754,-0.285694,0.0912513,0.060178,0.0928881,0.000510078,8
```

The numbers are in VGGT's units, in the first camera's frame (OpenCV axes: x right, y down, z forward). `cameras.csv`
has one row per frame, with the camera's centre, orientation and viewing direction. `question.json` records the
prompts and settings, and `"depth": "patch_points"` or `"dense"`. The columns are described in
[scene_sheet_plan.md](scene_sheet_plan.md#scene-sheet-format).

### 4. Compare with the perfect sheet (CPU)

```bash
.venv/Scripts/python.exe src/mvp/openvocab_adapter/scene_sheet_creator.py --source gold --scene scannet/scene0025_01 \
    --prompts "table,chair,door,telephone,fan" --size \
    --out-dir tasks/openvocab_adapter/out/scannet__scene0025_01/gold/sheets/table+chair+door+telephone+fan
```

```text
object_id,concept,track_id,x,y,z,size_x,size_y,size_z,size
0,table,1,0.886143,0.324196,-0.582737,3.13297,2.027,0.872624,5.54161
1,table,2,1.18117,-0.748299,-0.395667,2.50078,2.23334,1.24102,6.93125
2,chair,1,0.915205,-1.33816,-0.566614,1.3365,1.29523,1.13499,1.96475
3,chair,2,1.08988,0.715364,-0.465222,0.758704,0.555743,1.02553,0.432408
4,door,1,-2.35493,0.453892,0.0330963,0.40896,1.17544,2.1944,1.05486
5,telephone,1,1.64108,0.00681922,-0.0872051,0.33143,0.349815,0.19677,0.0228133
6,fan,1,-1.64329,-0.455992,-0.51139,0.414052,0.376798,0.980983,0.153047
```

The gold sheet is in metres, with z up. The pipeline's local run listed 7 tables, 4 chairs, 2 doors, 1 telephone and
no fan, against 2 tables, 2 chairs, 1 door, 1 telephone and 1 fan here. The two sheets' coordinates can't be compared
directly, because their units and frames differ.

### 5. Ask another question about the same scene

Run `question_scene_sheet.py` again with the new question. The geometry is reused, and SAM 3 and tracking run only
for prompts that have no tracks yet. A question that names only objects already tracked just rebuilds the sheet.
Tracks are made again from the saved masks when the settings change (re-identification, SAM 3 backend or
thresholds). `--fresh` starts the scene over.

### 6. Every question of the scene at once

```bash
.venv/Scripts/python.exe src/mvp/openvocab_adapter/scene_sheet_report.py scannet/scene0025_01 --source gold   # CPU
slurm/submit.sh -J sheet_report -t 02:00:00 -- .venv-ov4d/bin/python src/mvp/openvocab_adapter/scene_sheet_report.py \
    scannet/scene0025_01                                                                                      # pipeline, GPU
```

For pipeline sheets, the report first runs SAM 3 and tracking once for the prompts of all the scene's questions, so the
models load once per scene; each question then only builds its sheet. The sheets are the same as from one question at
a time, because SAM 3 and the tracker handle each prompt on its own. The report prints, for each question, its
prompts, `OK` or `FAILED` for its sheet, the prompts with no rows, and the files. It exits with status 1 if any sheet
failed.

### 7. Every question of every exported scene

```bash
slurm/submit.sh -J sheets -t 02:00:00 --gres gpu:1 -c 48 --mem 256G -- bash src/mvp/openvocab_adapter/hpc/run_sheets.sh
```

Runs the report of step 6, pipeline and gold, on 8 scenes at a time on one GPU (the 75 scenes take about 15 minutes
on a B200 or B300), skipping scenes already done. Then `sheet_index.py` writes `out/index.csv`, which gives each
question's prompts and the folders of its two sheets, and compares the object counts of the counting questions with
the true answers, and zips it all to `tasks/openvocab_adapter/scene_sheets_75.zip` (about 70 MB). The script's header
lists its settings. [scene_sheets_75.md](scene_sheets_75.md), copied into the zip as its README, explains the output:
how to find a question's sheets, the columns, the coordinate frames and the results.

### From Python

```python
import sys

sys.path.insert(0, "src/mvp/openvocab_adapter")
import question_scene_sheet

objects_csv, cameras_csv = question_scene_sheet.create_scene_sheet(
    "scannet/scene0025_01", question, frames="subset:uniform16", reid=True
)
```

The keyword arguments are the command-line options with underscores (`patch_depth_fallback=True`, `include_size=False`,
and so on). Errors stop the function with `SystemExit` and a message that starts with `[question]`. Catch it if you
call the function in a loop.

## Frame selections

Every script that takes `--frames` or `--preset` uses the same selections (`common.py`). Indices count the 128
frames of the saved VGGT pass.

| Selection | Frames |
|---|---|
| `all` | All 128 |
| `half` (preset) | `window:0:64:2`: 0, 2, 4, …, 126 |
| `eval32` (preset) | `subset:uniform32` |
| `smoke5` (preset) | `window:118:5:2`: 118, 120, …, 126 |
| `uniform:K` | `np.linspace(0, 127, K)`, rounded down |
| `subset:NAME` | A subset in `meta.json`: `uniform16`, `uniform32`, `space_aware16`, `space_aware32` |
| `window:START:COUNT[:STRIDE]` | `COUNT` frames from `START`, `STRIDE` apart (default 1) |
| `list:a,b,c` | These frames. They must be increasing |

In the sheet, `frame_<t>` counts the selected frames, not the original indices. `cameras.csv` names each frame's
image.

## The scripts and their settings

Every Python script lists its options with `--help`. The defaults below are the code's.

### `question_scene_sheet.py`: one question to its scene sheet

Runs the five steps for one scene and one question, and keeps the work for the next question. Arguments:
`<dataset>/<scene>` (or the scene name alone) and the question, as one string. The question doesn't have to come from
`questions.jsonl`.

| Option | Default | Meaning |
|---|---|---|
| `--frames` | `subset:uniform16` | 16 uniformly sampled frames of the 128, the VLM's. A preset or selection (see [Frame selections](#frame-selections)) |
| `--dense-dir` | `$DENSE_ROOT` or `data/vggt`, then `/<dataset>/<scene>` | The folder with `depth.npy` and `depth_conf.npy` |
| `--patch-depth-fallback` | off | Without dense depth, approximate it from the patch points (with a warning) instead of stopping |
| `--data-root` | searched (see [Inputs](#inputs)) | The folder holding the exported `scenes/` |
| `--out-dir` | `<work-dir>/sheets/<prompts>` | Where the CSVs and `question.json` go |
| `--work-dir` | `tasks/openvocab_adapter/out/<dataset>__<scene>/<frames>` | Where geometry, masks and tracks are kept |
| `--fresh` | off | Delete the scene's kept geometry, masks and tracks first |
| `--syntactic` | off | Use every noun phrase as a prompt, without WordNet or word lists |
| `--no-reid` | off (re-identification on) | Track without DINOv2. The adapter test ran this way |
| `--no-size` | off (sizes on) | Leave out `size_x`, `size_y`, `size_z`, `size` |
| `--relative-positions` | off | Per-frame positions in camera coordinates in place of the 0/1 flags |
| `--size-trim` | `0.02` | Percent of points ignored at each end of each axis when measuring size |
| `--conf-percentile` | `55` | Pixels below this percentile of depth confidence are ignored, in tracking and in the sheet |
| `--min-points` | `300` | Confident pixels an object needs to count as seen in a frame |
| `--backend` | `transformers` | How SAM 3 is run: `transformers` (the cached snapshot) or `sam3` (Meta's package) |
| `--checkpoint` | none | `--backend sam3` only: a local `sam3.pt` |
| `--min-score` | `0.5` | SAM 3 detections are saved from this score up. Tracking then applies its own 0.55 |
| `--overlays` | off | Also save each frame with SAM 3's masks drawn on it |

If the full extractor finds no object, the syntactic mode is tried. If that finds nothing either, `objects.csv` has
only its header, and `cameras.csv` is written as usual.

### `prompt_object_extractor.py`: question to SAM 3 prompts

Parses the question with spaCy and keeps the nouns that WordNet says are physical things, with their descriptive
modifiers ("red chair", "kitchen table"). It drops spatial words, generic nouns and words for the video or viewer.
It prints one comma-separated line of prompts per question.

| Option | Default | Meaning |
|---|---|---|
| `questions` | — | One or more questions |
| `--file` | — | A text file with one question per line |
| `--explain` | off | List every candidate with the reason it was kept or dropped |
| `--syntactic` | off | Keep every noun phrase: no WordNet and no word lists |
| `--model` | `en_core_web_md` | The spaCy pipeline |

From Python: `main(question)` returns the prompts, and `analyse(question)` returns every candidate with its verdict.
Both take `syntactic=True`. The word lists (`SPATIAL_WORDS`, `GENERIC_NOUNS`, `VIEWPOINT_NOUNS`) and thresholds
(`MIN_PHYSICAL_SHARE = 0.3`, `MIN_UNIT_SHARE = 1/3`) are at the top of the file.

### `prompt_extractor_report.py`: the extractor on real questions

Prints the extractor's decisions for every question of a few exported scenes, then tables of the most often kept and
dropped phrases. It is meant for spotting a word that is wrongly kept or dropped.

| Option | Default | Meaning |
|---|---|---|
| `--scenes-dir` | `tasks/vggt_data/scenes` | The folder with `<dataset>/<scene>/questions.jsonl` |
| `--per-dataset` | `1` | Scenes per dataset |
| `--seed` | none (name order) | Pick the scenes at random with this seed |
| `--types` | all | Only these question types, e.g. `route_planning` |
| `--max-per-scene` | all | At most this many questions per scene |
| `--top` | `40` | Rows in each summary table |
| `--model`, `--syntactic` | | As in `prompt_object_extractor.py` |

### `sam3_infer.py`: SAM 3 masks

Runs SAM 3 on a folder of frames, once per prompt, and saves one `<stem>__masks.npz` per frame. It writes
`manifest.json` last, to mark a finished run. Needs a GPU, except with `--dry-run`.

| Option | Default | Meaning |
|---|---|---|
| `--images` | required | The folder of frames |
| `--frames` | `all` | Which frames, as positions in the sorted folder |
| `--meta` | `meta.json` next to the frames folder | For `subset:NAME` |
| `--prompts` | required | Comma-separated prompts, e.g. `"chair,brick wall"` |
| `--out` | required | The output folder |
| `--backend` | `transformers` | `transformers` (cached snapshot, no login) or `sam3` (Meta's package, gated download) |
| `--checkpoint` | download | `--backend sam3` only: a local `sam3.pt` |
| `--min-score` | `0.5` | Keep detections from this score up |
| `--overlays` | off | Also save the masks drawn on each frame |
| `--dry-run` | off | Check the inputs and print the plan without loading SAM 3 |
| `--overwrite` | off | Replace a finished run in `--out` |

### `adapt_vggt.py`: the saved VGGT outputs in OpenVocab-4D's format

Validates the saved cameras and depth, then writes `frames/`, `cameras.npz`, `depth.npz` and `adapter_report.json`.
If a check fails, nothing is written. It runs no model.

| Option | Default | Meaning |
|---|---|---|
| `--scene` | `scannet/scene0025_01` | `<dataset>/<scene>` |
| `--data-root` | `tasks/vggt_data/out/vggt_vsibench_75` | The export holding `scenes/`. Locally this is usually `tasks/vggt_data` |
| `--preset` or `--frames` | one is required | A preset or a selection |
| `--label` | the preset or the spec | The selection's name in the output path |
| `--dense-dir` | none | The folder with `depth.npy` and `depth_conf.npy`. Without it, `depth.npz` isn't written and tracking can't run |
| `--patch-depth` | off | Write `depth.npz` approximated from the patch points instead |
| `--out` | `tasks/openvocab_adapter/out/<dataset>__<scene>/<label>/adapted` | The output folder |
| `--overwrite` | off | Replace an earlier run's files |

### `openvocab-4D-main/trackA/milestone4_video.py`: tracking (OpenVocab-4D)

This is OpenVocab-4D's own script, not part of this folder. Its `track` stage links per-frame detections into objects
by their 3D centres, with DINOv2 re-identification, and writes `tracks/<prompt>.npz`. Its `vggt` stage runs VGGT
itself, which only the adapter test's native run uses. The wrappers above call it with `--step 1` and `--masks-dir`.

| Option | Default | Meaning |
|---|---|---|
| `--images`, `--out`, `--prompts` | required | Frames, scene folder, comma-separated prompts |
| `--stages` | `vggt,track,fuse` | Which stages to run. The scene sheet needs only `track` |
| `--step` | `2` | Use every Nth frame. The wrappers pass `1` |
| `--masks-dir` | none | Reuse saved SAM 3 masks instead of running SAM 3. With `--no-reid` this runs on a CPU |
| `--conf-percentile`, `--min-points` | `55`, `300` | The confident-depth rule, as above |
| `--no-reid` | off | No DINOv2 re-identification |
| `--reid-sim` | `0.6` | Minimum DINOv2 cosine similarity to re-join an older or farther track |
| `--reid-radius-frac` | `0.15` | How far re-identification looks, as a fraction of the scene diagonal |
| `--chunk-size`, `--overlap` | `40`, `8` | `vggt` stage only: frames per VGGT chunk, and frames shared between chunks |
| `--track-res` | `1008` | Longest side of the frames given to SAM 3 |

`question_scene_sheet.py` doesn't pass `--reid-sim` or `--reid-radius-frac`. To change them, run the track stage by
hand, then build the sheet with `scene_sheet_creator.py`.

### `scene_sheet_creator.py`: tracks or gold labels to the sheet

**`--source pipeline`** (the default) reads a scene folder that has `cameras.npz`, `depth.npz` and
`tracks/<prompt>.npz`. Any prompt without tracks is tracked first. With `--images`, a scene with no geometry is
reconstructed with OpenVocab-4D's own VGGT. **`--source gold`** reads VSI-Bench's boxes.

| Option | Default | Meaning |
|---|---|---|
| `--source` | `pipeline` | `pipeline` or `gold` |
| `--scene` | required | Pipeline: the scene folder (milestone 4's `--out`). Gold: a VSI-Bench scene, e.g. `scannet/scene0025_01` |
| `--prompts` | required for pipeline | `"table,chair"` or `table chair`. For gold, every class when left out |
| `--out-dir` | `<scene>/scene_sheet` (gold: `tasks/openvocab_adapter/out/<dataset>__<scene>/gold_sheet`) | Where the CSVs go |
| `--size` | **off** | Add the size columns. This is the opposite of `question_scene_sheet.py`, which adds them unless `--no-size` |
| `--size-trim` | `0.02` | As above |
| `--relative-positions` | off | Pipeline: per-frame camera-coordinate positions in place of the 0/1 flags |
| `--no-frames` | off | Pipeline: leave out `num_frames` and the per-frame columns (gold never has them) |
| `--no-cameras` | off | Pipeline: leave out `cameras.csv` (gold never has one) |
| `--conf-percentile`, `--min-points` | `55`, `300` | As above |
| `--max-frame-points` | `5000` | Points per object per frame used for the median |
| `--masks-dir` | none | Saved SAM 3 masks to track from |
| `--no-reid` | off | As above |
| `--images`, `--step`, `--chunk-size`, `--overlap` | none, `1`, `40`, `8` | Pipeline, for a scene with no geometry yet: reconstruct it from these frames |
| `--gold-dir` | `data/raw/VSI-Bench/meta_info` | Gold: the metadata folder, downloaded when missing |

Use `--no-frames --no-cameras` to make a pipeline sheet with the same columns as a gold sheet. From Python:
`build_scene_sheet(scene, prompts, ...)` and `build_gold_scene_sheet(scene, prompts, include_size=True, ...)`.

### `scene_sheet_report.py`: every question of a scene

Builds and checks the sheet of every question of one scene, and prints the results for checking by eye. With
`--source gold`, it runs on a CPU for any of the 288 VSI-Bench scenes. Questions come from the exported
`questions.jsonl`, or else from `data/raw/VSI-Bench/test.jsonl`. Pipeline sheets use `question_scene_sheet.py`'s
defaults, and SAM 3 and tracking run once for all the questions' prompts (`question_scene_sheet.prepare_scene`).

| Option | Default | Meaning |
|---|---|---|
| `scene` | required | `<dataset>/<scene>` or the scene name |
| `--source` | `pipeline` | `pipeline` (GPU) or `gold` (CPU) |
| `--questions` | the export's | A `.jsonl` of questions in VSI-Bench's format |
| `--types` | all | Only these question types |
| `--limit` | all | Only the first N questions |
| `--data-root`, `--dense-dir`, `--patch-depth-fallback` | | As in `question_scene_sheet.py` |
| `--work-dir` | `out/<dataset>__<scene>/subset-uniform16` (gold: `.../gold`) | Where the work and sheets are kept |
| `--gold-dir` | as in `scene_sheet_creator.py` | Gold: the metadata folder |

### The adapter test: `hpc/`, `compare.py`, `summarize.py`, `export_tables.py`

The adapter test runs OpenVocab-4D twice on each scene, once on the saved VGGT outputs (adapted) and once on its own
VGGT (native), with the same masks, then compares the results. It runs on the cluster, as one job:

```bash
bash src/mvp/openvocab_adapter/hpc/start.sh
RUNS=smoke5 GPUS=2 TIME=01:00:00 SCENES_FILE=src/mvp/openvocab_adapter/hpc/scenes_smoke.tsv \
    bash src/mvp/openvocab_adapter/hpc/start.sh       # a short test on two scenes first
```

| Script | What it does | Settings |
|---|---|---|
| `hpc/start.sh` | Checks the inputs, creates `.venv-ov4d`, and submits one job: the setup, every scene, then the summary | `GPUS` (5), `TIME` (`02:00:00`), `RUNS` (`half`), `SCENES_FILE` (`hpc/scenes.tsv`), `DENSE_ROOT` (`data/vggt`), `DATA`, `SETUP` (`1` reruns the setup) |
| `hpc/run_packed.sh` | The job `start.sh` submits: one scene per GPU at a time. Scenes with results are skipped | As `run_all.sh`, plus `SCENES_FILE`, `SETUP` |
| `hpc/run_all.sh` | The whole test for one scene: tests, SAM 3, adapt, track (both runs), sheets, compare | `SCENE` (`scannet/scene0025_01`), `PROMPTS`, `RUNS` (`half`; several allowed, e.g. `"half all"`), `REID` (`0`), `PY` (`.venv-ov4d/bin/python`), `DENSE`, `DATA`, `WORK`, `RESULTS` |
| `hpc/submit_scenes.sh` | The older way: one job per scene, then a summary job | `RUNS`, `SCENES_FILE`, `TIME` (`01:00:00`), `AFTER` (a job id to wait for) |
| `hpc/setup_env.sh` | Installs and checks `.venv-ov4d` (as a job) | `NUMPY_SPEC` (`numpy>=1.26,<2`) |
| `hpc/make_scene_list.py` | Writes `hpc/scenes.tsv`: the scenes, and each one's prompts from its question templates | `--data-root`, `--per-dataset` (`scannet=4 scannetpp=3 arkitscenes=3`), `--max-prompts` (`12`), `--out` |
| `compare.py` | Compares an adapted run with a native run and gives the verdict: MATCHES, DIFFERS or INCOMPLETE. The pass marks are `CRITERIA` at the top | `--run-dir` (holds `adapted/` and `native/`), or `--adapted` and `--native`; `--masks`; `--out` |
| `summarize.py` | Writes `results/SUMMARY.md` from every `comparison.json` | `--results` (`tasks/openvocab_adapter/results`) |
| `export_tables.py` | Writes flat CSVs of every scene's objects, cameras, comparisons and detections to `results/data/` | `--results` |

`hpc/scenes.tsv` has one scene per line: `<dataset>/<scene>`, a tab, then comma-separated prompts. To run steps by
hand, or to read the results, see [DETAILS.md](../src/mvp/openvocab_adapter/DETAILS.md).

### `run_tests.sh` and `tests/`: CPU tests

```bash
bash src/mvp/openvocab_adapter/run_tests.sh                 # Linux, or Git Bash on Windows
bash src/mvp/openvocab_adapter/run_tests.sh -k GoldSheet    # extra arguments go to unittest
PYTHON=.venv-ov4d/bin/python bash src/mvp/openvocab_adapter/run_tests.sh
```

The tests run the real scripts on a small synthetic scene, with SAM 3, VGGT, OpenCV and Rerun replaced by small
fakes. By default they use the project's `.venv`, and `PYTHON` picks another. The prompt-extractor tests are skipped
without spaCy and NLTK. The run should end with `OK`.

### `common.py`

This file holds the shared paths (`DATA_ROOTS`, `DENSE_ROOT`, `OUT_ROOT`), the frame presets (`PRESETS`) and
`select_frames`. Change a preset or a search path here.

## Troubleshooting

| You see | Do this |
|---|---|
| `no dense depth for <scene>` | Run on the cluster, pass `--dense-dir` (or set `DENSE_ROOT`), or pass `--patch-depth-fallback` |
| `no exported scene ...` | The scene isn't in the export. Pass `--data-root`, or use `--source gold`, which works for all 288 scenes |
| `scene ... is in several datasets` | Give `<dataset>/<scene>` in place of the scene name alone |
| `spaCy model 'en_core_web_md' is not installed` | Install the requirements (see [The environment](#the-environment)) |
| `re-identification needs CUDA and DINOv2 weights` | Run on a GPU, or pass `--no-reid` |
| `401`, `gated repo`, `facebook/sam3` | Accept SAM 3's license on Hugging Face and run `hf auth login` |
| `[adapter] FAIL ...` and nothing written | The export and `data/vggt/` disagree. The line names the check that failed |
| A question with an apostrophe fails through `slurm/submit.sh` | `submit.sh` can't quote it. Put the command in a small shell script and submit that |
| A sheet has prompts like `distance` and no rows | The syntactic fallback was used (`prompt_mode` in `question.json`). The question names no physical object |
| A prompt has no rows | SAM 3 found nothing at score ≥ 0.55 with ≥ 300 confident pixels, or the track was seen in fewer frames than the minimum (1 of 16, 4 of 128, linear in between). Look at the masks with `--overlays` |

DETAILS.md has the full troubleshooting table for the adapter test.
