# Scene sheets for the 75 exported VSI-Bench scenes

A scene sheet for every one of the 2,728 VSI-Bench questions about the 75 scenes of the shared VGGT export
(`vggt_vsibench_75`): one from the pipeline (VGGT + SAM 3 + tracking) and one from the VSI-Bench gold labels. All
5,456 sheets were made; none failed. `hpc/run_sheets.sh` makes them (see the end); this file is the `README.md` of
its output folder, `scene_sheets_75/`.

## Quick start

`index.csv` has one row per question. Look a question up there, then open its two sheet folders:

```python
import csv
from pathlib import Path

root = Path("scene_sheets_75")
index = {row["id"]: row for row in csv.DictReader(open(root / "index.csv"))}

q = index["3137"]                       # VSI-Bench question id
print(q["question"], q["options"], q["ground_truth"])
pipeline = list(csv.DictReader(open(root / q["pipeline_sheet"] / "objects.csv")))
gold = list(csv.DictReader(open(root / q["gold_sheet"] / "objects.csv")))
cameras = list(csv.DictReader(open(root / q["pipeline_sheet"] / "cameras.csv")))
frames = root / q["scene"].replace("/", "__") / "subset-uniform16/scene/frames"
image = frames / Path(cameras[0]["image"]).name   # the image column holds a cluster path; use its file name
```

## `index.csv`

| Column | Meaning |
|---|---|
| `id` | VSI-Bench question id (unique across all scenes) |
| `scene` | `<dataset>/<scene>`, as in the VGGT export |
| `question_type` | VSI-Bench question type |
| `question`, `options`, `ground_truth` | As in VSI-Bench; `options` joined with ` \| `, empty for numerical questions |
| `prompts` | The objects the question names, as given to SAM 3 |
| `pipeline_sheet`, `gold_sheet` | The two sheet folders, relative to `scene_sheets_75/` |
| `pipeline_ok`, `gold_ok` | Whether the sheet was made and passed its checks (all `True`) |

Sheets are named by their prompts, so questions that name the same objects share a folder.

## Folder layout

```text
scene_sheets_75/
├── README.md                this file
├── index.csv                one row per question
├── reports/                 each scene's run: <dataset>__<scene>_pipeline.txt and _gold.txt
└── <dataset>__<scene>/
    ├── subset-uniform16/            the pipeline, on 16 frames evenly spread over the 128 exported ones
    │   ├── sheets/<prompts>/        objects.csv, cameras.csv, question.json
    │   ├── scene/                   frames/<t>.jpg, cameras.npz, adapter_report.json, tracks/<prompt>.npz
    │   └── masks/<prompts>/         SAM 3's masks
    └── gold/sheets/<prompts>/       objects.csv, sheet.json, question.json
```

## The sheets

**`objects.csv`**: one row per object.

| Column | Meaning |
|---|---|
| `object_id` | Row number |
| `concept` | The prompt it was found with |
| `track_id` | The object's number within its concept: the track (pipeline) or the labelled box (gold) |
| `x`, `y`, `z` | Centre: the median of its 3D points (pipeline), or the box centre (gold) |
| `size_x`, `size_y`, `size_z`, `size` | Extent along each axis (pipeline: of the points, 0.02nd to 99.98th percentile; gold: of the box), and their product |
| `num_frames` | Pipeline only: the number of frames it was seen in |
| `frame_000` … `frame_015` | Pipeline only: 1 if it was seen in that frame (the k-th row of `cameras.csv`) |

**`cameras.csv`** (pipeline only): one row per frame, with the camera's centre (`x`, `y`, `z`), its rotation as a
quaternion (`qw`, `qx`, `qy`, `qz`, camera to world), and its viewing direction (`forward_*`).

**`question.json`**: the question, its prompts and the settings used. **`sheet.json`** (gold): where the labels
come from and how prompts were matched to the dataset's classes.

The full column definitions are in
`docs/scene_sheet_plan.md` (*Scene sheet format*) on branch `vggt_output_adapter`.

## Coordinates and frames

- **Pipeline sheets** use the VGGT export's world frame and scale, the same as its `cameras.npz`: the camera of
  exported frame 0, with OpenCV axes (x right, y down, z forward), in VGGT's arbitrary per-scene units, not metres.
  So they line up with the shared point clouds and depth maps.
- **Gold sheets** are in the dataset's own world frame, in metres, with z up. The two can't be compared point by
  point; compare counts, sizes relative to each other, and orderings.
- **Frames.** The 16 frames are exported frames 0, 8, 16, 25, 33, 42, 50, 59, 67, 76, 84, 93, 101, 110, 118 and 127.
  `frames/<t>.jpg` is exported frame `t`, the same image as `frames/<t>.jpg` in the VGGT export.
  `scene/adapter_report.json` lists the matching frames of the original video (`source_video_frames`).
- **Paths.** `cameras.csv` (`image`) and `question.json` (`geometry`) hold paths on the cluster. Only the file and
  folder names at their ends matter.

## How good the pipeline sheets are

On the 223 counting questions ("How many chairs are in this room?"), here are the pipeline sheet's rows of the
asked class against the true answer:

| Result | Questions |
|---|---|
| Exact | 38 |
| Too many | 97 |
| Too few, but not zero | 52 |
| None found | 36 |

That is 732 rows for 662 objects. Doors are the worst (12 for 2, 15 for 5, 14 for 4). So one object is often
split into several rows, while small or rare objects are missed. The gold sheets are exact for 218 of the 223, so
the comparison itself is sound. Many pipeline sheets have a prompt with no rows at all; the reports in `reports/`
list them per question.

**Room-size questions** (75, one per scene) name no object, so their prompt is `room`. The pipeline sheet then holds
pieces of the room SAM 3 found in single frames, and the gold sheet is empty. Neither says anything about the room's
size, which has to come from the scene's points instead.

## How they were made

Branch `vggt_output_adapter`, with `.venv-ov4d` set up by `hpc/setup_env.sh` and the default settings: 16 uniform
frames, SAM 3 (transformers backend, score ≥ 0.5), the dense VGGT depth, and re-identification on. One B300 took
about 15 minutes for all 75 scenes:

```bash
slurm/submit.sh -J sheets -t 02:00:00 --gres gpu:1 -c 48 --mem 256G -- bash src/mvp/openvocab_adapter/hpc/run_sheets.sh
```

The script skips finished scenes, writes `index.csv`, prints the counting check, and copies this file into the
output folder. The per-scene depth maps (`scene/depth.npz`) are left out of the zip, because they are the shared
VGGT depth, cut to the 16 frames.
