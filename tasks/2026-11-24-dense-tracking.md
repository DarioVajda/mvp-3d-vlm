---
id: 2026-11-24-dense-tracking
title: Pipeline scene sheets of the 75 scenes on all 128 frames, with overlap tracking and the mask-relative depth rule
author: Nicolás Ramos Fernandez
created: 2026-11-24
status: queued            # queued | needs-info | running | done | failed | cancelled
priority: high            # low | normal | high
kind: feature-extraction  # eval | eda | feature-extraction | training | other
training: false
downloads: []             # SAM 3 (facebook/sam3@3c879f3) and DINOv2 are in the caches run_sheets.sh already uses
gpu_hours: 2.5            # B300: about 2 h for the dense run (SAM 3 on 8x the frames), 0.25 h for the check
disk_gb: 8                # out_all/ with the 128-frame depth.npz (about 50 MB a scene), plus the zip and the check
depends_on: []
papers: []
results: task-results/2026-11-24-dense-tracking/
job_ids: []               # filled in by Dario
---

# Pipeline scene sheets of the 75 scenes on all 128 frames, with overlap tracking and the mask-relative depth rule

## Question

Do the tracking fixes give pipeline sheets closer to gold on the 75 exported scenes: fewer rows per object, more of the
named objects found, counts closer to gold? The run produces the SAM 3 masks on all 128 frames, from which the new
tracking settings are then tuned locally on the dev scenes, without another GPU run.

## Hypothesis and what would change our mind

- Expected result:
  - Against `scene_sheets_75` (the 16-frame sheets) on the dev scenes (`experiments/2026-11-18-scene-sheet-formats/dev_scenes.txt`):
    - rows per unique object much closer to 1 (now 2.5)
    - named objects found clearly higher (now 74.5%)
    - count = gold clearly higher (now 27.1%)
    - all measured with `pipeline_scorecard.py --scale oracle`
  - A first look on 3 dev scenes, only re-tracking the existing 16-frame masks locally, already gave:
    - named objects found 79% → 93%
    - count = gold 34% → 53%
    - rows per unique object 1.87 → 1.58
- Result that would prove the hypothesis wrong:
  - no improvement in rows per unique object or count = gold, or
  - counts falling below gold for classes with several objects (distinct objects joined)
- What we do next in each case:
  - Better: tune the provisional overlap settings on `dev_scenes_15.txt` from the saved masks (CPU or local GPU),
    validate them unchanged on `dev_scenes_35.txt`, then pre-register a VLM test on scenes no condition has used.
  - Not better: look at the tracks of the failing classes before any VLM run.

## Grounding in previous work

No paper. The fixes come from measurements on `data/scene_sheets_75` (2026-11-24):

- The original tracker's matching radius is 5% of the first frame's diagonal, about 0.15 m (2–6 cm in many
  ARKitScenes scenes). Between the 16 frames, the same object's visible centre moves about 0.65 m, so only 16% of
  same-object pairs can match. 3D voxel overlap catches 56% of them, against 4% of pairs of different objects.
- The confident-depth rule drops 38% of SAM 3's detections of named objects. (Scene, class) coverage falls from 90%
  (SAM 3) to 75%.
- The 16 frames are about 57° of camera rotation apart; all 128 have VGGT depth already.

## Method

The code is in the commit that adds this file. The defaults of `question_scene_sheet.py`, and so of
`hpc/run_sheets.sh`, are now:

- all 128 frames (`--frames all`)
- tracking by 3D overlap (`--association overlap`)
- the mask-relative depth rule (`--depth-rule mask`)
- upright axes (`--axes upright`: z up from a fitted floor, `src/mvp/openvocab_adapter/upright.py`)

The original pipeline is `FRAMES=subset:uniform16 ASSOCIATION=centroid DEPTH_RULE=global AXES=camera0`.

### Inputs

- Data: the 75 exported scenes (`tasks/vggt_data/out`) and their dense depth (`data/vggt`), as for `scene_sheets_75`.
- Models: SAM 3 `facebook/sam3` at `3c879f3` (transformers backend, cached), DINOv2 ViT-S/14 through `torch.hub`
  (cached in `TORCH_HOME`, as `run_sheets.sh` sets it).
- Settings:
  - 128 frames
  - SAM 3 masks saved from score 0.5; tracking from 0.55
  - re-identification on
  - overlap settings as `question_scene_sheet.OVERLAP_OPTIONS` (provisional)
  - depth rule `mask`, 50 pixels
  - the prompts from the extractor, as in `scene_sheets_75`

### Commands

```bash
git fetch && git checkout scene_sheet_formats && git pull     # the commit that adds this file

# smoke test: three questions of one scene, dense defaults (about 5 minutes)
slurm/submit.sh -J sheets_dense_smoke -t 00:30:00 --gres gpu:1 -- .venv-ov4d/bin/python \
    src/mvp/openvocab_adapter/scene_sheet_report.py scannet/scene0025_01 --data-root tasks/vggt_data/out \
    --work-dir tasks/openvocab_adapter/out_smoke/scannet__scene0025_01/all --limit 3

# full run: every question of the 75 scenes, into tasks/openvocab_adapter/out_all/ and scene_sheets_75_all.zip
slurm/submit.sh -J sheets_dense -t 04:00:00 --gres gpu:1 -c 48 --mem 256G -- bash src/mvp/openvocab_adapter/hpc/run_sheets.sh

# regression check, independent of the full run: the original tracking, from the saved 16-frame masks
cp -r tasks/openvocab_adapter/out tasks/openvocab_adapter/out_legacy_check
rm -rf tasks/openvocab_adapter/out_legacy_check/reports tasks/openvocab_adapter/out_legacy_check/*/subset-uniform16/scene/tracks
FRAMES=subset:uniform16 ASSOCIATION=centroid DEPTH_RULE=global AXES=camera0 OUT=tasks/openvocab_adapter/out_legacy_check \
    ZIP=scene_sheets_75_legacy_check.zip \
    slurm/submit.sh -J sheets_legacy -t 01:00:00 --gres gpu:1 -c 48 --mem 256G -- bash src/mvp/openvocab_adapter/hpc/run_sheets.sh
.venv/bin/python - <<'EOF' > task-results/2026-11-24-dense-tracking/legacy_check.txt
import pathlib, numpy as np
a, b = pathlib.Path("tasks/openvocab_adapter/out"), pathlib.Path("tasks/openvocab_adapter/out_legacy_check")
tracks = sorted(b.glob("*/subset-uniform16/scene/tracks/*.npz"))
diff = [p for p in tracks if not np.array_equal(np.load(p)["labels"], np.load(a / p.relative_to(b))["labels"])]
sheets = sorted(b.glob("*/subset-uniform16/sheets/*/objects.csv"))
other = [p for p in sheets if p.read_bytes() != (a / p.relative_to(b)).read_bytes()]
print(f"tracks: {len(tracks) - len(diff)} of {len(tracks)} identical; sheets: {len(sheets) - len(other)} of {len(sheets)}")
print(*[p.relative_to(b) for p in diff + other][:50], sep="\n")
EOF
```

The legacy check's run does no SAM 3: the masks in the copy are reused, so it only tracks.

## Smoke test

`out_smoke/scannet__scene0025_01/all/` should have the following:

- **The report:** it ends with `3 questions: 3 sheets OK`.
- **`scene/tracks/summary.json`:** `"association": {"rule": "overlap", ...}`, `"depth_rule": {"rule": "mask", ...}`
  and `"min_track_frames": 4`.
- **Each mask folder:** a `manifest.json` that lists 128 frames.
- **Each `sheets/*/objects.csv`:** columns `frame_000` to `frame_127`.
- **`scene/upright.json`:** its `method` is `floor`, and `sheets/*/question.json` has `"axes": "upright"`.
- **Timing:** note how long SAM 3 and tracking took (`masks/*/manifest.json` has `seconds`). The full run's time is
  about 75 times the smoke test's per-scene share.

## Expected outputs

`task-results/2026-11-24-dense-tracking/`:

- `reports/`: the 150 reports of `tasks/openvocab_adapter/out_all/reports/`, a few MB
- `index.csv` of `out_all/`
- `legacy_check.txt`
- `README.md`: wall time, any scene that failed, and the scenes `sheet_index.py` lists as having no floor plane

`scene_sheets_75_all.zip` (`tasks/openvocab_adapter/`) holds the sheets, the masks, the tracks and the frames, without
`depth.npz`. Estimate: under 1 GB. It goes to the shared Drive folder where `scene_sheets_75.zip` is, not to git: the
local tuning reads its masks.

## Failure modes and checks

| Risk | How we'd notice | What to do |
| ---- | --------------- | ---------- |
| The full run takes longer than 4 h | the job ends with scenes left | Submit it again: finished scenes are skipped, and finished masks and tracks are reused |
| GPU memory with 8 streams at 128 frames | CUDA out of memory in a report | `STREAMS=4` (SAM 3 holds about 4 GB per stream, the same per frame as before) |
| Tracking at 128 frames is slow (re-identification embeds 8 times as many detections) | `tracks/summary.json` `seconds` large | Report it; nothing to change |
| A scene fails | `FAILED` in its report | Push the reports anyway; its report says why |
| The legacy check differs | `legacy_check.txt` lists files | Report the list; do not change anything. Locally, on an RTX 3070, the same check matched the cluster's tracks exactly on 3 scenes |
| The old `out/` is gone | the copy fails | Skip the legacy check and say so; it was checked locally |

## Resources

- One B200/B300 GPU, 48 CPUs and 256 GB of memory, as for `scene_sheets_75`.
- About 2 h for the full run and 15 min for the check.
- Disk: about 6 GB in `out_all/` and 1 GB for the zip. The legacy check's copy is about 1.5 GB and can be deleted
  after.
- No new downloads or licenses.

## Notes for Dario

- `run_sheets.sh` now writes into its own folder (`OUT`, by default `tasks/openvocab_adapter/out_<frames>`), so the
  dense run doesn't touch `tasks/openvocab_adapter/out/`, and the old run's finished reports don't make it skip scenes.
- If the full run has to stop early, the dev scenes matter most (`experiments/2026-11-18-scene-sheet-formats/dev_scenes.txt`).

## Notes from Dario

<!-- Dario writes here when status is needs-info, failed or done -->

## Grill log

| Question | Answer | Reason |
| --- | --- | --- |
| How are the 128 frames shown to a VLM that sees 16? | Not in the CSVs. When the prompt is written, an object counts as seen in one of the 16 frames if it is seen there, or in at least 2 of the 4 frames before and 2 of the 4 after. An object seen only between the 16 frames gets the nearest of them | One frame can miss an object that the frames around it show |
| Which depth rule? | A pixel counts when its confidence reaches the lower of the global threshold and the mask's own median, with 50 pixels needed | It keeps the masks the original rule drops entirely, and leaves confident masks as they were |
| Where do the GPU runs happen? | On the cluster, once. Tuning reads the saved masks locally | The tracking settings can be tuned without SAM 3 |
| Is milestone4_video.py's default changed? | No. The sheet pipeline passes the new options. The tracker's own defaults stay the original rules | The frozen runs (`rerun_detection.py`, hash-pinned) call it without options, and batch2_B must replicate them unchanged |
