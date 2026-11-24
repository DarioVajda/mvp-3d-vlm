# Testing OpenVocab-4D on the saved VGGT outputs

The code is here, in `src/mvp/openvocab_adapter/`. Its outputs go to `tasks/openvocab_adapter/`: `out/` holds the large
intermediate files (gitignored), and `results/` holds the small files that are committed. Every command below runs
from the repository root.

## Goal

OpenVocab-4D turns a video into a list of objects with 3D positions, and the scene sheets are built from that list.
It works in three stages:

1. **VGGT** reconstructs the scene: camera poses and a depth map for every frame.
2. **SAM 3** finds and segments the requested objects ("chair", "door", …) in each frame.
3. **Tracking and fusion** link each object across frames and place it in 3D, using VGGT's cameras and depth.

VGGT has already been run on all the VSI-Bench videos, and its outputs are saved in `data/vggt/`. OpenVocab-4D
therefore shouldn't need to run VGGT again. It can take **the saved VGGT outputs directly** and run only SAM 3,
tracking and fusion. That's quicker, and the whole project then uses the same 3D reconstruction.

`adapt_vggt.py` converts the saved VGGT outputs (cameras and depth) into the format OpenVocab-4D reads. This test
checks that the conversion is correct, meaning that OpenVocab-4D gives the same result on the saved VGGT outputs as
when it runs its own VGGT.

## What the test compares

For each scene, OpenVocab-4D runs twice on exactly the same frames:

| | **Native** (the normal pipeline) | **Adapted** (the saved VGGT outputs) |
|---|---|---|
| 3D reconstruction | OpenVocab-4D runs its own VGGT | the saved VGGT outputs, converted by `adapt_vggt.py` |
| Objects (SAM 3) | the same saved SAM 3 masks for both runs | the same saved SAM 3 masks for both runs |
| Tracking, fusion, scene sheet | yes | yes |

SAM 3 runs once, and both runs use its masks. **The only difference between the two runs is where the 3D
reconstruction comes from.** If the conversion is correct, both runs find the same objects in the same places. If
something is wrong (cameras in the wrong convention, depth from the wrong frames, a wrong scale), the adapted run puts
objects in the wrong places, or splits or merges them differently.

At the end, the two runs' final outputs are compared:

| What is compared | Passes if (provisional) |
|---|---|
| **Object tracks:** are the SAM 3 detections grouped into the same objects? | at least 90% of tracked pixels get the same object |
| **3D object positions:** is each object in the same place? | median error at most 5% of the scene's size |
| **Scene sheets:** do both sheets list the same objects? | at least 80% of rows appear in both |
| **Scene-sheet positions** | median error at most 5% of the scene's size |
| **Sanity checks:** the same frames, the same masks, valid files | all must pass |

Each scene gets a verdict: **MATCHES** if every pass mark is met, **DIFFERS** if not, or **INCOMPLETE** if a step
produced nothing to measure.

Some notes on the comparison:

- **The two runs can't agree perfectly.** The saved VGGT outputs come from one pass over all 128 frames of the video,
  while the native run sees only the frames being tested, so the two reconstructions differ slightly. Each also has
  its own coordinate frame and scale, so the two reconstructions are first lined up, using the depth maps only. The
  pass marks therefore allow small differences. They are a first guess, to be tuned once real numbers are in.
- **10 scenes** (listed in [`hpc/scenes.tsv`](hpc/scenes.tsv)): 4 from ScanNet, 3 from ScanNet++ and 3 from
  ARKitScenes, to see whether it works across datasets. For each scene, SAM 3 looks for the objects that the scene's
  VSI-Bench questions ask about.
- **Every other frame** (64 of the 128) keeps the cost down. This doesn't weaken the test, because both runs still
  see exactly the same frames.

## Where it runs and what it costs

The test runs on the cluster: SAM 3 and VGGT need a GPU, and the dense depth maps in `data/vggt/` are only there. A
single command submits **one job**. After that, the only thing left is to push the results once the job finishes.

**Cost (measured 2026-11-15, one B200):** about 1 minute of setup the first time, then about 3 minutes per scene with
64 frames: 10 scenes took 27 minutes on one GPU. The job runs one scene per GPU at a time (5 GPUs by default), so with
5 GPUs it takes about 10 minutes.

## How to run it

Run all of this on the cluster's login node, from the project checkout that has `data/vggt/`.

### Step 1: SAM 3 access

Nothing to do on this cluster. `sam3_infer.py` loads SAM 3 through transformers from the group's cached snapshot of
`facebook/sam3` (revision 3c879f3, the one the explainability results used), which needs no Hugging Face login.
Elsewhere, without that cache, SAM 3's weights are gated: accept the license at <https://huggingface.co/facebook/sam3>
and log in with `.venv/bin/hf auth login`. `--backend sam3` uses Meta's own package instead, as milestone 2 does.

### Step 2: start everything

```bash
git fetch origin
git checkout vggt_output_adapter
git pull
bash src/mvp/openvocab_adapter/hpc/start.sh
```

`start.sh` does the rest itself, and ends by printing `All submitted`:

1. It checks that the depth maps are there for all 10 scenes. If any are missing, it submits nothing and lists them.
2. It creates a separate environment, `.venv-ov4d`, and clones SAM 3 and VGGT at OpenVocab-4D's pinned commits. It
   doesn't touch `.venv`.
3. It submits one job (`hpc/run_packed.sh`, 5 GPUs by default): the setup if no earlier setup finished, then the
   scenes as a queue, one scene per GPU at a time, then the summary. Scenes that already have results are skipped, so
   submitting again continues where a job stopped.

Settings: `GPUS`, `TIME`, `RUNS`, `SCENES_FILE`, `DENSE_ROOT` and `DATA` (the last two when running from a checkout
without `data/vggt/` and `tasks/vggt_data/out/`). A short test first, on two scenes with five frames each:

```bash
RUNS=smoke5 GPUS=2 TIME=01:00:00 SCENES_FILE=src/mvp/openvocab_adapter/hpc/scenes_smoke.tsv \
    bash src/mvp/openvocab_adapter/hpc/start.sh
```

`hpc/submit_scenes.sh` still submits one job per scene, for rerunning a single scene.

### Step 3: wait

```bash
squeue -u $USER
```

When the job is gone from the list, it has finished. Its log is `logs/ov4d/ov4d_compare_<job id>.out`, and each
scene's log is in `logs/ov4d/scenes_<job id>/`.

### Step 4: push the results

```bash
git add tasks/openvocab_adapter/results
git commit -m 'OpenVocab-4D comparison results'
git push
```

These are small text files only (reports, scene-sheet CSVs, logs), never frames: ScanNet's terms don't allow sharing
those. **Push even if something failed,** because each scene's `run.log` shows where it stopped.

### If something goes wrong

| You see | Do this |
|---|---|
| `start.sh` stops with `missing: data/vggt/...` | Run it from the checkout that has `data/vggt/` |
| The setup fails on `facebook/sam3`, `401` or `gated repo` | The cached snapshot is missing: see step 1, then run `start.sh` again |
| The setup fails on anything else | Its log is `logs/ov4d/scenes_<job id>/setup.out`. Check [DETAILS.md](DETAILS.md#troubleshooting), or push the results and share the log |
| A scene fails | Push the results anyway, because its `run.log` says why |

Running `start.sh` again is safe. It skips the setup if that already finished, and reuses SAM 3's masks.

## The results

Flat tables of every scene's output, for analysis, are in `tasks/openvocab_adapter/results/data/` (`objects.csv`,
`cameras.csv`, `comparisons.csv`, `detections.csv`; columns in its `README.md`). Rebuild them with
`python src/mvp/openvocab_adapter/export_tables.py`.

The results go to `tasks/openvocab_adapter/results/`. **`SUMMARY.md`** has one row per scene: the verdict, how many
objects each run found, and the numbers behind the verdict. Each scene's folder has the details, including both runs'
scene sheets (`adapted/objects.csv` and `native/objects.csv`) for comparing by eye.

- **Most scenes MATCH:** the conversion works, and the saved VGGT outputs can be used directly. The next step is the
  same test with Nicolas's prompt extractor choosing the SAM 3 prompts.
- **Some scenes DIFFER:** their `comparison.json` shows which objects or prompts disagree, and by how much.

## A scene sheet for one question

Separately from the test, `question_scene_sheet.py` goes from one scene and one question to that question's scene
sheet. It picks the SAM 3 prompts from the question, runs SAM 3 on all 128 VGGT frames, and tracks the objects with the
saved VGGT cameras and depth: a detection continues the track whose 3D points it overlaps, and a mask's depth counts by
a confidence threshold relative to the mask. The prompt maps the 128 frames onto the 16 the VLM sees
(`mvp.benchmarks.scene_sheet`). `--frames subset:uniform16 --association centroid --depth-rule global` is the original
pipeline, which made every sheet before 2026-11-24 (details in [DETAILS.md](DETAILS.md#a-scene-sheet-for-one-question)).
It prints the paths of `objects.csv` and `cameras.csv`:

```bash
slurm/submit.sh -J scene_sheet -t 00:30:00 -- .venv-ov4d/bin/python src/mvp/openvocab_adapter/question_scene_sheet.py \
    scannet/scene0025_01 "Which of these objects (table, chair, door, telephone) is the closest to the fan?"
```

Like the test, it needs the dense depth in `data/vggt/`, so it runs on the cluster. Elsewhere,
`--patch-depth-fallback` approximates the depth from the exported scene's patch points, with a warning: coarser, but
enough to try it on a local GPU. See
[DETAILS.md](DETAILS.md#a-scene-sheet-for-one-question) for what it does, what it reuses, and its options.

---

More detail is in [DETAILS.md](DETAILS.md): settings, running a single scene or other frame selections, what each step
does, how the comparison works, the full results layout, the complete troubleshooting table, and the tests that run
without a GPU.
