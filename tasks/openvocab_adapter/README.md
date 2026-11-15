# Testing OpenVocab-4D on Dario's saved VGGT output

## What I'm trying to do

The scene sheets come from OpenVocab-4D. It turns a video into a list of objects with 3D positions in three stages:

1. **VGGT** reconstructs the scene: camera poses and a depth map for every frame.
2. **SAM 3** finds and segments the objects we ask about ("chair", "door", …) in each frame.
3. **Tracking and fusion** link each object across frames and place it in 3D, using VGGT's cameras and depth. The
   scene sheet is built from the result.

Dario has already run VGGT on all the VSI-Bench videos (`data/vggt/`). So OpenVocab-4D shouldn't need to run VGGT
again: we can **feed it Dario's saved VGGT output directly** and run only SAM 3, tracking and fusion. That's quicker,
and every part of the project then uses the same 3D reconstruction.

To make that work, `adapt_vggt.py` converts Dario's cameras and depth into the format OpenVocab-4D reads. **Before we
build on it, I need to know whether the conversion is right**, meaning whether OpenVocab-4D gives the same answer on
Dario's VGGT output as when it runs its own VGGT. That is what this test checks.

## What the test compares

For each scene, OpenVocab-4D runs twice on exactly the same frames:

| | **Native** (the normal pipeline) | **Adapted** (what I want to use) |
|---|---|---|
| 3D reconstruction | OpenVocab-4D runs its own VGGT | Dario's saved VGGT output, converted by `adapt_vggt.py` |
| Objects (SAM 3) | the same saved SAM 3 masks for both runs | the same saved SAM 3 masks for both runs |
| Tracking, fusion, scene sheet | yes | yes |

SAM 3 runs once, and both runs use its masks. **The only difference between the two runs is where the 3D
reconstruction comes from.** If the conversion is right, they find the same objects in the same places. If something
is off (cameras in the wrong convention, depth from the wrong frames, a wrong scale), the adapted run puts objects in
the wrong places, or splits or merges them differently.

At the end, the two runs' final outputs are compared:

| What is compared | Passes if (provisional) |
|---|---|
| **Object tracks:** do both runs group the SAM 3 detections into the same objects? | at least 90% of tracked pixels get the same object |
| **3D object positions:** is each object in the same place? | median error at most 5% of the scene's size |
| **Scene sheets:** do both sheets list the same objects? | at least 80% of rows appear in both |
| **Scene-sheet positions** | median error at most 5% of the scene's size |
| **Sanity checks:** the same frames, the same masks, valid files | all must pass |

Each scene gets a verdict: **MATCHES** if every pass mark is met, **DIFFERS** if not, or **INCOMPLETE** if a step
produced nothing to measure.

Some things to know about the comparison:

- **The two runs can't agree perfectly.** Dario's VGGT saw all 128 frames of the video in one pass, and the native run
  sees only the frames being tested, so the two reconstructions are slightly different. Each also has its own
  coordinate frame and scale, so the test first lines the two reconstructions up, using the depth maps only. That's
  why the pass marks allow small differences. They are a first guess, and we can tune them once real numbers are in.
- **10 scenes** (listed in [`hpc/scenes.tsv`](hpc/scenes.tsv)): 4 from ScanNet, 3 from ScanNet++ and 3 from
  ARKitScenes, so we see whether it works across datasets. For each scene, SAM 3 looks for the objects that the
  scene's VSI-Bench questions ask about.
- **Every other frame** (64 of the 128) to keep it cheap. That doesn't weaken the test, because both runs still see
  exactly the same frames.

## What I need from you (Dario)

This has to run on the cluster: SAM 3 and VGGT need a GPU, and the dense depth maps (`data/vggt/`) are only there. I
need you to **run one command, wait, and push the results back.** Everything is automated, and the steps are below.

**Cost:** roughly 5–10 minutes per scene, about 1–2 GPU-hours in total. The 10 scenes run in parallel, so it should
take about half an hour once the jobs start. This is an estimate, because it hasn't run on the cluster yet. Each job
has a 1-hour limit, so it can't run away.

## How to run it

Run all of this on the cluster's login node, from the project checkout that has `data/vggt/`.

### Step 1: SAM 3 access (once)

SAM 3's weights are gated on Hugging Face.

1. Accept the license at <https://huggingface.co/facebook/sam3> with your Hugging Face account.
2. Check that the cluster is logged in to that account:
   ```bash
   .venv/bin/hf auth whoami      # should print your user name
   .venv/bin/hf auth login       # only if it doesn't
   ```

### Step 2: start everything

```bash
git fetch origin
git checkout vggt_output_adapter
git pull
bash tasks/openvocab_adapter/hpc/start.sh
```

`start.sh` does the rest itself, and ends by printing `All submitted`:

1. It checks that the depth maps are there for all 10 scenes. If any are missing, it submits nothing and lists them.
2. It creates a separate environment, `.venv-ov4d`, with SAM 3 and VGGT. It doesn't touch `.venv`.
3. It submits a setup job (~20 minutes), then one job per scene, then a summary job. The scene jobs wait for the setup
   job, and they are cancelled automatically if it fails.

### Step 3: wait

```bash
squeue -u $USER
```

This lists the jobs still queued or running. When it's empty, everything has finished. The logs are in `logs/ov4d/`.

### Step 4: push the results

```bash
git add tasks/openvocab_adapter/results
git commit -m 'OpenVocab-4D comparison results'
git push
```

These are small text files only (reports, scene-sheet CSVs, logs), never frames: ScanNet's terms don't allow sharing
those. **Please push even if something failed,** because each scene's `run.log` shows where it stopped.

### If something goes wrong

| You see | Do this |
|---|---|
| `start.sh` stops with `missing: data/vggt/...` | Run it from the checkout that has `data/vggt/` |
| The setup job fails on `facebook/sam3`, `401` or `gated repo` | Do step 1, then run `start.sh` again |
| The setup job fails on anything else | Its log is `logs/ov4d/ov4d_setup_<job id>.out`. Check [DETAILS.md](DETAILS.md#troubleshooting), or push and send me the log |
| A scene's job fails | Nothing to do: push the results anyway, because its `run.log` says why |

Running `start.sh` again is safe. It skips the setup if that already finished, and reuses SAM 3's masks.

## What comes back

The results go to `tasks/openvocab_adapter/results/`. **`SUMMARY.md`** has one row per scene: the verdict, how many
objects each run found, and the numbers behind the verdict. Each scene's folder has the details, including both runs'
scene sheets (`adapted/objects.csv` and `native/objects.csv`) for comparing by eye.

- **Most scenes MATCH:** the conversion works, and we can use Dario's VGGT output directly. The next step is the same
  test with Nicolas's prompt extractor choosing the SAM 3 prompts.
- **Some scenes DIFFER:** their `comparison.json` shows which objects or prompts disagree, and by how much.

---

More detail is in [DETAILS.md](DETAILS.md): settings, running a single scene or other frame selections, what each step
does, how the comparison works, the full results layout, the complete troubleshooting table, and the tests that run
without a GPU.
