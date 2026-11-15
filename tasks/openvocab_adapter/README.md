# Testing OpenVocab-4D on the saved VGGT outputs

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
single command starts everything. After that, the only thing left is to push the results once the jobs finish.

**Cost:** roughly 5–10 minutes per scene, about 1–2 GPU-hours in total. The 10 scenes run in parallel, so it should
take about half an hour once the jobs start. This is an estimate, because it hasn't run on the cluster yet. Each job
has a 1-hour limit.

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
those. **Push even if something failed,** because each scene's `run.log` shows where it stopped.

### If something goes wrong

| You see | Do this |
|---|---|
| `start.sh` stops with `missing: data/vggt/...` | Run it from the checkout that has `data/vggt/` |
| The setup job fails on `facebook/sam3`, `401` or `gated repo` | Do step 1, then run `start.sh` again |
| The setup job fails on anything else | Its log is `logs/ov4d/ov4d_setup_<job id>.out`. Check [DETAILS.md](DETAILS.md#troubleshooting), or push the results and share the log |
| A scene's job fails | Push the results anyway, because its `run.log` says why |

Running `start.sh` again is safe. It skips the setup if that already finished, and reuses SAM 3's masks.

## The results

The results go to `tasks/openvocab_adapter/results/`. **`SUMMARY.md`** has one row per scene: the verdict, how many
objects each run found, and the numbers behind the verdict. Each scene's folder has the details, including both runs'
scene sheets (`adapted/objects.csv` and `native/objects.csv`) for comparing by eye.

- **Most scenes MATCH:** the conversion works, and the saved VGGT outputs can be used directly. The next step is the
  same test with Nicolas's prompt extractor choosing the SAM 3 prompts.
- **Some scenes DIFFER:** their `comparison.json` shows which objects or prompts disagree, and by how much.

---

More detail is in [DETAILS.md](DETAILS.md): settings, running a single scene or other frame selections, what each step
does, how the comparison works, the full results layout, the complete troubleshooting table, and the tests that run
without a GPU.
