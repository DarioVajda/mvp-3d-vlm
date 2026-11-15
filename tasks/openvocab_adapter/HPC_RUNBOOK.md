# Running the OpenVocab-4D comparison on the cluster

This job checks whether OpenVocab-4D gives the same result on Dario's saved VGGT reconstruction as on its own VGGT.
It runs SAM 3 once, then runs tracking, fusion and a scene sheet on both geometries with the same masks, and compares
them. Everything runs in one Slurm job on one B200/B300. The only input it needs is already on the cluster:
`data/vggt/scannet/scene0025_01/`.

## 1. One-time setup (about 20 minutes)

```bash
git fetch origin && git checkout vggt_output_adapter
/usr/bin/python3 -m venv .venv-ov4d          # on the login node: the container has no ensurepip
slurm/submit.sh -J ov4d_setup -t 00:45:00 -- bash tasks/openvocab_adapter/hpc/setup_env.sh
```

- This makes a separate `.venv-ov4d` and leaves the project's `.venv` alone. It uses the project's PyTorch
  (CUDA 13.0). OpenVocab-4D's own `install.sh` installs CUDA 12.6, which has no kernels for Blackwell GPUs.
- The log should end with `GPU ok`, `imports ok` and `access to facebook/sam3 ok`.
- SAM 3's weights are gated: accept the license at <https://huggingface.co/facebook/sam3> first.
- If the container has no `git`, the script prints the two `git clone` commands to run on the login node.

## 2. Run (about 30–60 minutes)

```bash
slurm/submit.sh -J ov4d_compare -t 02:00:00 -- bash tasks/openvocab_adapter/hpc/run_all.sh
```

It runs CPU tests, SAM 3 on the 128 frames, and then compares three frame selections: `smoke5` (5 nearby frames),
`eval32` (the uniform32 subset) and `all` (128 frames). For each selection it runs OpenVocab-4D on the adapted
geometry and on its own VGGT, then compares them.

- Frames: it uses the export in `tasks/vggt_data/out/` if one is there. Otherwise it re-exports this scene with
  `tasks/vggt_data/export.py` into `tasks/openvocab_adapter/out/export/`, using the project's `.venv`.
- To run only one selection, set `RUNS=eval32` (the variable reaches the job through `--export=ALL`). The other
  settings are listed at the top of `run_all.sh`.
- SAM 3's masks are kept in `tasks/openvocab_adapter/out/`, so a rerun skips SAM 3.

## 3. Send the results back

The job ends with a summary per selection and a `git add ... && git commit ... && git push` line to copy. It
commits only `tasks/openvocab_adapter/results/scannet__scene0025_01/` (a few hundred KB of JSON, CSV and the log). The
frames, depth, masks and point clouds stay in `out/`, which is gitignored. Please don't commit those, because
ScanNet's terms apply to the frames.

If the job fails, `results/.../run.log` holds the whole log. Commit it anyway, because it shows where it stopped.

## What to look at

`results/scannet__scene0025_01/SUMMARY.txt` shows each selection's verdict:

| Verdict | Meaning |
|---|---|
| MATCHES | every provisional pass mark is met |
| DIFFERS | a pass mark is missed, or a hard check failed (different frames, different masks, a wrong file format) |
| INCOMPLETE | a step produced nothing to measure |

The pass marks are in `CRITERIA` in `compare.py`. They are provisional, because the two geometries come from
different VGGT passes and can never agree exactly. The full numbers are in `<selection>/comparison.json`.

## If something goes wrong

| Message | Fix |
|---|---|
| `no CUDA GPU here` / `no CUDA GPU visible` | Run it through `slurm/submit.sh`, not on the login node |
| `no kernel image is available` | PyTorch does not support this GPU: check that `.venv-ov4d` has `torch 2.14.1+cu130` |
| `401`, `gated repo` or `no access to facebook/sam3` | Accept the license, then `hf auth login` |
| numpy fails to install | The container's Python may be too new for numpy<2: rerun setup with `NUMPY_SPEC=numpy` |
| `[adapter] ... check(s) failed, nothing written` | The export and `data/vggt/` disagree: send the log, it names the check |
| `masks are ... the image is ...` | SAM 3 returned masks at another size: send the log |
