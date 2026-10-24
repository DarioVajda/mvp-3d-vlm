# Task results

One folder per task, named exactly like the task file in `tasks/` without `.md`:
`tasks/2026-10-21-vggt-layer-similarity.md` → `task-results/2026-10-21-vggt-layer-similarity/`.

Each folder holds:

- `README.md`: what was run (commit, job ids, settings that differ from the task), the main
  results, and anything surprising. If the task failed, why.
- Small outputs: summaries (`*.json`, `*.csv`), predictions (`*.jsonl`), figures (`*.png`, `*.jpg`).
- The Slurm log, or the relevant part of it, if it helps explain the result.

Large binaries (feature dumps, checkpoints) are gitignored. Leave them on the cluster and note
their path in the folder's `README.md`.

After adding the results, set `status: done` (or `failed`) in the task file.
