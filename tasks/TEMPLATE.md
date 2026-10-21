---
id: YYYY-MM-DD-short-slug
title: One-line description of the run
author: Name
created: YYYY-MM-DD
status: queued            # queued | needs-info | running | done | failed | cancelled
priority: normal          # low | normal | high
kind: eda                 # eval | eda | feature-extraction | training | other
training: false           # true if anything is trained, fine-tuned or probed with fitted weights
downloads: []             # e.g. [facebook/VGGT-1B (~5 GB)]; [] if everything is already on the cluster
gpu_hours: 0.5            # estimate, B200/B300
disk_gb: 1                # estimate for new downloads + outputs
depends_on: []            # task ids that must be done first
papers: []                # files in papers/ this task builds on
results: task-results/YYYY-MM-DD-short-slug/
job_ids: []               # filled in by Dario
---

# <Title>

## Question

What we want to learn, in one or two sentences.

## Hypothesis and what would change our mind

- Expected result:
- Result that would prove the hypothesis wrong:
- What we do next in each case:

## Grounding in previous work

For each claim taken from a paper: the file in `papers/`, the page or section, and the verbatim
quote. Mark anything not backed by a paper as an assumption.

## Method

Exact steps. Reuse existing code in `src/` where possible and say which file.

### Inputs

- Data (path on the cluster, subset, number of items):
- Models (Hugging Face id and revision or commit):
- Settings (frames, resolution, layers, seeds, prompt):

### Commands

```bash
# smoke test first
slurm/submit.sh -J <name>_smoke -t 00:20:00 -- python src/... --limit 5 --out task-results/<id>/smoke
# full run
slurm/submit.sh -J <name> -t 01:00:00 -- python src/... --out task-results/<id>/
```

## Smoke test

What the small run is and what to check in its output before starting the full run.

## Expected outputs

Files that should appear in `task-results/<id>/`, with approximate sizes. Large binaries
(`*.pt`, `*.safetensors`, `*.parquet`, ...) are gitignored; say where they go instead.

## Failure modes and checks

| Risk | How we'd notice | What to do |
| ---- | --------------- | ---------- |
|      |                 |            |

## Resources

GPU type and count, wall time, memory, disk, and any downloads or license acceptance needed.

## Notes for Dario

Open questions or decisions left to the person running the task.

## Notes from Dario

<!-- Dario writes here when status is needs-info, failed or done -->

## Grill log

Summary of the decisions taken while writing the task: question, answer, reason.
