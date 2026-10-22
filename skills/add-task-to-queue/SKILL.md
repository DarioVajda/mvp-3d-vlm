---
name: add-task-to-queue
description: Add Task to Queue. Write a cluster task for Dario into tasks/ in the standard format, after grilling the author on decisions, failure modes and the prior research in papers/. Use when someone wants an experiment, eval, feature extraction or training run done on the cluster, or says "add a task", "queue this", "ask Dario to run".
argument-hint: <what you want run>
---

# Add Task to Queue

Only Dario can use the cluster. Other people ask for runs by adding a file to `tasks/`; Dario
runs it, sets its `status`, and puts outputs in `task-results/<task id>/`. The protocol is in
`tasks/README.md` and the format is `tasks/TEMPLATE.md`. Read both before starting.

The request is: $ARGUMENTS

## 1. Gather context before asking anything

Answer whatever you can yourself first, so the author is only asked what the repo can't answer.

- `README.md`: how runs are submitted (`slurm/submit.sh`), what data and models are on the
  cluster, the evaluation protocol, and the existing results.
- `src/`: code that can be reused. Prefer extending an existing script to writing a new one.
- `results/` and `results/summary.json`: what has already been measured.
- `tasks/*.md`: existing tasks. Check for duplicates and for tasks this one depends on.
- `task-results/`: outputs of finished tasks that this task could build on.
- `papers/`: the source of truth for previous research (see below).

### Using `papers/`

`papers/README.md` indexes the PDFs. Any PDF in `papers/` that is not in the index still
counts; add a row for it. To read a paper, extract its text with
`pdftotext papers/<file>.pdf <scratch dir>/<file>.txt` (into a scratch directory, never into the
repo) and search it.

For the task, find:

- the papers that ran the same or a similar experiment, and their exact settings (models,
  layers, frames, resolution, benchmarks, metrics);
- results the task should reproduce or compare against;
- claims the task tests, quoted verbatim with file and page or section.

Never state a paper's claim or setting from memory. Quote the PDF. If no paper in `papers/`
supports something, say so and record it in the task as an assumption.

## 2. Grill the author

Interview the author relentlessly about the task until every decision below is settled. Ask
**one question at a time**, each with your recommended answer and why (cite the paper or file
it comes from). If a question can be answered by reading the repo or the papers, read them
instead of asking. Walk down each branch and resolve dependencies between decisions in order:
for example, the models fix the downloads, which fix disk and time.

Cover at least:

1. **Question and hypothesis.** What exactly do we want to learn? What result do we expect, what
   result would prove us wrong, and what would we do next in each case? If no result would
   change a decision, push back on whether the task is worth running.
2. **Prior work.** Which papers in `papers/` did this already, and what did they find? Is this a
   reproduction, an extension or something new? Where does our setup differ from theirs, and
   does that difference matter?
3. **Method.** Which existing script, if any? Which models (exact Hugging Face id and revision),
   layers, frames, resolution, prompt, seeds? Is the protocol the same as the baselines in
   `README.md`, so results are comparable?
4. **Data.** Which benchmark or subset, how many items, how they are chosen (random with a
   seed, or a fixed id list in `data/subsets/`), and whether the data is already on the cluster.
5. **Downloads and training.** Does the task download any model or dataset that is not already
   on the cluster? Does it train, fine-tune or fit anything, including linear probes? Both must
   be explicit in the frontmatter, with sizes, and flagged to the author before the task is
   written. Gated models need a license accepted on Hugging Face; say which ones.
6. **Resources.** GPU hours, wall time, memory, disk for downloads and outputs. Base the
   estimate on timings in `README.md` or earlier `task-results/`, and say which.
7. **Failure modes.** For this task specifically, ask what could go wrong and how we would
   notice. Typical ones:
   - out of memory (long videos, many frames, many layers saved at full resolution);
   - output too large (per-layer features: compute the size before running);
   - silent protocol mismatch (wrong frame sampling, prompt, resolution or chat template)
     that makes results incomparable with the baselines;
   - answer parsing failures counted as wrong;
   - wrong layer indexing (counting from 0 or 1; with or without special tokens);
   - nondeterminism (no seed, sampling instead of greedy decoding);
   - the job hits its time limit and does not resume;
   - a version mismatch with `requirements.txt` (for example transformers);
   - a model or dataset that is gated, renamed or not in the cache.
   Each risk the author accepts goes in the failure-modes table with a check.
8. **Smoke test.** What is the smallest run that proves the pipeline works, and what in its
   output should be checked before the full run?
9. **Outputs.** Which files appear in `task-results/<task id>/`, their size, and which of them
   are too large for git (the `.gitignore` drops `*.pt`, `*.safetensors`, `*.parquet`, videos and
   archives). What figures or tables should the `README.md` there contain?
10. **Handover.** What can Dario decide alone, and what should he ask about first?

Stop when every branch is resolved. Summarise the decisions back to the author and get a yes
before writing the file.

## 3. Write the task

- Task id: `YYYY-MM-DD-<short-kebab-slug>` with today's date. Check that `tasks/<id>.md` and
  `task-results/<id>/` don't exist yet.
- Copy `tasks/TEMPLATE.md` to `tasks/<id>.md` and fill in every section. Set `status: queued`,
  `results: task-results/<id>/` and leave `job_ids: []`.
- Fill `## Grounding in previous work` with the verbatim quotes, each with its file and page.
- Fill `## Grill log` with a short summary of the questions, answers and reasons.
- Commands must use `slurm/submit.sh` and write into `task-results/<id>/`. Never run models on
  the login node.
- Do not create `task-results/<id>/`; Dario creates it when he adds the outputs.
- Never put tokens, passwords or other secrets in the task file.

Show the author the finished file. Commit only if they ask, with a plain message such as
`Add task <id>`.
