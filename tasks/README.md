# Task queue

Only Dario has access to the cluster. Everyone else asks for runs by adding a task file here.
Dario runs it, updates its status, and puts the outputs in `task-results/<task id>/`.

## Files

```
tasks/README.md                 this file
tasks/TEMPLATE.md               the task format (the add-task-to-queue skill fills it in)
tasks/<task id>.md              one file per task
task-results/<task id>/         outputs of that task, same name as the task file (no .md)
```

The task id is `YYYY-MM-DD-<short-kebab-slug>`, for example `2026-10-21-vggt-layer-similarity`.
The date is the day the task was written.

## Status

The `status` field in the task file's frontmatter is the only place a task's state lives.

| Status       | Set by    | Meaning                                                             |
| ------------ | --------- | ------------------------------------------------------------------- |
| `queued`     | author    | Ready to run                                                        |
| `needs-info` | Dario     | Can't run as written; questions are under `## Notes from Dario`     |
| `running`    | Dario     | Submitted to Slurm (`job_ids` filled in)                            |
| `done`       | Dario     | Finished; outputs are in `task-results/<task id>/`                  |
| `failed`     | Dario     | Ran but failed; the reason is in `task-results/<task id>/README.md` |
| `cancelled`  | either    | No longer needed                                                    |

## Adding a task

Use the `add-task-to-queue` skill (`/add-task-to-queue <what you want run>`). It interviews you
about the decisions and failure modes, checks the papers in `papers/`, and writes the file in
this format. You can also copy `TEMPLATE.md` by hand.

Rules:

- A task that downloads a model or dataset, or trains or fine-tunes anything, must say so in
  its frontmatter (`downloads`, `training`). Dario decides whether it runs.
- A task must include a smoke test (a small version that runs first).
- Never put tokens, passwords or other secrets in a task file.
