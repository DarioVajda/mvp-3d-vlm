"""Join the per-question VSI-Bench scores logged by run_eval.sh shards (task vsibench_shard) and aggregate them with
VG-LLM's own vsibench_aggregate_results. Checks that the shards together hold every question exactly once.
    merge_shards.py <output dir> <shard dir> [<shard dir> ...]"""
import json
import sys
from pathlib import Path

from loguru import logger
from lmms_eval.tasks.vsibench.utils import vsibench_aggregate_results

out, shards = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
scores = []
for d in shards:
    files = sorted(d.glob("**/*_samples_vsibench_shard.jsonl"))
    assert len(files) == 1, (d, files)
    scores += [json.loads(line)["vsibench_score"] for line in open(files[0])]
ids = [s["id"] for s in scores]
assert len(ids) == len(set(ids)) == 5130, (len(ids), len(set(ids)))
out.mkdir(parents=True, exist_ok=True)
with open(out / "per_question.jsonl", "w") as f:
    for s in sorted(scores, key=lambda s: s["id"]):
        f.write(json.dumps(s) + "\n")
# vsibench_aggregate_results returns only the overall score and logs the per-category ones; keep that log line
logger.add(out / "aggregate.log", format="{message}")
overall = vsibench_aggregate_results(scores)
(out / "metrics.json").write_text(json.dumps({"overall": overall, "n": len(scores)}, indent=2))
print(f"overall {overall:.2f} over {len(scores)} questions; per category in {out / 'aggregate.log'}")
