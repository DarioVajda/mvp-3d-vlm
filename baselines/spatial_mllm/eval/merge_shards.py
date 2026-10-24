"""Join the results of run_eval.sh shards and score them with Spatial-MLLM's own calculate_metrics.
    merge_shards.py <output dir> <shard dir> [<shard dir> ...]
Checks that the shards together hold every VSI-Bench question exactly once."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3] / "third_party" / "Spatial-MLLM"
sys.path.insert(0, str(REPO))
from src.evaluation.vsibench.eval_vsibench import calculate_metrics  # noqa: E402

out, shards = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
results = []
for d in shards:
    results += json.loads((d / "eval_result" / "results_spatial-mllm.json").read_text())
ids = [r["sample"]["id"] for r in results]
n_expected = sum(1 for _ in open(REPO / "datasets/evaluation/vsibench/test.jsonl"))
assert len(ids) == len(set(ids)) == n_expected, (len(ids), len(set(ids)), n_expected)
out.mkdir(parents=True, exist_ok=True)
(out / "results_spatial-mllm.json").write_text(json.dumps(results, indent=2))
metrics = calculate_metrics(results)
(out / "metrics_spatial-mllm.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: metrics[k] for k in ("acc", "mra", "all")}, indent=2))
