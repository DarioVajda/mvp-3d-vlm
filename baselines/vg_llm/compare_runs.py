"""Compare two runs of the same VSI-Bench shard question by question (e.g. pinned environment on H100/A100 against the
Blackwell cross-check environment): identical raw outputs, per-question score differences, and the shard score of
each run under VG-LLM's own vsibench_aggregate_results.
    compare_runs.py <run dir A> <run dir B>"""
import json
import sys
from pathlib import Path

from lmms_eval.tasks.vsibench.utils import vsibench_aggregate_results


def load(d):
    files = sorted(Path(d).glob("**/*_samples_vsibench_shard.jsonl"))
    assert len(files) == 1, (d, files)
    return {s["id"]: s for s in (json.loads(line)["vsibench_score"] for line in open(files[0]))}


a, b = load(sys.argv[1]), load(sys.argv[2])
assert a.keys() == b.keys(), "the runs cover different questions"
score = lambda s: s.get("accuracy", s.get("MRA:.5:.95:.05"))
same_text = sum(a[k]["prediction"] == b[k]["prediction"] for k in a)
diff = [k for k in a if score(a[k]) != score(b[k])]
print(f"questions: {len(a)}")
print(f"identical raw outputs: {same_text} ({100 * same_text / len(a):.1f}%)")
print(f"questions whose score differs: {len(diff)}")
for k in diff[:20]:
    print(f"  id {k} {a[k]['question_type']}: {a[k]['prediction']!r} ({score(a[k])}) vs {b[k]['prediction']!r} ({score(b[k])})")
sa, sb = vsibench_aggregate_results(list(a.values())), vsibench_aggregate_results(list(b.values()))
print(f"shard overall: A {sa:.2f}  B {sb:.2f}  difference {sb - sa:+.2f}")
