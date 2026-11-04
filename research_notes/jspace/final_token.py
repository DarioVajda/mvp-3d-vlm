"""True final-token and 12th-from-last readouts, text lens, for all 10 questions."""
import json, re

OUT = "/shared/workspace/povejmo/math/projects/mvp/results/jspace_v7"
clean = lambda n: (n.split("(")[0] if "(" in n else n).strip()
for m in ("lora_s0", "additive_s0"):
    rs = [json.loads(l) for l in open(f"{OUT}/{m}/readouts.jsonl")]
    for i in dict.fromkeys(r["id"] for r in rs):
        out = []
        for l in (14, 17, 20, 22):
            rr = [r for r in rs if r["id"] == i and r["lens"] == "text" and r["layer"] == l and r["kind"] == "text"]
            out.append(f"L{l} {[n for n, _ in rr[-1]['raw'][:4]]}")
        first = [r for r in rs if r["id"] == i and r["lens"] == "text" and r["layer"] == 14 and r["kind"] == "text"][0]
        print(f"{m[:4]} {first['type'][7:22]:15} gt {first['gt'][:5]:5} ans {first['answer']:4} first-tail {first['tok']!r} last {rr[-1]['tok']!r} | " + " ".join(out))
