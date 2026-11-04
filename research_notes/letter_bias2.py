"""Exact C counts per run; route-planning letter shares; and which prompt token the J-lens 'final token' table really read, with the top letter at every tail position."""
import json, re
from collections import Counter
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
R = f"{ROOT}/results/train"
L = "ABCD"
rng = np.random.default_rng(0)
letter = lambda s: (str(s).strip()[0].upper() if str(s).strip() and str(s).strip()[0].upper() in L else "?")
print("=== Counts on all MC questions (2,490) ===")
print(f"{'run':24}{'correct':>8}{'correct & C':>12}{'share':>7}{'wrong':>7}{'chose C':>8}{'share':>7}{'exp C':>7}{'chose non-C':>12}")
rp = {}
for run in ("2B_v9_fmix_lora_s1", "2B_v9_fmix_additive_s0", "2B_v9_fmix_additive_s1", "2B_v9_fmix_xa3_both_s0", "2B_v9_fmix_xa3_both_s1"):
    rows = [x for x in map(json.loads, open(f"{R}/{run}/branches/step5750/eval/vsi_f32_full.jsonl")) if x.get("options") is not None and len(x["options"])]
    ok = [x for x in rows if x["score"] == 1]; bad = [x for x in rows if x["score"] != 1]
    cc = sum(letter(x["ground_truth"]) == "C" for x in ok); wc = sum(letter(x["prediction"]) == "C" for x in bad)
    exp = sum(1 / (len(x["options"]) - 1) for x in bad if len(x["options"]) >= 3 and letter(x["ground_truth"]) != "C")
    print(f"{run[8:]:24}{len(ok):8}{cc:12}{100 * cc / len(ok):6.1f}%{len(bad):7}{wc:8}{100 * wc / len(bad):6.1f}%{exp:7.0f}{len(bad) - wc:12}")
    rp[run] = [x for x in rows if x["task"] == "route_planning"]
print("\n=== Route planning, pred − truth share per letter, 95% CI (n=194) ===")
for run, rows in rp.items():
    g = np.array([letter(x["ground_truth"]) for x in rows]); p = np.array([letter(x["prediction"]) for x in rows]); out = []
    for k in L:
        d = (p == k).astype(float) - (g == k).astype(float)
        bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(2000)]
        out.append(f"{k} {d.mean() * 100:+.1f}[{np.percentile(bs, 2.5):+.1f},{np.percentile(bs, 97.5):+.1f}]")
    print(f"{run[8:]:24}", " ".join(out))
print("\n=== J-lens, text lens, tail tokens of the 6 MC questions: token, and first A-D letter in the top-15 ===")
for m in ("lora_s0", "additive_s0"):
    rs = [json.loads(l) for l in open(f"{ROOT}/results/jspace_v7/{m}/readouts.jsonl")]
    ids = [i for i in dict.fromkeys(r["id"] for r in rs) if not re.search(r"\d", next(r["gt"] for r in rs if r["id"] == i))]
    for l_ in (14, 20):
        print(f"--- {m} L{l_}")
        for i in ids:
            rr = [r for r in rs if r["id"] == i and r["lens"] == "text" and r["layer"] == l_ and r["kind"] == "text"]
            cells = []
            for r in rr:
                ls = [re.sub(r"[^A-D]", "", n) for n, _ in r["raw"]]
                ls = [x for x in ls if len(x) == 1]
                cells.append(f"{r['tok']!r}:{ls[0] if ls else '-'}")
            print(f"  id {i} {rr[0]['type'][7:22]:15} gt {rr[0]['gt']} ans {rr[0]['answer']} | " + " ".join(cells))
    # does the first tail token's top letter follow the answer?
c = Counter()
for m in ("lora_s0", "additive_s0"):
    rs = [json.loads(l) for l in open(f"{ROOT}/results/jspace_v7/{m}/readouts.jsonl")]
    for r in rs:
        if r["lens"] == "text" and r["kind"] == "text" and r["layer"] == 20 and not re.search(r"\d", r["gt"]):
            ls = [re.sub(r"[^A-D]", "", n) for n, _ in r["raw"]]; ls = [x for x in ls if len(x) == 1]
            c[(r["tok"], ls[0] if ls else "-")] += 1
print("\nL20 (token, top letter) counts:", c.most_common(30))
