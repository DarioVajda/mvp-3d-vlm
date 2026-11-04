"""Summarise the J-lens readouts: salient vs random cells, what geometry pushes toward, the J-space components, answer emergence."""
import json
from collections import Counter, defaultdict
import numpy as np

OUT = "/shared/workspace/povejmo/math/projects/mvp/results/jspace_v7"
MODELS = ["lora_s0", "additive_s0"]
LAYERS = [0, 1, 2, 5, 8, 11, 14, 17, 20, 22]
FUNC = set(" and , . is in on the a to of with or are as at by for it its an be this that - / ( ) : ; ' \" 's".split()) | {"", " "}
recs = {m: [json.loads(l) for l in open(f"{OUT}/{m}/readouts.jsonl")] for m in MODELS}
print({m: len(v) for m, v in recs.items()}, "questions:", sorted({r["id"] for r in recs["lora_s0"]}))
q = json.load(open(f"{OUT}/additive_s0/lens_quality.json"))
print("lens quality (median rank of the model's next token; lower is better):", {ln: {l: (v["jlens_median_rank"], v["logit_median_rank"]) for l, v in d.items() if l in ("5", "8", "14", "20")} for ln, d in q.items()})


def best(rk, prefix):
    v = [r for k, r in rk.items() if k.startswith(prefix)]
    return min(v) if v else None


def clean(name):
    return name.split("(")[0].strip() if "(" in name else name.strip()


summary = {}
# 1. salient vs random cells: best rank of a spatial word / question word / gt, median over records
print("\n=== Ranks (median over cells) at salient vs random cells, visual lens.  sp = best spatial word, q = best question word, gt = correct answer ===")
for m in MODELS:
    for lens in ("visual", "text"):
        for kind in ("top", "rand"):
            row = {}
            for l in LAYERS:
                rr = [r for r in recs[m] if r["lens"] == lens and r["layer"] == l and r["kind"] == kind]
                row[l] = tuple(int(np.median([best(r["ranks"], p) for r in rr])) for p in ("sp:", "q:")) + (int(np.median([r["ranks"]["gt"] for r in rr])),)
            summary[f"{m}/{lens}/{kind}"] = row
            if lens == "visual":
                print(f"{m:12} {kind:4} " + "  ".join(f"L{l}: sp{row[l][0]:>6} q{row[l][1]:>6} gt{row[l][2]:>6}" for l in (2, 8, 14, 20)))
# 2. contrast words at salient cells (what is specific to the cell vs the video), visual lens, content words only
print("\n=== Most frequent cell-specific words (contrast readout top-5), salient cells, visual lens ===")
for m in MODELS:
    for l in (5, 11, 17):
        c = Counter()
        for r in recs[m]:
            if r["lens"] == "visual" and r["layer"] == l and r["kind"] == "top":
                for name, _ in r["contrast"][:5]:
                    w = clean(name)
                    if w.lower() not in FUNC and len(w.strip()) > 1:
                        c[w] += 1
        print(f"{m:12} L{l:>2}: {c.most_common(14)}")
# 3. geometry delta: what the injection pushes the cell toward (additive only)
print("\n=== Geometry delta h_on - h_off at salient cells: top words (visual lens) and the rank of spatial / question / gt words ===")
for l in LAYERS:
    rr = [r for r in recs["additive_s0"] if r["lens"] == "visual" and r["layer"] == l and r["kind"] == "top" and "delta" in r]
    if not rr:
        continue
    c = Counter()
    for r in rr:
        for name, _ in r["delta"]["top"][:5]:
            w = clean(name)
            if w.lower() not in FUNC and len(w.strip()) > 1:
                c[w] += 1
    ratio = np.median([r["delta"]["ratio"] for r in rr])
    print(f"L{l:>2} |delta|/|h| {ratio:.2f}  sp{int(np.median([best(r['delta']['ranks'], 'sp:') for r in rr])):>6} q{int(np.median([best(r['delta']['ranks'], 'q:') for r in rr])):>6} gt{int(np.median([r['delta']['ranks']['gt'] for r in rr])):>6}  {c.most_common(10)}")
rr = [r for r in recs["additive_s0"] if r["lens"] == "visual" and r["layer"] == 11 and r["kind"] == "rand" and "delta" in r]
print("   random cells at L11, for comparison:", Counter(clean(n) for r in rr for n, _ in r["delta"]["top"][:5] if clean(n).lower() not in FUNC).most_common(10))
# 4. J-space components at salient cells
print("\n=== J-space decomposition (k<=25 nonneg) at salient cells, visual lens: R^2 and most common components ===")
for m in MODELS:
    for l in (5, 11, 17):
        rr = [r for r in recs[m] if r["lens"] == "visual" and r["layer"] == l and r["kind"] == "top" and "jspace" in r]
        c = Counter(clean(n) for r in rr for n, _ in r["jspace"]["top"][:8])
        print(f"{m:12} L{l:>2} R2 median {np.median([r['jspace']['r2'] for r in rr]):.2f}  {[(w, n) for w, n in c.most_common(12) if w.lower() not in FUNC][:10]}")
# 5. answer emergence at the last text token: gt rank by layer
print("\n=== Rank of the correct answer token at the final prompt token, by layer (text lens / visual lens) ===")
for m in MODELS:
    for lens in ("text", "visual"):
        row = []
        for l in LAYERS:
            rr = [r for r in recs[m] if r["lens"] == lens and r["layer"] == l and r["kind"] == "text"]
            last = {r["id"]: r for r in rr}  # the last text position per question is the last record of that id
            row.append(int(np.median([r["ranks"]["gt"] for r in last.values()])))
        print(f"{m:12} {lens:6} " + " ".join(f"L{l}:{v}" for l, v in zip(LAYERS, row)))
# 6. per-question: at the last token, top raw words at L14 / L20 (text lens) for a look
print("\n=== Final-token readouts, text lens, per question (L14 → L20 top-5) ===")
for m in MODELS:
    seen = set()
    for r in recs[m]:
        if r["lens"] == "text" and r["kind"] == "text" and r["layer"] == 14 and r["id"] not in seen:
            seen.add(r["id"])
            r20 = next(x for x in recs[m] if x["lens"] == "text" and x["kind"] == "text" and x["layer"] == 20 and x["id"] == r["id"] and x["tok"] == r["tok"])
            print(f"{m:12} {r['type'][:22]:22} ans {r['answer']!r:5} gt {r['gt'][:5]!r:7} L14 {[clean(n) for n, _ in r['raw'][:5]]}  L20 {[clean(n) for n, _ in r20['raw'][:5]]}")
json.dump(summary, open(f"{OUT}/summary_ranks.json", "w"))
