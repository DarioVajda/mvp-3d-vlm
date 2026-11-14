"""Build the static pages in explainability-analysis/ from the saved results (CPU only; light enough for the login node).

  python -m mvp.explain.pages.build objsal     object_saliency_v10/single_question_3004/   (results/objsal_test*, detail_v10_3004)
  python -m mvp.explain.pages.build two        object_saliency_v10/two_direction_questions/ (results/detail_v10)
  python -m mvp.explain.pages.build final      final_results_v10_v11/                       (branch evals, results/ablation_v10)
  python -m mvp.explain.pages.build jspace     jspace_v7/                                   (results/jspace_v7; mvp.explain.jspace.build_page)
  python -m mvp.explain.pages.build all

Each page is a template in pages/templates with its data inlined at /*DATA*/ (and /*EXAMPLE*/). Frames are the
evaluation's 32 uniform frames, saved at 320x240; mask sheets outline the cells a mask covers by more than 25%.
"""
import json, os, runpy, sys

import numpy as np

from mvp.explain import ROOT

TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
OUT = f"{ROOT}/explainability-analysis"
q8 = lambda v, s: np.clip(np.round(255 * v / max(s, 1e-12)), 0, 255).astype(int).tolist()
dumps = lambda x: json.dumps(x)


def fill(template, dest, **parts):
    h = open(f"{TPL}/{template}").read()
    for k, v in parts.items():
        assert h.count(f"/*{k}*/") == 1, (template, k)
        h = h.replace(f"/*{k}*/", v)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "w").write(h)
    print(dest, f"{len(h) // 1024} KB")


def frames(qid, dest):
    """The question's 32 evaluation frames as dest/fNN.jpg (320x240)."""
    from mvp.benchmarks.vsibench import load_vsibench
    from mvp.fusion.inputs import uniform_frames
    d = load_vsibench("full").set_index("id", drop=False).loc[qid]
    fr, _ = uniform_frames(d["video_path"], 32)
    os.makedirs(dest, exist_ok=True)
    for t, im in enumerate(fr):
        im.convert("RGB").resize((320, 240)).save(f"{dest}/f{t:02d}.jpg", quality=72)


def mask_sheet(masks, colours, frame_dir, dest):
    """8x4 contact sheet of the frames with each object's cells outlined (mask covers > 25% of the cell)."""
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (240 * 8, 180 * 4))
    for t in range(32):
        im = Image.open(f"{frame_dir}/f{t:02d}.jpg").resize((240, 180))
        d = ImageDraw.Draw(im)
        for p, c in colours.items():
            if p not in masks.files:
                continue
            f = masks[p][t * 300:(t + 1) * 300].reshape(15, 20)
            for r in range(15):
                for cc in range(20):
                    if f[r, cc] > 0.25:
                        d.rectangle([cc * 12, r * 12, cc * 12 + 11, r * 12 + 11], outline=c, width=2)
        d.rectangle([2, 2, 22, 16], fill=(0, 0, 0))
        d.text((5, 3), str(t), fill=(255, 255, 255))
        sheet.paste(im, ((t % 8) * 240, (t // 8) * 180))
    sheet.save(dest, quality=78)


def objsal():
    """Question 3004: OWLv2 + SAM 2 masks (results/objsal_test) vs SAM 3 (results/objsal_test_sam3), saliency of the v10
    cross-attention arm and the additive control, and the attention rows of results/detail_v10_3004."""
    D = f"{OUT}/object_saliency_v10/single_question_3004"
    owl, sam3 = f"{ROOT}/results/objsal_test/3004", f"{ROOT}/results/objsal_test_sam3/3004"
    frames(3004, f"{D}/frames")
    cols = {"sofa": "#e34948", "table": "#2a78d6", "nightstand": "#6250d6", "door": "#eda100", "towel": "#e87ba4"}
    mask_sheet(np.load(f"{owl}/masks.npz"), cols, f"{D}/frames", f"{D}/mask_sheet.jpg")
    mask_sheet(np.load(f"{sam3}/masks.npz"), cols, f"{D}/frames", f"{D}/mask_sheet_sam3.jpg")

    r = json.load(open(f"{owl}/report.json"))
    out = {"question": r["question"], "options": r["options"], "gt": r["ground_truth"], "maps": {}, "masks": {},
           "models": {"add": {}, "xa": {}}}
    for run in r["models"]:  # saliency maps of the first (OWLv2) run
        mp = np.load(f"{owl}/{run}_maps.npz")
        dest = sum(mp[k] for k in mp.files if k.startswith("dest_"))
        out["maps"]["xa" if "xa3" in run else "add"] = {n: q8(v, np.percentile(v, 99)) for n, v in
                                                       {"vis": mp["vis"], "dest": dest}.items()}
    for mk, dirn in (("owl", owl), ("sam3", sam3)):
        r = json.load(open(f"{dirn}/report.json"))
        M = np.load(f"{dirn}/masks.npz")
        out["masks"][mk] = {p: {"role": m["role"], "frames": m["frames_detected"], "cells": m["mean_cells_covered"],
                                "f": q8(M[p], 1.0), "det_frames": sorted(map(int, m["detections"])),
                                "scores": [d[0][0] for _, d in sorted(m["detections"].items(), key=lambda x: int(x[0]))]}
                            for p, m in r["masks"].items()}
        for run, x in r["models"].items():
            out["models"]["xa" if "xa3" in run else "add"][mk] = {
                "margin": x["margin"], "top": x["top_token"],
                "enr": {n: {p: (v["enrichment"] if v else None) for p, v in e.items()} for n, e in x["enrichment"].items()},
                "attn": x.get("object_attention")}

    Dd = f"{ROOT}/results/detail_v10_3004/2B_v10_xa3_qknu_s0"
    R, M = np.load(f"{Dd}/3004_attn_rows.npz"), np.load(f"{Dd}/3004_masks.npz")
    fr, rep = M["frame_of_cell"], json.load(open(f"{Dd}/3004_report.json"))
    P = list(cols)
    ex = {"objects": P, "present": {p: sorted(set(fr[M[p] > 0].tolist())) for p in P}, "rows": {}}
    for L in "012":
        for p in ("sofa", "table"):
            for kind in ("learned", "bias"):
                v = R[f"{L}:{p}:{kind}"]
                ex["rows"][f"{L}:{p}:{kind}"] = {
                    "frames": np.round(np.bincount(fr, weights=v, minlength=32), 4).tolist(),
                    "on": {o: round(float(v @ M[o]), 4) for o in P},
                    "cells": np.clip(np.round(255 * v / np.percentile(v, 99.5)), 0, 255).astype(int).tolist()}
    ex["scale"] = {L: rep["object_attention"][L]["scale"] for L in "012"}
    ex["flow"] = {L: rep["object_attention"][L]["flow_mass"] for L in "012"}
    fill("objsal_3004.html", f"{D}/index.html", DATA=dumps(out), EXAMPLE=dumps(ex))


def two():
    """Questions 3584 (cross-attention right, additive wrong) and 3655 (the reverse): SAM 3 masks, saliency, attention
    rows and J-lens readouts of results/detail_v10."""
    D = f"{OUT}/object_saliency_v10/two_direction_questions"
    B, RUNS = f"{ROOT}/results/detail_v10", {"xa": "2B_v10_xa3_qknu_s0", "add": "2B_v9_fmix_additive_s0"}
    info = json.load(open(f"{B}/{RUNS['xa']}/cells.json"))
    RD = {k: [json.loads(l) for l in open(f"{B}/{r}/readouts.jsonl")] for k, r in RUNS.items()}
    LAY = [0, 1, 2, 5, 8, 11, 14, 17, 20, 22]
    out = {}
    for q in (3584, 3655):
        rep = {k: json.load(open(f"{B}/{r}/{q}_report.json")) for k, r in RUNS.items()}
        M = np.load(f"{B}/{RUNS['xa']}/{q}_masks.npz")
        fr, P = M["frame_of_cell"], list(rep["xa"]["phrases"])
        frames(q, f"{D}/q{q}")
        mask_sheet(M, dict(zip(P, ["#e34948", "#2a78d6", "#6250d6"])), f"{D}/q{q}", f"{D}/q{q}/masks.jpg")
        Q = {"id": q, "question": rep["xa"]["question"], "options": rep["xa"]["options"], "gt": rep["xa"]["ground_truth"],
             "video": rep["xa"]["video"].split("/")[-2:], "objects": P, "roles": rep["xa"]["phrases"],
             "masks": {p: {"f": q8(M[p], 1), "frames": rep["xa"]["masks"][p]["frames_detected"],
                           "cells": rep["xa"]["masks"][p]["mean_cells_covered"],
                           "present": sorted(set(fr[M[p] > 0].tolist()))} for p in P},
             "models": {}, "maps": {}, "attn": {}, "jl": {}}
        for k in RUNS:
            r = rep[k]
            Q["models"][k] = {"answer": r["top_token"].strip(), "margin": r["margin"],
                              "enr": {n: {p: (v["enrichment"] if v else None) for p, v in r["enrichment"][n].items()}
                                      for n in ("vis", "dest_sum")},
                              "area": {p: (r["enrichment"]["vis"][p]["area"] if r["enrichment"]["vis"][p] else None)
                                       for p in P}}
            mp = np.load(f"{B}/{RUNS[k]}/{q}_maps.npz")
            d = sum(mp[x] for x in mp.files if x.startswith("dest_"))
            Q["maps"][k] = {"vis": q8(mp["vis"], np.percentile(mp["vis"], 99)), "dest": q8(d, np.percentile(d, 99))}
            R = [x for x in RD[k] if x["id"] == q]
            cellobj = {c: o for o, cs in info[str(q)]["own"].items() for c in cs}
            jl = {}
            for lens in ("visual", "text"):
                sub = [x for x in R if x["lens"] == lens]
                ranks = {kind: [] for kind in ("top", "own", "rand")}
                delta = {kind: [] for kind in ("top", "own", "rand")}
                for L in LAY:
                    for kind in ranks:
                        rows = [x for x in sub if x["layer"] == L and x["kind"] == kind]
                        if kind == "own":
                            v = [x["ranks"][f"q:{cellobj[x['cell']]}"] for x in rows]
                        else:
                            v = [min(x["ranks"][f"q:{o}"] for o in P) for x in rows]
                        ranks[kind].append(float(np.median(v)) if v else None)
                        dv = [x["delta"]["ratio"] for x in rows if "delta" in x]
                        delta[kind].append(float(np.median(dv)) if dv else None)
                fin = []
                for L in LAY:
                    x = [x for x in sub if x["layer"] == L and x["kind"] == "text"][-1]
                    fin.append({"gt": x["ranks"]["gt"], "front": x["ranks"]["sp:front"], "back": x["ranks"]["sp:back"],
                                "left": x["ranks"]["sp:left"], "right": x["ranks"]["sp:right"],
                                "top": [t for t, _ in x["raw"][:5]]})
                cells = {}
                for p in P:
                    c0 = info[str(q)]["own"][p][:1]
                    if not c0:
                        continue
                    rows = {x["layer"]: x for x in sub if x["kind"] == "own" and x["cell"] == c0[0]}
                    cells[p] = {"cell": c0[0], "frame": int(c0[0] // 300),
                                "raw": [[t for t, _ in rows[L]["raw"][:5]] for L in LAY],
                                "jspace": [[t for t, _ in rows[L].get("jspace", {}).get("top", [])[:5]] for L in LAY],
                                "delta": [[t for t, _ in rows[L]["delta"]["top"][:5]] if "delta" in rows[L] else []
                                          for L in LAY]}
                top0 = info[str(q)]["top"][0]
                rows = {x["layer"]: x for x in sub if x["kind"] == "top" and x["cell"] == top0}
                cells["_top"] = {"cell": top0, "frame": int(top0 // 300),
                                 "raw": [[t for t, _ in rows[L]["raw"][:5]] for L in LAY] if rows else []}
                jl[lens] = {"ranks": ranks, "delta": delta, "final": fin, "cells": cells}
            Q["jl"][k] = jl
        A = np.load(f"{B}/{RUNS['xa']}/{q}_attn_rows.npz")
        oa = rep["xa"]["object_attention"]
        Q["attn"] = {"pairs": {L: oa[L]["pairs"] for L in "012"}, "scale": {L: oa[L]["scale"] for L in "012"},
                     "flow": {L: oa[L]["flow_mass"] for L in "012"}, "rows": {}}
        for L in "012":
            for p in P:
                for kind in ("learned", "bias"):
                    v = A[f"{L}:{p}:{kind}"]
                    e = {"frames": np.round(np.bincount(fr, weights=v, minlength=32), 4).tolist(),
                         "on": {o: round(float(v @ M[o]), 4) for o in P}}
                    if kind == "learned":
                        e["cells"] = q8(v, np.percentile(v, 99.5))
                    Q["attn"]["rows"][f"{L}:{p}:{kind}"] = e
        out[q] = Q
    out["layers"] = LAY
    fill("two_questions.html", f"{D}/index.html", DATA=dumps(out))


def final():
    """VSI-Bench of the four final arms (2 seeds, decay branches 5750 / 8250 / 10750) and the attention-swap summary."""
    arms = {"additive": "2B_v9_fmix_additive", "xa3_qknu": "2B_v10_xa3_qknu", "xa3_nu": "2B_v10_xa3_nu",
            "xa3_nu_text": "2B_v11_xa3_nu_text"}
    out = {"scores": {}, "ablation": json.load(open(f"{ROOT}/results/ablation_v10/2B_v10_xa3_qknu_s0/summary_step5750.json"))}
    for a, r in arms.items():
        for s in (0, 1):
            for st in (5750, 8250, 10750):
                d = json.load(open(f"{ROOT}/results/train/{r}_s{s}/branches/step{st}/eval/vsi_f32_full_summary.json"))
                out["scores"].setdefault(a, {}).setdefault(str(s), {})[str(st)] = {
                    k: round(v, 2) for k, v in d.items() if isinstance(v, float)}
        args = json.load(open(f"{ROOT}/results/train/{r}_s0/args.json"))
        out.setdefault("args", {})[a] = {k: args.get(k) for k in ("qk_lr", "text_queries", "nu", "lr", "fusion_lr", "arm")}
    fill("final_results.html", f"{OUT}/final_results_v10_v11/index.html", DATA=dumps(out))


def jspace():
    sys.argv = ["build_page"]
    runpy.run_module("mvp.explain.jspace.build_page", run_name="__main__")
    print(f"{OUT}/jspace_v7/index.html")


if __name__ == "__main__":
    steps = {"objsal": objsal, "two": two, "final": final, "jspace": jspace}
    which = sys.argv[1:] or ["all"]
    for w in (list(steps) if which == ["all"] else which):
        if w not in steps:
            raise SystemExit(__doc__)
        steps[w]()
