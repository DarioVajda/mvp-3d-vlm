"""Summarise the attention-swap ablation (mvp.explain.swap_eval): accuracy per question type for the learned
cross-attention (the ordinary step eval), each swap mode and the additive control, with paired differences against
learned (bootstrap 95% CI over questions) and the flip counts. CPU only.

  python -m mvp.explain.swap_summary --run 2B_v10_xa3_qknu_s0 --control 2B_v9_fmix_additive_s0 --step 5750
"""
import argparse, json, os
import numpy as np

GROUPS = {"rel_direction": ("object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"),
          "rel_direction_easy": ("object_rel_direction_easy",), "rel_direction_medium": ("object_rel_direction_medium",),
          "rel_direction_hard": ("object_rel_direction_hard",), "rel_distance": ("object_rel_distance",),
          "route_planning": ("route_planning",)}


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--step", type=int, default=5750)
    ap.add_argument("--modes", nargs="+", default=["bias", "uniform", "diag"])
    ap.add_argument("--dir", default="results/ablation_v10")
    args = ap.parse_args()
    ev = lambda r: f"results/train/{r}/branches/step{args.step}/eval/vsi_f32_full.jsonl"
    learned, control = load(ev(args.run)), load(ev(args.control))
    modes = {m: load(f"{args.dir}/{args.run}/step{args.step}_{m}.jsonl") for m in args.modes
             if os.path.exists(f"{args.dir}/{args.run}/step{args.step}_{m}.jsonl")}
    ids = sorted(set.intersection(*(set(d) for d in modes.values())))
    rng = np.random.default_rng(0)
    out = {"run": args.run, "control": args.control, "step": args.step, "n": len(ids), "groups": {}}
    for g, types in GROUPS.items():
        q = [i for i in ids if learned[i]["question_type"] in types]
        if not q:
            continue
        s = lambda d: np.array([float(d[i]["score"]) for i in q])
        base = s(learned)
        row = {"n": len(q), "learned": round(100 * base.mean(), 2), "control": round(100 * s(control).mean(), 2)}
        boot = rng.integers(0, len(q), (2000, len(q)))
        for m, d in modes.items():
            x = s(d)
            diff = (x - base)[boot].mean(1) * 100
            row[m] = {"acc": round(100 * x.mean(), 2), "delta": round(100 * (x - base).mean(), 2),
                      "ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)],
                      "lost": int(((base >= .5) & (x < .5)).sum()), "gained": int(((base < .5) & (x >= .5)).sum())}
        out["groups"][g] = row
    os.makedirs(f"{args.dir}/{args.run}", exist_ok=True)
    json.dump(out, open(f"{args.dir}/{args.run}/summary_step{args.step}.json", "w"), indent=1)
    hdr = f"{'group':22s} {'n':>5s} {'learned':>8s} {'additive':>9s} " + " ".join(f"{m:>22s}" for m in modes)
    print(hdr)
    for g, r in out["groups"].items():
        cells = " ".join(f"{r[m]['acc']:6.1f} ({r[m]['delta']:+.1f} [{r[m]['ci95'][0]:+.1f},{r[m]['ci95'][1]:+.1f}])"
                         for m in modes)
        print(f"{g:22s} {r['n']:5d} {r['learned']:8.1f} {r['control']:9.1f} {cells}")


if __name__ == "__main__":
    main()
