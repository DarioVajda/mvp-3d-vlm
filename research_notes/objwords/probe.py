"""Probes on the recorded object-word activations (CPU). Fit on half the videos, test on the other half; shuffled-label control.

  rel_direction: multiclass direction word from the target / facing / start / answer token
  rel_distance:  per-candidate binary "is the closest" from each cand_k token (and from the ref token: index of the closest)
  route:         per-fill-in turn (L/R/B) from the fill_k token, and the cumulative heading (product of turns so far)
  abs_distance:  ridge regression to log distance from a, b and the answer token

  probe.py --dir results/objwords_v7
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np

LAYERS = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
HEAD = {"L": 1, "R": -1, "B": 2}


def load(d):
    meta = [json.loads(l) for l in open(f"{d}/meta.jsonl")]
    out = []
    for r in meta:
        p = f"{d}/{r['id']}.npz"
        if os.path.exists(p):
            z = np.load(p)
            out.append((r, {k: z[k] for k in z.files}))
    return out


def ridge_cls(X, y, Xt, lam=1.0):
    """One-hot ridge classifier."""
    cls = sorted(set(y)); Y = np.eye(len(cls))[[cls.index(v) for v in y]]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn, Xtn = (X - mu) / sd, (Xt - mu) / sd
    W = np.linalg.solve(Xn.T @ Xn + lam * np.eye(X.shape[1]), Xn.T @ (Y - Y.mean(0)))
    return np.array(cls)[(Xtn @ W + Y.mean(0)).argmax(1)]


def ridge_reg(X, y, Xt, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn, Xtn = (X - mu) / sd, (Xt - mu) / sd
    w = np.linalg.solve(Xn.T @ Xn + lam * np.eye(X.shape[1]), Xn.T @ (y - y.mean()))
    return Xtn @ w + y.mean()


def split(items, frac=0.5):
    vids = sorted({r["video"] for r, _ in items})
    rng = np.random.default_rng(0); rng.shuffle(vids)
    cal = set(vids[: int(len(vids) * frac)])
    return [it for it in items if it[0]["video"] in cal], [it for it in items if it[0]["video"] not in cal]


def feats(items, role_pat, cond, li):
    X, y, g = [], [], []
    for r, z in items:
        roles = {rr[0]: i for i, rr in enumerate(r["roles"])}
        for role, idx in roles.items():
            if not re.fullmatch(role_pat, role):
                continue
            lab = label(r, role)
            if lab is None:
                continue
            h = z[cond][li, idx].astype(np.float32)
            X.append(h); y.append(lab); g.append(r["video"])
    return np.array(X), np.array(y), g


def label(r, role):
    L, t = r["labels"], r["task"]
    if t == "object_rel_direction":
        return L["dir"]
    if t == "object_rel_distance":
        if role.startswith("cand_"):
            return int(int(role.split("_")[1]) == L["closest"])
        return L["closest"] if L["closest"] >= 0 else None
    if t == "route_planning":
        if role.startswith("fill_"):
            k = int(role.split("_")[1])
            return L["turns"][k] if k < len(L["turns"]) else None
        return None
    if t == "object_abs_distance":
        return L["log_dist"]
    return None


def heading_label(r, role):
    """Cumulative heading after the turns up to and including fill_k, as a signed quarter-turn count mod 4."""
    if r["task"] != "route_planning" or not role.startswith("fill_"):
        return None
    k = int(role.split("_")[1]); T = r["labels"]["turns"]
    if k >= len(T):
        return None
    return sum(HEAD[x] for x in T[:k + 1]) % 4


def run_probe(items, task, role_pat, cond, kind, name, out, lab_fn=None):
    its = [it for it in items if it[0]["task"] == task and cond in it[1]]
    if not its:
        return
    cal, test = split(its)
    rng = np.random.default_rng(1)
    res = {}
    for li, l in enumerate(LAYERS):
        global label
        if lab_fn is not None:
            saved = label; label = lab_fn
        X, y, _ = feats(cal, role_pat, cond, li); Xt, yt, _ = feats(test, role_pat, cond, li)
        if lab_fn is not None:
            label = saved
        if len(X) < 20 or len(Xt) < 10 or len(set(y.tolist())) < 2:
            continue
        if kind == "cls":
            acc = float((ridge_cls(X, y, Xt) == yt).mean())
            ys = y.copy(); rng.shuffle(ys)
            sh = float((ridge_cls(X, ys, Xt) == yt).mean())
            maj = float((yt == max(set(y.tolist()), key=list(y).count)).mean())
            res[l] = {"acc": round(acc, 3), "shuffled": round(sh, 3), "majority": round(maj, 3), "n_cal": len(X), "n_test": len(Xt)}
        else:
            pred = ridge_reg(X, y, Xt)
            r2 = 1 - ((pred - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
            ys = y.copy(); rng.shuffle(ys)
            r2s = 1 - ((ridge_reg(X, ys, Xt) - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
            res[l] = {"r2": round(float(r2), 3), "shuffled_r2": round(float(r2s), 3), "corr": round(float(np.corrcoef(pred, yt)[0, 1]), 3), "n_cal": len(X), "n_test": len(Xt)}
    out[name] = res
    key = "acc" if kind == "cls" else "r2"
    base = f" majority {next(iter(res.values()))['majority']:.2f}" if kind == "cls" and res else ""
    print(f"  {name:48} " + " ".join(f"L{l}:{v[key]:.2f}" for l, v in res.items()) + f" | shuffled ~{np.mean([v['shuffled' if kind == 'cls' else 'shuffled_r2'] for v in res.values()]):.2f}{base}" if res else f"  {name}: no data")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/objwords_v7"); args = ap.parse_args()
    report = {}
    for md in sorted(glob.glob(f"{args.dir}/*/meta.jsonl")):
        name = md.split("/")[-2]
        items = load(os.path.dirname(md))
        print(f"== {name}: {len(items)} questions", {t: sum(r['task'] == t for r, _ in items) for t in sorted({r['task'] for r, _ in items})})
        out = report.setdefault(name, {})
        for cond in ("h_on", "h_off"):
            if not any(cond in z for _, z in items):
                continue
            tag = "on" if cond == "h_on" else "off"
            for role in ("target", "facing", "start", "answer"):
                run_probe(items, "object_rel_direction", role, cond, "cls", f"rel_direction[{tag}] {role} -> direction", out)
            run_probe(items, "object_rel_distance", r"cand_\d", cond, "cls", f"rel_distance[{tag}] cand_k -> is closest", out)
            run_probe(items, "object_rel_distance", "answer", cond, "cls", f"rel_distance[{tag}] answer -> closest index", out)
            run_probe(items, "route_planning", r"fill_\d", cond, "cls", f"route[{tag}] fill_k -> turn k", out)
            run_probe(items, "route_planning", r"fill_\d", cond, "cls", f"route[{tag}] fill_k -> heading so far", out, lab_fn=heading_label)
            run_probe(items, "route_planning", "answer", cond, "cls", f"route[{tag}] answer -> first turn", out, lab_fn=lambda r, role: r["labels"]["turns"][0] if r["labels"]["turns"] else None)
            for role in ("a", "b", "answer"):
                run_probe(items, "object_abs_distance", role, cond, "reg", f"abs_distance[{tag}] {role} -> log distance", out)
        # behavioural baselines from the same records
        for t in sorted({r["task"] for r, _ in items}):
            rs = [r for r, _ in items if r["task"] == t]
            out[f"baseline {t}"] = {"margin_on": round(float(np.mean([r["margin_on"] for r in rs])), 3),
                                   "margin_off": round(float(np.mean([r["margin_off"] for r in rs if "margin_off" in r])), 3) if any("margin_off" in r for r in rs) else None, "n": len(rs)}
    json.dump(report, open(f"{args.dir}/probes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
