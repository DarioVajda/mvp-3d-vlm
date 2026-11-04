"""Object-word study on v7 checkpoints: record hidden states at the question's object words (geometry on / off), answers and
margins; and a small activation-patching pilot. No training.

Positions per question (text region only): the last subword of each object mention (role: start / facing / target / ref / cand_k /
a / b / landmark_k), each "[please fill in]" in route questions (fill_k), the final prompt token (answer), and 5 random non-object
text tokens (rand_k). Stored layers: LAYERS (output of decoder layer k, after the fusion hook).

Patching pilot (--patch N per task, --patch-layers): geometry-off forward with the hidden state at layer l overwritten, at chosen
positions, by the geometry-on values of the same question, measured by margin and generated answer. Conditions: obj (object words),
rand (as many random text tokens), donor (object words from the previous question of the same task, geometry on), allvis (every
visual token: re-inserts geometry at its source).

  objwords.py --models lora_s0 additive_s0 xa3_both_s0 --cap 200 --patch 50 --patch-layers 3 11 19 --out results/objwords_v7
"""
import argparse, json, os, re, sys, time
from collections import defaultdict
sys.path.insert(0, "src")
import numpy as np
import torch
from mvp.benchmarks.vsibench import NA_QUESTION_TYPES, build_prompt, load_vsibench
from mvp.fusion.inputs import uniform_frames
from mvp.fusion.model import FusionVL, stop_tokens

LAYERS = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
TASKS = ["object_rel_direction", "object_rel_distance", "route_planning", "object_abs_distance"]
AC = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
TURN = {"left": "L", "right": "R", "back": "B", "around": "B"}


def objects(doc):
    """[(role, phrase)] for the question, plus task labels."""
    q, t = doc["question"], doc["task"]
    lab = {}
    if t == "object_rel_direction":
        m = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) to (?:the|my) ", q)
        if not m: return None, None
        roles = [("start", m[1]), ("facing", m[2]), ("target", m[3])]
        opt = next(o for o in doc["options"] if o.strip()[0].upper() == str(doc["ground_truth"]).strip()[0].upper())
        lab["dir"] = opt.split(".", 1)[1].strip().lower()
    elif t == "object_rel_distance":
        m = re.search(r"which of these objects \((.+?)\) is the closest to the (.+?)\?", q)
        if not m: return None, None
        cands = [c.strip() for c in m[1].split(",")]
        roles = [("ref", m[2])] + [(f"cand_{i}", c) for i, c in enumerate(cands)]
        opt = next(o for o in doc["options"] if o.strip()[0].upper() == str(doc["ground_truth"]).strip()[0].upper())
        name = opt.split(".", 1)[1].strip().lower()
        lab["closest"] = next((i for i, c in enumerate(cands) if c.lower() == name), -1)
    elif t == "object_abs_distance":
        m = re.search(r"distance between the (.+?) and the (.+?) \(in meters\)", q)
        if not m: return None, None
        roles = [("a", m[1]), ("b", m[2])]
        lab["log_dist"] = float(np.log(max(float(doc["ground_truth"]), 0.05)))
    elif t == "route_planning":
        m = re.search(r"beginning at the (.+?) (?:and )?facing the (.+?)\. You want to navigate to the (.+?)\.", q)
        if not m: return None, None
        roles = [("start", m[1]), ("facing", m[2]), ("goal", m[3])]
        for i, lm in enumerate(re.findall(r"Go forward until the (.+?)(?= \d+\.| You have|\.)", q)):
            roles.append((f"landmark_{i}", lm))
        opt = next(o for o in doc["options"] if o.strip()[0].upper() == str(doc["ground_truth"]).strip()[0].upper())
        lab["turns"] = [TURN[x] for x in re.findall(r"turn (left|right|back|around)", opt.lower())]
    else:
        return None, None
    return roles, lab


def locate(tok, ids, roles, image_id, rng):
    """Token index of the last subword of each role's first mention after the images, the fill-ins, answer, random text tokens."""
    toks = [tok.decode([t]) for t in ids]
    text, off = "", []
    for s in toks:
        off.append((len(text), len(text + s))); text += s
    last_img = max(i for i, t in enumerate(ids) if t == image_id)
    region = off[last_img][1]
    low = text.lower()
    pos, spans = [], []
    for role, phrase in roles:
        m = re.search(r"(?<![a-z])" + re.escape(phrase.lower()) + r"(?![a-z])", low[region:])
        if not m:
            continue
        a, b = region + m.start(), region + m.end()
        idx = [i for i, (s, e) in enumerate(off) if s < b and e > a]
        pos.append((role, idx[-1], phrase)); spans += idx
    for k, m in enumerate(re.finditer(r"\[please fill in\]", low[region:])):
        a, b = region + m.start(), region + m.end()
        idx = [i for i, (s, e) in enumerate(off) if s < b and e > a]
        pos.append((f"fill_{k}", idx[-1], "[please fill in]")); spans += idx
    pos.append(("answer", len(ids) - 1, toks[-1]))
    cand = [i for i in range(last_img + 1, len(ids) - 1) if i not in spans and re.search(r"[a-z]", toks[i].lower())
            and not toks[i].lstrip().startswith("<") and not re.fullmatch(r"\s?[A-D]", toks[i])]  # no special tokens, no option letters
    for k, i in enumerate(rng.choice(cand, min(5, len(cand)), replace=False)):
        pos.append((f"rand_{k}", int(i), toks[i]))
    return pos


def targets(tok, doc):
    enc = lambda s: tok(s, add_special_tokens=False).input_ids[0]
    if doc["question_type"] in NA_QUESTION_TYPES:
        alts = [enc(str(i)) for i in range(10)]
        gt = enc(next(c for c in str(doc["ground_truth"]) if c.isdigit()))
    else:
        alts = [enc(o.strip()[0]) for o in doc["options"]]
        gt = enc(str(doc["ground_truth"]).strip()[0])
    return gt, [t for t in alts if t != gt]


class Hooks:
    """Record layer outputs at `pos` (or whole layers in `full`), and optionally overwrite positions at one layer (patch)."""

    def __init__(self, layers, record=(), pos=None, full=(), patch=None):
        self.rec, self.fullrec, self.handles = {}, {}, []
        self.pos, self.patch = pos, patch  # patch: (layer, index tensor, values [n, D])
        for i, layer in enumerate(layers):
            if i in record or i in full or (patch and i == patch[0]):
                self.handles.append(layer.register_forward_hook(self._hook(i, i in record, i in full)))

    def _hook(self, i, rec, full):
        def hook(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.patch and i == self.patch[0] and h.shape[1] > 1:  # prefill only
                h = h.clone(); h[0, self.patch[1]] = self.patch[2].to(h.dtype)
            if rec and h.shape[1] > 1:
                self.rec[i] = h[0, self.pos].detach().to(torch.float16).cpu()
            if full and h.shape[1] > 1:
                self.fullrec[i] = h[0].detach().clone()
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    def __enter__(self): return self

    def __exit__(self, *a):
        for h in self.handles: h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lora_s0", "additive_s0", "xa3_both_s0"])
    ap.add_argument("--cap", type=int, default=200, help="questions per task for recording")
    ap.add_argument("--patch", type=int, default=0, help="questions per task for the patching pilot")
    ap.add_argument("--patch-layers", type=int, nargs="+", default=[3, 11, 19])
    ap.add_argument("--out", default="results/objwords_v7")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_vsibench("full")
    df = df[df["task"].isin(TASKS)] if "task" in df else df
    picks = []
    for t in TASKS:
        g = df[df["question_type"].str.startswith(t)].sample(frac=1, random_state=0)
        picks.append(g.head(args.cap))
    import pandas as pd
    df = pd.concat(picks)
    docs = df.to_dict("records")
    for d in docs:
        d["task"] = re.sub(r"_(easy|medium|hard)$", "", d["question_type"])
    print(f"{len(docs)} questions", {t: sum(d['task'] == t for d in docs) for t in TASKS}, flush=True)
    for name in args.models:
        t0 = time.time()
        m = FusionVL.load(f"results/train/2B_v7_{name}/branches/step10750/final.pt", grad_ckpt=False).cuda().eval().requires_grad_(False)
        tok = m.processor.tokenizer
        layers = m.qwen.model.language_model.layers
        geo_arm = m.vggt is not None
        od = f"{args.out}/{name}"; os.makedirs(od, exist_ok=True)
        meta = open(f"{od}/meta.jsonl", "a")
        done = {json.loads(l)["id"] for l in open(f"{od}/meta.jsonl")} if os.path.getsize(f"{od}/meta.jsonl") else set()
        pat = open(f"{od}/patch.jsonl", "a")
        prev_obj = {}  # task -> (layer -> values) of the previous question's object words, for the donor control
        n_patch = defaultdict(int)
        unparsed = 0
        video, vis_in = None, None
        for qi, doc in enumerate(sorted(docs, key=lambda d: (d["task"], d["video_path"]))):
            roles, lab = objects(doc)
            if roles is None:
                unparsed += 1; continue
            if doc["video_path"] != video:
                video = doc["video_path"]
                frames, idx = uniform_frames(video, 32)
                vis_in = m.inputs.visual(frames)
            text = m.inputs.text([build_prompt(doc)], vis_in["image_grid_thw"], padding_side="left")
            batch = {k: v.cuda() for k, v in {**vis_in, **text}.items()}
            ids = batch["input_ids"][0].tolist()
            rng = np.random.default_rng(int(doc["id"]))
            pos = locate(tok, ids, roles, m.inputs.image_token_id, rng)
            P = torch.tensor([p[1] for p in pos], device="cuda")
            gt, alts = targets(tok, doc)
            inp = m.llm_inputs(batch)
            want_patch = n_patch[doc["task"]] < args.patch and geo_arm
            patch_layers = args.patch_layers if want_patch else []

            def run(geo_on, patch=None, gen=True, full=()):
                m._geo = None
                if geo_on and geo_arm:
                    m.set_geometry(batch)
                with Hooks(layers, record=set(LAYERS) if patch is None else (), pos=P, full=full, patch=patch) as hk, torch.no_grad(), AC():
                    hs = m.qwen.model(**inp, use_cache=False).last_hidden_state
                    lg = m.qwen.lm_head(hs[0, -1]).float()
                    margin = (lg[gt] - torch.logsumexp(lg[alts], 0)).item()
                    ans = None
                    if gen:
                        out = m.qwen.generate(**inp, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None, **stop_tokens(tok))
                        ans = tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()
                m._geo = None
                return margin, ans, hk
            if int(doc["id"]) not in done:
                m_on, a_on, h_on = run(True, full=patch_layers)
                arrays = {"pos": np.array([p[1] for p in pos]), "h_on": torch.stack([h_on.rec[l] for l in LAYERS]).numpy()}
                rec = {"id": int(doc["id"]), "task": doc["task"], "type": doc["question_type"], "video": video, "scene": doc.get("scene_name"), "gt": str(doc["ground_truth"]),
                       "roles": [[p[0], p[1], p[2]] for p in pos], "labels": lab, "n_tokens": len(ids), "margin_on": m_on, "answer_on": a_on}
                if geo_arm:
                    m_off, a_off, h_off = run(False)
                    arrays["h_off"] = torch.stack([h_off.rec[l] for l in LAYERS]).numpy()
                    rec.update(margin_off=m_off, answer_off=a_off)
                np.savez_compressed(f"{od}/{doc['id']}.npz", **arrays)
                meta.write(json.dumps(rec) + "\n"); meta.flush()
            else:
                h_on = run(True, full=patch_layers)[2] if want_patch else None
            # ---- patching pilot ----
            if want_patch and h_on is not None:
                obj_idx = [i for i, p in enumerate(pos) if not p[0].startswith(("rand_", "answer", "fill_"))]
                rand_idx = [i for i, p in enumerate(pos) if p[0].startswith("rand_")][:len(obj_idx)]
                vis_idx = torch.nonzero(batch["input_ids"][0] == m.inputs.image_token_id)[:, 0]
                prec = {"id": int(doc["id"]), "task": doc["task"], "gt": str(doc["ground_truth"]), "n_obj": len(obj_idx), "results": {}}
                for l in patch_layers:
                    H = h_on.fullrec[l]
                    conds = {"obj": (P[obj_idx], H[P[obj_idx]]), "rand": (P[rand_idx], H[P[rand_idx]]), "allvis": (vis_idx, H[vis_idx])}
                    if doc["task"] in prev_obj and l in prev_obj[doc["task"]]:
                        dv = prev_obj[doc["task"]][l]
                        k = min(len(obj_idx), len(dv))
                        conds["donor"] = (P[obj_idx][:k], dv[:k])
                    for c, (ix, vals) in conds.items():
                        mg, an, _ = run(False, patch=(l, ix, vals), gen=doc["task"] == "object_abs_distance")
                        prec["results"][f"{c}@{l}"] = {"margin": mg, "answer": an}
                    prev_obj.setdefault(doc["task"], {})[l] = H[P[obj_idx]].clone()
                pat.write(json.dumps(prec) + "\n"); pat.flush()
                n_patch[doc["task"]] += 1
                del h_on
            if qi % 20 == 0:
                print(f"  {name} [{qi + 1}/{len(docs)}] {time.time() - t0:.0f}s {doc['task']} roles {[p[0] for p in pos][:6]} mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GB", flush=True)
        print(f"== {name} done in {time.time() - t0:.0f}s, unparsed {unparsed}, patched {dict(n_patch)}", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
