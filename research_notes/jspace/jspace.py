"""J-lens / J-space readout on trained v7 checkpoints, at the cells the saliency maps marked as influential. No training.

Follows Anthropic's estimator (transformer-circuits.pub/2026/workspace; github.com/anthropics/jacobian-lens, fitting.py):
J_l = mean over source positions p of sum_{p' >= p} d h_final[p'] / d h_l[p]; the lens of h is unembed(J_l h).

Two lenses per model, because the paper fits on text only and our question is about visual tokens:
  text    wikitext prompts, 128 tokens, source = all positions past the first 16
  visual  VSI videos at 8 frames (the training protocol) with geometry on; source = the visual tokens only
Readouts, 32 frames (the evaluation protocol), same cells for every model:
  cells   top-5 cells of additive_s0's saliency map (sum of dest_0..2) and 5 random cells as the control; last 12 text tokens
  raw       top tokens of the lens at that (layer, position)
  contrast  top tokens of the lens minus the mean lens over 64 random cells of the video (what is specific to this cell)
  delta     geometry arms: top tokens of W_U J_l (h_on - h_off), what the injection itself pushes toward
  jspace    sparse nonnegative decomposition of h - mean(h) over the rows of W_U J_l (greedy, k = 25, NNLS refit)
  ranks     rank of spatial words, of the question's own words and of the correct answer, for the statistics

  jspace.py [--models lora_s0 additive_s0 xa3_both_s0] [--prompts 100] [--videos 6] [--per-type 2]
"""
import argparse, gzip, json, math, os, re, sys, time
sys.path.insert(0, "src")
import numpy as np
import torch
from mvp.benchmarks.vsibench import build_prompt, load_vsibench
from mvp.fusion.inputs import uniform_frames
from mvp.fusion.model import FusionVL

OUT = "results/jspace_v7"
SAL = "results/explain_v7/sal"
GLOSS = "/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal/research_notes/ext/jacobian-lens/assets/qwen_gloss.json.gz"
SOURCES = [0, 1, 2, 5, 8, 11, 14, 17, 20, 22]
SPATIAL = ["left", "right", "behind", "front", "near", "far", "close", "distance", "meter", "meters", "depth", "height", "width",
           "wide", "tall", "above", "below", "under", "over", "corner", "wall", "floor", "ceiling", "room", "direction", "position",
           "next", "between", "across", "side", "back", "forward", "camera", "view", "size", "large", "small", "big"]
STOP = set("the a an of in on to is are at by and or from with this that these those what which how many much if i am it be do does "
           "you your for as its each measuring closest point object objects answer question option letter given choices directly "
           "please using single numerical value standing facing first time frames video".split())
AC = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


class Rec:
    """Records the residual after each layer in `at` (after the fusion hooks, which were registered earlier). `leaf`: that
    layer's output becomes a leaf, so nothing before it enters the graph."""

    def __init__(self, layers, at, leaf=None):
        self.act, self.handles = {}, []
        for i in at:
            self.handles.append(layers[i].register_forward_hook(self._hook(i, leaf)))

    def _hook(self, i, leaf):
        def hook(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if i == leaf:
                h = h.detach().requires_grad_(True)
            self.act[i] = h
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def jacobian(m, run_forward, seq_mask, src_mask, dim_batch):
    """One sequence. seq_mask: valid target positions; src_mask: source positions averaged over. {layer: [D, D] CPU}."""
    layers = m.qwen.model.language_model.layers
    final, D = len(layers) - 1, m.qwen.config.text_config.hidden_size
    J = {l: torch.zeros(D, D) for l in SOURCES}
    with Rec(layers, [*SOURCES, final], leaf=SOURCES[0]) as rec, torch.enable_grad():
        run_forward()
        tgt = rec.act[final]
        srcs = [rec.act[l] for l in SOURCES]
        tpos, spos = seq_mask.nonzero()[:, 0].to(tgt.device), src_mask.nonzero()[:, 0].to(tgt.device)
        b = torch.arange(dim_batch, device=tgt.device)
        co = torch.zeros_like(tgt)
        starts = list(range(0, D, dim_batch))
        for pi, s in enumerate(starts):
            n = min(dim_batch, D - s)
            co.zero_()
            co[b[:n, None], tpos[None, :], s + b[:n, None]] = 1.0
            grads = torch.autograd.grad(tgt, srcs, grad_outputs=co, retain_graph=pi < len(starts) - 1)
            for l, g in zip(SOURCES, grads):
                J[l][s:s + n] = g[:n][:, spos].float().mean(1).cpu()
    return J


def fit_text(m, tok, prompts, dim_batch, path):
    if os.path.exists(path):
        return torch.load(path)
    tot, t0 = {l: 0 for l in SOURCES}, time.time()
    n = 0
    for pi, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt", truncation=True, max_length=128).input_ids.cuda()
        L = ids.shape[1]
        if L < 32:
            continue
        ids = ids.expand(dim_batch, -1)
        def run():
            with AC():
                m.qwen.model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
        mask = torch.zeros(L, dtype=torch.bool); mask[16:L - 1] = True
        m._geo = None
        J = jacobian(m, run, mask, mask, dim_batch)
        tot = {l: tot[l] + J[l] for l in SOURCES}
        n += 1
        if pi % 10 == 0:
            print(f"  text lens {pi + 1}/{len(prompts)} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GB", flush=True)
    out = {l: tot[l] / n for l in SOURCES}
    torch.save(out, path)
    return out


def fit_visual(m, docs, dim_batch, path):
    if os.path.exists(path):
        return torch.load(path)
    tot, t0 = {l: 0 for l in SOURCES}, time.time()
    for vi, doc in enumerate(docs):
        frames, _ = uniform_frames(doc["video_path"], 8)
        vis = m.inputs.visual(frames)
        text = m.inputs.text([build_prompt(doc)] * dim_batch, vis["image_grid_thw"], padding_side="left")
        batch = {k: v.cuda() for k, v in {**vis, **text}.items()}
        if m.vggt is not None:
            m.set_geometry(batch)
        inp = m.llm_inputs(batch)
        def run():
            with AC():
                m.qwen.model(**inp, use_cache=False)
        vm = (batch["input_ids"][0] == m.inputs.image_token_id).cpu()
        first = int(vm.nonzero()[0])
        tmask = torch.zeros_like(vm); tmask[first:-1] = True
        J = jacobian(m, run, tmask, vm, dim_batch)
        tot = {l: tot[l] + J[l] for l in SOURCES}
        m._geo = None
        print(f"  visual lens {vi + 1}/{len(docs)} {time.time() - t0:.0f}s, {vm.sum().item()} visual of {len(vm)} tokens, mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GB", flush=True)
    out = {l: tot[l] / len(docs) for l in SOURCES}
    torch.save(out, path)
    return out


class Reader:
    def __init__(self, m):
        self.m, self.tok = m, m.processor.tokenizer
        self.W = m.qwen.lm_head.weight.detach().float()
        self.norm = m.qwen.model.language_model.norm
        gl = json.load(gzip.open(GLOSS)) if os.path.exists(GLOSS) else {}
        self.gloss = {int(k): v for k, v in gl.items()}

    def name(self, t):
        s = self.tok.decode([t])
        g = self.gloss.get(int(t))
        return f"{s}({g})" if g else s

    def logits(self, h, J=None, use_norm=True):
        """h [N, D] float on GPU. J None is the logit lens."""
        x = h.float() if J is None else h.float() @ J.T
        if use_norm:
            x = self.norm(x.to(self.norm.weight.dtype if hasattr(self.norm, "weight") else torch.bfloat16)).float()
        return x @ self.W.T

    def top(self, lg, k=15):
        v, i = lg.topk(k)
        return [[self.name(t), round(x, 2)] for t, x in zip(i.tolist(), v.tolist())]

    def word_ids(self, words):
        out = {}
        for w in words:
            ids = set()
            for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                t = self.tok(f, add_special_tokens=False).input_ids
                if len(t) == 1:
                    ids.add(t[0])
            if ids:
                out[w] = sorted(ids)
        return out

    def ranks(self, lg, groups):
        order = lg.argsort(descending=True)
        rank = torch.zeros_like(order); rank[order] = torch.arange(len(order), device=order.device)
        return {g: min(int(rank[t]) for t in ids) for g, ids in groups.items()}


@torch.no_grad()
def decompose(x, V, Vn, k=25):
    """Greedy nonnegative decomposition of x [D] over the rows of V (normalised rows Vn): pick the row with the largest
    positive correlation with the residual, refit the coefficients by projected gradient. Returns ids, coefs, R^2."""
    x = x.float()
    sel, c = [], None
    r = x.clone()
    for _ in range(k):
        s = Vn @ r
        if sel:
            s[sel] = -1e9
        j = int(s.argmax())
        if s[j] <= 0:
            break
        sel.append(j)
        A = Vn[sel]
        G = A @ A.T
        b = A @ x
        c = torch.zeros(len(sel), device=x.device) if c is None else torch.cat([c, torch.zeros(1, device=x.device)])
        step = 1 / torch.linalg.matrix_norm(G, 2).clamp_min(1e-6)
        for _ in range(60):
            c = (c - step * (G @ c - b)).clamp_min(0)
        r = x - A.T @ c
    r2 = 1 - (r.norm() / x.norm().clamp_min(1e-9)).pow(2).item()
    return sel, ([] if c is None else c.tolist()), r2


def select_questions(per_type):
    load = lambda n: {r["id"]: r for r in map(json.loads, open(f"{SAL}/{n}/summary.jsonl"))}
    add, lo = load("additive_s0"), load("lora_s0")
    add1, lo1 = load("additive_s1"), load("lora_s1")
    by = {}
    for i, a in add.items():
        ok = lambda r: r[i]["top_token"] == r[i]["gt_token"]
        score = (ok(add) and ok(add1)) * 2 - (ok(lo) or ok(lo1)) * 1
        by.setdefault(a["question_type"], []).append((-score, i))
    pick = []
    for t, v in sorted(by.items()):
        v.sort()
        pick += [(i, -s) for s, i in v[:per_type]]
    return pick


def sanity(R, lenses, out):
    prompts = [("Fact: The currency used in the country shaped like a boot is", -2),
               ("The capital of the state containing Dallas is", -2),
               ("Q: What is the opposite of hot? A: The opposite of hot is", -2)]
    m, rows = R.m, []
    layers = m.qwen.model.language_model.layers
    for p, pos in prompts:
        ids = R.tok(p, return_tensors="pt").input_ids.cuda()
        m._geo = None
        with torch.no_grad(), Rec(layers, SOURCES) as rec, AC():
            m.qwen.model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
        for name, J in lenses.items():
            for l in SOURCES:
                lg = R.logits(rec.act[l][0, pos][None], J[l].cuda())[0]
                rows.append({"prompt": p, "lens": name, "layer": l, "top": R.top(lg, 8)})
    json.dump(rows, open(out, "w"))
    for r in rows:
        if r["layer"] in (8, 14, 20):
            print("  sanity", r["lens"], r["layer"], r["prompt"][:40], [x[0] for x in r["top"][:6]], flush=True)


def lens_quality(R, J, prompts, n=20):
    """Median rank of the model's own next token under the lens vs the logit lens, per layer: the paper's basic check."""
    m, res = R.m, {}
    layers = m.qwen.model.language_model.layers
    final = len(layers) - 1
    acc = {l: ([], []) for l in SOURCES}
    for p in prompts[-n:]:
        ids = R.tok(p, return_tensors="pt", truncation=True, max_length=128).input_ids.cuda()
        m._geo = None
        with torch.no_grad(), Rec(layers, [*SOURCES, final]) as rec, AC():
            m.qwen.model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
        idx = torch.arange(16, ids.shape[1] - 1, 8, device=ids.device)
        tgt = R.logits(rec.act[final][0, idx]).argmax(-1)
        for l in SOURCES:
            for out, Jx in zip(acc[l], (J[l].cuda(), None)):
                lg = R.logits(rec.act[l][0, idx], Jx)
                out.extend((lg > lg.gather(1, tgt[:, None])).sum(1).tolist())
    return {l: {"jlens_median_rank": float(np.median(a)), "logit_median_rank": float(np.median(b))} for l, (a, b) in acc.items()}


@torch.no_grad()
def readout(R, lenses, doc, cells_top, cells_rand, fout, model_name, cells_own=()):
    m = R.m
    frames, idx = uniform_frames(doc["video_path"], 32)
    vis = m.inputs.visual(frames)
    text = m.inputs.text([build_prompt(doc)], vis["image_grid_thw"], padding_side="left")
    batch = {k: v.cuda() for k, v in {**vis, **text}.items()}
    layers = m.qwen.model.language_model.layers
    if m.vggt is not None:
        m.set_geometry(batch)
    geo = m._geo
    inp = m.llm_inputs(batch)
    with Rec(layers, SOURCES) as rec, AC():
        hs = m.qwen.model(**inp, use_cache=False).last_hidden_state
    on = {l: rec.act[l][0].detach().float() for l in SOURCES}
    answer = R.tok.decode(m.qwen.lm_head(hs[0, -1]).argmax().item())
    off = None
    if geo is not None:
        m._geo = None
        with Rec(layers, SOURCES) as rec, AC():
            m.qwen.model(**inp, use_cache=False)
        off = {l: rec.act[l][0].detach().float() for l in SOURCES}
    ids = batch["input_ids"][0]
    vm = ids == m.inputs.image_token_id
    vidx = vm.nonzero()[:, 0]
    N = len(vidx)
    grid = batch["image_grid_thw"].tolist()
    W_m = grid[0][2] // m.inputs.merge
    per = N // len(grid)
    place = lambda n: (n // per, (n % per) // W_m, (n % per) % W_m)
    tail = (~vm).nonzero()[:, 0][-12:]
    words = [w for w in re.findall(r"[a-z]+", doc["question"].lower()) if w not in STOP and len(w) > 2]
    gt = str(doc["ground_truth"]).strip()
    from mvp.benchmarks.vsibench import NA_QUESTION_TYPES
    gt_first = next((c for c in gt if c.isdigit()), gt[:1]) if doc["question_type"] in NA_QUESTION_TYPES else gt[:1]
    groups = {**{f"sp:{w}": v for w, v in R.word_ids(SPATIAL).items()}, **{f"q:{w}": v for w, v in R.word_ids(sorted(set(words))).items()},
              "gt": sorted({t for ids_ in (R.tok(f, add_special_tokens=False).input_ids for f in (gt_first, " " + gt_first)) if len(ids_) == 1 for t in ids_})}
    rng = np.random.default_rng(int(doc["id"]))
    bg = torch.as_tensor(rng.choice(N, min(64, N), replace=False), device=vidx.device)
    for name, J in lenses.items():
        for l in SOURCES:
            Jl = J[l].cuda()
            V = R.W @ Jl
            Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            bg_lg = R.logits(on[l][vidx[bg]], Jl).log_softmax(-1).mean(0)
            mean_h = on[l][vidx].mean(0)
            items = [("top", int(c), int(vidx[c])) for c in cells_top] + [("rand", int(c), int(vidx[c])) for c in cells_rand] + \
                    [("own", int(c), int(vidx[c])) for c in cells_own] + [("text", -1, int(p)) for p in tail.tolist()]
            lgs = R.logits(on[l][[i[2] for i in items]], Jl)
            for (kind, c, p), lg in zip(items, lgs):
                rec_ = {"model": model_name, "id": int(doc["id"]), "type": doc["question_type"], "lens": name, "layer": l, "kind": kind, "cell": c,
                        "place": place(c) if c >= 0 else None, "tok": R.tok.decode(ids[p].item()), "answer": answer, "gt": gt,
                        "raw": R.top(lg), "contrast": R.top(lg.log_softmax(-1) - bg_lg), "ranks": R.ranks(lg, groups)}
                if kind != "rand" and name == "visual":
                    sel, co, r2 = decompose(on[l][p] - mean_h, V, Vn)
                    order = np.argsort(co)[::-1]
                    rec_["jspace"] = {"r2": round(r2, 4), "top": [[R.name(sel[o]), round(co[o], 3)] for o in order[:25]]}
                if off is not None and kind != "text":
                    d = on[l][p] - off[l][p]
                    dl = R.logits(d[None], Jl)[0]  # RMS scaling is a scalar, so this only applies the norm's channel weights
                    rec_["delta"] = {"norm": round(d.norm().item(), 3), "ratio": round((d.norm() / off[l][p].norm().clamp_min(1e-9)).item(), 3),
                                     "top": R.top(dl), "ranks": R.ranks(dl, groups)}
                fout.write(json.dumps(rec_, ensure_ascii=False) + "\n")
    fout.flush()
    m._geo = None
    return answer


def wiki_prompts(n):
    cached = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikitext_prompts.json")
    if os.path.exists(cached):  # prefetched on the login node; compute nodes may have no internet
        return json.load(open(cached))[:n]
    from datasets import load_dataset
    out = []
    for r in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True):
        if len(r["text"].strip()) >= 600:
            out.append(r["text"])
            if len(out) == n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lora_s0", "additive_s0", "xa3_both_s0"])
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prompts = wiki_prompts(args.prompts + 20)
    fit_p, eval_p = prompts[:args.prompts], prompts[args.prompts:]
    df = load_vsibench("full")
    pick = select_questions(args.per_type)
    sel_ids = {i for i, _ in pick}
    json.dump(pick, open(f"{args.out}/questions.json", "w"))
    vids = df[~df["video_path"].isin(df[df["id"].isin(sel_ids)]["video_path"])].drop_duplicates("video_path").sample(args.videos, random_state=0)
    fit_docs = vids.to_dict("records")
    by_id = {int(r["id"]): r for r in df[df["id"].isin(sel_ids)].to_dict("records")}
    print(f"{len(pick)} questions, {len(fit_p)} text prompts, {len(fit_docs)} fit videos", flush=True)
    for name in args.models:
        t0 = time.time()
        m = FusionVL.load(f"results/train/2B_v7_{name}/branches/step10750/final.pt", grad_ckpt=False).cuda().eval().requires_grad_(False)
        R = Reader(m)
        print(f"== {name} arm {m.arm}", flush=True)
        os.makedirs(f"{args.out}/{name}", exist_ok=True)
        lenses = {"text": fit_text(m, R.tok, fit_p, 32, f"{args.out}/{name}/lens_text.pt"),
                  "visual": fit_visual(m, fit_docs, 16, f"{args.out}/{name}/lens_visual.pt")}
        print(f"  lenses fitted {time.time() - t0:.0f}s", flush=True)
        sanity(R, lenses, f"{args.out}/{name}/sanity.json")
        q = {n: lens_quality(R, J, eval_p, min(20, len(eval_p))) for n, J in lenses.items()}
        json.dump(q, open(f"{args.out}/{name}/lens_quality.json", "w"), indent=1)
        print("  quality", {n: {l: v[l] for l in (8, 14, 20)} for n, v in q.items()}, flush=True)
        with open(f"{args.out}/{name}/readouts.jsonl", "w") as fout:
            for i, (qid, _) in enumerate(pick):
                z = np.load(f"{SAL}/additive_s0/{qid}.npz")
                sal = sum(z[f"dest_{k}"] for k in ("0", "1", "2"))
                top = np.argsort(sal)[::-1][:5]
                rand = np.random.default_rng(qid).choice(len(sal), 5, replace=False)
                own = ()  # cross-attention: also read the 5 most influential cells of its own saliency map ("own")
                if name.startswith("xa3"):
                    z2 = np.load(f"{SAL}/{name}/{qid}.npz")
                    own = np.argsort(sum(z2[f"dest_{k}"] for k in ("0", "1", "2")))[::-1][:5]
                ans = readout(R, lenses, by_id[qid], top, rand, fout, name, own)
                print(f"  [{i + 1}/{len(pick)}] id {qid} {by_id[qid]['question_type']} answer {ans!r} gt {by_id[qid]['ground_truth']!r} {time.time() - t0:.0f}s", flush=True)
        del m, R, lenses
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
