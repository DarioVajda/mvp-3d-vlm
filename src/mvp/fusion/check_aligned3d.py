"""GPU checks of the 3D position signals of the xattn_aligned arm (pe="3d", mvp.fusion.points) before training.
The grid path (v5) is covered by mvp.fusion.check_aligned; check 0 here shows it is unchanged.

  Module (random weights, synthetic point maps, no Qwen):
  0. pe="grid" matches the committed (v5) AlignedXAttn: same init, bit-identical outputs, gradients within 1e-3
     relative (the committed module's own run-to-run difference, printed alongside, is ~1e-5: SDPA backward is not
     bitwise deterministic).
  1. cell_points against a brute-force reference: 28 x 28 px cell means in Qwen's token order, the low-confidence
     cells (at VGGT's confidence floor, or in the bottom 10% of the rest), the scale (median 3D gap of horizontally adjacent confident cells; garbage in low cells does not move
     it), distances (clip at d_max, low-confidence keys at d_max, own cell 0); a frame shifted by two cells over a
     plane has its corresponding cells at distance 0.
  2. 3D RoPE against an independent implementation of the plan (8 pairs each for t~, x, y, z; t~ frequencies
     100^(-j/8); x, y, z frequencies 2 pi / w_j, w_j geometric from 1.5 to 100 cells; NeoX half split); q.k
     depends only on the (t~, x, y, z) offset (shared translation of the scene leaves it unchanged); equal positions
     leave q.k unchanged; a shift by one wavelength of a pair turns that pair once around; dims 64-127 untouched.
  3. 3D bias: mask = -nu_h * dist, against the plan's formula.
  4. SDPA (bf16, with the 3D mask) against an eager fp32 reference built from the independent RoPE and bias:
     outputs and gradients of nu, W_q, W_k, W_v, W_o and the inputs; for all three (rope, bias) settings.
  5. Special cases: zero W_o -> zero update; no content, steep nu -> token n reads geometry token n where no other
     cell is at distance 0, and the mean of its 3D duplicates where there are; no content, no bias -> the mean;
     correspondence mass is 0 when no other-frame cell is within 1 cell and 1 when the bias puts all mass there.
  6. Memory of a 32-frame forward (N = 9600) with the 3D bias, 8 questions, no grad.
  FusionVL (Qwen3.5 + VGGT point head, a real 8-frame micro-batch), for (rope, bias) in (0, 1), (1, 0), (1, 1):
  7. Point head output on real frames: shapes, scale, how far apart adjacent cells and other-frame nearest cells are
     after scaling, low-confidence fraction.
  8. Init loss == base model's; own-cell / own-frame / correspondence mass at init (nu = 1.5).
  9. Perturbed W_o changes the loss; gradients reach LoRA, mergers, W_q/W_k/W_v/W_o and nu.
  10. Generation, save/load round trip (config carries pe), time and peak memory of 8 questions fwd + bwd.

  slurm/submit.sh -J check_aligned3d -t 00:40:00 -- python -m mvp.fusion.check_aligned3d
"""
import argparse
import importlib.util
import math
import os
import subprocess
import tempfile
import time

import torch
import torch.nn.functional as F

from mvp.fusion.check import micro_batch
from mvp.fusion.check_aligned import grid, ok


def warn(cond, msg):  # a plan expectation on real data: reported, reviewed before training, does not stop the checks
    print(("PASS " if cond else "WARN ") + msg, flush=True)
from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import FusionVL, count_params
from mvp.fusion.modules import AlignedXAttn
from mvp.fusion.points import ScenePoints, cell_points
from mvp.paths import ROOT
from mvp.train.data import to_device

DEV = "cuda"
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---- independent implementations of the plan (spec §9.2-9.4), used as references ----
def ref_cells(points, conf, grid_, d_max, low_q=0.1):
    """Brute force: loop over cells."""
    xyz, cc = [], []
    for t, (_, h, w) in enumerate(grid_.tolist()):
        for r in range(h // 2):
            for c in range(w // 2):
                xyz.append(points[t, 28 * r:28 * r + 28, 28 * c:28 * c + 28].reshape(-1, 3).double().mean(0))
                cc.append(conf[t, 28 * r:28 * r + 28, 28 * c:28 * c + 28].double().mean())
    xyz, cc = torch.stack(xyz), torch.stack(cc)
    floor = cc <= 1 + 1e-6
    thr = torch.quantile(cc[~floor].float(), low_q).double()
    low = floor | (cc < thr)
    gaps, n = [], 0
    for t, (_, h, w) in enumerate(grid_.tolist()):
        hc, wc = h // 2, w // 2
        for r in range(hc):
            for c in range(wc - 1):
                i = n + r * wc + c
                if not low[i] and not low[i + 1]:
                    gaps.append((xyz[i + 1] - xyz[i]).norm())
        n += hc * wc
    s = torch.stack(gaps).median()
    xyz = xyz / s
    N = len(xyz)
    dist = torch.zeros(N, N, dtype=torch.double, device=xyz.device)
    for i in range(N):
        dist[i] = (xyz - xyz[i]).norm(dim=-1)
    dist = dist.clamp(max=d_max)
    dist[:, low] = d_max
    dist[torch.arange(N), torch.arange(N)] = 0
    return xyz, low, dist, s


def ref_angles(pos, xyz, T, train_frames=8):
    """[N, 32] rotary angles: pairs 0-7 t~, 8-15 x, 16-23 y, 24-31 z."""
    tt = pos[:, 0].double() * (train_frames - 1) / max(T - 1, 1)
    f_t = torch.tensor([100.0 ** (-j / 8) for j in range(8)], dtype=torch.double, device=pos.device)
    w = torch.tensor([1.5 * (100 / 1.5) ** (j / 7) for j in range(8)], dtype=torch.double, device=pos.device)
    f_s = 2 * math.pi / w
    return torch.cat([tt[:, None] * f_t, xyz[:, :1].double() * f_s, xyz[:, 1:2].double() * f_s, xyz[:, 2:3].double() * f_s], 1)


def ref_rotate(x, ang):  # x [..., N, 128]; NeoX half split over the first 64 dims: pairs (i, i + 32), i < 32
    x = x.double()
    c, s = ang.cos(), ang.sin()
    x1, x2, rest = x[..., :32], x[..., 32:64], x[..., 64:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, rest], -1)


def ref_forward(blk, y, g, pos, pts):
    """Eager fp64 AlignedXAttn.forward from the plan's formulas."""
    B, N, D = y.shape
    H, d = blk.heads, blk.head_dim
    split = lambda t: t.view(t.shape[0], N, H, d).transpose(1, 2)
    gn = F.layer_norm(g.double(), (D,), blk.norm_kv.weight.double(), blk.norm_kv.bias.double())[None]
    q = split(F.layer_norm(y.double(), (D,), blk.norm_q.weight.double(), blk.norm_q.bias.double()) @ blk.q.weight.double().T)
    k, v = split(gn @ blk.k.weight.double().T), split(gn @ blk.v.weight.double().T)
    if blk.rope:
        ang = ref_angles(pos, pts.xyz, int(pos[:, 0].max()) + 1)
        q, k = ref_rotate(q, ang), ref_rotate(k, ang)
    logits = q @ k.transpose(-1, -2) / math.sqrt(d)
    if blk.local_bias:
        logits = logits - blk.nu.double()[:, None, None] * pts.dist.double()
    out = logits.softmax(-1) @ v
    return out.transpose(1, 2).reshape(B, N, D) @ blk.o.weight.double().T


def synthetic_scene(T=8, a=0.01, shift=2, seed=0):
    """Frames of a textured plane z = 3 seen by a camera that moves `shift` cells to the right per frame: pixel (v, u)
    of frame t is the plane point (x, a v, 3 + small relief(x)), x = a (u + 28 shift t). Confidence is constant per cell:
    15% of cells at the floor (1.0), 10% of the others in (1.01, 1.49), the rest in (1.5, 2.5); all 23.5% low cells
    carry garbage points (1e3) that must not move the scale."""
    g = torch.Generator().manual_seed(seed)
    v, u = torch.meshgrid(torch.arange(420.0), torch.arange(560.0), indexing="ij")
    x = lambda t: a * (u + 28 * shift * t)  # world x of pixel column u in frame t; relief depends on the world point only
    pts = torch.stack([torch.stack([x(t), a * v, 3 + 0.001 * torch.sin(x(t) / 0.07)], -1) for t in range(T)])
    cconf = 1.5 + torch.rand(T, 15, 20, generator=g)
    perm = torch.randperm(T * 300, generator=g)
    floorc = torch.zeros(T * 300, dtype=torch.bool)  # 15% of cells at VGGT's confidence floor (exactly 1.0)
    floorc[perm[:T * 45]] = True
    lowc = torch.zeros(T * 300, dtype=torch.bool)  # plus 10% of the other cells clearly below the rest
    n_rest = T * 300 - T * 45
    lowc[perm[T * 45:T * 45 + n_rest // 10]] = True
    floorc, lowc = floorc.view(T, 15, 20), lowc.view(T, 15, 20)
    cconf[lowc] = 1.01 + 0.48 * torch.rand(int(lowc.sum()), generator=g)
    cconf[floorc] = 1.0
    lowc = lowc | floorc
    conf = cconf.repeat_interleave(28, 1).repeat_interleave(28, 2)
    pts[lowc.repeat_interleave(28, 1).repeat_interleave(28, 2)] = 1e3
    return pts.to(DEV), conf.to(DEV), a


def head_module():
    """The committed (HEAD) mvp.fusion.modules, imported from git under another name."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:src/mvp/fusion/modules.py"], capture_output=True,
                         text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "modules_head.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("modules_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module_checks():
    D, H = 2048, 16
    T = 8
    pos = AlignedXAttn.positions(grid(T))
    N = len(pos)

    # 0. grid path unchanged
    old = head_module().AlignedXAttn
    y = torch.randn(2, N, D, device=DEV)
    g = torch.randn(N, D, device=DEV)
    def run(cls, rope, bias):  # init with a fixed seed, perturbed W_o, bf16 forward + backward
        torch.manual_seed(1); m = cls(D, H, rope, bias).to(DEV)
        torch.manual_seed(2); torch.nn.init.normal_(m.o.weight, std=0.02)
        yy = y.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(yy, g, pos)
        o.float().square().sum().backward()
        return m, o, [yy.grad] + [p.grad for p in m.parameters()]
    for rope, bias in [(False, False), (True, False), (False, True), (True, True)]:
        a, oa, ga = run(old, rope, bias)
        a2, oa2, ga2 = run(old, rope, bias)  # run-to-run noise of the committed module itself (SDPA backward kernels)
        b, ob, gb = run(AlignedXAttn, rope, bias)
        sa, sb = a.state_dict(), b.state_dict()
        same_init = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        noise = max(rel(x, z) for x, z in zip(ga, ga2))
        diff = max(rel(x, z) for x, z in zip(ga, gb))
        ok(same_init and torch.equal(oa, ob) and diff < 1e-3,  # SDPA backward is not bitwise reproducible (~1e-5)
           f"pe='grid' rope={int(rope)} bias={int(bias)}: same init {same_init}, output bit-identical {torch.equal(oa, ob)} "
           f"(old vs old {torch.equal(oa, oa2)}), gradient diff {diff:.2e} vs the committed module's own run-to-run {noise:.2e}")

    # 1. cell_points
    P, C, a_px = synthetic_scene(T)
    sc = cell_points(P, C, grid(T), d_max=10.0)
    ok(bool(sc.low[(C[:, ::28, ::28] <= 1).flatten()].all()), "every floor cell is low")
    rx, rlow, rdist, rs = ref_cells(P, C, grid(T), 10.0)
    ok(sc.xyz.shape == (N, 3) and sc.dist.shape == (N, N), f"ScenePoints shapes xyz {tuple(sc.xyz.shape)}, dist {tuple(sc.dist.shape)}")
    expect = (T * 45 + (T * 255) // 10) / (T * 300)
    ok(torch.equal(sc.low.cpu(), rlow.cpu()) and abs(sc.low.float().mean().item() - expect) < 1e-3,
       f"low-confidence cells (floor or bottom 10% of the rest) = reference, fraction {sc.low.float().mean().item():.3f} (expected {expect:.3f})")
    ok(abs(sc.scale - rs.item()) < 1e-6 * rs.item() and abs(sc.scale - 28 * a_px) < 1e-4,
       f"scale {sc.scale:.5f} = reference {rs.item():.5f} = 28 px x {a_px} (garbage in low cells ignored)")
    ok(torch.allclose(sc.xyz.double(), rx, rtol=1e-5, atol=1e-3), f"cell means in token order = reference (max err {(sc.xyz.double() - rx).abs().max().item():.2e})")
    ok(torch.allclose(sc.dist.double(), rdist, atol=1e-3), f"distances = reference (max err {(sc.dist.double() - rdist).abs().max().item():.2e})")
    i = 87  # frame 0, (r 4, c 7); frame 1 moved 2 cells right, so the same surface is at (4, 5) of frame 1
    j = 300 + 4 * 20 + 5
    if not (sc.low[i] or sc.low[j]):
        ok(sc.dist[i, j].item() < 0.01 and abs(sc.dist[i, i + 1].item() - 1) < 0.01,
           f"same surface in the next frame at dist {sc.dist[i, j].item():.4f}; in-frame neighbour {sc.dist[i, i + 1].item():.4f} cells")
    ok(sc.dist.max().item() == 10.0 and bool((sc.dist[:, sc.low][~sc.low] == 10.0).all()) and bool((sc.dist.diagonal() == 0).all()),
       "dist clipped at d_max = 10, low-confidence keys at d_max, own cell 0")

    # 2. 3D RoPE
    blk = AlignedXAttn(D, H, rope=True, local_bias=True, pe="3d").to(DEV)
    ok(blk.rope_axis.tolist() == [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8, "rotary pairs 0-7 t~, 8-15 x, 16-23 y, 24-31 z")
    cos, sin = blk.rotary(pos, sc)
    ang = ref_angles(pos, sc.xyz, T)
    err = torch.maximum((cos.double() - ang.cos()).abs(), (sin.double() - ang.sin()).abs())
    bound = 4 * 2 ** -23 * ang.abs() + 1e-6  # fp32 rounding of the angle; the garbage low cells sit ~3600 cells out
    ok(err[~sc.low].max().item() < 1e-5 and bool((err <= bound).all()),
       f"angles = plan (wavelengths {[round(2 * math.pi / f, 2) for f in blk.rope_freq[8:16].tolist()]} cells; max err "
       f"{err[~sc.low].max().item():.1e} on confident cells, within fp32 rounding on the far low-confidence ones)")
    d = blk.head_dim
    q, k = torch.randn(1, 1, 1, d, device=DEV), torch.randn(1, 1, 1, d, device=DEV)

    def dot(pq, pk):  # pq, pk: (t, x, y, z)
        P4 = torch.tensor([pq, pk], device=DEV, dtype=torch.float)
        ps = torch.cat([P4[:, :1].long(), torch.zeros(2, 2, dtype=torch.long, device=DEV)], 1)
        ps = torch.cat([ps, torch.tensor([[T - 1, 0, 0]], device=DEV)])  # T = 8 so t~ = t
        cs, sn = blk.rotary(ps, ScenePoints(torch.cat([P4[:, 1:], torch.zeros(1, 3, device=DEV)]), None, None, 1.0))
        rq, rk = blk.apply_rotary(q, cs[:1], sn[:1]), blk.apply_rotary(k, cs[1:2], sn[1:2])
        return (rq * rk).sum().item(), rq
    base, _ = dot([1, 2.0, -3.5, 7.25], [4, 9.5, 1.0, 3.0])
    moved, _ = dot([1, 2.0 + 13.7, -3.5 - 4.2, 7.25 + 30.1], [4, 9.5 + 13.7, 1.0 - 4.2, 3.0 + 30.1])  # scene translated
    ok(abs(base - moved) < 1e-3 * max(1, abs(base)), f"q.k invariant to translating the scene ({base:.4f} vs {moved:.4f})")
    other, _ = dot([1, 2.0, -3.5, 7.25], [4, 9.5, 1.5, 3.0])
    ok(abs(base - other) > 1e-3, f"q.k changes with the 3D offset ({base:.4f} vs {other:.4f})")
    same, rq = dot([3, 5.0, 6.0, 7.0], [3, 5.0, 6.0, 7.0])
    ok(abs(same - (q * k).sum().item()) < 1e-3, "equal positions leave q.k unchanged")
    ok(torch.allclose(rq.norm(), q.norm(), atol=1e-4) and torch.equal(rq[..., 64:], q[..., 64:]), "norm preserved, dims 64-127 untouched")
    one = blk.rotary(torch.zeros(2, 3, dtype=torch.long, device=DEV),
                     ScenePoints(torch.tensor([[0.0, 0, 0], [1.5, 0, 0]], device=DEV), None, None, 1.0))
    ok(abs(one[0][1, 8].item() - 1) < 1e-5 and abs(one[1][1, 8].item()) < 1e-5 and abs(one[0][1, 9].item() - 1) > 0.1,
       "an x shift of 1.5 cells turns pair 8 (shortest wavelength) exactly once around, pair 9 not")
    t32 = AlignedXAttn.positions(grid(32))
    sc32 = ScenePoints(torch.zeros(len(t32), 3, device=DEV), None, None, 1.0)
    c32, _ = blk.rotary(t32, sc32)
    c8, _ = blk.rotary(pos, ScenePoints(torch.zeros(N, 3, device=DEV), None, None, 1.0))
    ok(torch.allclose(c32[-1], c8[-1]), "t~ rescaling as in the grid path: last of 32 frames = last of 8")

    # 3. 3D bias
    with torch.no_grad():
        blk.nu.copy_(torch.linspace(0.5, 3, H, device=DEV))
    m = blk.bias(pos, torch.float, sc)[0]
    ok(torch.allclose(m, -blk.nu[:, None, None] * rdist.float(), atol=1e-4), "bias = -nu_h * min(|xyz_i - xyz_j|, d_max) (low keys at d_max)")

    # 4. SDPA vs eager reference, all three settings
    for rope, bias in [(False, True), (True, False), (True, True)]:
        torch.manual_seed(3)
        b = AlignedXAttn(D, H, rope=rope, local_bias=bias, pe="3d").to(DEV)
        torch.nn.init.normal_(b.o.weight, std=0.02)
        if bias:
            with torch.no_grad():
                b.nu.copy_(torch.linspace(0.5, 3, H, device=DEV))
        yy = torch.randn(2, N, D, device=DEV, requires_grad=True)
        gg = torch.randn(N, D, device=DEV, requires_grad=True)
        params = [yy, gg, b.q.weight, b.k.weight, b.v.weight, b.o.weight] + ([b.nu] if bias else [])
        names = ["y", "g", "W_q", "W_k", "W_v", "W_o"] + (["nu"] if bias else [])
        ref = ref_forward(b, yy, gg, pos, sc)
        gref = torch.autograd.grad(ref.square().sum(), params)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = b(yy, gg, pos, sc)
        gout = torch.autograd.grad(out.float().square().sum(), params)
        tag = f"rope={int(rope)} bias={int(bias)}"
        ok(rel(out, ref) < 0.03, f"[{tag}] SDPA bf16 output vs eager fp64 plan: rel err {rel(out, ref):.4f}")
        for name, x, r in zip(names, gout, gref):
            ok(r.norm() > 0 and rel(x, r) < 0.05, f"[{tag}] grad {name}: rel err {rel(x, r):.4f}, |grad| {r.norm():.3e}")

    # 5. special cases
    with torch.no_grad():
        b0 = AlignedXAttn(D, H, pe="3d").to(DEV)
        ok(b0(y, g, pos, sc).abs().max().item() == 0, "zero-initialised W_o: the update is exactly 0")
        gn = F.layer_norm(g, (D,))
        for bias in (True, False):
            b = AlignedXAttn(D, H, rope=True, local_bias=bias, pe="3d").to(DEV)
            b.q.weight.zero_(); b.k.weight.zero_(); b.v.weight.copy_(torch.eye(D)); b.o.weight.copy_(torch.eye(D))
            if bias:
                b.nu.fill_(60.0)
            o = b(y, g, pos, sc)[0]
            if bias:  # steep: softmax over the keys at distance exactly 0 (own cell, and its duplicates in 3D)
                zero = (sc.dist < 1e-3).float()
                target = (zero / zero.sum(1, keepdim=True)) @ gn
            else:
                target = gn.mean(0, keepdim=True).expand_as(gn)
            ok(rel(o, target) < 2e-3, ("steep nu, no content: each token reads the mean of the cells at 3D distance 0 (its own, "
                                       "and the same surface in other frames)" if bias else "no bias, no content: every token reads the mean") + f", rel err {rel(o, target):.2e}")
        # correspondence mass: distinct points everywhere -> 0; bias that puts all mass on the next frame's duplicate -> 1
        lone = ScenePoints(torch.randn(N, 3, device=DEV) * 100, torch.zeros(N, dtype=torch.bool, device=DEV), None, 1.0)
        lone.dist = torch.cdist(lone.xyz, lone.xyz).clamp_max(10.0)
        st = b.attn_stats(y, g, pos, lone)
        ok(st["corr_mass"] < 1e-3, f"correspondence mass with no 3D duplicates: {st['corr_mass']:.2e}")
        bb = AlignedXAttn(D, H, rope=False, local_bias=True, pe="3d").to(DEV)
        bb.q.weight.zero_(); bb.k.weight.zero_(); bb.nu.fill_(60.0)
        dup = ScenePoints(sc.xyz, sc.low, sc.dist.clone(), sc.scale)
        dup.dist.fill_diagonal_(10.0)  # own cell made expensive: the mass goes to its duplicates in other frames
        st = bb.attn_stats(y, g, pos, dup)
        has_dup = ((sc.dist < 1e-3).sum(1) > 1).float().mean().item()
        print(f"     {has_dup:.2f} of the synthetic cells have a 3D duplicate in another frame; correspondence mass {st['corr_mass']:.3f}", flush=True)
        ok(st["corr_mass"] > 0.9 * has_dup, "correspondence mass counts the attention on other-frame cells within 1 cell")
        st = AlignedXAttn(D, H, pe="3d").to(DEV).attn_stats(y, g, pos, sc)
        print(f"     init nu 1.5, random W_q, W_k, synthetic plane: {st}", flush=True)

    # 6. 32 frames
    pos32 = AlignedXAttn.positions(grid(32))
    P32, C32, _ = synthetic_scene(32)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sc32 = cell_points(P32, C32, grid(32))
        blk(torch.randn(8, len(pos32), D, device=DEV, dtype=torch.bfloat16), torch.randn(len(pos32), D, device=DEV), pos32, sc32)
        torch.cuda.synchronize()
    print(f"     32 frames (N {len(pos32)}), 8 questions, one point, incl. ScenePoints: peak +{(torch.cuda.max_memory_allocated() - before) / 2**30:.1f} GB", flush=True)


def fusion_checks(model_id):
    torch.manual_seed(0)
    base = FusionVL(model_id, "base", pixels=TRAIN_PIXELS).cuda()
    batch = micro_batch(base, 3)
    with torch.no_grad():
        ref = base(batch)
    big = micro_batch(base, 8)
    del base
    torch.cuda.empty_cache()
    first = True
    for rope3, bias in [(False, True), (True, False), (True, True)]:
        tag = f"3d rope={int(rope3)} bias={int(bias)}"
        torch.manual_seed(0)
        model = FusionVL(model_id, "xattn_aligned", rope3=rope3, local_bias=bias, pe="3d", pixels=TRAIN_PIXELS,
                         grad_ckpt=True).cuda()
        model.train()
        new = {k: v for k, v in count_params(model).items() if k != "lora"}
        print(f"[{tag}] trainable {({k: f'{v / 1e6:.2f}M' for k, v in count_params(model).items()})}, new {sum(new.values()) / 1e6:.2f}M", flush=True)
        ok(model.point_head is not None and not any(p.requires_grad for p in model.point_head.parameters())
           and {4, 11, 17, 23} <= model.vggt.cached_layer_indices, f"[{tag}] point head loaded, frozen; aggregator caches 4/11/17/23")
        # 7.
        with torch.no_grad():
            loss = model(batch)
        sc = model._pts
        n_vis = int((batch["input_ids"][0] == model.inputs.image_token_id).sum())
        ok(sc is not None and sc.xyz.shape == (n_vis, 3) and model._pos.shape == (n_vis, 3), f"[{tag}] ScenePoints for {n_vis} visual tokens")
        if first:
            first = False
            T = len(batch["image_grid_thw"])
            frame = model._pos[:, 0]
            hc, wc = (batch["image_grid_thw"][0, 1:] // 2).tolist()
            adj = (sc.xyz.view(T, hc, wc, 3)[:, :, 1:] - sc.xyz.view(T, hc, wc, 3)[:, :, :-1]).norm(dim=-1)
            vert = (sc.xyz.view(T, hc, wc, 3)[:, 1:] - sc.xyz.view(T, hc, wc, 3)[:, :-1]).norm(dim=-1)
            raw = torch.cdist(sc.xyz, sc.xyz)
            raw[frame[:, None] == frame[None]] = float("inf")
            nn_other = raw.min(1).values[~sc.low]
            q = lambda x: [round(v, 2) for v in torch.quantile(x.float().flatten(), torch.tensor([0.1, 0.5, 0.9], device=DEV)).tolist()]
            print(f"     [{tag}] real frames: scale {sc.scale:.4f} VGGT units per cell, low {sc.low.float().mean():.3f}; "
                  f"horizontal neighbours p10/50/90 {q(adj)}, vertical {q(vert)}; nearest cell in another frame {q(nn_other)}; "
                  f"scene extent {[round(v, 1) for v in (sc.xyz[~sc.low].max(0).values - sc.xyz[~sc.low].min(0).values).tolist()]} cells", flush=True)
            good = ~sc.low.view(T, hc, wc)
            ok(abs(adj[good[:, :, 1:] & good[:, :, :-1]].median().item() - 1) < 1e-4,  # the scale's own (lower) median
               f"[{tag}] median adjacent gap of confident cells = 1 cell after scaling")
            warn(0.2 < torch.quantile(nn_other, 0.5).item() < 2.5, f"[{tag}] the same surface in another frame is within 1-2 cells (median nearest {torch.quantile(nn_other, 0.5).item():.2f})")
        ok(torch.allclose(loss, ref, atol=1e-4), f"[{tag}] init loss == base {loss.tolist()} vs {ref.tolist()}")
        # 8.
        model.eval()
        model.collect_attn = True
        with torch.no_grad():
            model(batch)
        model.collect_attn = False
        ok(set(model.attn_stats) == {"0", "1", "2"} and all("corr_mass" in s for s in model.attn_stats.values()),
           f"[{tag}] attn stats at init {model.attn_stats['0']}")
        if bias:
            own = model.attn_stats["0"]["self_mass"]
            warn(0.2 <= own <= 0.4, f"[{tag}] own-cell mass at init {own:.3f} within 0.2-0.4 (v5 grid: 0.31)")
        model.attn_stats = {}
        model.train()
        # 9.
        with torch.no_grad():
            for b in model.aligned.values():
                torch.nn.init.normal_(b.o.weight, std=0.02)
        loss = model(batch)
        ok((loss - ref).abs().max().item() > 1e-3, f"[{tag}] perturbed W_o changes the loss ({(loss - ref).abs().max().item():.4f})")
        loss.sum().backward()
        names = ["lora_", "mergers.0.mlp.0.weight", "mergers.2.mlp.2.weight", "aligned.0.q.weight", "aligned.1.k.weight",
                 "aligned.2.v.weight", "aligned.2.o.weight"] + (["aligned.0.nu", "aligned.2.nu"] if bias else [])
        gn = {}
        for n, p in model.named_parameters():
            for key in names:
                if key in n and p.grad is not None:
                    gn[key] = gn.get(key, 0) + p.grad.float().norm().item()
        ok(all(gn.get(k, 0) > 0 for k in names), f"[{tag}] gradients reach " + ", ".join(f"{k} {gn.get(k, 0):.2e}" for k in names))
        model.zero_grad(set_to_none=True)
        ok(model.bias_slopes() == ({k: {"nu": 1.5} for k in "012"} if bias else {}), f"[{tag}] bias_slopes {model.bias_slopes()}")
        # 10.
        model.eval()
        gb = {k: batch[k] for k in ("pixel_values", "image_grid_thw", "vggt_images")}
        qs = ["How many chairs are in this room?", "What is the size of the room in square meters?", "Is the door left of the bed?"]
        gb.update(to_device(model.inputs.text(qs, batch["image_grid_thw"], padding_side="left"), DEV))
        print(f"     [{tag}] generate: {model.generate(gb)}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            model.save(path)
            with torch.no_grad():
                l1 = model(batch)
            loaded = FusionVL.load(path, pixels=TRAIN_PIXELS).cuda().eval()
            ok(loaded.config == model.config and loaded.config["pe"] == "3d" and loaded.point_head is not None,
               f"[{tag}] loaded config {({k: loaded.config[k] for k in ('rope3', 'local_bias', 'pe', 'bias_nu', 'pe_dmax')})}")
            with torch.no_grad():
                l2 = loaded(batch)
            ok(torch.allclose(l1, l2, atol=1e-4), f"[{tag}] save/load round trip: same loss {l1.tolist()}")
            del loaded
        model.train()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        model(big).sum().backward()
        torch.cuda.synchronize()
        print(f"     [{tag}] 8 questions fwd+bwd: {time.time() - t:.2f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--module-only", action="store_true")
    args = ap.parse_args()
    module_checks()
    if not args.module_only:
        fusion_checks(args.model)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
