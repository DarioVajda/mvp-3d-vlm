"""Attention-swap ablation of a trained AlignedXAttn arm: score VSI-Bench with the learned cross-attention replaced at
inference by a fixed read pattern, everything else (weights, geometry tokens, prompts, frames) unchanged.

  --mode bias     attention = softmax(3D bias b_ij) alone: the learned q.k term is dropped (where the fixed locality prior
                  alone would look)
  --mode uniform  every visual token reads the mean geometry token of the video (no spatial selection at all)
  --mode diag     visual token n reads geometry token n only (the additive arm's pairing, through W_v and W_o)
  --mode learned  unchanged: the ordinary eval (for a sanity check against results/train/<run>/branches/.../eval)

The patch is applied at runtime to the loaded model's AlignedXAttn instances; no module on disk is changed. The eval
loop, scoring and summary are mvp.eval.vsibench's.

  python -m mvp.explain.swap_eval --run 2B_v10_xa3_qknu_s0 --step 5750 --mode bias --ids results/ablation_v10/ids.txt \
      --out results/ablation_v10/2B_v10_xa3_qknu_s0/step5750_bias.jsonl
"""
import argparse, sys, types
sys.path.insert(0, "src")
import torch

from mvp.fusion.modules import AlignedXAttn


def _finish(self, out, B, N, D):
    return self.o(out.transpose(1, 2).reshape(B, N, D))


def forward_bias(self, y, g, pos, pts=None, kv=None):
    B, N, D = y.shape
    _, _, v = self.project(y, g, pos, pts, kv)
    outs = []
    for i in range(0, N, self.QUERY_CHUNK):
        a = self.bias(pos, torch.float, pts, slice(i, i + self.QUERY_CHUNK))[0].softmax(-1)  # [H, n, N]
        outs.append(torch.einsum("hnm,bhmd->bhnd", a.to(v.dtype), v))
    return _finish(self, torch.cat(outs, 2), B, N, D)


def forward_uniform(self, y, g, pos, pts=None, kv=None):
    B, N, D = y.shape
    _, _, v = self.project(y, g, pos, pts, kv)
    return _finish(self, v.mean(2, keepdim=True).expand(-1, -1, N, -1), B, N, D)


def forward_diag(self, y, g, pos, pts=None, kv=None):
    B, N, D = y.shape
    _, _, v = self.project(y, g, pos, pts, kv)
    return _finish(self, v, B, N, D)


MODES = {"learned": None, "bias": forward_bias, "uniform": forward_uniform, "diag": forward_diag}


def patch(model, mode):
    fwd = MODES[mode]
    n = 0
    for m in model.modules():
        if isinstance(m, AlignedXAttn):
            n += 1
            if fwd is not None:
                m.forward = types.MethodType(fwd, m)
    assert n > 0, "no AlignedXAttn in this model (not an aligned cross-attention arm)"
    print(f"swap_eval: mode={mode}, patched {n} AlignedXAttn blocks", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=5750)
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--ids", required=True, help="file with one question id per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nframes", type=int, default=32)
    args = ap.parse_args()
    ckpt = f"results/train/{args.run}/branches/step{args.step}/final.pt"

    from mvp.fusion import model as M
    orig = M.FusionVL.load
    M.FusionVL.load = staticmethod(lambda path, **kw: patch(orig(path, **kw), args.mode))

    from mvp.eval import vsibench as EV
    sys.argv = ["vsibench", "--fusion-ckpt", ckpt, "--nframes", str(args.nframes), "--split", "full",
                "--ids", args.ids, "--out", args.out]
    EV.main()


if __name__ == "__main__":
    main()
