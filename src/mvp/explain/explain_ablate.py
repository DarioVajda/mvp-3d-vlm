"""VSI-Bench through the unchanged evaluation (mvp.eval.vsibench) with the geometry path of a trained checkpoint edited.

  zero       no injection (additive: last merger linear; xattn: W_o), the LoRA trained with geometry stays
  shuffle    geometry tokens permuted among the cells of their frame (right content, wrong place)
  self       xattn: each token reads only its own cell, i.e. the additive wiring inside the trained xattn
  nonself    xattn: everything except its own cell
  frame      xattn: own frame only
  otherframe xattn: other frames only

  explain_ablate.py <mode> --fusion-ckpt ... --ids ... --nframes 32 --out ...   (other arguments go to mvp.eval.vsibench)
"""
import sys
sys.path.insert(0, "src")
import torch
from mvp.eval import vsibench
from mvp.fusion.model import FusionVL


def edit(model, mode):
    keys = [str(k) for k in model.pairs]
    if mode == "zero":
        for k in keys:
            if model.arm == "additive":
                model.mergers[k].mlp[2].weight.data.zero_(); model.mergers[k].mlp[2].bias.data.zero_()
            else:
                model.aligned[k].o.weight.data.zero_()
    elif mode == "shuffle":
        gen = torch.Generator().manual_seed(0)
        for k in keys:
            mg = model.mergers[k]
            orig = mg.forward
            def forward(grid, orig=orig):
                out = orig(grid)
                return out[torch.randperm(len(out), generator=gen).to(out.device)]
            mg.forward = forward
    else:
        assert model.arm == "xattn_aligned" and mode in ("self", "nonself", "frame", "otherframe"), (model.arm, mode)
        for k in keys:
            blk = model.aligned[k]
            orig = blk.bias
            def bias(pos, dtype, pts=None, orig=orig):
                b = orig(pos, dtype, pts)
                N = len(pos)
                eye = torch.eye(N, dtype=torch.bool, device=pos.device)
                same = pos[None, :, 0] == pos[:, None, 0]
                allow = {"self": eye, "nonself": ~eye, "frame": same, "otherframe": ~same}[mode]
                return b.masked_fill(~allow, float("-inf"))
            blk.bias = bias
    return model


mode = sys.argv.pop(1)
_load = FusionVL.load
FusionVL.load = classmethod(lambda cls, path, **kw: edit(_load(path, **kw), mode))
vsibench.main()
