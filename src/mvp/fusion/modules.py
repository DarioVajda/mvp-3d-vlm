"""New trainable modules for geometry fusion (docs/idea.md, "Architecture details"; sizes in docs/param_budget.md).

  PerceiverResampler  VGGT tokens of all frames -> a fixed set of latents (Flamingo Sec. 2.1, without the separate
                      latent self-attention: K/V are [media; latents]).
  GatedXAttnDense     Flamingo-style cross-attention + FFW block added to the residual stream after a frozen decoder
                      layer; a no-op at init either through tanh gates (a = 0) or zero-initialised output projections.
  AdditiveMerger      SpatialStack's language-side merger (2603.27437 App. A.2; their deepstack_language_add):
                      RMSNorm per VGGT patch, 2x2 window concat, MLP 8192 -> 4096 -> D, last linear zero-initialised
                      (their reset_residual_branches_to_noop), so it is also a no-op at init.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head attention with separate query and key/value inputs, no biases (4 * dim^2 weights), SDPA kernel."""

    def __init__(self, dim, heads, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(kv_dim, dim, bias=False)
        self.v = nn.Linear(kv_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, kv):  # x [B, Nq, dim], kv [B, Nk, kv_dim]
        B, Nq, _ = x.shape
        split = lambda t: t.view(B, t.shape[1], self.heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(kv)), split(self.v(kv)))
        return self.o(out.transpose(1, 2).reshape(B, Nq, -1))


def ffw(dim, mult=4):
    return nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))


class PerceiverResampler(nn.Module):
    def __init__(self, in_dim=2048, dim=1024, out_dim=2560, num_latents=256, num_layers=8, heads=8):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "norm_x": nn.LayerNorm(dim), "norm_q": nn.LayerNorm(dim), "attn": Attention(dim, heads),
            "norm_ffw": nn.LayerNorm(dim), "ffw": ffw(dim)}) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, x):  # x [N, in_dim]: VGGT tokens of all frames of one video -> [num_latents, out_dim]
        x = self.inp(x)[None]
        lat = self.latents[None]
        for l in self.layers:
            q = l["norm_q"](lat)
            lat = lat + l["attn"](q, torch.cat([l["norm_x"](x), q], dim=1))  # Flamingo: K/V include the latents
            lat = lat + l["ffw"](l["norm_ffw"](lat))
        return self.out(self.norm_out(lat))[0]


class GatedXAttnDense(nn.Module):
    """Cross-attention (LLM tokens query the geometry latents) + FFW, both residual.

    gate="tanh": Flamingo's tanh(a) gates with a = 0 at init (the pilot's arm 4; the gates stayed at 0 because the
                 untrained branches were 2-7x the residual stream, so any opening raised the loss).
    gate="none": no gate; the output projections of both branches are zero-initialised instead, exactly like the
                 AdditiveMerger's last linear, so the block is a no-op at init and the gradient to the output
                 projection decides how fast it opens (matched to the additive arm).
    """

    def __init__(self, dim, heads, gate="tanh"):
        super().__init__()
        assert gate in ("tanh", "none"), gate
        self.gate = gate
        self.norm_q, self.norm_ffw = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.xattn = Attention(dim, heads)
        self.ffw = ffw(dim)
        if gate == "tanh":
            self.a_xattn = nn.Parameter(torch.zeros(()))
            self.a_ffw = nn.Parameter(torch.zeros(()))
        else:
            nn.init.zeros_(self.xattn.o.weight)
            nn.init.zeros_(self.ffw[2].weight)
            nn.init.zeros_(self.ffw[2].bias)

    def forward(self, y, latents):  # y [B, T, D] LLM hidden states (all tokens query); latents [L, D] of this video
        gx, gf = (torch.tanh(self.a_xattn), torch.tanh(self.a_ffw)) if self.gate == "tanh" else (1.0, 1.0)
        y = y + gx * self.xattn(self.norm_q(y), latents[None].expand(y.shape[0], -1, -1))
        return y + gf * self.ffw(self.norm_ffw(y))

    def gates(self):
        if self.gate != "tanh":
            return {}
        return {"xattn": torch.tanh(self.a_xattn).item(), "ffw": torch.tanh(self.a_ffw).item()}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class AdditiveMerger(nn.Module):
    def __init__(self, in_dim=2048, hidden=4096, out_dim=2560, merge=2):
        super().__init__()
        self.merge = merge
        self.ln = RMSNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim * merge * merge, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, grid):  # grid [h, w, in_dim] patch tokens of one frame -> [(h/2)*(w/2), out_dim]
        h, w, c = grid.shape  # in the row-major order of Qwen's merged visual tokens
        m = self.merge
        x = self.ln(grid).view(h // m, m, w // m, m, c).permute(0, 2, 1, 3, 4).reshape(-1, m * m * c)
        return self.mlp(x)
