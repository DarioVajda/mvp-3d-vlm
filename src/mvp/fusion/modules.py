"""New trainable modules for geometry fusion (docs/idea.md, "Architecture details"; sizes in docs/param_budget.md).

  PerceiverResampler  VGGT tokens of all frames -> a fixed set of latents (Flamingo Sec. 2.1, without the separate
                      latent self-attention: K/V are [media; latents]).
  GatedXAttnDense     Flamingo-style cross-attention + FFW block added to the residual stream after a frozen decoder
                      layer; a no-op at init either through tanh gates (a = 0) or zero-initialised output projections.
  AdditiveMerger      SpatialStack's language-side merger (2603.27437 App. A.2; their deepstack_language_add):
                      RMSNorm per VGGT patch, 2x2 window concat, MLP 8192 -> 4096 -> D, last linear zero-initialised
                      (their reset_residual_branches_to_noop), so it is also a no-op at init.
  AlignedXAttn        Qwen's visual tokens (queries) attend over the AdditiveMerger's geometry tokens of the same video
                      (keys / values), which are aligned one-to-one with them; optional 3-axis RoPE on (frame, row,
                      col) and an optional learned locality bias; output projection zero-initialised, visual tokens
                      only, no FFW (the aligned cross-attention arm).
"""
import math

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
    def __init__(self, in_dim=2048, hidden=4096, out_dim=2560, merge=2, zero_init=True):
        super().__init__()
        self.merge = merge
        self.ln = RMSNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim * merge * merge, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        if zero_init:  # off when the tokens feed AlignedXAttn's keys / values: there the zero sits in its W_o instead
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, grid):  # grid [h, w, in_dim] patch tokens of one frame -> [(h/2)*(w/2), out_dim]
        h, w, c = grid.shape  # in the row-major order of Qwen's merged visual tokens
        m = self.merge
        x = self.ln(grid).view(h // m, m, w // m, m, c).permute(0, 2, 1, 3, 4).reshape(-1, m * m * c)
        return self.mlp(x)


class AlignedXAttn(nn.Module):
    """Cross-attention from Qwen's visual tokens to the geometry tokens of the same video, one per visual token.

    Token n of the N = T * h * w visual tokens and geometry token n describe the same merged 2x2 cell (t, r, c) (both
    are in Qwen's row-major order per frame), so the additive arm is the special case "token n reads geometry token n".
    Here every visual token may read every geometry token of the video:

      Q = RoPE3(W_q LN(h_visual)), K = RoPE3(W_k LN(G)), V = W_v LN(G)
      out = W_o softmax(Q K^T / sqrt(d) - lam_h |r - r'| - lam_h |c - c'| - mu_h [t != t']) V

    RoPE3 (rope=True): the first `rope_pairs` of each head's d / 2 rotary pairs (NeoX half-split layout) are rotated by
    the token's (t~, r, c): 10 pairs by t~, 11 by r, 11 by c, fixed frequencies theta^(-j / n_axis); the rest of the
    head carries no position. t~ = t * (train_frames - 1) / (T - 1) puts the frames of any video on the training range,
    so 32 evaluation frames fall between the 8 trained positions instead of past them. q.k then depends only on the
    offset between query and key. The locality bias (local_bias=True) has one learned slope pair (lam_h, mu_h) per head.
    W_o is zero-initialised: the block adds nothing at init. Returns the update to add to the visual tokens only.
    """

    AXES = (10, 11, 11)  # rotary pairs for (t~, r, c)

    def __init__(self, dim, heads, rope=True, local_bias=True, lam=1.5, mu=3.0, train_frames=8, theta=100.0):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.rope, self.local_bias, self.train_frames = rope, local_bias, train_frames
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim, bias=False) for _ in range(4))
        nn.init.zeros_(self.o.weight)
        if local_bias:
            self.lam = nn.Parameter(torch.full((heads,), float(lam)))
            self.mu = nn.Parameter(torch.full((heads,), float(mu)))
        if rope:
            assert 2 * sum(self.AXES) <= self.head_dim, (self.AXES, self.head_dim)
            axis = torch.cat([torch.full((n,), a) for a, n in enumerate(self.AXES)])
            freq = torch.cat([theta ** (-torch.arange(n, dtype=torch.float) / n) for n in self.AXES])
            self.register_buffer("rope_axis", axis, persistent=False)
            self.register_buffer("rope_freq", freq, persistent=False)

    @staticmethod
    def positions(grid, merge=2):
        """(t, r, c) of the merged visual tokens of one video, in token order: [N, 3] long. grid: image_grid_thw [T, 3]."""
        out = []
        for t, (_, h, w) in enumerate(grid.tolist()):
            r, c = torch.meshgrid(torch.arange(h // merge), torch.arange(w // merge), indexing="ij")
            out.append(torch.stack([torch.full_like(r, t), r, c], -1).reshape(-1, 3))
        return torch.cat(out).to(grid.device)

    def rotary(self, pos):
        """cos, sin [N, n_pairs] of the RoPE3 angles for positions [N, 3] (t, r, c)."""
        T = int(pos[:, 0].max()) + 1
        p = pos.float()
        p[:, 0] = p[:, 0] * (self.train_frames - 1) / max(T - 1, 1)  # t~
        ang = p[:, self.rope_axis] * self.rope_freq
        return ang.cos(), ang.sin()

    def apply_rotary(self, x, cos, sin):  # x [B, H, N, d]; rotates pairs (i, i + n) for i < n = number of rotary pairs
        n = cos.shape[-1]
        x1, x2, rest = x[..., :n], x[..., n:2 * n], x[..., 2 * n:]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)

    def bias(self, pos, dtype):
        """Additive attention mask [1, H, N, N]: -lam_h * (|dr| + |dc|) - mu_h * [t != t']."""
        p = pos.float()
        d = (p[:, None, 1:] - p[None, :, 1:]).abs().sum(-1)
        other = (p[:, None, 0] != p[None, :, 0]).float()
        return -(self.lam[:, None, None] * d + self.mu[:, None, None] * other)[None].to(dtype)

    def project(self, y, g, pos):
        B, N, _ = y.shape
        split = lambda t: t.view(t.shape[0], N, self.heads, self.head_dim).transpose(1, 2)
        gn = self.norm_kv(g)[None]
        q, k, v = split(self.q(self.norm_q(y))), split(self.k(gn)), split(self.v(gn))
        if self.rope:
            cos, sin = self.rotary(pos)
            q, k = self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)
        return q, k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)

    def forward(self, y, g, pos):  # y [B, N, D] visual tokens of each sequence; g [N, D] geometry; pos [N, 3]
        B, N, D = y.shape
        q, k, v = self.project(y, g, pos)
        mask = self.bias(pos, q.dtype) if self.local_bias else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, N, D))

    @torch.no_grad()
    def attn_stats(self, y, g, pos, n_query=512):
        """Attention diagnostics on up to n_query visual tokens of the first sequence, averaged over heads and
        queries: mass on the query's own cell, on its own frame, and the entropy of the attention row (nats)."""
        q, k, _ = self.project(y[:1], g, pos)
        idx = torch.linspace(0, y.shape[1] - 1, min(n_query, y.shape[1]), device=y.device).long()
        logits = (q[0][:, idx].float() @ k[0].float().transpose(-1, -2)) / math.sqrt(self.head_dim)  # [H, n, N]
        if self.local_bias:
            logits = logits + self.bias(pos, torch.float)[0][:, idx]
        a = logits.softmax(-1)
        own = a[:, torch.arange(len(idx)), idx]
        same = (pos[None, :, 0] == pos[idx, 0][:, None]).float()  # [n, N]
        out = {"self_mass": own.mean().item(), "frame_mass": (a * same).sum(-1).mean().item(),
               "entropy": -(a * a.clamp_min(1e-12).log()).sum(-1).mean().item()}
        if self.local_bias:
            out.update(lam=self.lam.mean().item(), mu=self.mu.mean().item())
        return out
