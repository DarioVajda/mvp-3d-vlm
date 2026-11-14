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
                      only, no FFW (the aligned cross-attention arm). pe="3d" swaps both position signals for their 3D
                      versions from VGGT's point head (mvp.fusion.points): RoPE on (frame, x, y, z) and a 3D-distance
                      bias.
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
    """xa_head (hybrid arm): a second output layer `xa` of the same shape on the shared trunk (ln, mlp[0], GELU), with
    the default (non-zero) init; forward then returns (mlp[2] output, xa output): the zero-initialised tokens to add and
    the tokens for AlignedXAttn's keys / values."""

    def __init__(self, in_dim=2048, hidden=4096, out_dim=2560, merge=2, zero_init=True, xa_head=False):
        super().__init__()
        self.merge = merge
        self.ln = RMSNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim * merge * merge, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        if zero_init:  # off when the tokens feed AlignedXAttn's keys / values: there the zero sits in its W_o instead
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)
        self.xa = nn.Linear(hidden, out_dim) if xa_head else None

    def forward(self, grid):  # grid [h, w, in_dim] patch tokens of one frame -> [(h/2)*(w/2), out_dim]
        h, w, c = grid.shape  # in the row-major order of Qwen's merged visual tokens
        m = self.merge
        x = self.ln(grid).view(h // m, m, w // m, m, c).permute(0, 2, 1, 3, 4).reshape(-1, m * m * c)
        if self.xa is None:
            return self.mlp(x)
        z = self.mlp[1](self.mlp[0](x))
        return self.mlp[2](z), self.xa(z)


def per_head(init, heads):
    """Initial values of a per-head slope: `init` for every head, or (lo, hi) log-spaced over the heads."""
    if isinstance(init, (int, float)):
        return torch.full((heads,), float(init))
    lo, hi = init
    return torch.logspace(math.log10(lo), math.log10(hi), heads)


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
    offset between query and key. The locality bias (local_bias=True) has one learned slope pair (lam_h, mu_h) per head,
    init lam (or log-spaced over the heads from lam = (lo, hi), as nu below) and mu.
    W_o is zero-initialised: the block adds nothing at init. Returns the update to add to the visual tokens only.

    pe="3d" (needs the video's ScenePoints, mvp.fusion.points; xyz in merged-cell units, camera-0 frame) replaces both:
      RoPE: 8 pairs each for (t~, x, y, z); t~ as above with theta^(-j / 8); x, y, z with 2 pi / w_j, wavelengths w_j
            geometric from 1.5 to 100 cells (W3D)
      bias: -nu_h * min(|xyz_i - xyz_j|, d_max) (ScenePoints.dist), one learned slope nu_h per head, init nu (or
            log-spaced over the heads from nu = (lo, hi), so heads start at different locality scales)

    Text queries (`text`): any other token of the sequence reads the same keys and values through the same W_q, W_o.
    It has no position in the scene, so its query carries no position signal either: its rotated pairs are zero (it
    matches keys on the unrotated pairs only) and its bias row is zero.
    """

    AXES = (10, 11, 11)   # rotary pairs for (t~, r, c)
    AXES_3D = (8, 8, 8, 8)  # rotary pairs for (t~, x, y, z)
    W3D = (1.5, 100.0)    # shortest and longest wavelength of the x, y, z pairs, in cells
    QUERY_CHUNK = 4096    # query rows per attention call at inference (forward)

    def __init__(self, dim, heads, rope=True, local_bias=True, lam=1.5, mu=3.0, train_frames=8, theta=100.0, pe="grid",
                 nu=1.5):
        super().__init__()
        assert pe in ("grid", "3d"), pe
        self.heads, self.head_dim = heads, dim // heads
        self.rope, self.local_bias, self.train_frames, self.pe = rope, local_bias, train_frames, pe
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim, bias=False) for _ in range(4))
        nn.init.zeros_(self.o.weight)
        if local_bias and pe == "grid":  # lam: one init for every head, or (lo, hi): log-spaced over the heads
            self.lam = nn.Parameter(per_head(lam, heads))
            self.mu = nn.Parameter(torch.full((heads,), float(mu)))
        if local_bias and pe == "3d":  # nu: likewise
            self.nu = nn.Parameter(per_head(nu, heads))
        if rope:
            axes = self.AXES if pe == "grid" else self.AXES_3D
            assert 2 * sum(axes) <= self.head_dim, (axes, self.head_dim)
            axis = torch.cat([torch.full((n,), a) for a, n in enumerate(axes)])
            if pe == "grid":
                freq = torch.cat([theta ** (-torch.arange(n, dtype=torch.float) / n) for n in axes])
            else:
                n_t, n_s = axes[0], axes[1]
                w = torch.logspace(math.log10(self.W3D[0]), math.log10(self.W3D[1]), n_s)
                freq = torch.cat([theta ** (-torch.arange(n_t, dtype=torch.float) / n_t)] + [2 * math.pi / w] * 3)
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

    def rotary(self, pos, pts=None):
        """cos, sin [N, n_pairs] of the rotary angles: coordinates (t~, r, c) from positions [N, 3] (t, r, c), or
        (t~, x, y, z) with x, y, z from ScenePoints (pe="3d")."""
        T = int(pos[:, 0].max()) + 1
        p = pos.float()
        p[:, 0] = p[:, 0] * (self.train_frames - 1) / max(T - 1, 1)  # t~
        if self.pe == "3d":
            p = torch.cat([p[:, :1], pts.xyz.float()], 1)
        ang = p[:, self.rope_axis] * self.rope_freq
        return ang.cos(), ang.sin()

    def apply_rotary(self, x, cos, sin):  # x [B, H, N, d]; rotates pairs (i, i + n) for i < n = number of rotary pairs
        n = cos.shape[-1]
        x1, x2, rest = x[..., :n], x[..., n:2 * n], x[..., 2 * n:]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)

    def bias(self, pos, dtype, pts=None, rows=slice(None)):
        """Additive attention mask [1, H, n, N] of the query rows `rows` (all N by default): -lam_h * (|dr| + |dc|) -
        mu_h * [t != t'], or -nu_h * dist (pe="3d")."""
        if self.pe == "3d":
            return -(self.nu[:, None, None] * pts.dist[rows])[None].to(dtype)
        p = pos.float()
        d = (p[rows, None, 1:] - p[None, :, 1:]).abs().sum(-1)
        other = (p[rows, None, 0] != p[None, :, 0]).float()
        return -(self.lam[:, None, None] * d + self.mu[:, None, None] * other)[None].to(dtype)

    def keys(self, g, pos, pts=None):
        """Keys (rotated) and values of the geometry tokens g [N, D]: [1, H, N, d] each. They depend on the video only,
        so a caller may compute them once and pass them as `kv` to every call on the same video."""
        assert self.pe == "grid" or pts is not None, "pe='3d' needs the video's ScenePoints"
        split = lambda t: t.view(1, len(g), self.heads, self.head_dim).transpose(1, 2)
        gn = self.norm_kv(g)[None]
        k, v = split(self.k(gn)), split(self.v(gn))
        if self.rope:
            cos, sin = self.rotary(pos, pts)
            k = self.apply_rotary(k, cos, sin)
        return k, v

    def project(self, y, g, pos, pts=None, kv=None):
        B, N, _ = y.shape
        k, v = kv if kv is not None else self.keys(g, pos, pts)
        q = self.q(self.norm_q(y)).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        if self.rope:
            cos, sin = self.rotary(pos, pts)
            q = self.apply_rotary(q, cos, sin)
        return q, k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)

    def text(self, x, kv):
        """Update of M text tokens x [M, D] (from any sequences about the video): attention over the keys / values
        kv = keys(...) with no position signal (rotated query pairs zero, no bias). [M, D]"""
        k, v = kv
        q = self.q(self.norm_q(x)).view(1, len(x), self.heads, self.head_dim).transpose(1, 2)
        if self.rope:
            q = torch.cat([torch.zeros_like(q[..., :2 * len(self.rope_freq)]), q[..., 2 * len(self.rope_freq):]], -1)
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        return self.o(out.transpose(1, 2).reshape(len(x), -1))

    def forward(self, y, g, pos, pts=None, kv=None):  # y [B, N, D] visual tokens; g [N, D] geometry; pos [N, 3]; pts: ScenePoints
        B, N, D = y.shape
        q, k, v = self.project(y, g, pos, pts, kv)
        if self.local_bias and not torch.is_grad_enabled() and N > self.QUERY_CHUNK:
            # inference on long videos: the [H, N, N] mask (tens of GB at 128 frames) is built QUERY_CHUNK rows at a
            # time; the same attention, row by row
            out = torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + self.QUERY_CHUNK], k, v,
                                                            attn_mask=self.bias(pos, q.dtype, pts,
                                                                                slice(i, i + self.QUERY_CHUNK)))
                             for i in range(0, N, self.QUERY_CHUNK)], 2)
        else:
            mask = self.bias(pos, q.dtype, pts) if self.local_bias else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, N, D))

    @torch.no_grad()
    def attn_stats(self, y, g, pos, pts=None, n_query=512):
        """Attention diagnostics on up to n_query visual tokens of the first sequence, averaged over heads and
        queries: mass on the query's own cell, on its own frame, and the entropy of the attention row (nats). With
        ScenePoints also the correspondence mass: on keys in other frames within 1 cell of the query in 3D."""
        q, k, _ = self.project(y[:1], g, pos, pts)
        idx = torch.linspace(0, y.shape[1] - 1, min(n_query, y.shape[1]), device=y.device).long()
        logits = (q[0][:, idx].float() @ k[0].float().transpose(-1, -2)) / math.sqrt(self.head_dim)  # [H, n, N]
        if self.local_bias:
            logits = logits + self.bias(pos, torch.float, pts, idx)[0]
        a = logits.softmax(-1)
        own = a[:, torch.arange(len(idx)), idx]
        same = (pos[None, :, 0] == pos[idx, 0][:, None]).float()  # [n, N]
        out = {"self_mass": own.mean().item(), "frame_mass": (a * same).sum(-1).mean().item(),
               "entropy": -(a * a.clamp_min(1e-12).log()).sum(-1).mean().item()}
        if pts is not None:
            corr = (1 - same) * (pts.dist[idx] <= 1).float()
            out["corr_mass"] = (a * corr).sum(-1).mean().item()
        if self.local_bias and self.pe == "grid":
            out.update(lam=self.lam.mean().item(), mu=self.mu.mean().item())
        if self.local_bias and self.pe == "3d":
            out.update(nu=self.nu.mean().item())
        return out
