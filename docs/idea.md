# 3D perception injection into a VLM with gated cross-attention

## Core idea

Give a small vision-language model (VLM) a sense of the three-dimensional layout of a room by letting its
language model read the internal features of a frozen 3D reconstruction network (VGGT) through newly added
gated cross-attention blocks, in the style of Flamingo. The VLM's own weights stay frozen apart from LoRA
adapters; only the new blocks, a resampler and the adapters are trained.

**Hypothesis.** Letting tokens *choose* what geometry to read (cross-attention) beats *adding* geometry to the
hidden states at fixed positions (SpatialStack-style additive injection), when both are given the same VGGT
features, the same data and the same LLM adapters.

### What the related papers did, and what nobody has tested

| Method | Where geometry enters | Operator | LLM training | Geometry gain over data-only fine-tune (VSI-Bench) |
|---|---|---|---|---|
| VG-LLM (as described in SpatialStack Sec. 3) | vision-encoder output | addition | full FT | +1.6 (66.36 vs 64.76, SpatialStack Table 2) |
| VLM-3R (2505.20279, Sec. 4.1) | before the LLM, visual tokens as queries | one cross-attention block | LoRA r=128 | +3.2 with CUT3R (60.90 vs 57.74, Table 6); +0.4 with VGGT (Table D) |
| SpatialStack (2603.27437, Eq. 5) | LLM layers 0, 1, 2, visual-token slice | addition | full FT | +2.8 (67.52 vs 64.76, Table 2) |
| Flamingo (2204.14198, Sec. 2.2) | before every n-th LM layer | gated cross-attention + FFW | frozen LM | n/a (2D vision into a text LM) |

None of them puts cross-attention *inside* the LLM for geometry, and none ablates how many or which LLM layers
receive geometry. SpatialStack's "three layers" is inherited from VGGT's DPT taps (4, 11, 17, 23 minus 4) and
its only placement is layers 0–2; it never ran "one VGGT layer injected inside the LLM", and its stacking gain
(67.52 vs 66.36 for layer 23 at the vision output) confounds "inside the LLM" with "three layers". Flamingo's
one depth ablation (Table 3 row v): a single block in the middle of a 24-layer LM scored 59.8, every 4th layer
68.8, every layer 70.7.

Most of the jump from untuned bases in these papers comes from the VSI-template training data, not the
geometry: SpatialStack's data-only fine-tune moves Qwen3.5 from 53.6 to 64.76 and geometry adds 2.8 on top.
Any arm here must therefore be read against a data-only control, and MMSI-Bench is a regression gate: the
SpatialStack checkpoint scores 28.0 on MMSI against 32.2 for plain Qwen3.5-4B (our runs), so VSI fine-tuning
can cost general spatial reasoning.

## Decisions (grill of 2026-10-24)

| Decision | Choice | Why |
|---|---|---|
| Claim to settle first | cross-attention beats additive injection | the one comparison nobody has run |
| Base model | Qwen3.5-4B | base of SpatialStack-Qwen3.5-4B (67.5 reproduced at 67.53), so the published additive recipe runs as a reference arm. Cost: 3.6 points weaker zero-shot than Qwen3-VL-4B at 16 frames (48.3 vs 51.9); hybrid attention (below) |
| LLM weights | LoRA in every arm, r = 64 on all linear layers | identical across arms so only the fusion differs; lets the LLM adapt its answer format |
| Training data | VLM-3R ScanNet QA, each of the 6 tasks capped at 3k (room size keeps all 1,201): ≈16.2k samples | balanced per-task signal; half the raw 51.8k is relative direction |
| Insertion points | one gated cross-attention block before each softmax-attention layer: 3, 7, 11, 15, 19, 23, 27, 31 | Flamingo's every-4th recipe mapped onto Qwen3.5's 3:1 linear/softmax pattern; injected geometry lands right in front of a layer that can retrieve it exactly |
| VGGT features | aggregator layer 23 (last), patch tokens + camera token + 4 register tokens | layer 23 alone gave most of SpatialStack's VSI gain (Table 1); VLM-3R's camera "view token" was worth 1.8 (Table 6) |
| Compression | Perceiver resampler, 8 layers, width 1536, 16 heads, 256 latents per video | fixed K/V size regardless of frame count, which is what makes 8-frame training and 32-frame eval possible. Other downsampling (grid-preserving) is a later variant |
| Frame tag into resampler | none | relies on VGGT's own features. Known risk: VGGT is permutation-equivariant for all but the first frame, so order-dependent tasks (appearance order, MMSI camera motion) get no help from geometry. These are the canaries; add a learned frame-index embedding if they drop |
| Block | full Flamingo gated xattn-dense: xattn at width 2560 (20 heads × 128) + FFW 2560→10240→2560, both branches gated by tanh(α), α init 0 | faithful to Flamingo; its ablations: no gate −4.2 and unstable (Table 3 iii), no FFW −3.8 (row iv, VANILLA XATTN) |
| Who queries | all tokens, text included | Flamingo's design; the answer is produced at text positions |
| Frames | train on 8 uniform frames; evaluate at 8 and 32 | SpatialStack trains on 4–8, evaluates at 32 (Table B) |
| Hyperparameters | lr 1e-4 for LoRA and for new modules, 3 epochs, effective batch 32, 3% warmup, cosine, AdamW, bf16, loss on answer tokens only | SpatialStack: lr 1e-5, batch 64, 1 epoch, 3% warmup, cosine (Table B), but that is full fine-tuning; fresh modules need more. Not from a paper |
| First evaluation | VSI-Bench stratified 100 per task type (~1.1k q) at 8 and 32 frames; full MMSI-Bench (1,000 q, 2048 new tokens, thinking off) | ~10 min per arm per frame setting; SE ≈1.5 overall, ≈5 per task. Success criteria deferred until first results |
| Success criterion | deferred: run the subset, look, then decide | user's call |

### Arms

All arms: Qwen3.5-4B, LoRA r=64, same data, frames, steps, seed; VGGT frozen, computed on the fly on the 8
training frames (the 128-frame cache in `data/vggt/` is a different joint pass and only serves the sampler).

| # | Arm | New trainable weights (excl. LoRA) | Purpose |
|---|---|---|---|
| 0a | Qwen3.5-4B zero-shot, 8 frames | 0 | untuned baseline at the training frame count (not yet run) |
| 0b | Qwen3.5-4B zero-shot, 32 frames | 0 | untuned baseline at the eval frame count (not yet run; 16-frame run = 48.3) |
| 1 | LoRA only, no geometry | 0 | data-only control (SpatialStack's 64.76 row) |
| 2 | Matched additive: VGGT layer 23 → one shared SpatialStack-style merger → added to the visual-token slice at the same 8 positions as the blocks | 44M | only the operator differs from arm 4; also SpatialStack's missing "one layer inside the LLM" ablation |
| 3 | Published SpatialStack: VGGT 11/17/23 → three mergers → LLM layers 0/1/2 (their code) | 132M (measured) | reference to the paper |
| 4 | Cross-attention: 8 full Flamingo blocks + 8-layer resampler | 863M | the hypothesis |

Known limitation: arm 4 has 6.5× the new weights of arm 3 and 20× arm 2. A win can be attributed to capacity
until a capacity-matched variant (no FFW, inner width 1024, 2-layer resampler: ~86M; or narrow FFW: ~190M) is
run. Chosen knowingly for the first experiment.

### Parameter counts

Full arithmetic and ASCII diagrams in [param_budget.md](param_budget.md). Summary (D = 2560, VGGT width G = 2048):

| Component | Arithmetic | Weights |
|---|---|---|
| Resampler input map | 2048 × 1536 | 3.1M |
| Resampler, 8 layers | 8 × (4·1536² + 2·1536·6144) | 226.5M |
| Resampler output map + latents | 1536 × 2560 + 256 × 1536 | 4.3M |
| One block, cross-attention | 4 × 2560² | 26.2M |
| One block, FFW | 2 × 2560 × 10240 | 52.4M |
| 8 blocks | 8 × 78.6M | 629M |
| **Arm 4 total** | | **863M** (Adam state ≈12 GB) |
| SpatialStack merger (checkpoint shapes: LN 2048; 4096×8192; 2560×4096; biases) | | 44.1M each, 132M for three |

### Speed and memory (estimates, to be replaced by the smoke test)

Basis: README kernel benchmark, frozen-model train step at 32 frames / 5k tokens = 0.31 s (B200), 0.49 s (H100);
VGGT 0.6 s and 8 GB for 32 frames on an H100. Assumptions: 4B MACs per token for the LLM, backward = 2× forward,
linear scaling with tokens.

- LLM step at 8 frames (≈1.3k tokens): ≈0.08 s B200 / 0.13 s H100. Blocks add ≈16% of LLM forward MACs, resampler
  ≈9%; with backward ≈ +25% → ≈0.10 / 0.16 s. VGGT forward at 8 frames ≈0.08 / 0.12 s (same in arms 2–4).
- One epoch of 16.2k samples, batch 1, compute only: ≈55 min B200 / ≈80 min H100; 3 epochs ≈3–4 GPU-h per arm
  before data loading and the Qwen vision encoder, which are not benchmarked for training.
- Eval: Qwen3.5-4B zero-shot at 16 frames took 18 min on an H100 for 5,130 questions; the 1.1k subset at 32 frames
  ≈10 min per arm; MMSI with 2048 new tokens ≈2 h 10 min full (README) — consider a 500-question half for first results.
- Blocks cost the same per token at 8 and 32 frames (always 256 latents); the resampler's K/V side and VGGT grow with frames.

## Architecture details

### Qwen3.5-4B layer structure (from its `config.json`)

32 decoder layers, hidden 2560. `layer_types`: 24 × `linear_attention` (Gated DeltaNet: fixed-size 128×128 state
per head, 16 key heads / 32 value heads, causal conv kernel 4, fp32 state, output gate, no RoPE) and 8 ×
`full_attention` at indices 3, 7, 11, 15, 19, 23, 27, 31 (16 heads × 256, 4 KV heads, multimodal RoPE on a quarter
of each head, output gate). No DeepStack (`deepstack_visual_indexes: []`). Vision encoder: 24 layers, 16 px patches,
2×2 spatial merge, temporal patch 2, output 2560.

A gated cross-attention block is a residual add, so it can sit before any layer; we put each one directly before a
softmax layer so the next layer can do exact token-level retrieval over what was just injected, instead of folding
it into a linear layer's running memory first. Whether geometry survives three linear-attention layers is untested;
this placement sidesteps the question.

### Block (Flamingo Fig. 4, at D = 2560)

```
y = y + tanh(a_xattn) * xattn(q = LN(y), kv = latents)      # 4 × 2560², 20 heads of 128
y = y + tanh(a_ffw)   * ffw(LN(y))                          # 2560 → 10240 → 2560
y = frozen_qwen_softmax_layer(y)                            # with LoRA
```

Both scalars `a_*` start at 0, so at step 0 the model is exactly Qwen3.5-4B.

### Resampler (Flamingo Sec. 2.1 / App. A.1.1, enlarged)

Input: VGGT layer-23 tokens for the 8 frames, (8 × 1036 patches + 8 × 5 special) × 2048 → Linear 2048→1536.
8 layers of: latents ← cross-attention over [tokens; latents] (K/V include the latents, as Flamingo), FFW 4×.
256 learned latents, width 1536, 16 heads. Output Linear 1536→2560. No frame or spatial embedding (see risk above).
At 32-frame eval the input is 4× longer; the output is still 256 × 2560.

### Additive arms

- **Matched (arm 2):** LayerNorm → 2×2 window concat (4 × 2048 = 8192) → MLP 8192→4096→2560 (the shapes in the
  released SpatialStack checkpoint) → add to the hidden state at visual-token positions, after the same 8 layers the
  blocks precede, one shared merger. Requires VGGT at 14 px patches on a grid aligned to Qwen's merged tokens: Qwen
  at 640×480 gives 40×30 patches → 20×15 merged; VGGT at 560×420 gives 40×30 patches → 20×15 windows. (The resampler
  arm needs no alignment.)
- **Published (arm 3):** SpatialStack's own training code and config (`feature_fusion_method: deepstack_language_add`,
  `geometry_encoder_layers: [11, 17, 23]`, `geometry_fusion_layers: [0, 1, 2]`, `geometry_merger_type: mlp`), with
  full fine-tuning replaced by the same LoRA as the other arms.

### Data and prompts

- VLM-3R ScanNet QA: 51.8k VSI-Bench-template questions on 1,201 ScanNet videos, none in VSI-Bench. Tasks and raw
  counts: rel. direction 25,781, rel. distance 11,038, abs. distance 6,960, obj. size 4,234, counting 2,565, room size
  1,201. No route planning or appearance order. Cap 3k per task → ≈16.2k.
- Prompts have no `<video>` placeholder ("These are frames of a video.\n..."); visual tokens go in front of the text,
  as SpatialStack does. Thinking off (`enable_thinking=False`), as in SpatialStack's lmms-eval config.
- 8 uniform frames per video; the same frames go to Qwen (640×480) and VGGT (518×392 for the resampler arm).
- VGGT outputs are scale-normalised (VGGT Sec. 3.4: "we do not apply such normalization to the predictions ... we
  force it to learn the normalization"), so metric answers (abs. distance, size, room size) must be learned from data.
  VLM-3R attributes VGGT's weak room-size result (63.7 → 54.0, Table D) to this.

## What to log (so first results can be read)

- Per step: loss, lr, the 16 gate values `tanh(a)` per block, gradient norm, step time, peak memory.
- Per eval: VSI per task at 8 and 32 frames, MMSI per category, parse rate, mean attention entropy per block, share
  of attention mass on camera/register tokens vs patches, answers dumped to jsonl as in `results/zeroshot/`.
- Per run: full config, seed, frame indices, VGGT confidence stats, git commit, freeze.txt.

## Failure modes and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| Gates stay near 0: the LLM never uses geometry | `tanh(a)` flat across training; arm 4 ≈ arm 1 | raise lr on `a` only, or init a = 0.1; check K/V scale after the 1536→2560 map |
| 863M fresh weights overfit 16k samples | train loss ≪ eval loss; VSI subset up, MMSI down | fewer epochs, dropout in blocks, or the capacity-matched variant |
| Template fitting, not geometry | arm 1 ≈ arms 2–4 on VSI | report per task; rel. direction and rel. distance are where geometry should show |
| No frame tag → order tasks degrade | appearance order / MMSI camera motion below arm 1 | add learned frame-index embedding to resampler input |
| 8-train / 32-eval mismatch | 32-frame scores below 8-frame for arm 4 only | fine-tune 1 epoch at 32 frames, or sample 8–32 frames during training |
| VGGT on 8 frames reconstructs poorly on dark ScanNet footage | low `depth_conf`; arm 4 loses on specific scenes | log confidence; try space-aware frame picks |
| Additive arm grid misaligned | silent; arm 2 ≈ arm 1 | assert Qwen merged grid == VGGT window grid per frame |
| Answer truncation | parse rate < 99% | 16 new tokens for VSI is enough for zero-shot Qwen3.5 (median 3 chars); recheck after training |
| Kernel fallback | step time 5× slower, transformers warning | ensure `flash-linear-attention` + `causal-conv1d` in the venv (slurm/install_linear_attn_kernels.sh) |
| LoRA through Gated DeltaNet layers | NaNs or no learning in linear layers | gradients through the Triton kernels were benchmarked (README); verify on the smoke test |

## Smoke test (before any full run)

50 training steps of arm 4 at batch 1 on 8 frames; then 20 stratified VSI questions at 8 and 32 frames and 20
MMSI questions. Check: loss decreases, all 16 gates move off zero, no NaN, step time and peak memory recorded
(these replace the estimates above), parse rate 100%, grid assertion for arm 2 passes.

## Superseded leans (kept for the record)

Earlier drafts leaned to proportional spatial downsampling over a Perceiver resampler, and to 2D RoPE on both
queries and keys to align VGGT's grid with Qwen's. The resampler was chosen for the first experiment because it
makes the K/V size independent of frame count (8-frame training, 32-frame eval). Grid-preserving compression with
2D RoPE stays on the list as the next variant of arm 4, and the matched additive arm already needs the grid alignment.

### Rough sketch for the resampler implementation
```python
import torch
import torch.nn as nn

class PerceiverResampler(nn.Module):
    def __init__(self, dim=1536, in_dim=2048, out_dim=2560, num_latents=256, num_layers=8, heads=16):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleDict({
            'cross_attn': nn.MultiheadAttention(dim, heads, batch_first=True),
            'ffw': nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
            'norm_q': nn.LayerNorm(dim), 'norm_x': nn.LayerNorm(dim), 'norm_ffw': nn.LayerNorm(dim),
        }) for _ in range(num_layers)])
        self.out = nn.Linear(dim, out_dim)

    def forward(self, x):                      # x: [B, frames*(1036+5), 2048] VGGT layer-23 tokens
        x = self.inp(x)
        latents = self.latents.expand(x.shape[0], -1, -1)
        for layer in self.layers:
            q = layer['norm_q'](latents)
            kv = torch.cat([layer['norm_x'](x), q], dim=1)   # Flamingo: K/V include the latents
            latents = latents + layer['cross_attn'](q, kv, kv)[0]
            latents = latents + layer['ffw'](layer['norm_ffw'](latents))
        return self.out(latents)               # [B, 256, 2560], same size at 8 or 32 frames


class GatedXAttnDense(nn.Module):
    def __init__(self, dim=2560, heads=20):
        super().__init__()
        self.norm_q, self.norm_ffw = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.xattn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffw = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.a_xattn = nn.Parameter(torch.zeros(1))
        self.a_ffw = nn.Parameter(torch.zeros(1))

    def forward(self, y, latents):             # y: LLM hidden states [B, T, 2560]; all tokens query
        y = y + torch.tanh(self.a_xattn) * self.xattn(self.norm_q(y), latents, latents)[0]
        y = y + torch.tanh(self.a_ffw) * self.ffw(self.norm_ffw(y))
        return y                               # then the frozen softmax layer (3, 7, ..., 31) runs
```

## Summary table of the experiment designs

Shared by every arm: Qwen3.5-4B (thinking off), VLM-3R ScanNet QA capped at 3k per task (≈16.2k), 8 uniform
training frames, lr 1e-4, 3 epochs, effective batch 32, 3% warmup, cosine, bf16, loss on answer tokens only, one
seed; evaluation on the stratified 100-per-task VSI-Bench subset at 8 and 32 frames plus full MMSI-Bench.

| | 0a / 0b · zero-shot | 1 · LoRA only | 2 · matched additive | 3 · published SpatialStack | 4 · cross-attention |
|---|---|---|---|---|---|
| Role | untuned baseline at 8 / 32 frames | data-only control | same positions and features as arm 4, addition instead of attention | reference to the paper | the hypothesis |
| Trained | nothing | LoRA r=64 | LoRA + 1 merger | LoRA + 3 mergers | LoRA + 8 blocks + resampler |
| Geometry source | none | none | VGGT layer 23, patch tokens | VGGT layers 11, 17, 23, patch tokens | VGGT layer 23, patch + camera + 4 register tokens |
| Compression | – | – | 2×2 window concat → MLP 8192→4096→2560 (one shared merger) | same, one merger per layer | Perceiver resampler, 8 layers, width 1536, 256 latents per video |
| Where it enters the LLM | – | – | added to visual-token hidden states after layers 3, 7, …, 31 | added to visual-token hidden states after layers 0, 1, 2 (VGGT 11→0, 17→1, 23→2) | gated cross-attention before layers 3, 7, …, 31; all tokens query |
| Operator | – | – | addition, position-matched | addition, position-matched | softmax attention over 256 latents, tanh gates init 0 |
| Grid alignment needed | – | – | yes: VGGT at 560×420 so 20×15 windows match Qwen's 20×15 merged tokens | yes (their code handles it) | no |
| Frame-order information | Qwen's own visual tokens | same | same | same | same; none inside the geometry path |
| New weights (excl. LoRA) | 0 | 0 | 44M | 132M (measured from checkpoint) | 863M (629M blocks + 234M resampler) |
| Added compute per sample at 8 frames | – | – | VGGT forward (≈1.9× an LLM forward) + negligible | same + negligible | VGGT forward + ≈25% of the LLM's training step |
| Expected if the hypothesis holds | 48.3 at 16 frames (known); 8 and 32 to run | well above 0b (SpatialStack: +11 from data alone) | above arm 1 by a point or two | ≈ arm 2, possibly slightly above | above arm 2 on VSI, especially rel. direction and rel. distance; MMSI not below 32.2 |
| What a null result would mean | – | – | – | – | attention adds nothing over addition at equal data; or capacity/overfitting masked it (check gates, train vs eval loss) |
| Comparisons it enables | fine-tune effect (1 − 0b) | geometry effect (2, 3, 4 − 1) | operator effect (4 − 2) | reproduction (3 vs 67.5 published, under LoRA instead of full FT); stacking effect (3 − 2) | – |

Caveat on arm 4: 6.5× the new weights of arm 3 and 20× arm 2. A capacity-matched variant (no FFW, inner width
1024, 2-layer resampler, ≈86M; or narrow FFW, ≈190M) is the first follow-up if arm 4 wins.
