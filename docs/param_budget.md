# Trainable-parameter budget of the fusion arms

VGGT-1B token width G = 2048. LoRA on the LLM is identical in every arm and is not counted. Counts are weight
matrices only (biases, norms and gate scalars are negligible) unless stated.

| Base | Width D | Softmax-attention layers (one block before each) | Blocks |
|---|---|---|---|
| Qwen3.5-4B (headline runs) | 2560 | 3, 7, 11, 15, 19, 23, 27, 31 | 8 |
| Qwen3.5-2B (first runs, debugging) | 2048 | 3, 7, 11, 15, 19, 23 | 6 |

## Perceiver resampler (chosen: R = 1024, 8 layers, 256 latents, 8 heads of 128)

Flamingo resampler layers only: the latents cross-attend to [tokens; latents], then an FFW; no separate latent
self-attention. One resampler runs once per video and every block reads its output.

```
VGGT layer-23 tokens                     learned latents
[frames x (1036 + 5), G=2048]            [256, R=1024]          256*1024 = 0.26M
        |                                      |
   Linear G->R  (2048*1024 = 2.1M)             |
        v                                      v
   +--------------------------------------------------+
   |  x8 layers:                                      |
   |   cross-attn  latents <- [tokens; latents]       |
   |     Wq, Wk, Wv, Wo : 4 * R*R                     |   4 * 1024^2      =  4.2M
   |   FFW  R -> 4R -> R                              |   2 * 1024*4096   =  8.4M
   |                                       per layer  |   12.6M     x8 = 100.7M
   +--------------------------------------------------+
        |
   Linear R->D   4B: 1024*2560 = 2.6M   2B: 1024*2048 = 2.1M   (only for the full blocks, which take D-wide K/V)
        v
   [256 latents, D]

   total   4B: 105.6M   2B: 105.1M        (without the output map: 103.0M)
```

The number of latents barely matters for the count (64 latents: 0.07M, 256: 0.26M); it changes compute only.
Superseded: width 1536 with 16 heads (232M resampler, 863M arm 4 on the 4B).

## Gated cross-attention-dense block, Flamingo style (default)

```
   residual stream y  [tokens, D]             latents x  [256, D]
          |                                       |
          |-- LN --> Q = y Wq   (D*D)             |-- K = x Wk  (D*D)
          |                                       |-- V = x Wv  (D*D)
          |          attn(Q,K,V) Wo  (D*D)                 cross-attn = 4*D^2
          y = y + tanh(a_x) * attn         <- gate a_x, 1 scalar, init 0
          |
          |-- LN --> FFW D -> 4D -> D               (2 * D*4D = 8*D^2)
          y = y + tanh(a_f) * ffw          <- gate a_f, 1 scalar, init 0
          v
   into the frozen Qwen softmax layer (with LoRA)

   4B (D=2560, 20 heads of 128): 26.2 + 52.4 = 78.6M per block   x8 = 629M
   2B (D=2048, 16 heads of 128): 16.8 + 33.6 = 50.3M per block   x6 = 302M
```

## Shrunk cross-attention-only block (alternative: inner width 1024, 8 heads of 128, no FFW)

Same gated cross-attention, but projected down to the resampler width instead of up to D, and without the FFW
branch (Flamingo Table 3 row iv: dropping the FFW cost 3.8 points). K and V read the 1024-wide latents directly,
so the resampler needs no output map.

```
   y [tokens, D]                         latents x [256, 1024]
     |-- LN --> Q = y Wq   D*1024              K = x Wk  1024*1024 = 1.05M
     |                                         V = x Wv  1024*1024 = 1.05M
     |          attn(Q,K,V) Wo   1024*D
     y = y + tanh(a) * attn
   4B: 2.62 + 1.05 + 1.05 + 2.62 = 7.3M per block   x8 = 59M
   2B: 2.10 + 1.05 + 1.05 + 2.10 = 6.3M per block   x6 = 38M
```

## SpatialStack geometry token merger (additive arms; 2603.27437v3 App. A.2, shapes from the released checkpoint)

```
   VGGT tokens [frames x 37x28, G=2048]
        |  LayerNorm (2 x 2048 params)
        |  2x2 window -> concat 4 tokens  ->  [frames x 18x14, 4G = 8192]
        |  W1  8192 -> 4096            8192*4096 = 33.6M   (hidden width 4096, read from the checkpoint)
        |  GELU
        |  W2  4096 -> D               4B: 4096*2560 = 10.5M   2B: 4096*2048 = 8.4M
        v
   [one D-wide vector per merged Qwen visual token]
        +  added into the residual at visual-token positions after LLM layer j

   per merger (with biases)   4B: 44.1M   2B: 42.0M   (2B keeps the 4096 hidden width; no released 2B checkpoint)

   Source: tensor shapes in `Journey9ni/SpatialStack-Qwen3.5-4B/model.safetensors` (header read over HTTP):
   `language_feature_fusion.fusion_layers.{0,1,2}.0.geo_ln.{weight,bias} [2048]`,
   `geo_mlp.0.weight [4096, 8192]`, `geo_mlp.0.bias [4096]`, `geo_mlp.2.weight [2560, 4096]`, `geo_mlp.2.bias [2560]`.
   The checkpoint stores each merger twice (under `language_feature_fusion.` and `model.language_feature_fusion.`),
   264M on disk, 132M distinct. The paper itself gives no count and no hidden width.
```

## Totals

| Arm | Qwen3.5-4B | Qwen3.5-2B |
|---|---|---|
| Control 1: LoRA only | 0 | 0 |
| Control 2: matched additive, VGGT 23 → one shared merger → the same block positions | 44M | 42M |
| Reference: published SpatialStack (3 mergers → LLM layers 0, 1, 2) | 132M (measured) | 126M |
| **Cross-attn: full Flamingo blocks + resampler (R = 1024, 8 layers, 256 latents)** | **735M** | **407M** |
| Cross-attn: shrunk blocks + same resampler (no output map) | 162M | 141M |

Training state in fp32 (weights + gradients + Adam moments, 16 bytes per parameter): 4B full arm ≈ 11.8 GB, 2B
full arm ≈ 6.5 GB, on top of the frozen model, VGGT and activations.
