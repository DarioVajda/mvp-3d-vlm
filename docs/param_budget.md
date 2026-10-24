# Trainable-parameter budget of the fusion arms

Qwen3.5-4B width D = 2560. VGGT-1B token width G = 2048. LoRA on the LLM is identical in every arm and
is not counted. Counts are weight matrices only (biases, norms and gate scalars are negligible).

## Perceiver resampler, Flamingo size (width R = 1536, 6 layers, 64 latents)

```
VGGT layer-23 tokens            learned latents
[frames x 1036, G=2048]         [64, R=1536]
        |                             |
   Linear G->R  (2048*1536 = 3.1M)    |
        v                             v
   +-----------------------------------------+
   |  x6 layers:                             |
   |   cross-attn  latents <- [tokens;latents]
   |     Wq, Wk, Wv, Wo : 4 * R*R           |   4 * 1536^2 = 9.4M
   |   FFN  R -> 4R -> R                     |   2 * 1536*6144 = 18.9M
   |                               per layer |   28.3M
   +-----------------------------------------+   x6 = 169.9M
        |
   Linear R->D  (1536*2560 = 3.9M)
        v
   [64 latents, D=2560]

   total  3.1 + 169.9 + 3.9 + latents 0.1  =  ~177M   (Flamingo reports 194M)
```

## Gated cross-attention-dense block, Flamingo style (one of 8, before LLM layers 3, 7, ..., 31)

```
   residual stream y  [tokens, D=2560]        latents x  [64, D]
          |                                       |
          |-- LN --> Q = y Wq   (D*D = 6.55M)     |-- K = x Wk  (6.55M)
          |                                       |-- V = x Wv  (6.55M)
          |          attn(Q,K,V) Wo  (6.55M)              cross-attn = 26.2M
          y = y + tanh(a_x) * attn         <- gate a_x, 1 scalar, init 0
          |
          |-- LN --> FFN D -> 4D -> D   (2 * 2560*10240 = 52.4M)
          y = y + tanh(a_f) * ffn          <- gate a_f, 1 scalar, init 0
          v
   into frozen Qwen layer

   per block  26.2 + 52.4  =  78.6M        x8 blocks  =  629M
```

## Shrunk cross-attention-only block (inner width 1024, 8 heads of 128, no FFN)

```
   y [tokens, 2560]                      latents x [64, 1024]
     |-- LN --> Q = y Wq   2560*1024 = 2.62M    K = x Wk  1024*1024 = 1.05M
     |                                           V = x Wv  1.05M
     |          attn(Q,K,V) Wo   1024*2560 = 2.62M
     y = y + tanh(a) * attn
   per block 7.3M        x8 = 59M
   + small resampler: width 1024, 2 layers
       in-proj 2048*1024 = 2.1M; per layer 4*1024^2 + 2*1024*4096 = 12.6M; x2 = 25.2M  -> 27M
   total  ~86M
```

## SpatialStack geometry token merger (additive arm; 2603.27437v3 App. A.2, shapes from the released checkpoint)

```
   VGGT tokens [frames x 37x28, G=2048]
        |  LayerNorm (2 x 2048 params)
        |  2x2 window -> concat 4 tokens  ->  [frames x 18x14, 4G = 8192]
        |  W1  8192 -> 4096            8192*4096 = 33.6M   (hidden width 4096, read from the checkpoint)
        |  GELU
        |  W2  4096 -> D               4096*2560 = 10.5M
        v
   [one 2560-d vector per merged Qwen visual token]
        +  added into the residual at visual-token positions after LLM layer j

   per merger 44.1M
   published (3 mergers, VGGT 11/17/23 -> LLM layers 0,1,2)  =  132M
   at the 8 block positions, separate mergers                 =  352M
   at the 8 positions, one shared merger                      =   44M

   Source: tensor shapes in `Journey9ni/SpatialStack-Qwen3.5-4B/model.safetensors` (header read over HTTP):
   `language_feature_fusion.fusion_layers.{0,1,2}.0.geo_ln.{weight,bias} [2048]`,
   `geo_mlp.0.weight [4096, 8192]`, `geo_mlp.0.bias [4096]`, `geo_mlp.2.weight [2560, 4096]`, `geo_mlp.2.bias [2560]`.
   The checkpoint stores each merger twice (under `language_feature_fusion.` and `model.language_feature_fusion.`),
   264M on disk, 132M distinct. The paper itself gives no count and no hidden width.
```

## Totals

| Arm | New weights |
|---|---|
| Cross-attn, full Flamingo blocks + Flamingo resampler | ~806M |
| Cross-attn, shrunk blocks + small resampler | ~86M |
| Additive, published SpatialStack (3 mergers) | 132M (measured) |
| Additive, 8 positions, separate mergers | ~352M |
| Additive, 8 positions, one shared merger | ~44M |
