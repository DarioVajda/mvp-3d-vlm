"""Qwen3.5 with LoRA and VGGT geometry fusion: one module for training and evaluation (docs/idea.md, "Arms").

Arms:
  base      the untouched Qwen3.5 (zero-shot arms 0a/0b)
  lora      LoRA only, no geometry (control 1)
  additive  SpatialStack's recipe: VGGT layer L_k -> its own AdditiveMerger -> added at the visual-token positions
            to the output of decoder layer k (control 2)
  xattn     same inputs and injection points, with the merger replaced by PerceiverResampler(VGGT layer L_k) +
            a cross-attention block whose output is added to the output of decoder layer k (arm 4)
  xattn_aligned  the additive arm's mergers (not zero-initialised), read through AlignedXAttn: each visual token
            attends over all geometry tokens of the video instead of receiving its own one by addition; options
            rope3 (3-axis RoPE on frame, row, col) and local_bias (learned locality bias, init lam, mu)

Defaults follow the released SpatialStack checkpoint: VGGT layers (11, 17, 23) paired in order with decoder layers
(0, 1, 2) (their geometry_encoder_layers / geometry_fusion_layers). The fusion runs in a forward hook on those decoder
layers, so the Qwen module tree, its checkpoint names and its gradient checkpointing are untouched; with checkpointing
on, the hooks are recomputed in backward too. VGGT is frozen, kept inside the model and run on the fly on the same
frames Qwen sees, once per video: a forward first computes the geometry of its frames (merged tokens or latents per
injection point), then the LLM pass reads it through the hooks.

  model = FusionVL("Qwen/Qwen3.5-2B", "xattn").cuda()
  per_question_loss = model(batch)        # batch from InputBuilder.build(frames, questions, answers)
  answers = model.generate(batch)         # batch built with padding_side="left", no answers

Pilot checkpoints (24 Oct, one shared merger / resampler, pre-hooks on the softmax layers, tanh gates) still load:
`load` translates their config and state names (see _upgrade).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from mvp.fusion.inputs import EVAL_PIXELS, InputBuilder
from mvp.fusion.modules import AdditiveMerger, AlignedXAttn, GatedXAttnDense, PerceiverResampler

ARMS = ("base", "lora", "additive", "xattn", "xattn_aligned")
# All linear layers of the language model except Gated DeltaNet's in_proj_a / in_proj_b, which output one scalar per
# head (16-32 wide), narrower than the rank
LORA_TARGETS = (r".*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                r"|mlp\.(gate|up|down)_proj)")
VGGT_ID = "facebook/VGGT-1B"


class FusionVL(nn.Module):
    def __init__(self, model_id, arm, lora_r=64, lora_alpha=128, lora_dropout=0.05, vggt_layers=(11, 17, 23),
                 fusion_after=(0, 1, 2), gate="none", resampler_dim=1024, resampler_layers=8, resampler_heads=8,
                 num_latents=256, special_tokens=False, rope3=True, local_bias=True, bias_lam=1.5, bias_mu=3.0,
                 train_frames=8, pixels=None, grad_ckpt=False):
        super().__init__()
        assert arm in ARMS, arm
        pixels = pixels or EVAL_PIXELS
        self.qwen = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.qwen.requires_grad_(False)
        tc = self.qwen.config.text_config
        D = tc.hidden_size
        vggt_layers, fusion_after = list(vggt_layers), list(fusion_after)
        if len(vggt_layers) == 1:
            vggt_layers = vggt_layers * len(fusion_after)
        assert len(vggt_layers) == len(fusion_after) == len(set(fusion_after)), (vggt_layers, fusion_after)
        assert all(0 <= k < tc.num_hidden_layers for k in fusion_after), fusion_after
        self.config = dict(model_id=model_id, arm=arm, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           vggt_layers=vggt_layers, fusion_after=fusion_after, gate=gate, resampler_dim=resampler_dim,
                           resampler_layers=resampler_layers, resampler_heads=resampler_heads, num_latents=num_latents,
                           special_tokens=special_tokens)
        if arm == "xattn_aligned":
            self.config.update(rope3=rope3, local_bias=local_bias, bias_lam=bias_lam, bias_mu=bias_mu,
                               train_frames=train_frames)
        self.arm = arm
        self.pairs = dict(zip(fusion_after, vggt_layers))  # decoder layer k -> VGGT layer whose features it receives
        self.processor = AutoProcessor.from_pretrained(model_id, **pixels)
        self.processor.tokenizer.padding_side = "left"
        self.inputs = InputBuilder(self.processor)

        if arm != "base":
            from peft import LoraConfig, inject_adapter_in_model
            inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                               target_modules=LORA_TARGETS), self.qwen.model.language_model)
            for n, p in self.qwen.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()  # fp32 trainable weights; matmuls run under bf16 autocast
                    p.requires_grad_(True)

        self.vggt = None
        if arm in ("additive", "xattn", "xattn_aligned"):
            from vggt.models.vggt import VGGT
            self.vggt = VGGT.from_pretrained(VGGT_ID).aggregator  # the heads are not needed
            self.vggt.cached_layer_indices = set(vggt_layers)  # keep only the layers we read
            self.vggt.requires_grad_(False).eval()
        keys = [str(k) for k in fusion_after]
        if arm == "xattn":
            self.resamplers = nn.ModuleDict({k: PerceiverResampler(2048, resampler_dim, D, num_latents, resampler_layers,
                                                                   resampler_heads) for k in keys})
            self.blocks = nn.ModuleDict({k: GatedXAttnDense(D, D // 128, gate) for k in keys})
        if arm == "additive":
            self.mergers = nn.ModuleDict({k: AdditiveMerger(2048, 4096, D) for k in keys})
        if arm == "xattn_aligned":
            self.mergers = nn.ModuleDict({k: AdditiveMerger(2048, 4096, D, zero_init=False) for k in keys})
            self.aligned = nn.ModuleDict({k: AlignedXAttn(D, D // 128, rope3, local_bias, bias_lam, bias_mu, train_frames)
                                          for k in keys})
        self._pos = None          # xattn_aligned: (t, r, c) of the current video's visual tokens [N, 3]
        self.collect_attn = False  # xattn_aligned: record AlignedXAttn.attn_stats on the next forward (dev diagnostics)
        self.attn_stats = {}

        self._geo = None          # geometry of the current video per injection point: {k: latents [L, D] or merged tokens [N_visual, D]}
        self._visual_mask = None  # [B, T] visual-token positions of the current forward's input_ids
        self.stats = {}           # {k: ||added||/||h|| of the last forward}, how far the injection point has opened
        if arm in ("additive", "xattn", "xattn_aligned"):
            layers = self.qwen.model.language_model.layers
            for k in fusion_after:
                layers[k].register_forward_hook(self._make_hook(str(k)))
            self.qwen.model.register_forward_pre_hook(self._capture_ids, with_kwargs=True)
        if grad_ckpt:
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- geometry ----
    def train(self, mode=True):
        # Qwen stays in train mode during training (HF gradient checkpointing only runs in train mode; Qwen3.5's
        # attention dropout is 0); VGGT is always in eval mode
        super().train(mode)
        if self.vggt is not None:
            self.vggt.eval()
        return self

    @torch.no_grad()
    def vggt_frames(self, images, grid):
        """{VGGT layer: per frame (special tokens [5, 2048], patch grid [h, w, 2048])} for the layers in use."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, ps = self.vggt(images[None])
        Hp, Wp = images.shape[-2] // 14, images.shape[-1] // 14
        res = {}
        for L in set(self.pairs.values()):
            tok = out[L][0]  # [T, 5 + P, 2048]
            patches = tok[:, ps:].reshape(len(tok), Hp, Wp, -1)
            res[L] = [(tok[t, :ps], patches[t, :h, :w]) for t, (_, h, w) in enumerate(grid.tolist())]
        return res

    def set_geometry(self, batch):
        per_layer = self.vggt_frames(batch["vggt_images"], batch["image_grid_thw"])
        self._geo = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k, L in self.pairs.items():
                frames = per_layer[L]
                if self.arm == "xattn":
                    x = torch.cat([torch.cat([s, p.reshape(-1, p.shape[-1])]) if self.config["special_tokens"]
                                   else p.reshape(-1, p.shape[-1]) for s, p in frames])
                    self._geo[str(k)] = self.resamplers[str(k)](x)
                else:
                    self._geo[str(k)] = torch.cat([self.mergers[str(k)](p) for _, p in frames])
        if self.arm == "xattn_aligned":
            self._pos = AlignedXAttn.positions(batch["image_grid_thw"], self.inputs.merge)

    def _capture_ids(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self._visual_mask = None if ids is None else ids == self.inputs.image_token_id

    def _make_hook(self, key):
        def hook(module, args, out):
            """Forward hook on decoder layer k: adds the geometry of this video to the layer's output."""
            if self._geo is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            geo = self._geo[key]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.arm == "xattn":
                    new = self.blocks[key](h, geo).to(h.dtype)
                    ref = h
                else:
                    m = self._visual_mask
                    if m is None or m.shape != h.shape[:2] or not m.any():
                        return None  # decoding steps after the prefill: no visual tokens
                    n = int(m[0].sum())
                    assert n == len(geo) and bool((m.sum(1) == n).all()), "visual tokens != VGGT windows"
                    delta = torch.zeros_like(h)
                    if self.arm == "xattn_aligned":  # visual tokens attend over the geometry tokens of the video
                        y = h[m].view(h.shape[0], n, -1)
                        if self.collect_attn and not torch.is_grad_enabled():
                            self.attn_stats[key] = self.aligned[key].attn_stats(y, geo, self._pos)
                        delta[m] = self.aligned[key](y, geo, self._pos).reshape(-1, h.shape[-1]).to(h.dtype)
                    else:
                        delta[m] = geo.to(h.dtype).repeat(h.shape[0], 1)
                    new = h + delta
                    ref = h[m]
                if torch.is_grad_enabled() or not self.training:
                    self.stats[key] = ((new - h).norm() / ref.norm().clamp(min=1e-6)).item()
            return (new, *out[1:]) if isinstance(out, tuple) else new
        return hook

    # ---- LLM ----
    @torch.no_grad()
    def image_features(self, batch, B):
        """Qwen vision encoder once per video; the merged tokens are repeated for the B questions about it."""
        out = self.qwen.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True)
        return {"image": BaseModelOutputWithPooling(pooler_output=tuple(out.pooler_output) * B)}

    def llm_inputs(self, batch):
        B = batch["input_ids"].shape[0]
        return dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    mm_token_type_ids=batch["mm_token_type_ids"], image_grid_thw=batch["image_grid_thw"].repeat(B, 1),
                    mm_encoder_outputs=self.image_features(batch, B))

    def forward(self, batch):
        """Mean answer-token loss of each question, [B]. Logits are only computed at supervised positions."""
        self._geo = None
        if self.vggt is not None:
            self.set_geometry(batch)  # kept until the next forward: checkpointed layers re-read it in backward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.qwen.model(**self.llm_inputs(batch), use_cache=False).last_hidden_state
            targets = batch["labels"][:, 1:]
            sel = targets != -100
            logits = self.qwen.lm_head(hs[:, :-1][sel])
        tok_loss = F.cross_entropy(logits.float(), targets[sel], reduction="none")
        rows = sel.nonzero()[:, 0]
        n = sel.sum(1).clamp(min=1)
        return torch.zeros(len(n), device=tok_loss.device).index_add(0, rows, tok_loss) / n

    @torch.no_grad()
    def generate(self, batch, max_new_tokens=16):
        self._geo = None
        if self.vggt is not None:
            self.set_geometry(batch)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.qwen.generate(**self.llm_inputs(batch), max_new_tokens=max_new_tokens, do_sample=False,
                                         temperature=None, top_p=None, top_k=None, **stop_tokens(self.processor.tokenizer))
        finally:
            self._geo = None
        out = out[:, batch["input_ids"].shape[1]:]
        return [s.strip() for s in self.processor.batch_decode(out, skip_special_tokens=True)]

    # ---- state ----
    def gates(self):
        if self.arm != "xattn":
            return {}
        return {k: g for k, b in self.blocks.items() if (g := b.gates())}

    def bias_slopes(self):
        """xattn_aligned with local_bias: {k: {"lam": mean over heads, "mu": ...}} of the learned locality slopes."""
        if self.arm != "xattn_aligned" or not self.config["local_bias"]:
            return {}
        return {k: {"lam": b.lam.mean().item(), "mu": b.mu.mean().item()} for k, b in self.aligned.items()}

    def fusion_modules(self):
        """{k: the new module(s) at injection point k}, for diagnostics."""
        if self.arm == "xattn":
            return {k: (self.resamplers[k], self.blocks[k]) for k in self.blocks}
        if self.arm == "additive":
            return {k: (self.mergers[k],) for k in self.mergers}
        if self.arm == "xattn_aligned":
            return {k: (self.mergers[k], self.aligned[k]) for k in self.mergers}
        return {}

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def save(self, path):
        torch.save({"config": self.config, "state": self.trainable_state()}, path)

    @classmethod
    def load(cls, path, **overrides):
        ckpt = torch.load(path, map_location="cpu")
        config, state = _upgrade(ckpt["config"], ckpt["state"])
        model = cls(**{**config, **overrides})
        missing, unexpected = model.load_state_dict(state, strict=False)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert not unexpected and not (trainable & set(missing)), (unexpected, sorted(trainable & set(missing))[:5])
        return model


def _upgrade(config, state):
    """Pilot checkpoints: one `merger` / `resampler` shared by pre-hooks on `fusion_layers`, tanh gates, special tokens.
    A pre-hook on layer i is a forward hook on layer i - 1; the shared module is copied to every injection point."""
    if "fusion_after" in config:
        return config, state
    config = dict(config)
    pre = config.pop("fusion_layers")
    after = [i - 1 for i in pre]
    config.update(vggt_layers=[config.pop("vggt_layer")] * len(after), fusion_after=after, gate="tanh")
    config.setdefault("special_tokens", True)
    new = {}
    for n, v in state.items():
        head, _, rest = n.partition(".")
        if head in ("merger", "resampler"):
            for k in after:
                new[f"{head}s.{k}.{rest}"] = v.clone()
        elif head == "blocks":
            i, _, rest = rest.partition(".")
            new[f"blocks.{int(i) - 1}.{rest}"] = v
        else:
            new[n] = v
    return config, new


def stop_tokens(tok):
    """Qwen3.5 ships no generation_config.json, so generate() stops only at the config's eos, <|endoftext|>. The base
    model emits it right after <|im_end|>; a model fine-tuned to end its answer with <|im_end|> does not, and runs on
    into invented turns. Stop at either."""
    ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    return {"eos_token_id": ids, "pad_token_id": tok.pad_token_id}


def count_params(model):
    groups = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            g = "lora" if "lora_" in n else n.split(".")[0]
            groups[g] = groups.get(g, 0) + p.numel()
    return groups
