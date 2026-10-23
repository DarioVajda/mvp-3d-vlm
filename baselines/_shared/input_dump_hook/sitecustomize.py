"""Input fingerprints for baseline runs, without changing the baselines' code.

Put this directory on PYTHONPATH and set DUMP_INPUTS=<file.jsonl>: every model.generate() call then appends one line
with the decoded prompt (special tokens kept), the shape and sha256 of every tensor argument (pixel values, grids,
geometry-encoder inputs) and the generation kwargs. DUMP_INPUTS_MAX limits the number of lines (default 100).
"""
import os

if os.environ.get("DUMP_INPUTS"):
    import hashlib
    import json

    import transformers

    _path, _max = os.environ["DUMP_INPUTS"], int(os.environ.get("DUMP_INPUTS_MAX", "100"))
    _orig = transformers.GenerationMixin.generate
    _count = [0]

    def _desc(v):
        import torch
        if isinstance(v, torch.Tensor):
            t = v.detach().to("cpu").contiguous()
            raw = t.view(torch.uint8) if t.dtype == torch.bfloat16 else t
            return {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16],
                    **({"min": float(t.float().min()), "max": float(t.float().max())} if t.numel() and t.is_floating_point() else {})}
        if isinstance(v, (list, tuple)):
            return [_desc(x) for x in v]
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        return str(type(v))

    def generate(self, *args, **kwargs):
        if _count[0] < _max:
            _count[0] += 1
            rec = {k: _desc(v) for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")}
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                tok = getattr(self, "_dump_tokenizer", None)
                if tok is None:
                    tok = transformers.AutoTokenizer.from_pretrained(self.config._name_or_path)
                    self._dump_tokenizer = tok
                rec["input_ids"] = _desc(ids)
                rec["n_tokens"] = int(ids.shape[-1])
                rec["prompt"] = [tok.decode(r, skip_special_tokens=False) for r in ids.tolist()]
            rec["generation_config"] = {k: v for k, v in (kwargs.get("generation_config").to_dict().items()
                                        if kwargs.get("generation_config") is not None else [])
                                        if k in ("do_sample", "temperature", "top_p", "top_k", "num_beams", "max_new_tokens")}
            with open(_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return _orig(self, *args, **kwargs)

    transformers.GenerationMixin.generate = generate
