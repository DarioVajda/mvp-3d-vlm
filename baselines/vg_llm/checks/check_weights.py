"""Load the checkpoint the way lmms_eval/models/vgllm.py does and check that every checkpoint tensor (VGGT encoder
and geometry merger included) equals the corresponding parameter of the loaded model.
    check_weights.py <checkpoint dir>"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

ckpt = sys.argv[1]
config = AutoConfig.from_pretrained(ckpt)
assert getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False)
model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
    ckpt, config=config, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2").eval()
print("attn:", model.config._attn_implementation, model.config.vision_config._attn_implementation)
sd = model.state_dict()
stats = {"equal": 0, "differ": 0, "missing_in_model": 0}
bad = []
seen = set()
for f in sorted(Path(ckpt).glob("*.safetensors")):
    with safe_open(f, framework="pt", device="cpu") as h:
        for k in h.keys():
            seen.add(k)
            if k not in sd:
                stats["missing_in_model"] += 1; bad.append(("missing", k)); continue
            if torch.equal(h.get_tensor(k).to(sd[k].dtype), sd[k].cpu()):
                stats["equal"] += 1
            else:
                stats["differ"] += 1; bad.append(("differ", k))
stats["model_params_not_in_checkpoint"] = len([k for k in sd if k not in seen])
print(stats)
for b in bad[:30]:
    print(*b)
print("not in checkpoint:", [k for k in sd if k not in seen][:30])
print("geometry params in model:", len([k for k in sd if k.startswith(("geometry_encoder", "geometry_merger"))]))
