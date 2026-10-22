"""Load the checkpoint the way lmms_eval/models/qwen3_5.py does and check that every checkpoint tensor (geometry
encoder and fusion layers included) equals the corresponding parameter of the loaded model."""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

ckpt = sys.argv[1]
model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
    ckpt, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2",
    geometry_encoder_path="facebook/VGGT-1B").eval()
print("attn:", model.config._attn_implementation, model.config.text_config._attn_implementation,
      model.config.vision_config._attn_implementation)
sd = model.state_dict()
stats = {"equal": 0, "differ": 0, "missing_in_model": 0}
bad = []
for f in sorted(Path(ckpt).glob("*.safetensors")):
    with safe_open(f, framework="pt", device="cpu") as h:
        for k in h.keys():
            if k not in sd:
                stats["missing_in_model"] += 1; bad.append(("missing", k)); continue
            if torch.equal(h.get_tensor(k).to(sd[k].dtype), sd[k].cpu()):
                stats["equal"] += 1
            else:
                stats["differ"] += 1; bad.append(("differ", k))
print(stats)
for b in bad[:30]:
    print(*b)
geo = [k for k in sd if "geometry_encoder" in k or "language_feature_fusion" in k]
print("geometry/fusion params in model:", len(geo))
print("fusion layers:", model.config.geometry_fusion_layers, "encoder layers:", model.config.geometry_encoder_layers)
