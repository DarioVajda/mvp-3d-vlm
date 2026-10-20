"""Thin Qwen3-VL wrapper, following lmms-eval's `qwen3_vl` model (lmms_eval/models/simple/qwen3_vl.py):
qwen_vl_utils frame sampling/resizing with image_patch_size=16, processor called with do_resize=False,
system prompt "You are a helpful assistant.", greedy decoding.
"""
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

SYSTEM_PROMPT = "You are a helpful assistant."


class QwenVL:
    def __init__(self, model_id, min_pixels=256 * 32 * 32, max_pixels=2048 * 32 * 32, attn="sdpa"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        self.min_pixels, self.max_pixels = min_pixels, max_pixels

    def load_video(self, path, nframes=None, fps=None, max_frames=2048, total_pixels=224000 * 32 * 32):
        """Uniformly sample `nframes` frames (lmms-eval default: nframes=max_num_frames=32), or sample at `fps`
        (Qwen3-VL report video protocol: 2 fps, <=2048 frames, <=224K video tokens in total)."""
        ele = {"type": "video", "video": path, "min_pixels": self.min_pixels}
        if fps is not None:
            ele.update(fps=fps, max_frames=max_frames, total_pixels=total_pixels)
        else:
            ele.update(nframes=nframes, max_pixels=self.max_pixels)
        msg = [{"role": "user", "content": [ele]}]
        _, videos, video_kwargs = process_vision_info(
            msg, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True)
        video, meta = videos[0]
        return video, meta, video_kwargs

    def load_images(self, images):
        msg = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]}]
        imgs, _ = process_vision_info(msg, image_patch_size=16)
        return imgs

    def _messages(self, prompt, visual_type, n_visuals=1):
        content = [{"type": visual_type} for _ in range(n_visuals)] + [{"type": "text", "text": prompt}]
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content}]

    @torch.inference_mode()
    def generate_video(self, video, meta, video_kwargs, prompts, max_new_tokens=16):
        """Answer several prompts about the same video in one batch."""
        texts = [self.processor.apply_chat_template(self._messages(p, "video"), tokenize=False,
                                                    add_generation_prompt=True) for p in prompts]
        inputs = self.processor(text=texts, videos=[video] * len(texts), video_metadata=[meta] * len(texts),
                                do_resize=False, padding=True, return_tensors="pt", **video_kwargs)
        return self._generate(inputs, max_new_tokens)

    @torch.inference_mode()
    def generate_images(self, images, prompt, max_new_tokens=16):
        text = self.processor.apply_chat_template(self._messages(prompt, "image", len(images)), tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, do_resize=False, padding=True, return_tensors="pt")
        return self._generate(inputs, max_new_tokens)

    def _generate(self, inputs, max_new_tokens):
        inputs = inputs.to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None)
        out = out[:, inputs["input_ids"].shape[1]:]
        return [s.strip() for s in self.processor.batch_decode(out, skip_special_tokens=True)], \
            int(inputs["input_ids"].shape[1])
