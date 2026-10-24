"""Thin Qwen3-VL / Qwen3.5 wrapper, following lmms-eval's `qwen3_vl` model (lmms_eval/models/simple/qwen3_vl.py):
qwen_vl_utils frame sampling/resizing with image_patch_size=16, processor called with do_resize=False,
system prompt "You are a helpful assistant.", greedy decoding. Qwen3.5 (same processor as Qwen3-VL) thinks by
default; thinking is turned off, as in SpatialStack's lmms-eval `qwen3_5` model (disable_thinking=true).
"""
import torch
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoProcessor

SYSTEM_PROMPT = "You are a helpful assistant."


class QwenVL:
    def __init__(self, model_id, min_pixels=256 * 32 * 32, max_pixels=2048 * 32 * 32, attn="sdpa"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.template_kwargs = {"enable_thinking": False} if "Qwen3.5" in model_id else {}

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

    def load_video_frames(self, path, indices):
        """Like load_video, but with given frame indices (e.g. from space-aware sampling). Frames are resized
        exactly as load_video would for the same number of frames, and the metadata carries the true indices
        so Qwen3-VL's timestamps match the chosen frames."""
        video, meta, video_kwargs = self.load_video(path, nframes=len(indices))
        vr = VideoReader(path, ctx=cpu(0))
        frames = torch.from_numpy(vr.get_batch(list(indices)).asnumpy()).permute(0, 3, 1, 2)
        frames = transforms.functional.resize(frames, list(video.shape[-2:]), interpolation=InterpolationMode.BICUBIC,
                                              antialias=True).float()
        return frames, {**meta, "frames_indices": list(map(int, indices))}, video_kwargs

    def load_images(self, images):
        msg = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]}]
        imgs, _ = process_vision_info(msg, image_patch_size=16)
        return imgs

    def _messages(self, prompt, visual_type, n_visuals=1, pre="", system=SYSTEM_PROMPT):
        """User turn = [pre text] + visuals + [prompt text]; empty text parts are left out."""
        content = [{"type": "text", "text": pre}] if pre else []
        content += [{"type": visual_type} for _ in range(n_visuals)]
        content += [{"type": "text", "text": prompt}] if prompt else []
        return [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]

    @torch.inference_mode()
    def generate_video(self, video, meta, video_kwargs, prompts, max_new_tokens=16, pres=None, system=SYSTEM_PROMPT):
        """Answer several prompts about the same video in one batch. `pres`: optional text placed before the video."""
        pres = pres or [""] * len(prompts)
        texts = [self.processor.apply_chat_template(self._messages(p, "video", pre=q, system=system), tokenize=False,
                                                    add_generation_prompt=True, **self.template_kwargs)
                 for p, q in zip(prompts, pres)]
        inputs = self.processor(text=texts, videos=[video] * len(texts), video_metadata=[meta] * len(texts),
                                do_resize=False, padding=True, return_tensors="pt", **video_kwargs)
        return self._generate(inputs, max_new_tokens)

    @torch.inference_mode()
    def generate_images(self, images, prompt, max_new_tokens=16):
        text = self.processor.apply_chat_template(self._messages(prompt, "image", len(images)), tokenize=False,
                                                  add_generation_prompt=True, **self.template_kwargs)
        inputs = self.processor(text=[text], images=images, do_resize=False, padding=True, return_tensors="pt")
        return self._generate(inputs, max_new_tokens)

    def _generate(self, inputs, max_new_tokens):
        inputs = inputs.to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None)
        out = out[:, inputs["input_ids"].shape[1]:]
        return [s.strip() for s in self.processor.batch_decode(out, skip_special_tokens=True)], \
            int(inputs["input_ids"].shape[1])
