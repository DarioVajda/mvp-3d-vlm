"""Thin Qwen3-VL / Qwen3.5 wrapper, following lmms-eval's `qwen3_vl` model (lmms_eval/models/simple/qwen3_vl.py):
qwen_vl_utils frame sampling/resizing with image_patch_size=16, processor called with do_resize=False,
system prompt "You are a helpful assistant.", greedy decoding. Qwen3.5 (same processor as Qwen3-VL) thinks by
default; thinking is turned off, as in SpatialStack's lmms-eval `qwen3_5` model (disable_thinking=true).
"""
import re

import torch
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoProcessor

from mvp.fusion.model import stop_tokens

SYSTEM_PROMPT = "You are a helpful assistant."


class QwenVL:
    def __init__(self, model_id, min_pixels=256 * 32 * 32, max_pixels=2048 * 32 * 32, attn="sdpa", weights=True):
        """weights=False loads only the processor: enough to build prompts (render_prompt), not to generate."""
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda").eval() if weights else None
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

    def video_inputs(self, video, meta, video_kwargs, prompts, pres=None, system=SYSTEM_PROMPT):
        """The batch generate_video sends to the model: each prompt's chat template (`pres`: text before the video),
        left-padded, with the video once per prompt. Needs only the processor (weights=False)."""
        pres = pres or [""] * len(prompts)
        texts = [self.processor.apply_chat_template(self._messages(p, "video", pre=q, system=system), tokenize=False,
                                                    add_generation_prompt=True, **self.template_kwargs)
                 for p, q in zip(prompts, pres)]
        return self.processor(text=texts, videos=[video] * len(texts), video_metadata=[meta] * len(texts),
                              do_resize=False, padding=True, return_tensors="pt", **video_kwargs)

    @torch.inference_mode()
    def generate_video(self, video, meta, video_kwargs, prompts, max_new_tokens=16, pres=None, system=SYSTEM_PROMPT):
        """Answer several prompts about the same video in one batch. `pres`: optional text placed before the video."""
        return self._generate(self.video_inputs(video, meta, video_kwargs, prompts, pres, system), max_new_tokens)

    def render_prompt(self, video, meta, video_kwargs, prompt, pre="", system=SYSTEM_PROMPT):
        """The full input of generate_video for one prompt, as text: the chat template with the processor's
        "<t seconds>" label before each merged frame, and each frame's run of video tokens shown as
        "<|video_pad|> x N"."""
        text = self.processor.apply_chat_template(self._messages(prompt, "video", pre=pre, system=system), tokenize=False,
                                                  add_generation_prompt=True, **self.template_kwargs)
        inputs = self.processor(text=[text], videos=[video], video_metadata=[meta], do_resize=False, return_tensors="pt",
                                **video_kwargs)
        decoded = self.processor.tokenizer.decode(inputs["input_ids"][0])
        pad = self.processor.video_token
        return re.sub(f"(?:{re.escape(pad)})+", lambda m: f"{pad} x {len(m.group(0)) // len(pad)}", decoded)

    def text_inputs(self, prompts, pres=None, system=SYSTEM_PROMPT):
        """The batch generate_text sends to the model: each prompt's chat template with no video or image (`pres`:
        text before the prompt), left-padded. Needs only the processor (weights=False)."""
        pres = pres or [""] * len(prompts)
        return self.processor(text=[self.render_text_prompt(p, q, system) for p, q in zip(prompts, pres)],
                              padding=True, return_tensors="pt")

    @torch.inference_mode()
    def generate_text(self, prompts, max_new_tokens=16, pres=None, system=SYSTEM_PROMPT):
        """Answer several prompts without any video or image (the eval's --no-video)."""
        return self._generate(self.text_inputs(prompts, pres, system), max_new_tokens)

    def render_text_prompt(self, prompt, pre="", system=SYSTEM_PROMPT):
        """The full input of generate_text for one prompt, as text: the chat template with no visual parts."""
        return self.processor.apply_chat_template(self._messages(prompt, None, 0, pre=pre, system=system),
                                                  tokenize=False, add_generation_prompt=True, **self.template_kwargs)

    @torch.inference_mode()
    def generate_images(self, images, prompt, max_new_tokens=16):
        text = self.processor.apply_chat_template(self._messages(prompt, "image", len(images)), tokenize=False,
                                                  add_generation_prompt=True, **self.template_kwargs)
        inputs = self.processor(text=[text], images=images, do_resize=False, padding=True, return_tensors="pt")
        return self._generate(inputs, max_new_tokens)

    def _generate(self, inputs, max_new_tokens):
        """(replies, padded input length). Also sets last_new_tokens, one (tokens, hit_limit) per reply
        (new_token_counts), and last_prompt_tokens, each prompt's own input length without the batch's padding."""
        inputs = inputs.to(self.model.device)
        self.last_prompt_tokens = inputs["attention_mask"].sum(dim=1).tolist()
        stops = stop_tokens(self.processor.tokenizer)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None, **stops)
        out = out[:, inputs["input_ids"].shape[1]:]
        self.last_new_tokens = new_token_counts(out, stops["eos_token_id"])
        return [s.strip() for s in self.processor.batch_decode(out, skip_special_tokens=True)], \
            int(inputs["input_ids"].shape[1])


def new_token_counts(generated, stop_ids):
    """For each generated row (prompt removed): (tokens before its first stop token, whether it has none). Generation
    ends when every row has stopped or at max_new_tokens, and stopped rows are padded after their stop token, so a row
    without one was cut off by max_new_tokens."""
    is_stop = torch.isin(generated, torch.tensor(stop_ids, device=generated.device))
    counts = []
    for row in is_stop:
        hits = row.nonzero()
        counts.append((int(hits[0]), False) if len(hits) else (int(row.numel()), True))
    return counts
