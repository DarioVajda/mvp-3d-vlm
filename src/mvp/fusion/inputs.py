"""Model inputs for one video (or one multi-image question): frames as separate images, SpatialStack's protocol.

Shared by training and evaluation, so both see exactly the same inputs:
  * frames: PIL images, e.g. 8 (train) or 32 (eval) uniformly spaced video frames (np.linspace(0, n-1, k), as in
    SpatialStack's data_qwen.read_video_images and lmms-eval qwen3_5._sample_video_frames);
  * Qwen: each frame is one image, resized by the processor within [min_pixels, max_pixels] to multiples of 32;
    a 640x480 frame gives a 40x30 patch grid = 300 visual tokens;
  * VGGT: each frame resized to Qwen's patch grid x 14 px (SpatialStack's build_qwen3_5_geometry_inputs), so VGGT's
    2x2 patch windows line up with Qwen's merged tokens (640x480 -> 560x420). Frames of different sizes (MMSI) are
    padded with 1.0 at the bottom/right, not centred as in SpatialStack, so each frame's grid is a top-left crop;
  * text: system prompt + [images..., question] + generation prompt with thinking off, which ends in
    "<think>\n\n</think>\n\n"; for training the answer and <|im_end|> follow and are the only supervised tokens.
"""
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import functional as TF

SYSTEM_PROMPT = "You are a helpful assistant."
VGGT_PATCH = 14
# SpatialStack's pixel limits: training script (train.sh) and lmms-eval qwen3_5 defaults
TRAIN_PIXELS = dict(min_pixels=16 * 28 * 28, max_pixels=576 * 28 * 28)
EVAL_PIXELS = dict(min_pixels=256 * 28 * 28, max_pixels=1605632)


def uniform_frames(path, k):
    """k uniformly spaced frames (all frames if the video is shorter), as PIL images, plus their indices."""
    vr = VideoReader(str(path), ctx=cpu(0))
    n = len(vr)
    idx = np.arange(n) if n <= k else np.linspace(0, n - 1, k).astype(int)
    return [Image.fromarray(f) for f in vr.get_batch(idx.tolist()).asnumpy()], idx.tolist()


class InputBuilder:
    def __init__(self, processor):
        self.processor = processor
        self.tok = processor.tokenizer
        self.image_token = processor.image_token
        self.image_token_id = self.tok.convert_tokens_to_ids(self.image_token)
        self.merge = processor.image_processor.merge_size
        self.end_ids = self.tok("<|im_end|>", add_special_tokens=False).input_ids

    def visual(self, frames):
        """Qwen pixel values and grids for the frames, and the VGGT input [T, 3, H, W] in [0, 1]."""
        out = self.processor.image_processor(images=frames, return_tensors="pt")
        grid = out["image_grid_thw"]  # [T, 3] = (1, h, w) patches
        vggt = [TF.to_tensor(f.convert("RGB").resize((int(w) * VGGT_PATCH, int(h) * VGGT_PATCH), Image.Resampling.BICUBIC))
                for f, (_, h, w) in zip(frames, grid.tolist())]
        H, W = max(t.shape[1] for t in vggt), max(t.shape[2] for t in vggt)
        vggt = torch.stack([F.pad(t, (0, W - t.shape[2], 0, H - t.shape[1]), value=1.0) for t in vggt])
        return {"pixel_values": out["pixel_values"], "image_grid_thw": grid, "vggt_images": vggt}

    def prompt_text(self, question, n_images):
        content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": question}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def expand(self, text, grid):
        """Replace each image placeholder by its number of merged visual tokens, as the Qwen processor does."""
        parts = text.split(self.image_token)
        assert len(parts) == len(grid) + 1, (len(parts) - 1, len(grid))
        n_tok = (grid.prod(-1) // self.merge ** 2).tolist()
        return parts[0] + "".join(self.image_token * n + p for n, p in zip(n_tok, parts[1:]))

    def text(self, questions, grid, answers=None, padding_side="right"):
        """Token ids for B questions about the same frames. With answers: labels on answer + <|im_end|> only."""
        rows, labels = [], []
        for i, q in enumerate(questions):
            ids = self.tok(self.expand(self.prompt_text(q, len(grid)), grid), add_special_tokens=False).input_ids
            lab = [-100] * len(ids)
            if answers is not None:
                ans = self.tok(answers[i], add_special_tokens=False).input_ids + self.end_ids
                ids, lab = ids + ans, lab + ans
            rows.append(ids)
            labels.append(lab)
        L = max(map(len, rows))
        pad = self.tok.pad_token_id
        out = {k: torch.full((len(rows), L), v, dtype=torch.long) for k, v in
               [("input_ids", pad), ("attention_mask", 0), ("labels", -100)]}
        for i, (ids, lab) in enumerate(zip(rows, labels)):
            s = slice(0, len(ids)) if padding_side == "right" else slice(L - len(ids), L)
            out["input_ids"][i, s] = torch.tensor(ids)
            out["attention_mask"][i, s] = 1
            out["labels"][i, s] = torch.tensor(lab)
        out["mm_token_type_ids"] = (out["input_ids"] == self.image_token_id).int()
        if answers is None:
            del out["labels"]
        return out

    def build(self, frames, questions, answers=None, padding_side="right"):
        vis = self.visual(frames)
        return {**vis, **self.text(questions, vis["image_grid_thw"], answers, padding_side)}
