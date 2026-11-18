"""Standalone SAM 3 inference: save masks for a list of frames, for reuse by OpenVocab-4D.

Runs on a machine with an NVIDIA GPU (see README.md). VGGT is neither needed nor loaded. --dry-run checks the
inputs and prints the plan without importing torch or SAM 3.

Two ways to run the same SAM 3 model (--backend):
  transformers  (default) transformers' Sam3Model, from the group's cached snapshot of facebook/sam3 at SAM3_REV, the
                revision the explainability results used. Needs no Hugging Face login when the snapshot is cached.
  sam3          Meta's sam3 package, as in milestone 2. Downloads sam3.pt, which needs an accepted license (or
                --checkpoint).
Both run SAM 3's promptable concept segmentation, one text prompt at a time, the image encoder once per frame.

Output, in --out:

  <stem>__masks.npz   one per frame, also when nothing was detected. The milestone 2 format:
                      masks [N,H,W] bool, boxes [N,4] xyxy, scores [N], prompt_ids [N], prompts, source
  <stem>__overlay.png with --overlays: the masks drawn on the frame, for a visual check
  manifest.json       written last, so it marks a finished run: prompts, score threshold and, per frame, the
                      image's SHA-256, its size and the detections per prompt

  python src/mvp/openvocab_adapter/sam3_infer.py --images <scene>/frames --frames all \
      --prompts "chair,door,fan" --out masks/scannet__scene0025_01
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FORMAT = "openvocab4d-sam3-masks/v1"
SAM3_REV = "3c879f39826c281e95690f02c7821c4de09afae7"  # as in src/mvp/explain/one_question.py


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of frames, e.g. a scene's frames/")
    p.add_argument("--frames", default="all",
                   help="positions in the sorted folder: all | uniform:K | subset:NAME | window:START:COUNT[:STRIDE] | list:a,b")
    p.add_argument("--meta", help="meta.json for subset:NAME (default: next to the frames folder)")
    p.add_argument("--prompts", required=True, help="comma-separated noun phrases, e.g. 'chair,brick wall'")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--backend", choices=["transformers", "sam3"], default="transformers",
                   help="transformers: the cached facebook/sam3 snapshot (no login); sam3: Meta's package (see above)")
    p.add_argument("--checkpoint", help="--backend sam3 only: local sam3.pt (default: download from Hugging Face)")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="keep detections from this score up; the pipeline later applies its own 0.55")
    p.add_argument("--overlays", action="store_true", help="also save a visual check per frame")
    p.add_argument("--dry-run", action="store_true", help="validate the inputs and print the plan, without SAM 3")
    p.add_argument("--overwrite", action="store_true", help="replace an earlier finished run in --out")
    return p.parse_args(argv)


def plan(args):
    images = common.list_images(args.images)
    if not images:
        sys.exit(f"[sam3] no images in {args.images}")
    meta_path = Path(args.meta) if args.meta else Path(args.images).resolve().parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    selected = [images[i] for i in common.select_frames(args.frames, len(images), meta)]
    prompts = list(dict.fromkeys(s.strip() for s in args.prompts.split(",") if s.strip()))
    if not prompts:
        sys.exit("[sam3] no prompts given")
    return selected, prompts


def detect(processor, image, prompts, min_score, to_numpy_masks, torch):
    """Every detection of every prompt, as in milestone 2: (mask, box, score, prompt)."""
    detections = []
    # SAM 3 must run under bf16 autocast (see milestone 2)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for prompt in prompts:
            output = processor.set_text_prompt(prompt=prompt, state=state)
            masks = to_numpy_masks(output["masks"])
            boxes = output["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
            scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
            for k in range(min(len(scores), len(masks))):
                if scores[k] >= min_score:
                    detections.append((masks[k], boxes[k], float(scores[k]), prompt))
    return detections


def transformers_detector(torch):
    """detect(image, prompts, min_score) with transformers' Sam3Model; the same (mask, box, score, prompt) tuples."""
    from transformers import Sam3Model, Sam3Processor

    hub = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    snapshot = os.path.join(hub, "models--facebook--sam3", "snapshots", SAM3_REV)
    # facebook/sam3 is gated: the cached snapshot needs no Hub access; otherwise download that revision
    source, kwargs = (snapshot, {}) if os.path.isdir(snapshot) else ("facebook/sam3", {"revision": SAM3_REV})
    print(f"[sam3] transformers Sam3Model from {source}")
    processor = Sam3Processor.from_pretrained(source, **kwargs)
    model = Sam3Model.from_pretrained(source, **kwargs).to("cuda").eval()
    texts = {}

    def detect(image, prompts, min_score):
        width, height = image.size
        detections = []
        with torch.no_grad():
            pixels = processor(images=image, return_tensors="pt").to("cuda")
            vision = model.get_vision_features(pixel_values=pixels.pixel_values)
            for prompt in prompts:
                if prompt not in texts:
                    texts[prompt] = processor(text=prompt, return_tensors="pt").to("cuda")
                output = model(vision_embeds=vision, input_ids=texts[prompt].input_ids,
                               attention_mask=texts[prompt].attention_mask)
                result = processor.post_process_instance_segmentation(
                    output, threshold=min_score, target_sizes=[(height, width)])[0]
                masks = result["masks"].bool().cpu().numpy()
                boxes = result["boxes"].float().cpu().numpy().reshape(-1, 4)
                scores = result["scores"].float().cpu().numpy().reshape(-1)
                for k in range(len(scores)):
                    detections.append((masks[k], boxes[k], float(scores[k]), prompt))
        return detections

    return detect, f"facebook/sam3@{SAM3_REV[:7]} (transformers Sam3Model)"


def sam3_detector(torch, checkpoint, to_numpy_masks):
    """detect(image, prompts, min_score) with Meta's sam3 package, as in milestone 2."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    processor = Sam3Processor(build_sam3_image_model(checkpoint_path=checkpoint))
    return (lambda image, prompts, min_score: detect(processor, image, prompts, min_score, to_numpy_masks, torch),
            "facebook/sam3 (sam3 package)")


def save_frame(out, path, image, detections, prompts):
    width, height = image.size
    if detections:
        masks = np.stack([d[0] for d in detections])
        if masks.shape[1:] != (height, width):
            sys.exit(f"[sam3] {path.name}: masks are {masks.shape[1:]}, the image is {(height, width)}")
        boxes = np.stack([d[1] for d in detections])
    else:  # an explicit empty result, so "nothing found" cannot be mistaken for a missing frame
        masks, boxes = np.zeros((0, height, width), bool), np.zeros((0, 4), np.float32)
    np.savez_compressed(out / f"{path.stem}__masks.npz", masks=masks, boxes=boxes,
                        scores=np.array([d[2] for d in detections], np.float64),
                        prompt_ids=np.array([prompts.index(d[3]) for d in detections], np.int64),
                        prompts=np.array(prompts), source=str(path))


def run(args, frames, prompts, out):
    import torch

    if not torch.cuda.is_available():
        sys.exit("[sam3] no CUDA GPU here. This script is meant for a machine with an NVIDIA GPU; see src/mvp/openvocab_adapter/README.md")

    sys.path.insert(0, str(common.TRACK_A))
    from milestone2_sam3_concepts import draw_overlay, to_numpy_masks

    started = time.perf_counter()
    if args.backend == "transformers":
        find, model_name = transformers_detector(torch)
    else:
        find, model_name = sam3_detector(torch, args.checkpoint, to_numpy_masks)
    print(f"[sam3] model loaded in {time.perf_counter() - started:.1f}s")

    records = []
    for i, path in enumerate(frames):
        image = Image.open(path).convert("RGB")
        detections = find(image, prompts, args.min_score)
        save_frame(out, path, image, detections, prompts)
        if args.overlays:
            draw_overlay(image, detections).save(out / f"{path.stem}__overlay.png")
        found = {p: sum(1 for d in detections if d[3] == p) for p in prompts}
        records.append({"stem": path.stem, "file": f"{path.stem}__masks.npz", "image_sha256": common.sha256_file(path),
                        "image_hw": [image.size[1], image.size[0]], "detections": found})
        print(f"[sam3] {i + 1}/{len(frames)} {path.stem}: {sum(found.values())} detections")

    (out / "manifest.json").write_text(json.dumps({
        "format": FORMAT, "model": model_name, "backend": args.backend, "prompts": prompts, "min_score": args.min_score,
        "frame_selection": args.frames, "gpu": torch.cuda.get_device_name(0),
        "seconds": round(time.perf_counter() - started, 1), "frames": records,
    }, indent=1))
    print(f"[sam3] done: {len(frames)} frames -> {out}")


def main(argv=None):
    args = parse_args(argv)
    frames, prompts = plan(args)
    print(f"[sam3] {len(frames)} frames ({frames[0].name} .. {frames[-1].name}) x {len(prompts)} prompts: {prompts}")
    if args.dry_run:
        print(f"[sam3] dry run: would write {len(frames)} mask files and manifest.json to {args.out}")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        if not args.overwrite:
            sys.exit(f"[sam3] {out} already holds a finished run; pass --overwrite to replace it")
        (out / "manifest.json").unlink()
        for stale in out.glob("*__masks.npz"):
            stale.unlink()
    run(args, frames, prompts, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
