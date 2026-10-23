"""MMSI-Bench data, prompt and answer extraction.

Prompt and answer extraction follow the official repo (InternRobotics/MMSI-Bench, vlmeval/dataset/mmsi_bench.py):
images, then "{question}\nAnswer with the option's letter from the given choices directly. Enclose the option's
letter within ``.", parsed with the regex extractor (``x`` -> `x` -> first standalone A-D). The official
pipeline falls back to an LLM judge for unparsed answers; we count them as wrong and report the parse rate.
"""
import io
import re

from PIL import Image

POST_PROMPT = "Answer with the option's letter from the given choices directly. Enclose the option's letter within ``."


def extract_choice(pred):
    m = re.search(r"``([^`]*)``", pred)
    if m:
        pred = m.group(1)
    m = re.search(r"`([^`]*)`", pred)
    if m:
        pred = m.group(1)
    m = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", pred)
    return m.group() if m else None


def to_pil(x):
    return Image.open(io.BytesIO(x["bytes"] if isinstance(x, dict) else x)).convert("RGB")

