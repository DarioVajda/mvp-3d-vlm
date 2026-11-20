"""VSI-Bench data loading, prompts and metrics.

Prompts and scoring mirror lmms-eval's `vsibench` task (lmms_eval/tasks/vsibench/utils.py), which is also
the prompt format given in the Qwen3-VL technical report (Appendix B.6):
  * MCA: "These are frames of a video.\n{question}\nOptions:\n{options}\nAnswer with the option's letter ..."
  * NA:  "These are frames of a video.\n{question}\nPlease answer the question using a single word or phrase."
  * MCA scored by exact match on the first token of the answer, NA by Mean Relative Accuracy (MRA).
  * Overall = macro average over the 8 task types, with rel_direction easy/medium/hard averaged into one.
"""
import json
import re
import numpy as np
import pandas as pd

from mvp.paths import VSI_DIR

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]
# The 8 reported task types (column order of the VSI-Bench paper, Table 1)
TASKS = [
    "object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
    "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order",
]

PRE_PROMPT = "These are frames of a video."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NA_POST_PROMPT = "Please answer the question using a single word or phrase."


def load_vsibench(split="full"):
    """split: 'full' (5130 qs), 'debiased' (2362) or 'pruned' (2768)."""
    if split == "full":
        df = pd.read_json(VSI_DIR / "test.jsonl", lines=True)
    else:
        df = pd.read_parquet(VSI_DIR / f"test_{split}.parquet")
    df["video_path"] = [str(VSI_DIR / d / f"{s}.mp4") for d, s in zip(df["dataset"], df["scene_name"])]
    return df


def task_of(question_type):
    return "object_rel_direction" if question_type.startswith("object_rel_direction") else question_type


def build_prompt(doc, sheet=None, reasoning="direct", sheet_kind=None):
    """sheet: a rendered scene sheet (mvp.benchmarks.scene_sheet), put between the video sentence and the question.
    reasoning: "direct" (lmms-eval's prompt: answer at once) or "cot" (reason step by step first, then end on
    "Answer: X", which score_final reads; cot_prompt). sheet_kind: where the sheet comes from, its source ('pipeline'
    or 'gold'), for cot's instructions."""
    pre = PRE_PROMPT if sheet is None else PRE_PROMPT + "\n" + sheet
    if reasoning not in REASONING:
        raise ValueError(f"reasoning must be one of {REASONING}, not {reasoning!r}")
    if reasoning == "cot":
        return "\n".join([pre, question_block(doc), cot_prompt(doc, sheet is not None, sheet_kind)])
    if doc["question_type"] in NA_QUESTION_TYPES:
        return pre + "\n" + doc["question"] + "\n" + NA_POST_PROMPT
    options = "Options:\n" + "\n".join(doc["options"])
    return "\n".join([pre, doc["question"], options, MCA_POST_PROMPT])


# ---- prompt layouts for the prompt-echoing experiment (Abhinandan et al., arXiv 2607.15565) ----
# S = system, T = question, I = video. The baseline above is SIT: video first, then the lmms-eval text.
LAYOUTS = ["sit", "sti", "stit"]
ECHO_SYSTEM_PROMPT = ("You are a helpful assistant. The question is given before the video frames and repeated "
                      "after them; use the frames to answer it.")


def question_block(doc):
    """The question (and options) alone, without the lmms-eval pre/post prompts."""
    if doc["question_type"] in NA_QUESTION_TYPES:
        return doc["question"]
    return doc["question"] + "\nOptions:\n" + "\n".join(doc["options"])


def build_prompt_parts(doc, layout="sit", sheet=None, reasoning="direct", sheet_kind=None):
    """(text before the video, text after the video). The chat template joins parts with no separator, so text
    before the video ends in a newline to put the first frame timestamp on its own line.
      sit:  ("", baseline)                      question after the video (lmms-eval)
      sti:  (baseline + "\n", "")               the same text moved before the video
      stit: (question_block + "\n", baseline)   question echoed: before the video, and unchanged after it
    A scene sheet goes between the video and the question, so only with sit. reasoning and sheet_kind: build_prompt's.
    """
    if sheet is not None and layout != "sit":
        raise ValueError("a scene sheet goes between the video and the question: layout sit only")
    if layout == "sit":
        return "", build_prompt(doc, sheet, reasoning, sheet_kind)
    if layout == "sti":
        return build_prompt(doc, reasoning=reasoning) + "\n", ""
    if layout == "stit":
        return question_block(doc) + "\n", build_prompt(doc, reasoning=reasoning)
    raise ValueError(layout)


# ---- chain-of-thought instructions (reasoning="cot"; not part of lmms-eval) ----
# They take the place of the "answer directly" line, so nothing in the prompt asks for a bare answer. The procedure is
# the same for every question; each question type adds one hint on how to get its quantity, and the scene
# information's origin adds what to watch for in it. The answer line is the one score_final reads.
REASONING = ["direct", "cot"]
COT_STEPS = (
    "Think it through step by step before you answer. First, say which objects or places the question is about and "
    "find them in the video{where}. Then write down the facts you need, such as positions, distances, sizes, counts or "
    "the order in which things appear, and work out the answer from them one step at a time, showing any arithmetic. "
    "Before answering, check that the result agrees with what the video shows.")
COT_HINTS = {
    "object_counting": ("Count each physical object once, even if it appears in many frames, and remember that some "
                        "may be out of view in a given frame."),
    "object_abs_distance": ("Find where each of the two objects is, then work out the straight-line distance between "
                            "them in metres, measured as the question says."),
    "object_size_estimation": ("Work out the object's length, width and height, and give the longest of the three in "
                               "centimetres."),
    "room_size_estimation": ("Work out the room's length and width from its walls, the furniture along them and how far "
                             "the camera moves, and multiply them to get the floor area in square metres; for several "
                             "rooms, add up their areas."),
    "object_rel_distance": ("Estimate the distance from the reference object to each of the options, then pick the one "
                            "the question asks for."),
    "object_rel_direction": ("Picture yourself standing where the question puts you and facing the way it says. Work "
                             "on the floor plan, ignoring height: decide whether the target is in front of you or "
                             "behind you and whether it is on your left or your right, then pick the matching option."),
    "route_planning": ("Walk the route one step at a time from the given start and facing: at each step say which way "
                       "you face, and choose the turn that points you towards the next object."),
    "obj_appearance_order": "Find the first frame in which each object appears, and order the objects by it.",
}
METRIC_TASKS = {"object_abs_distance", "object_size_estimation", "room_size_estimation"}
COT_PIPELINE = ("The scene information was computed automatically from the video and can be wrong: an object can be "
                "missed, counted twice or misplaced. Where it disagrees with what you see, weigh the two.")
COT_SCALE = ("Its distances and sizes are in the 3D reconstruction's own units, not metres: to answer in metres or "
             "centimetres, first work out its scale from an object whose real size you know, such as a door (about 2 "
             "metres tall) or a chair seat (about 0.45 metres off the floor).")
COT_ANSWER = {
    "mca": 'End your reply with a last line of the form "Answer: X", where X is only the letter of the correct option.',
    "na": ('End your reply with a last line of the form "Answer: X", where X is only a number, in the unit the question '
           'asks for.'),
}


def cot_prompt(doc, has_sheet=False, sheet_kind=None):
    """The chain-of-thought instructions that end the prompt. has_sheet: the prompt carries scene information;
    sheet_kind: where it comes from ('pipeline': computed from the video, in relative units; 'pipeline_priors': the
    same, approximately in metres; 'gold': labels, in metres; None: not said)."""
    task = task_of(doc["question_type"])
    text = COT_STEPS.format(where=" and in the scene information above" if has_sheet else "") + " " + COT_HINTS[task]
    if has_sheet and sheet_kind in ("pipeline", "pipeline_priors"):
        text += " " + COT_PIPELINE + (" " + COT_SCALE if task in METRIC_TASKS and sheet_kind == "pipeline" else "")
    return text + "\n" + COT_ANSWER["na" if doc["question_type"] in NA_QUESTION_TYPES else "mca"]


# ---- scoring (verbatim logic from lmms-eval) ----
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                      "ninety": 90, "hundred": 100, "thousand": 1000})
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs(pred - target) / target <= 1 - conf_intervs
    return accuracy.mean()


def extract_number(text):
    """First number (digits or English number words) in the response, else None."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    digit_match = _NUM_RE.search(s)
    word_run_start, word_run_tokens = None, []
    for m in re.finditer(r"[a-z]+", s):
        tok = m.group(0)
        if word_run_start is None:
            if tok in _NUMBER_WORDS:
                word_run_start = m.start()
                word_run_tokens.append(tok)
        elif tok == "and":
            continue
        elif tok in _NUMBER_WORDS:
            word_run_tokens.append(tok)
        else:
            break
    if digit_match is not None and (word_run_start is None or digit_match.start() < word_run_start):
        return float(digit_match.group(0))
    if word_run_start is None:
        return None
    total, current = 0, 0
    for tok in word_run_tokens:
        val = _NUMBER_WORDS[tok]
        if val == 100:
            current = max(current, 1) * 100
        elif val == 1000:
            total += max(current, 1) * 1000
            current = 0
        else:
            current += val
    return float(total + current)


def score(doc, prediction):
    if doc["question_type"] in MCA_QUESTION_TYPES:
        return exact_match(fuzzy_matching(prediction), doc["ground_truth"])
    pred = extract_number(prediction)
    if pred is None:
        return 0.0
    return float(mean_relative_accuracy(pred, float(doc["ground_truth"])))


def parsed(doc, prediction, final=False):
    """Whether the extraction rule of score() could read an answer: an option letter for multiple choice (the first
    word, as fuzzy_matching reads it), a number for numeric questions. An unread answer scores 0. With final=True,
    the rule of score_final() instead: the letter or number the reply ends on."""
    if doc["question_type"] in MCA_QUESTION_TYPES:
        letters = {o.split(".", 1)[0].strip().lower() for o in doc["options"]}
        answer = final_letter(prediction) if final else fuzzy_matching(prediction)
        return answer is not None and answer.lower() in letters
    return (final_number(prediction) if final else extract_number(prediction)) is not None


# ---- final-answer scoring, for replies that reason before answering (not part of lmms-eval) ----
_ANSWER_PHRASE = re.compile(r"answer(?: is|:)?\s*(?:\*\*)?\s*\(?([A-D])\b(?![a-z])", re.I)
_LONE_LETTER = re.compile(r"(?<![A-Za-z])\(?([A-D])\)?(?=[.):*]|\s*$)")


def final_letter(text):
    """The option letter a free-form reply ends on: the last "answer is X" / "Answer: X", else the last letter that
    stands alone ("B", "(B)", "B.", "**B**") in the reply's last non-empty line. None if there is neither."""
    m = list(_ANSWER_PHRASE.finditer(text))
    if m:
        return m[-1].group(1).upper()
    lines = [l for l in text.strip().splitlines() if l.strip()]
    m = list(_LONE_LETTER.finditer(lines[-1])) if lines else []
    return m[-1].group(1) if m else None


def final_number(text):
    """The last number in a free-form reply (the first, for a bare-number reply)."""
    nums = _NUM_RE.findall(str(text).replace(",", ""))
    return float(nums[-1]) if nums else extract_number(text)


def score_final(doc, prediction):
    """Like score(), but reads the answer from the end of the reply instead of its first word."""
    if doc["question_type"] in MCA_QUESTION_TYPES:
        letter = final_letter(prediction)
        return 0.0 if letter is None else exact_match(letter, doc["ground_truth"])
    pred = final_number(prediction)
    if pred is None:
        return 0.0
    return float(mean_relative_accuracy(pred, float(doc["ground_truth"])))


def summarize(df):
    """df has columns question_type, score. Returns {task: score*100, 'overall': macro avg} like lmms-eval."""
    per_qtype = df.groupby("question_type")["score"].mean()
    out = {}
    for t in TASKS:
        if t == "object_rel_direction":
            vals = [per_qtype[q] for q in per_qtype.index if q.startswith("object_rel_direction")]
        else:
            vals = [per_qtype[t]] if t in per_qtype.index else []
        if vals:
            out[t] = 100 * float(np.mean(vals))
    out["overall"] = float(np.mean(list(out.values())))
    return out


def load_predictions(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(l) for l in f])
