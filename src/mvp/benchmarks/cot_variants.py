"""Follow-up structured chain-of-thought variants for the computing tasks with the gold sheet
(experiments/2026-11-25-concise-math-cot). Each takes the place of the "answer directly" line, as reasoning="cot"
(condition D of 2026-11-19-structured-cot) does, and each adds one intervention to the one before:

  cot_f1  concise: brief instructions without the compulsory check against the video, the video only where needed,
          the task's hint (vsibench.COT_HINTS, unchanged) and a fixed output structure
  cot_f2  F1 + the geometric operation and formula for the task (MATH)
  cot_f3  F2 + one synthetic worked example for the task (EXAMPLES)

The output structure (FORMAT) and the answer line (vsibench.COT_ANSWER, which score_final reads) are the same in all
three. The pieces are module constants so that a variant can be changed or a new one added in one place; fingerprint()
hashes what a variant's prompts are made of, so that predictions made with other text are never mixed with new ones.
Only the computing tasks with the gold sheet are defined: anything else raises ValueError.
"""
import hashlib
import json

PROMPT_VERSION = "2026-11-25.1"  # bump with any change to the text below
VARIANTS = ("cot_f1", "cot_f2", "cot_f3")
COMPUTING_TYPES = ("object_abs_distance", "object_rel_distance", "object_size_estimation")

# F1: what condition D's COT_STEPS asked for, without the compulsory check against the video and with a fixed structure
CONCISE = ("Reason briefly before you answer. Use the scene information above as your source; look at the video only if "
           "you need it to tell which object the question means. Do not restate the question, describe the video, "
           "repeat a calculation or explain more than needed.")
FORMAT = ("Write only these lines:\n"
          "Objects: the ids of the scene information's rows the question is about\n"
          "Attributes: only the values from those rows that you need\n"
          "Calculation: the steps that give the answer")

# F2: the operation each computing task needs, as the gold sheet writes boxes (x, y, z the centre; size_x, size_y,
# size_z the full extent along x, y, z; metres) and as VSI-Bench measures (closest points; sizes in centimetres)
BOX = ("Each row is a box: x, y, z is its centre and size_x, size_y, size_z its full length along x, y and z, all in "
       "metres.")
GAP = ("The distance between two boxes A and B is measured between their closest points, not their centres: for each "
       "axis k of x, y and z, gap_k = max(0, |A_k - B_k| - (A_size_k + B_size_k) / 2), and the distance is "
       "sqrt(gap_x^2 + gap_y^2 + gap_z^2), which is 0 when the boxes overlap along all three axes.")
MATH = {
    "object_abs_distance": (BOX + " " + GAP + " If an object has several rows, use the pair of rows with the smallest "
                            "distance. The sheet is in metres, as the answer is: do not convert."),
    "object_rel_distance": (BOX + " " + GAP + " An option's distance to the reference object is the smallest distance "
                            "between any of its rows and any row of the reference object. Compute it for every option "
                            "and choose the option with the smallest distance."),
    "object_size_estimation": ("Each row's size_x, size_y and size_z are the object's full extent along x, y and z, in "
                               "metres; x, y and z are its centre's position, not sizes. The longest dimension is the "
                               "largest of size_x, size_y and size_z. Convert it to centimetres by multiplying by "
                               "100 (0.85 m = 85 cm). If the object has several rows, use the largest of them."),
}

# F3: one synthetic worked example per task, in the gold sheet's csv and the FORMAT structure (no benchmark scene,
# question or answer). Their numbers are checked by tests/test_cot_variants.py.
EXAMPLE_INTRO = "Worked example, for a different scene (do not use its numbers):"
EXAMPLE_OUTRO = "End of the example. Now answer the question above."
EXAMPLES = {
    "object_abs_distance": """id,label,x,y,z,size_x,size_y,size_z
sofa_1,sofa,1.000,2.000,0.400,2.000,0.900,0.800
lamp_1,lamp,3.200,3.000,0.800,0.400,0.400,1.600
Question: Measuring from the closest point of each object, what is the distance between the sofa and the lamp (in meters)?
Objects: sofa_1, lamp_1
Attributes: sofa_1 centre (1.000, 2.000, 0.400), size (2.000, 0.900, 0.800); lamp_1 centre (3.200, 3.000, 0.800), size (0.400, 0.400, 1.600)
Calculation: gap_x = max(0, 2.200 - 1.200) = 1.000; gap_y = max(0, 1.000 - 0.650) = 0.350; gap_z = max(0, 0.400 - 1.200) = 0; distance = sqrt(1.000^2 + 0.350^2 + 0^2) = 1.059
Answer: 1.059""",
    "object_rel_distance": """id,label,x,y,z,size_x,size_y,size_z
desk_1,desk,0.000,0.000,0.400,1.600,0.800,0.800
bed_1,bed,2.300,0.200,0.300,2.600,2.000,0.600
shelf_1,shelf,0.000,1.600,1.000,1.200,0.400,2.000
chair_1,chair,-1.800,0.000,0.450,0.500,0.500,0.900
Question: Measuring from the closest point of each object, which of these objects (bed, shelf, chair) is the closest to the desk?
Options:
A. bed
B. shelf
C. chair
Objects: desk_1, bed_1, shelf_1, chair_1
Attributes: desk_1 centre (0.000, 0.000, 0.400), size (1.600, 0.800, 0.800); bed_1 centre (2.300, 0.200, 0.300), size (2.600, 2.000, 0.600); shelf_1 centre (0.000, 1.600, 1.000), size (1.200, 0.400, 2.000); chair_1 centre (-1.800, 0.000, 0.450), size (0.500, 0.500, 0.900)
Calculation: bed: gaps (0.200, 0, 0), distance 0.200; shelf: gaps (0, 1.000, 0), distance 1.000; chair: gaps (0.750, 0, 0), distance 0.750; smallest: bed
Answer: A""",
    "object_size_estimation": """id,label,x,y,z,size_x,size_y,size_z
cabinet_1,cabinet,0.800,-1.200,0.915,0.452,1.830,0.600
Question: What is the length of the longest dimension (length, width, or height) of the cabinet, measured in centimeters?
Objects: cabinet_1
Attributes: cabinet_1 size (0.452, 1.830, 0.600)
Calculation: longest = max(0.452, 1.830, 0.600) = 1.830 m; 1.830 x 100 = 183 cm
Answer: 183""",
}


def check(question_type, variant, has_sheet, sheet_kind):
    """ValueError unless the variant is defined for this question: a computing task, with the gold sheet."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    if question_type not in COMPUTING_TYPES:
        raise ValueError(f"{variant} is defined for the computing tasks {COMPUTING_TYPES} only, not {question_type}")
    if not has_sheet or sheet_kind != "gold":
        raise ValueError(f"{variant} is defined with the gold sheet only (sheet_kind {sheet_kind!r})")


def instructions(question_type, variant, hint, answer_line, has_sheet=True, sheet_kind="gold"):
    """The text that ends the prompt for a computing-task question under this variant. hint: the task's
    vsibench.COT_HINTS entry; answer_line: vsibench.COT_ANSWER for the question's answer type."""
    check(question_type, variant, has_sheet, sheet_kind)
    parts = [CONCISE + " " + hint]
    if variant in ("cot_f2", "cot_f3"):
        parts.append(MATH[question_type])
    if variant == "cot_f3":
        parts.append(EXAMPLE_INTRO + "\n" + EXAMPLES[question_type] + "\n" + EXAMPLE_OUTRO)
    parts += [FORMAT, answer_line]
    return "\n".join(parts)


def fingerprint(variant, hints, answer_lines):
    """SHA-256 of everything a variant's prompts are made of (its text pieces for every computing task, the hints and
    answer lines it uses, PROMPT_VERSION): equal fingerprints, equal prompts for the same question and sheet."""
    pieces = {"version": PROMPT_VERSION, "variant": variant, "concise": CONCISE, "format": FORMAT,
              "hints": {t: hints[t] for t in COMPUTING_TYPES}, "answer": answer_lines}
    if variant in ("cot_f2", "cot_f3"):
        pieces["math"] = MATH
    if variant == "cot_f3":
        pieces["examples"] = [EXAMPLE_INTRO, EXAMPLES, EXAMPLE_OUTRO]
    return hashlib.sha256(json.dumps(pieces, sort_keys=True).encode()).hexdigest()
