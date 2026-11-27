---
id: definitive-7-model-size
title: Does a larger Qwen3.5 (2B, 4B, 9B) use the gold scene sheet better on VSI-Bench, answering directly and with CoT
owner: Nicolás Ramos Fernandez
created: 2026-11-27
status: draft                # draft | planned | running | analysing | done | abandoned
tags: [gold-sheet, chain-of-thought, model-size, definitive]
varies: [visual.model, prompt.reasoning]
depends_on: [definitive-5-structured-cot (runs R3, R4), TODOs C1, C4, C5]
---

# Model size

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder). Text in `<angle brackets>` is filled in once
TODOs C1, C4 and C5 are settled.

## Question

Does a larger Qwen3.5 use the gold scene sheet better, answering directly and with CoT? And does CoT's effect grow
with model size?

## Hypothesis and what would change our mind

- **Expected:**
  - CoT's gain grows with size: it is positive for the 9B, and larger for the 9B than for the 2B.
  - Direct answering with the sheet also improves with size.
- **Would prove it wrong:** S9C − S9D ≤ 0, so model size is not why CoT fails, at least up to 9B. Or the interaction
  is ≤ 0.
- **Next:**
  - size is a reason: use the larger model for the sheet with CoT, and repeat experiment 6 on it;
  - size is not the reason: the prompt and the sheet's form are the next suspects.

## Background

- **Ji et al., arXiv 2507.13362, Table III.** A structured two-step CoT raised CV-Bench for Llama-4-Scout (53.8 →
  75.7) and Qwen2.5-VL-7B (60.5 → 68.1), but barely for the 3B PaLI-Gemma2 (52.2 → 53.6).
- **The 2B, all 5,130 questions, video + gold csv, direct:** 48.9, 98.7 on counting
  (`results/zeroshot/vsibench/gold_sheet_relations_full/`, at `0f7c5f2`). Background only.
- **`experiments/2026-11-25-concise-math-cot/README.md`, "Model size"** planned the 9B against the 2B on 30
  development questions. It was never run; this experiment replaces it.

## Design

### Conditions

All six have the video and the gold sheet (csv).

| ID | `visual.model` | `prompt.reasoning` | Run |
|---|---|---|---|
| S2D | Qwen/Qwen3.5-2B | direct | R21 |
| S2C | Qwen/Qwen3.5-2B | CoT | R22 |
| S4D | Qwen/Qwen3.5-4B | direct | R3 |
| S4C | Qwen/Qwen3.5-4B | CoT | R4 |
| S9D | Qwen/Qwen3.5-9B | direct | R23 |
| S9C | Qwen/Qwen3.5-9B | CoT | R24 |

### Comparisons

All on the overall and on the computing tasks, each also without counting.

- **Primary, one Holm family:** CoT − direct for each model: S2C − S2D, S4C − S4D and S9C − S9D.
- **Secondary, one Holm family:** S4D − S2D, S9D − S4D, S4C − S2C and S9C − S4C.
- **Third, one Holm family:** the interaction, as a per-question contrast: (S9C − S9D) − (S2C − S2D), and (S4C − S4D) −
  (S2C − S2D).

### Decisions

| Decision | Choice | Why |
|---|---|---|
| The same for every model | every flag but `--model`, plus `--device-map auto` for the 9B | the model is the only difference |
| The 9B's loading | bf16, `--device-map auto`; `QwenVL` stops if any weight would go to the CPU or disk | no quantization, so the 9B is the same model family at full precision |
| CoT prompt | `<TODO C4>` | depends on C1 |
| Batches | `<TODO C5>` | the 9B may not fit 8 questions at a time |

## Configuration

```yaml
# README.md section 2, except:
visual:
  model: varies                # Qwen/Qwen3.5-2B | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B; each revision in model_revision_<model>.txt
scene_sheet:
  source: gold
prompt:
  reasoning: varies            # direct | CoT (TODO C4)
```

## Metrics and analysis

As `README.md` 2.7, with its CoT measures, per model. In addition, measured seconds per question and peak GPU memory
per model and condition.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| Larger models write longer replies, and the 2,048-token limit cuts more of them | the share cut off rises with size | report it per model; read the score with cut-off replies counted wrong too |
| Batch sizes differ between models (if TODO C5 says so) | — | stated as a deviation; greedy decoding in a batch can differ in the last digits |
| The gold sheet contains the counting answer | counting near 100 for every model | read everything without counting too |

## Runs and cost

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
BASE="--nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
GOLD="--sheets-root tasks/openvocab_adapter/out_full --sheet-format csv --sheet-source gold"
COT="--reasoning <TODO C4>"
M2="--model Qwen/Qwen3.5-2B"
M9="--model Qwen/Qwen3.5-9B --device-map auto <TODO C5: --batch-size>"
```

**Checks** (`README.md` 2.8):

1. **Inputs and gold sheets.** As experiment 2, check 1.
2. **Smoke test** per model, direct and CoT, then each model's revision. Report each model's peak GPU memory and
   seconds per question.

   ```bash
   for m in 2 9; do
     M=M$m
     python -m mvp.eval.vsibench ${!M} $BASE $GOLD      --ids $C/smoke_ids.txt --out $S/qwen3_5_${m}B_direct.jsonl
     python -m mvp.eval.vsibench ${!M} $BASE $GOLD $COT --ids $C/smoke_ids.txt --out $S/qwen3_5_${m}B_cot.jsonl
   done
   python $C/cluster_checks.py revision Qwen/Qwen3.5-2B $OUT/model_revision_qwen3_5_2B.txt
   python $C/cluster_checks.py revision Qwen/Qwen3.5-9B $OUT/model_revision_qwen3_5_9B.txt
   ```

**Runs:** R3 and R4 come from experiment 5. The other four each report 5,130 questions:

```bash
python -m mvp.eval.vsibench $M2 $BASE $GOLD      --out $OUT/qwen3_5_2B_R21.jsonl
python -m mvp.eval.vsibench $M2 $BASE $GOLD $COT --out $OUT/qwen3_5_2B_R22.jsonl
python -m mvp.eval.vsibench $M9 $BASE $GOLD      --out $OUT/qwen3_5_9B_R23.jsonl
python -m mvp.eval.vsibench $M9 $BASE $GOLD $COT --out $OUT/qwen3_5_9B_R24.jsonl
```

**Analysis:** `<make_table_7.py>` (`README.md` section 5, item 6) writes the Results table.

## TODO

- **C1:** the structured-CoT prompt (experiment 5).
- **C4:** the CoT prompt here. Either C1's prompt for every question type, or a concise variant (experiment 6) on the
  three computing tasks with C1's prompt elsewhere.
- **C5:** the 9B's batch size. Either the 2B's and 4B's 8, if memory allows, or one smaller size, for the 9B alone or
  for all three models, stated as a deviation.

## Results

<!-- make_table:start -->
Not run yet.
<!-- make_table:end -->

### Examples

### What the results show

## Conclusion

- **Answer to the question:**
- **Hypothesis:**
- **Caveats:**
- **Next:**

## Log
