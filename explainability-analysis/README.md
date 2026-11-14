# Explainability analysis

These are static HTML reports. Open each `index.html` in a browser. The images next to a page are loaded by relative path, so keep each folder intact.

| Folder | Report | Checkpoints |
|---|---|---|
| `jspace_v7/` | **J-Lens Readouts.** The first J-space analysis: text and visual Jacobian lenses read at salient and random cells, across layers. | v7 LoRA, additive and cross-attention, seed 0, decay branch at step 10,750 |
| `object_saliency_v10/single_question_3004/` | **Object Saliency Test.** Object-based saliency on question 3004, with OWLv2 + SAM 2 masks compared against SAM 3 masks, and object-to-object attention in the cross-attention arm. | v10 `xa3_qknu` s0 and v9 additive s0, step 5,750 |
| `object_saliency_v10/two_direction_questions/` | **Two Direction Questions, Up Close.** Questions 3584 (cross-attention right, additive wrong) and 3655 (the reverse), with SAM 3 masks, saliency, attention rows and J-lens readouts across layers. | v10 `xa3_qknu` s0 and v9 additive s0, step 5,750 |
| `final_results_v10_v11/` | **Final results.** VSI-Bench for the four final arms (2 seeds, decay branches at 5,750 / 8,250 / 10,750) and the attention-swap ablation. | v9 additive, v10 `xa3_qknu` and `xa3_nu`, v11 `xa3_nu_text` |

## Reproducing

The code is in `src/mvp/explain/`. The raw outputs (maps, masks, attention rows, readouts, ablation scores) are in `results/explain_v7/`, `results/jspace_v7/`, `results/objwords_v7/`, `results/objsal_test*/`, `results/detail_v10*/` and `results/ablation_v10/`.

1. **Fetch the J-lens reference code** (pinned commit; only its Qwen token glosses are used):

   ```bash
   scripts/fetch_jacobian_lens.sh
   ```

2. **Regenerate the question-id lists** (CPU only). These reproduce the committed files exactly:

   ```bash
   python -m mvp.explain.ids v7       # results/explain_v7/ids_200.txt
   python -m mvp.explain.ids swap     # results/ablation_v10/ids.txt
   python -m mvp.explain.ids screen   # results/detail_v10/screen_ids.json
   ```

3. **Re-run the GPU analyses** with the arguments they were run with. Each group submits its own Slurm jobs; run the script without arguments to list the groups.

   ```bash
   slurm/explain.sh <v7_saliency | v7_ablate | v7_analysis | jspace_v7 | objwords_v7 | objsal | screen | detail | swap>
   python -m mvp.explain.swap_summary --run 2B_v10_xa3_qknu_s0 --control 2B_v9_fmix_additive_s0 --step 5750
   ```

4. **Rebuild the pages** (CPU only; fine on the login node):

   ```bash
   python -m mvp.explain.pages.build all
   ```

   The object-saliency and final-results pages, with their frames and mask sheets, rebuild byte-identical from the saved results. The templates are in `src/mvp/explain/pages/templates/`.

   The J-lens page also needs `results/jspace_v7/xa3_both_s0/readouts.jsonl`, which is not in the repository. Regenerate it with `slurm/explain.sh jspace_v7`. Without it, the page is built for the LoRA and additive models only.

Model outputs vary slightly in bf16 across GPU types. For example, the answer margin for question 3004 differs in the second decimal between two runs of the same checkpoint. Re-running the analyses on another GPU can therefore move numbers slightly, while the pages built from the saved results do not change.
