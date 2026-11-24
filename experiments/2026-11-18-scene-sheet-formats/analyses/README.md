# Analyses of the pipeline runs

Each `.py` here writes the `.md` of the same name (`analyse_<x>.py` writes `<x>_analysis.md`) to stdout, except
`compare_variants.py`, which writes `variants_table.md` into its work folder. All of them read the predictions in
`../results/<run>/`, as each notebook saved them; most also read sheets from `data/` (not in git, see the README's
"Running the pipeline"). Run from the repository root with `PYTHONPATH=src`. `<M>` is a checkout whose `data/` has the
sheets named, and `<E>` is this experiment folder.

Which results are confirmatory is set by the pre-registrations alone (`../pipeline_tests.md`). Everything here is
descriptive or exploratory unless it is marked as a registered result recomputed from the predictions.

## Runs (`../results/`)

| Folder | Notebook | Questions | Conditions |
|---|---|---|---|
| `explore200_v3` | `explore200_v3.ipynb` | `explore200_ids.txt` (200) | A_video, pipeline_v3 |
| `dev200` | `dev200.ipynb` | `dev200_ids.txt` (200) | A_video, pipeline_v3, pipeline_v3_fixed |
| `dev200_dedup` | `dev200_dedup.ipynb` | `dev200_ids.txt` | A_video, pipeline_v3, pipeline_v3_dedup |
| `dev200_relations` | `dev200_relations.ipynb` | `dev200_ids.txt` | the above plus pipeline_v3_dedup_relations |
| `confirm_v3` | `confirm_v3.ipynb` | `confirm_ids.txt` (180), pre-registered | A_video, pipeline_v3 |
| `final` | `final_test.ipynb` | `final_ids.txt` (538), pre-registered | A_video, pipeline_v3_dedup_relations |
| `dev200_v4` | `dev200_v4.ipynb` | `dev200_distance_ids.txt` (50) | A_video, pipeline_v4 |
| `final2` | `final_test2.ipynb` | `batch2_A_ids.txt` (798), pre-registered | A_video, pipeline_v4 |

## Scripts

| Script | Arguments | Output | What it does |
|---|---|---|---|
| `analyse_v3.py` | `<E>/results/explore200_v3 <M>/data/rerun_sheets/pipeline_v3_explore200` | `v3_analysis.md` | pipeline_v3 against video per task, by named-object coverage, and selective rules (in-sample, with a split-half estimate) |
| `analyse_dev.py` | `<E>/results/dev200 <M>/data/rerun_sheets/pipeline_v3_dev50` | `dev_analysis.md` | the explore200 rules tested unchanged on dev200, and the "Not detected" line |
| `score_dedup.py` | `<M>/data/rerun_sheets <E>` (`PYTHONPATH=src:<E>`) | `dedup_scores.md` | dedup and size-aware-merge variants against gold on the 50 dev scenes |
| `analyse_dedup.py` | `<E>/results/dev200_dedup <M>/data/rerun_sheets` | `dedup_analysis.md` | pipeline_v3_dedup against video and pipeline_v3 |
| `analyse_relations.py` | `<E>/results/dev200_relations <work> <E>` | `relations_analysis.md` | pipeline_v3_dedup_relations; `<work>` has `data/scene_sheets_pipeline_v3_dedup2_priors` etc. as the notebook unpacked them |
| `analyse_confirm.py` | `<E>/results/confirm_v3 <M>/data/rerun_sheets/pipeline_v3_75` | `confirm_analysis.md` | `prereg_v3_relative_direction.md`'s result recomputed, then descriptive analyses |
| `analyse_final.py` | `<E>/results/final <E> <work>/data/scene_sheets_pipeline_v3_dedup2_priors` | `final_analysis.md` | `prereg_full_method.md`'s H1 and H2 recomputed, then exploratory analyses |
| `relation_fixes.py` | `<M> <E> <work>` | `relation_fixes.md` | offline scores of scale fixes for the precomputed relations (dev scenes; the final test's scenes only as a diagnostic) |
| `box_fixes.py` | `<M> <E> <work>` | `box_fixes.md` | trimmed boxes for the distance relations, with the median scale; `<work>/boxfix/` holds the variant sheets |
| `size_fixes.py` | `<M> <E> <work>` | `size_fixes.md` | whether trimmed boxes improve object and room sizes (dev scenes) |
| `size_diagnosis.py` | `<E>/results/explore200_v3 <E> <M>` | `size_diagnosis.md` | object-size errors split into scale, merge and box shape |
| `analyse_final2.py` | `<E>/results/final2 <E> <M>/data/rerun_sheets/pipeline_v4_batch2_A` | `final2_analysis.md` | `prereg_batch2_A.md`'s H1–H3 recomputed, then exploratory analyses |
| `compare_variants.py` | `<work> <M>/data/scene_sheets_75` | `variants_table.md` | merge and sizing variants scored with `pipeline_scorecard.py` (`PYTHONPATH=src:<E>`) |
| `paired_errors.py` | (imported) | | size and distance errors per (scene, class), for comparing sheet folders on the same pairs |

`relation_fixes.py`, `box_fixes.py` and `size_fixes.py` read the variant sheets they compare from `<work>`; those were
built with `pipeline_v3.py` and `merge_points.py` with other `size_trim` values, and are not in git.

## Tables of the detection reruns

`rerun14_table.md`, `rerun_threshold_table.md`, `rerun_fix1_table.md`, `rerun_fixed2_table.md`: the detection settings
of `rerun_detection.ipynb` scored with `pipeline_scorecard.py` on the 14 scenes of `dev_scenes_15.txt` that every run
has; `pipeline_v3`'s settings (`rerun_detection.FROZEN`) were chosen from the last one.
`pipeline_v3_validation.md`: those settings on the 35 other dev scenes (`dev_scenes_35.txt`), unchanged.
`cap_table.md`, `variants_table.md`, `pipeline_scorecard_oldcode.md`, `pipeline_scorecard_rebuild.md`: the merge caps,
point merge and sizing variants on the dev scenes. The full scorecards of the reruns are `../pipeline_scorecard_*.md`.
