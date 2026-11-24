#!/usr/bin/env bash
# Pipeline and gold scene sheets for every question of every exported scene, on one GPU. After hpc/setup_env.sh:
#
#   slurm/submit.sh -J sheets -t 04:00:00 --gres gpu:1 -c 48 --mem 256G -- bash src/mvp/openvocab_adapter/hpc/run_sheets.sh
#
# Runs scene_sheet_report.py on STREAMS scenes at a time (SAM 3 needs about 4 GB of GPU memory per scene), then
# sheet_index.py, then zips the results to share. A scene whose two reports both finished with no failed sheet is
# skipped, so the same command picks up where a stopped run left off. With the defaults (all 128 frames) SAM 3 does
# about 8 times the work of the 16-frame sheets, whose 75 scenes took about 15 minutes on one B300: expect about 2 hours.
#
# Outputs (gitignored), in $OUT: the sheets in <dataset>__<scene>/<frames>/ and <dataset>__<scene>/gold/, each scene's
# two reports in reports/<dataset>__<scene>_{pipeline,gold}.txt, index.csv, and README.md (a copy of
# docs/scene_sheets_75.md). Then tasks/openvocab_adapter/$ZIP: all of it as ${ZIP%.zip}/, without the per-scene depth
# maps (the shared VGGT depth, cut to the selected frames).
#
# Optional settings:
#   DATA_ROOT  the exported scenes (default: tasks/vggt_data/out, of the main checkout when run from a worktree)
#   DENSE_ROOT the saved per-pixel VGGT depth (default: data/vggt, likewise)
#   STREAMS    scenes at a time (default 8)
#   FRAMES     question_scene_sheet.py's --frames (default all; subset:uniform16 is the VLM's 16 frames)
#   ASSOCIATION, DEPTH_RULE  question_scene_sheet.py's --association and --depth-rule (default: its defaults, overlap
#              and mask; centroid and global are the original tracking)
#   OUT        where everything goes (default tasks/openvocab_adapter/out_<frames>, e.g. out_all). The sheets of
#              docs/scene_sheets_75.md were made in tasks/openvocab_adapter/out with FRAMES=subset:uniform16 and the
#              original tracking (ASSOCIATION=centroid DEPTH_RULE=global).
#   ZIP        the zip's name (default scene_sheets_75_<frames>.zip)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")  # the main checkout, also from a worktree
DATA_ROOT=${DATA_ROOT:-$MAIN/tasks/vggt_data/out}
export DENSE_ROOT=${DENSE_ROOT:-$MAIN/data/vggt}
export NLTK_DATA=$PWD/.venv-ov4d/nltk_data TORCH_HOME=${TORCH_HOME:-/shared/workspace/povejmo/torch_cache}
STREAMS=${STREAMS:-8}
FRAMES=${FRAMES:-all}
LABEL=${FRAMES//:/-}; LABEL=${LABEL//,/_}  # the work folder's name, as question_scene_sheet.frame_selection makes it
OUT=$(realpath -m "${OUT:-tasks/openvocab_adapter/out_$LABEL}")
ZIP=${ZIP:-scene_sheets_75_$LABEL.zip}
PY=.venv-ov4d/bin/python
REPORT=src/mvp/openvocab_adapter/scene_sheet_report.py
REPORTS=$OUT/reports
mkdir -p "$REPORTS"
PIPELINE_OPTIONS=(--frames "$FRAMES")
[[ -n "${ASSOCIATION:-}" ]] && PIPELINE_OPTIONS+=(--association "$ASSOCIATION")
[[ -n "${DEPTH_RULE:-}" ]] && PIPELINE_OPTIONS+=(--depth-rule "$DEPTH_RULE")
echo "frames $FRAMES, tracking ${PIPELINE_OPTIONS[*]:2}, out $OUT"

finished() {  # the report ran to its summary line and no sheet failed
    tail -1 "$1" 2>/dev/null | grep -q " questions: .* 0 failed$"
}

# largest scenes first, dealt round-robin, so the streams end at about the same time
todo=$(tail -n +2 "$DATA_ROOT/scenes.csv" | sort -t, -k5,5nr | cut -d, -f1,2 | tr , / | while read -r sc; do
    n=${sc/\//__}
    finished "$REPORTS/${n}_pipeline.txt" && finished "$REPORTS/${n}_gold.txt" || echo "$sc"
done)
echo "$(echo "$todo" | grep -c .) scenes to do, $STREAMS at a time"

for s in $(seq 0 $((STREAMS - 1))); do
    ( echo "$todo" | awk -v s="$s" -v k="$STREAMS" 'NF && (NR - 1) % k == s' | while read -r sc; do
        n=${sc/\//__}
        $PY $REPORT "$sc" --data-root "$DATA_ROOT" --work-dir "$OUT/$n/$LABEL" "${PIPELINE_OPTIONS[@]}" \
            > "$REPORTS/${n}_pipeline.txt" 2>&1
        $PY $REPORT "$sc" --data-root "$DATA_ROOT" --work-dir "$OUT/$n/gold" --source gold > "$REPORTS/${n}_gold.txt" 2>&1
        echo "$sc: $(tail -1 "$REPORTS/${n}_pipeline.txt") | gold: $(tail -1 "$REPORTS/${n}_gold.txt")"
    done ) &
done
wait
$PY src/mvp/openvocab_adapter/sheet_index.py --data-root "$DATA_ROOT" --out "$OUT"
cp docs/scene_sheets_75.md "$OUT/README.md"
NAME=${ZIP%.zip}
cd tasks/openvocab_adapter && rm -f "$ZIP" && ln -sfn "$OUT" "$NAME" \
    && zip -qr "$ZIP" "$NAME/" -x '*/depth.npz' -x '*/__pycache__/*'; rm -f "$NAME"
ls -lh "$ZIP"
