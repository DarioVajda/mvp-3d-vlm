#!/usr/bin/env bash
# Pipeline and gold scene sheets for every question of every exported scene, on one GPU. After hpc/setup_env.sh:
#
#   slurm/submit.sh -J sheets -t 02:00:00 --gres gpu:1 -c 48 --mem 256G -- bash src/mvp/openvocab_adapter/hpc/run_sheets.sh
#
# Runs scene_sheet_report.py on STREAMS scenes at a time (SAM 3 needs about 4 GB of GPU memory per scene), then
# sheet_index.py, then zips the results to share. A scene whose two reports both finished with no failed sheet is
# skipped, so the same command picks up where a stopped run left off. The 75 scenes take about 15 minutes on one B300.
#
# Outputs (gitignored), in tasks/openvocab_adapter/out/: the sheets in <dataset>__<scene>/, each scene's two reports in
# reports/<dataset>__<scene>_{pipeline,gold}.txt, index.csv, and README.md (a copy of docs/scene_sheets_75.md). Then
# tasks/openvocab_adapter/scene_sheets_75.zip: all of it as scene_sheets_75/, without the per-scene depth maps (the
# shared VGGT depth, cut to the selected frames), about 70 MB.
#
# Optional settings:
#   DATA_ROOT  the exported scenes (default: tasks/vggt_data/out, of the main checkout when run from a worktree)
#   DENSE_ROOT the saved per-pixel VGGT depth (default: data/vggt, likewise)
#   STREAMS    scenes at a time (default 8)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")  # the main checkout, also from a worktree
DATA_ROOT=${DATA_ROOT:-$MAIN/tasks/vggt_data/out}
export DENSE_ROOT=${DENSE_ROOT:-$MAIN/data/vggt}
export NLTK_DATA=$PWD/.venv-ov4d/nltk_data TORCH_HOME=${TORCH_HOME:-/shared/workspace/povejmo/torch_cache}
STREAMS=${STREAMS:-8}
PY=.venv-ov4d/bin/python
REPORT=src/mvp/openvocab_adapter/scene_sheet_report.py
REPORTS=tasks/openvocab_adapter/out/reports
mkdir -p "$REPORTS"

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
        $PY $REPORT "$sc" --data-root "$DATA_ROOT" > "$REPORTS/${n}_pipeline.txt" 2>&1
        $PY $REPORT "$sc" --data-root "$DATA_ROOT" --source gold > "$REPORTS/${n}_gold.txt" 2>&1
        echo "$sc: $(tail -1 "$REPORTS/${n}_pipeline.txt") | gold: $(tail -1 "$REPORTS/${n}_gold.txt")"
    done ) &
done
wait
$PY src/mvp/openvocab_adapter/sheet_index.py --data-root "$DATA_ROOT"
cp docs/scene_sheets_75.md tasks/openvocab_adapter/out/README.md
cd tasks/openvocab_adapter && rm -f scene_sheets_75.zip && ln -sfn out scene_sheets_75 \
    && zip -qr scene_sheets_75.zip scene_sheets_75/ -x '*/depth.npz' -x '*/__pycache__/*'; rm -f scene_sheets_75
ls -lh scene_sheets_75.zip
