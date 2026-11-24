#!/usr/bin/env bash
# The whole OpenVocab-4D comparison for one scene, on one GPU. After hpc/setup_env.sh:
#
#   slurm/submit.sh -J ov4d_compare -t 02:00:00 -- bash src/mvp/openvocab_adapter/hpc/run_all.sh
#
#   1. CPU tests of the adapter, loader and comparison
#   2. SAM 3 masks for the frames the selections need, saved once and reused by every run below
#   for each frame selection in RUNS:
#   3. adapted: the saved 128-frame VGGT outputs (cameras + dense depth from data/vggt/) in OpenVocab-4D's format
#   4. OpenVocab-4D tracking and fusion on the adapted geometry, and on its own VGGT (native), with the same masks
#   5. a scene sheet of each, then compare.py
#
# The small results (reports, scene sheets, comparison, environment, log; no frames) go to
# tasks/openvocab_adapter/results/<dataset>__<scene>/, to be committed. Everything else stays in tasks/openvocab_adapter/out/ (gitignored).
#
# Optional settings, e.g.  RUNS=eval32 slurm/submit.sh ... -- bash src/mvp/openvocab_adapter/hpc/run_all.sh
#   SCENE    scannet/scene0025_01
#   PROMPTS  objects to find (default: those named in scene0025_01's VSI-Bench questions)
#   RUNS     "half": every other frame (64). Others: smoke5 (5 nearby frames), eval32 (the uniform32 subset),
#            all (128 frames); several are run one after the other, e.g. RUNS="half all"
#   REID     0; 1 turns on DINOv2 re-identification, in both runs
#   PY       .venv-ov4d/bin/python
#   DENSE    data/vggt/<scene>, the folder with depth.npy and depth_conf.npy
#   DATA     the export holding scenes/<scene>/ (default: found under tasks/vggt_data/out, else exported)
#   WORK, RESULTS   tasks/openvocab_adapter/out/<name>, tasks/openvocab_adapter/results/<name>
set -euo pipefail
cd "$(dirname "$0")/../../../.."
SCENE=${SCENE:-scannet/scene0025_01}
PROMPTS=${PROMPTS:-chair,door,fan,keyboard,monitor,pillow,sofa,table,telephone,window}
RUNS=${RUNS:-half}
PY=${PY:-.venv-ov4d/bin/python}
REID_FLAG=--no-reid
[[ ${REID:-0} == 1 ]] && REID_FLAG=

A=src/mvp/openvocab_adapter  # the code
T=tasks/openvocab_adapter     # its outputs: out/ (gitignored) and results/ (committed)
OV=openvocab-4D-main
NAME=${SCENE/\//__}
WORK=${WORK:-$T/out/$NAME}
MASKS=$WORK/masks
RESULTS=${RESULTS:-$T/results/$NAME}
DENSE=${DENSE:-${DENSE_ROOT:-data/vggt}/$SCENE}

mkdir -p "$WORK" "$RESULTS"
exec > >(tee "$RESULTS/run.log") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $* ==="; }

step "inputs"
command -v "$PY" >/dev/null || { echo "no $PY: run src/mvp/openvocab_adapter/hpc/setup_env.sh first" >&2; exit 1; }
[[ -f $DENSE/depth.npy && -f $DENSE/depth_conf.npy ]] || { echo "no dense depth in $DENSE" >&2; exit 1; }
FOUND=""
for root in ${DATA:-} tasks/vggt_data/out/vggt_vsibench_75 tasks/vggt_data/out $T/out/export; do
    [[ -f $root/scenes/$SCENE/meta.json ]] && { FOUND=$root; break; }
done
DATA=$FOUND
if [[ -z $DATA ]]; then  # re-export the one scene from data/vggt/, with the project's environment
    DATA=$T/out/export
    .venv/bin/python tasks/vggt_data/export.py --scenes "$SCENE" --out "$DATA"
fi
FRAMES=$DATA/scenes/$SCENE/frames
echo "scene $SCENE | frames $FRAMES | dense depth $DENSE | runs: $RUNS | prompts: $PROMPTS | ${REID_FLAG:-with re-id}"
{
    echo "date: $(date -Is)"; echo "host: $(hostname)"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "settings: SCENE=$SCENE RUNS='$RUNS' REID=${REID:-0} PROMPTS=$PROMPTS"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    $PY --version; $PY -m pip freeze
} > "$RESULTS/environment.txt"

step "1/5 CPU tests"
$PY -m unittest discover -s $A/tests

step "2/5 SAM 3 masks"
# only the frames the selections in RUNS use; earlier masks are reused if they cover those frames and prompts
SAM3_FRAMES=$($PY - "$A" "$DATA/scenes/$SCENE/meta.json" $RUNS <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import common
meta = json.load(open(sys.argv[2]))
print(common.frames_for(sys.argv[3:], meta["num_frames"], meta))
EOF
)
COVERED=$($PY - "$A" "$MASKS/manifest.json" "$SAM3_FRAMES" "$FRAMES" "$PROMPTS" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import common
manifest, spec, frames, prompts = Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
if not manifest.is_file():
    print("no"); sys.exit()
images = common.list_images(frames)
need = {images[int(i)].stem for i in spec.split(":", 1)[1].split(",")}
saved = json.loads(manifest.read_text())
wanted = {p.strip() for p in prompts.split(",") if p.strip()}
print("yes" if need <= {f["stem"] for f in saved["frames"]} and wanted <= set(saved["prompts"]) else "no")
EOF
)
if [[ $COVERED == yes ]]; then
    echo "reusing the finished run in $MASKS (delete it to segment again)"
else
    $PY $A/sam3_infer.py --images "$FRAMES" --frames "$SAM3_FRAMES" --prompts "$PROMPTS" --out "$MASKS" --overlays --overwrite
fi
cp "$MASKS/manifest.json" "$RESULTS/masks_manifest.json"

for RUN in $RUNS; do
    R=$WORK/$RUN
    rm -rf "$R"

    step "3/5 $RUN: adapt the saved VGGT outputs"
    $PY $A/adapt_vggt.py --scene "$SCENE" --data-root "$DATA" --preset "$RUN" --dense-dir "$DENSE" --out "$R/adapted"

    step "4/5 $RUN: OpenVocab-4D on the adapted geometry, then on its own VGGT"
    $PY -u $OV/trackA/milestone4_video.py --images "$R/adapted/frames" --out "$R/adapted" --prompts "$PROMPTS" \
        --step 1 --stages track,fuse --masks-dir "$MASKS" $REID_FLAG
    $PY -u $OV/trackA/milestone4_video.py --images "$R/adapted/frames" --out "$R/native" --prompts "$PROMPTS" \
        --step 1 --stages vggt,track,fuse --masks-dir "$MASKS" $REID_FLAG

    step "5/5 $RUN: scene sheets and comparison"
    for SIDE in adapted native; do  # the original rules: compare.py aligns the two runs in the first camera's axes
        $PY $A/scene_sheet_creator.py --scene "$R/$SIDE" --prompts "$PROMPTS" --size --axes camera0 --depth-rule global
    done
    $PY $A/compare.py --run-dir "$R" --masks "$MASKS" || echo "compare.py found a FAIL (see above)"

    # the small files only: no frames, depth, masks or point clouds
    out=$RESULTS/$RUN
    rm -rf "$out"
    mkdir -p "$out/adapted" "$out/native"
    cp "$R/comparison.json" "$out/"
    cp "$R/adapted/adapter_report.json" "$out/adapted/"
    for SIDE in adapted native; do
        cp "$R/$SIDE/report.json" "$R/$SIDE/tracks/summary.json" "$out/$SIDE/" 2>/dev/null || true
        mv "$out/$SIDE/summary.json" "$out/$SIDE/tracks_summary.json" 2>/dev/null || true
        cp "$R/$SIDE/scene_sheet/objects.csv" "$R/$SIDE/scene_sheet/cameras.csv" "$out/$SIDE/"
    done
    cp "$R/native/align_report.json" "$out/native/"
done

step "summary"
$PY - "$RESULTS" $RUNS <<'EOF' | tee "$RESULTS/SUMMARY.txt"
import json, sys
from pathlib import Path
results = Path(sys.argv[1])
for run in sys.argv[2:]:
    report = json.loads((results / run / "comparison.json").read_text())
    print(f"{run}: {report['verdict']} (provisional pass marks)  {report['counts']}")
    for c in report["criteria"]:
        mark = {True: "ok  ", False: "FAIL", None: "n/a "}[c["ok"]]
        print(f"   {mark} {c['criterion']}: {c['value']} (needs {c['limit']})")
EOF
echo
echo "done. Commit and push $RESULTS (small text files only):"
echo "  git add $RESULTS && git commit -m 'OpenVocab-4D comparison results for $SCENE' && git push"
