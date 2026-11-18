#!/usr/bin/env bash
# The whole scene-sheet format experiment on the cluster, from the repository root, with no manual steps:
#
#   bash experiments/2026-11-18-scene-sheet-formats/run_all.sh
#
# 1. checks that the sheets (index.csv) and every scene's data/vggt/<dataset>/<scene>/meta.json are there;
# 2. prints the full prompt of one question per dataset (and 680, whose timestamps sit on a rounding boundary) with
#    --print-prompt, and checks that the video's <t seconds> labels are the sheet's first_seen timestamps;
# 3. runs a 20-question smoke test (2 per question type) of A and B, records the model revision, and stops if either
#    run fails, answers too few questions, or the sheet doesn't reach the prompt;
# 4. submits the 11 runs A–K with slurm/submit.sh and writes their job ids to jobs.txt next to the outputs.
# Steps 2 and 3 are Slurm jobs this script waits for (sbatch --wait). Then, once the jobs are done:
#   PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/make_table.py
#
# SHEETS (default tasks/openvocab_adapter/out) and VGGT (default data/vggt) override the two input folders.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/2026-11-18-scene-sheet-formats
SHEETS=${SHEETS:-tasks/openvocab_adapter/out}
VGGT=${VGGT:-data/vggt}
MODEL=Qwen/Qwen3.5-2B
OUT=results/zeroshot/vsibench/scene_sheet_formats
CHECKS=results/_scratch/scene_sheet_formats_checks  # smoke outputs and logs; not kept in git
COMMON=(--model "$MODEL" --nframes 16 --sampling uniform128 --vggt-root "$VGGT")
export LOGDIR=scene_sheet_formats
fail() { echo "STOP: $*" >&2; exit 1; }

# Steps 2 and 3 run inside Slurm jobs as `run_all.sh --job <step> <sheets> <vggt>`; plain arguments, because
# slurm/submit.sh passes the command through printf %q.
if [[ ${1:-} == --job ]]; then
    SHEETS=$3; VGGT=$4; COMMON=(--model "$MODEL" --nframes 16 --sampling uniform128 --vggt-root "$VGGT")
    case $2 in
        prompts)
            python -m mvp.eval.vsibench "${COMMON[@]}" --sheet-source pipeline --sheet-format csv --sheets-root "$SHEETS" \
                --ids "$EXP/prompt_check_ids.txt" --print-prompt 4 > "$CHECKS/prompts.txt"
            python "$EXP/cluster_checks.py" prompts "$CHECKS/prompts.txt" --sheets-root "$SHEETS" --vggt-root "$VGGT" ;;
        smoke)
            python -m mvp.eval.vsibench "${COMMON[@]}" --ids "$EXP/smoke_ids.txt" --out "$CHECKS/smoke_A.jsonl"
            python -m mvp.eval.vsibench "${COMMON[@]}" --ids "$EXP/smoke_ids.txt" --sheet-source pipeline \
                --sheet-format csv --sheets-root "$SHEETS" --out "$CHECKS/smoke_B.jsonl"
            python "$EXP/cluster_checks.py" smoke "$CHECKS/smoke_A.jsonl" "$CHECKS/smoke_B.jsonl" --n 20
            python "$EXP/cluster_checks.py" revision "$MODEL" "$OUT/model_revision.txt" ;;
        *) fail "unknown job step $2" ;;
    esac
    exit 0
fi

echo "== 1. inputs"
[[ -f $SHEETS/index.csv ]] || fail "no $SHEETS/index.csv (set SHEETS=<the scene sheets folder>)"
# index.csv has quoted questions with line breaks, so it is read with the csv module (stdlib python3 is enough)
python3 - "$SHEETS" "$VGGT" "$EXP/question_ids.txt" <<'PY' || fail "inputs incomplete (see above)"
import csv, json, sys
from pathlib import Path
sheets, vggt, ids = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = list(csv.DictReader(open(sheets / "index.csv", newline="")))
want = {int(line) for line in open(ids) if line.strip()}
ok = {int(r["id"]) for r in rows} == want
print(f"  index.csv: {len(rows)} questions; question_ids.txt: {len(want)}; same set: {ok}")
bad = []
for scene in sorted({r["scene"] for r in rows}):
    meta = vggt / scene / "meta.json"
    try:
        json.loads(meta.read_text())["video_fps"]
    except (OSError, KeyError, ValueError):
        bad.append(str(meta))
missing = [r[k] for r in rows for k in ("pipeline_sheet", "gold_sheet") if not (sheets / r[k] / "objects.csv").exists()]
print(f"  meta.json with video_fps: {len({r['scene'] for r in rows}) - len(bad)} of {len({r['scene'] for r in rows})} "
      f"scenes; sheet folders without objects.csv: {len(missing)}")
for line in bad[:10] + missing[:10]:
    print("  missing:", line)
sys.exit(0 if ok and not bad and not missing else 1)
PY
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "  warning: uncommitted changes; the runs record commit $(git rev-parse --short HEAD) but run the working tree"
fi

echo "== 2. prompts and timestamps"
mkdir -p "$CHECKS" "$OUT"
rm -f "$CHECKS"/*
slurm/submit.sh -J ssf_prompts -t 00:30:00 --wait -o "$CHECKS/prompts_%j.out" -- \
    bash "$EXP/run_all.sh" --job prompts "$SHEETS" "$VGGT" > /dev/null \
    || fail "prompt check failed; see $CHECKS/prompts_*.out and $CHECKS/prompts.txt"
grep -E "^question|^OK" "$CHECKS"/prompts_*.out

echo "== 3. smoke test (20 questions, A and B) and model revision"
slurm/submit.sh -J ssf_smoke -t 00:30:00 --wait -o "$CHECKS/smoke_%j.out" -- \
    bash "$EXP/run_all.sh" --job smoke "$SHEETS" "$VGGT" > /dev/null \
    || fail "smoke test failed; see $CHECKS/smoke_*.out"
grep -E "^(video|sheet|the sheet|model|revision|OK)" "$CHECKS"/smoke_*.out

echo "== 4. the 11 runs"
# condition, sheet source, sheet format (README, Conditions); outputs qwen3_5_2B_<condition>_[gold_]<format>.jsonl
runs=("A - -"
      "B pipeline csv" "C pipeline markdown_table" "D pipeline json" "E pipeline markdown_kv" "F pipeline text"
      "G gold csv" "H gold markdown_table" "I gold json" "J gold markdown_kv" "K gold text")
{
    echo "# submitted $(date -Iseconds) at commit $(git rev-parse HEAD)"
    for run in "${runs[@]}"; do
        read -r cond source fmt <<< "$run"
        if [[ $source == - ]]; then
            name=A_video; sheet=()
        else
            name=${cond}_$fmt
            [[ $source == pipeline ]] || name=${cond}_${source}_$fmt
            sheet=(--sheet-source "$source" --sheet-format "$fmt" --sheets-root "$SHEETS")
        fi
        job=$(slurm/submit.sh -J "ssf_$cond" -t 01:00:00 -- python -m mvp.eval.vsibench "${COMMON[@]}" \
            --ids "$EXP/question_ids.txt" ${sheet[@]+"${sheet[@]}"} --out "$OUT/qwen3_5_2B_$name.jsonl")
        echo "$cond $job $OUT/qwen3_5_2B_$name.jsonl"
    done
} | tee "$OUT/jobs.txt"
echo "logs in logs/$LOGDIR/; when all 11 are done: PYTHONPATH=src python $EXP/make_table.py"
