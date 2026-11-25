"""Runs of the follow-up CoT variants (mvp.benchmarks.cot_variants: F1 cot_f1, F2 cot_f2, F3 cot_f3) on the computing
tasks with the gold sheet, through the eval (python -m mvp.eval.vsibench), resumable and kept apart by configuration.

  subset     nested_subset: k questions per computing task from a pool (default: explore200's 75 computing questions,
             the development set), in a fixed order per task drawn from SEED, so the questions of k are among those of
             any larger k and a larger run reuses a smaller one's answers. heldout_pool: the computing questions of
             --questions whose scenes no development question uses, for the evaluation after the prompts are frozen
  config     run_config: everything that decides an answer except the question (model, eval settings, the variant's
             prompt fingerprint and version, the gold sheets' index hash, test.jsonl's hash); its hash names nothing,
             it is checked: guard() stops a run whose folder holds answers made under another config
  conditions the variants (cot_f1, cot_f2, cot_f3), and the pilot's direct conditions under the pilot's names
             (DIRECT: A_video, gold_csv, gold_relations), for running them with another model
  models     --model other than MODEL (Qwen3.5-2B) writes to <out>/<model's name, lower case>/ (model_dir), so another
             model's answers never land in the 2B's folders; --device-map auto (not the default) for a model loaded
             across the GPU by accelerate, which stops if any weight would be offloaded to the CPU; --batch-size 1 for
             a model run one question at a time (the default, SETTINGS' 8, is the pilot's)
  layout     <out>/<variant>/config.json, predictions.jsonl (+ the eval's _summary.json), prompts.jsonl (the exact
             text after the video, question by question), timing.jsonl (one line per call: questions before and after,
             seconds, GPU), ids_k<k>.txt (the questions of each call)

  PYTHONPATH=src python experiments/2026-11-25-concise-math-cot/followup.py --variants cot_f1 cot_f2 cot_f3 --k 1 \
      --sheets-root tasks/openvocab_adapter/out_full --out results/zeroshot/vsibench/concise_math_cot/dev
  PYTHONPATH=src python experiments/2026-11-25-concise-math-cot/followup.py --model Qwen/Qwen3.5-9B --device-map auto \
      --batch-size 1 --variants gold_csv cot_f1 gold_relations A_video --k 10 --sheets-root ... --out ...  # -> <out>/qwen3.5-9b/
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMPUTING = ("object_abs_distance", "object_rel_distance", "object_size_estimation")
DEV_IDS = REPO / "experiments/2026-11-18-scene-sheet-formats/explore200_ids.txt"
QUESTIONS = REPO / "data/raw/VSI-Bench/test.jsonl"
SEED = 0
MODEL = "Qwen/Qwen3.5-2B"
# the structured-CoT pilot's settings (explore_cot.ipynb), unchanged: thinking off (the eval's default for Qwen3.5),
# 16 frames uniform128, 2,048 new tokens, the final answer scored, up to 8 questions of a video per batch
SETTINGS = {"nframes": 16, "sampling": "uniform128", "max_new_tokens": 2048, "final_answer": True, "batch_size": 8,
            "token_budget": 120000, "sheet_source": "gold", "sheet_format": "csv", "layout": "sit"}
STAGES = {"smoke": 1, "small": 5, "extended": 10, "full": 25}  # questions per task; explore200 has 25 per task
VARIANTS = ("cot_f1", "cot_f2", "cot_f3")
# the pilot's direct conditions (explore_cot.ipynb's CONDITIONS), name -> sheet source: A, C and E
DIRECT = {"A_video": None, "gold_csv": "gold", "gold_relations": "gold_relations"}
DEVICE_MAP = "cuda"  # the eval's default (mvp.models.qwen_vl.QwenVL)


class ConfigMismatch(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_questions(path=QUESTIONS):
    return {int(q["id"]): q for q in map(json.loads, Path(path).read_text().splitlines())}


def order_key(seed, qid):
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def nested_subset(pool, questions, k, seed=SEED):
    """k questions per computing task from pool (ids), each task's in a fixed order drawn from seed, tasks in COMPUTING
    order: the first k of each task's order, so a smaller k's questions are always among a larger one's."""
    out = []
    for task in COMPUTING:
        mine = sorted((q for q in pool if questions[q]["question_type"] == task), key=lambda q: order_key(seed, q))
        if len(mine) < k:
            raise ValueError(f"the pool has {len(mine)} {task} questions, fewer than k = {k}")
        out += mine[:k]
    return out


def dev_pool(questions, ids_file=DEV_IDS):
    """explore200's computing questions: the development set, which the prompts were designed on."""
    ids = [int(x) for x in Path(ids_file).read_text().split()]
    return [q for q in ids if questions[q]["question_type"] in COMPUTING]


def heldout_pool(questions, dev_ids, available=None):
    """Computing questions from scenes no development question comes from (and, if given, among `available` ids,
    e.g. those with a gold sheet): for evaluating frozen prompts on unseen questions and scenes."""
    scene = lambda q: (questions[q]["dataset"], questions[q]["scene_name"])
    dev_scenes = {scene(q) for q in dev_ids}
    return sorted(q for q, d in questions.items() if d["question_type"] in COMPUTING and scene(q) not in dev_scenes
                  and (available is None or q in available))


def model_dir(out_root, model=MODEL):
    """The folder of a model's conditions: out_root for MODEL (as before), out_root/<model's name, lower case> otherwise."""
    return Path(out_root) if model == MODEL else Path(out_root) / model.split("/")[-1].lower()


def run_config(variant, sheets_root, model=MODEL, questions_path=QUESTIONS, prompt_version=None, device_map=DEVICE_MAP,
               batch_size=None):
    """Everything that decides an answer other than the question itself. A direct condition's prompts are the eval's
    (unchanged since the pilot); write_prompts stops if a logged prompt changes."""
    extra = {} if device_map == DEVICE_MAP else {"device_map": device_map}
    if batch_size is not None:
        extra["batch_size"] = batch_size  # replaces SETTINGS' value
    if variant in DIRECT:
        source = DIRECT[variant]
        return {"variant": variant, "model": model, **SETTINGS, "reasoning": "direct", "sheet_source": source,
                "sheet_format": "csv" if source else None, "sheets_index_sha256": sha256(Path(sheets_root) / "index.csv"),
                "questions_sha256": sha256(questions_path), **extra}
    from mvp.benchmarks import cot_variants
    from mvp.benchmarks.vsibench import COT_ANSWER, COT_HINTS
    if prompt_version is not None and prompt_version != cot_variants.PROMPT_VERSION:
        raise ConfigMismatch(f"prompt version {prompt_version!r} asked for, the code has {cot_variants.PROMPT_VERSION!r}")
    return {"variant": variant, "model": model, **SETTINGS, "prompt_version": cot_variants.PROMPT_VERSION,
            "prompt_fingerprint": cot_variants.fingerprint(variant, COT_HINTS, COT_ANSWER),
            "sheets_index_sha256": sha256(Path(sheets_root) / "index.csv"),
            "questions_sha256": sha256(questions_path), **extra}


def guard(run_dir, config):
    """Write config.json in a new run folder; in one that has answers, stop unless they were made under this config."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path, preds = run_dir / "config.json", run_dir / "predictions.jsonl"
    if path.is_file():
        old = json.loads(path.read_text())
        if old != config:
            changed = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            raise ConfigMismatch(f"{run_dir}: its answers were made under another configuration ({', '.join(changed)} "
                                 "differ). Use a new --out, or delete the folder to start again.")
    elif preds.is_file():
        raise ConfigMismatch(f"{run_dir}: answers without a config.json; their configuration is unknown")
    else:
        path.write_text(json.dumps(config, indent=1, sort_keys=True))


def answered(run_dir):
    path = Path(run_dir) / "predictions.jsonl"
    if not path.is_file():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def eval_command(config, ids_file, out, sheets_root, vggt_root=None):
    cmd = [sys.executable, "-m", "mvp.eval.vsibench", "--model", config["model"], "--nframes", str(config["nframes"]),
           "--sampling", config["sampling"], "--max-new-tokens", str(config["max_new_tokens"]), "--batch-size",
           str(config["batch_size"]), "--token-budget", str(config["token_budget"]), "--reasoning",
           config.get("reasoning", config["variant"])]
    if config["sheet_source"]:
        cmd += ["--sheet-source", config["sheet_source"], "--sheet-format", config["sheet_format"], "--sheets-root",
                str(sheets_root)]
    cmd += ["--ids", str(ids_file), "--out", str(out)]
    if config["final_answer"]:
        cmd.append("--final-answer")
    if vggt_root:
        cmd += ["--vggt-root", str(vggt_root)]
    if config.get("device_map", DEVICE_MAP) != DEVICE_MAP:
        cmd += ["--device-map", config["device_map"]]
    return cmd


def write_prompts(run_dir, ids, variant, sheets_root, questions):
    """prompts.jsonl: the exact text the eval puts after the video for each question not logged yet (the eval's own
    prompt_parts; the video's tokens and timestamps come before it and are the same in every condition). A question
    logged already must get the same text again, or the run stops: its answers were made with another prompt."""
    from mvp.eval import vsibench as ev
    path = Path(run_dir) / "prompts.jsonl"
    have = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.is_file() else {}
    source = DIRECT.get(variant, "gold")
    args = argparse.Namespace(sheets=source is not None, sheet_source=source, sheet_format="csv" if source else None,
                              cameras=False, sheets_root=str(sheets_root), vggt_root="unused", time_unit="seconds",
                              visibility="auto", step_order="move-turn", object_directions="floor", metric_scale=None,
                              layout="sit", reasoning="direct" if variant in DIRECT else variant, no_video=False)
    with open(path, "a") as f:
        for q in ids:
            pre, post = ev.prompt_parts(questions[q], args)
            if q not in have:
                f.write(json.dumps({"id": q, "before_video": pre, "after_video": post}) + "\n")
            elif (have[q]["before_video"], have[q]["after_video"]) != (pre, post):
                raise ConfigMismatch(f"{run_dir}: question {q}'s prompt is not the one its answer was made with")


def run(variant, k, sheets_root, out_root, pool=None, seed=SEED, questions_path=QUESTIONS, model=MODEL,
        vggt_root=None, prompt_version=None, dry_run=False, env=None, device_map=DEVICE_MAP, batch_size=None):
    """One condition on k questions per task: checks the folder's config, logs the prompts, runs the eval on the
    questions not answered yet, and times the call (the model's loading included). Returns the run folder."""
    questions = load_questions(questions_path)
    pool = dev_pool(questions) if pool is None else pool
    ids = nested_subset(pool, questions, k, seed)
    config = run_config(variant, sheets_root, model, questions_path, prompt_version, device_map, batch_size)
    run_dir = model_dir(out_root, model) / variant
    guard(run_dir, {**config, "seed": seed, "pool_sha256": hashlib.sha256(json.dumps(sorted(pool)).encode()).hexdigest()})
    ids_file = run_dir / f"ids_k{k}.txt"
    ids_file.write_text("\n".join(map(str, ids)) + "\n")
    write_prompts(run_dir, ids, variant, sheets_root, questions)
    before = answered(run_dir)
    todo = [q for q in ids if q not in before]
    cmd = eval_command(config, ids_file, run_dir / "predictions.jsonl", sheets_root, vggt_root)
    if dry_run or not todo:
        return run_dir, cmd, todo
    started = time.time()
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    seconds = time.time() - started
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    with open(run_dir / "timing.jsonl", "a") as f:
        f.write(json.dumps({"k": k, "questions": len(ids), "new": len(todo), "seconds": round(seconds, 1),
                            "seconds_per_new_question": round(seconds / len(todo), 1), "gpu": gpu}) + "\n")
    return run_dir, cmd, todo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=[*VARIANTS, *DIRECT])
    ap.add_argument("--k", type=int, default=STAGES["smoke"], help="questions per computing task")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sheets-root", required=True, help="the gold sheets (build_gold_sheets.py's out_full)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--pool", help="a file of question ids to draw from (default: explore200's computing questions)")
    ap.add_argument("--model", default=MODEL, help="another model's answers go to <out>/<its name, lower case>/")
    ap.add_argument("--device-map", default=DEVICE_MAP, choices=[DEVICE_MAP, "auto"])
    ap.add_argument("--batch-size", type=int, help=f"questions of a video per batch (default {SETTINGS['batch_size']})")
    ap.add_argument("--vggt-root")
    ap.add_argument("--dry-run", action="store_true", help="check the configs, log the prompts, print the commands")
    args = ap.parse_args()
    pool = [int(x) for x in Path(args.pool).read_text().split()] if args.pool else None
    for variant in args.variants:
        run_dir, cmd, todo = run(variant, args.k, args.sheets_root, args.out, pool, args.seed, args.questions,
                                 args.model, args.vggt_root, dry_run=args.dry_run, device_map=args.device_map,
                                 batch_size=args.batch_size)
        print(f"{variant}: {len(todo)} questions to answer in {run_dir}" + (f"\n  {' '.join(cmd)}" if args.dry_run else ""))


if __name__ == "__main__":
    main()
