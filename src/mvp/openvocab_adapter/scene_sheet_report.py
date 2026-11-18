"""Scene sheets for every question of one scene, printed for checking by eye.

For each VSI-Bench question about the scene, in the order of the scene's questions, it prints:

  question   the question's type, id and text
  prompts    the SAM 3 prompts prompt_object_extractor.py picks: its full mode (the semantic-syntactic one), with the
             syntactic fallback that question_scene_sheet.py uses when the full mode keeps nothing
  sheet      OK when the scene sheet was made and passes the checks below, FAILED and why otherwise, and the prompts
             that got no rows (SAM 3 found nothing, or the gold labels have no such class)
  files      where the sheet's files are

The sheet comes from one of two sources, each with its default settings:

  pipeline (default)  question_scene_sheet.create_scene_sheet(scene, question): SAM 3, then tracking on the saved VGGT
                      outputs, 128 frames, size columns, one 0/1 column per frame, cameras.csv. Needs what
                      question_scene_sheet.py needs: the OpenVocab-4D environment, a GPU, and the scene's dense depth.
  --source gold       the same prompts, then scene_sheet_creator.build_gold_scene_sheet: VSI-Bench's ground-truth
                      boxes, in metres, with the size columns (question_scene_sheet.py's default). Needs spaCy, NLTK
                      and numpy only, no GPU, and works for any of the 288 VSI-Bench scenes.

Checks of a pipeline sheet: objects.csv and cameras.csv exist; the columns are object_id ... size, num_frames, then one
frame column per row of cameras.csv; every row's concept is one of the prompts, its numbers are finite, and num_frames
is at least 1 and equals its frame flags; question.json lists the same prompts.
Checks of a gold sheet: the columns are object_id ... size, with no per-frame columns and no cameras.csv; every prompt
that matches a gold class has exactly that class's objects (a class two prompts name, under the first only); the
numbers are finite and size is the product of the extents; sheet.json lists the same prompts.

The questions come from the exported scene (scenes/<dataset>/<scene>/questions.jsonl), or from --questions: a .jsonl in
VSI-Bench's format (data/raw/VSI-Bench/test.jsonl, which is tried last for a gold report of a scene not exported).

  python src/mvp/openvocab_adapter/scene_sheet_report.py scannet/scene0025_01 --source gold
  slurm/submit.sh -J sheet_report -t 02:00:00 -- .venv-ov4d/bin/python src/mvp/openvocab_adapter/scene_sheet_report.py \
      scannet/scene0025_01

The report goes to stdout; the steps' own logs go to stderr (add 2> log.txt to keep them apart). Work is kept and
reused as in question_scene_sheet.py, so a second report of the same scene only rebuilds the sheets. Exit status 1 if
any sheet failed.
"""
import argparse
import contextlib
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import question_scene_sheet  # noqa: E402
import scene_sheet_creator as sheet  # noqa: E402

VSI_QUESTIONS = common.REPO / "data" / "raw" / "VSI-Bench" / "test.jsonl"
GLOBAL_COLUMNS = ["object_id", "concept", "track_id", "x", "y", "z", "size_x", "size_y", "size_z", "size"]
CAMERA_COLUMNS = ["frame", "image", "x", "y", "z", "qw", "qx", "qy", "qz", "forward_x", "forward_y", "forward_z"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(scene, source="pipeline", data_root=None, questions=None, gold_dir=None):
    """("<dataset>/<scene>", the scene's questions) from --questions, the exported scene, or VSI-Bench's test.jsonl."""
    try:
        data_root, scene_id = common.find_scene(scene, data_root)
    except (FileNotFoundError, ValueError) as error:
        if source == "pipeline":
            raise ValueError(str(error)) from error
        data_root, scene_id = None, sheet.load_gold_scene(scene, gold_dir)[0]  # a gold report needs no export
    if questions is None and data_root is not None:
        path = Path(data_root) / "scenes" / scene_id / "questions.jsonl"
        if path.is_file():
            return scene_id, read_jsonl(path)
    path = Path(questions) if questions else VSI_QUESTIONS
    if not path.is_file():
        raise ValueError(f"no questions for {scene_id}: no questions.jsonl in the export and no {path}")
    rows = [r for r in read_jsonl(path) if f"{r['dataset']}/{r['scene_name']}" == scene_id]
    return scene_id, rows


def finite(row, columns):
    try:
        return all(math.isfinite(float(row[c])) for c in columns)
    except (KeyError, TypeError, ValueError):
        return False


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        return header, [dict(zip(header, values)) for values in reader]


def count_rows(rows, prompts):
    return {p: sum(r.get("concept") == p for r in rows) for p in prompts}


def check_pipeline_sheet(objects_csv, cameras_csv, prompts):
    """(problems, rows per prompt) of a pipeline sheet made with the default settings."""
    objects_csv, cameras_csv = Path(objects_csv), Path(cameras_csv) if cameras_csv else None
    if not objects_csv.is_file():
        return [f"no {objects_csv}"], {}
    if cameras_csv is None or not cameras_csv.is_file():
        return [f"no cameras.csv next to {objects_csv}"], {}
    header, rows = read_csv(objects_csv)
    camera_header, cameras = read_csv(cameras_csv)
    problems = []
    frames = header[len(GLOBAL_COLUMNS) + 1:]
    if header[:len(GLOBAL_COLUMNS) + 1] != GLOBAL_COLUMNS + ["num_frames"]:
        problems.append(f"objects.csv starts with {header[:len(GLOBAL_COLUMNS) + 1]}, not "
                        f"{GLOBAL_COLUMNS + ['num_frames']}")
    if camera_header != CAMERA_COLUMNS:
        problems.append(f"cameras.csv has the columns {camera_header}")
    if not cameras:
        problems.append("cameras.csv has no frames")
    if len(frames) != len(cameras) or not all(f.startswith("frame_") for f in frames):
        problems.append(f"{len(frames)} frame columns in objects.csv, {len(cameras)} frames in cameras.csv")
    for i, row in enumerate(rows):
        where = f"row {i} ({row.get('concept')})"
        if row.get("object_id") != str(i):
            problems.append(f"{where}: object_id {row.get('object_id')}")
        if row.get("concept") not in prompts:
            problems.append(f"{where}: not one of the prompts")
        if not finite(row, GLOBAL_COLUMNS[3:]):
            problems.append(f"{where}: a position or size that is not a finite number")
        flags = [row.get(f) for f in frames]
        if any(flag not in ("0", "1") for flag in flags):
            problems.append(f"{where}: a frame flag that is not 0 or 1")
        elif row.get("num_frames") != str(flags.count("1")) or not flags.count("1"):
            problems.append(f"{where}: num_frames {row.get('num_frames')}, {flags.count('1')} frame flags set")
    record = objects_csv.parent / "question.json"
    if not record.is_file():
        problems.append("no question.json")
    elif json.loads(record.read_text()).get("prompts") != list(prompts):
        problems.append(f"question.json lists other prompts: {json.loads(record.read_text()).get('prompts')}")
    return problems, count_rows(rows, prompts)


def check_gold_sheet(objects_csv, prompts, info):
    """(problems, rows per prompt) of a gold sheet with the size columns, against the scene's gold labels."""
    objects_csv = Path(objects_csv)
    if not objects_csv.is_file():
        return [f"no {objects_csv}"], {}
    header, rows = read_csv(objects_csv)
    problems = []
    if header != GLOBAL_COLUMNS:
        problems.append(f"objects.csv has the columns {header}, not {GLOBAL_COLUMNS}")
    if (objects_csv.parent / "cameras.csv").exists():
        problems.append("a cameras.csv next to a gold sheet")
    counts = count_rows(rows, prompts)
    listed = set()
    for prompt in prompts:
        cls = sheet.match_gold_class(info, prompt)
        expected = 0 if cls is None or cls in listed else len(info["object_bbox"][cls])
        listed.add(cls)
        if counts[prompt] != expected:
            problems.append(f"{counts[prompt]} rows for {prompt!r}, the gold labels have {expected}")
    for i, row in enumerate(rows):
        where = f"row {i} ({row.get('concept')})"
        if row.get("concept") not in prompts:
            problems.append(f"{where}: not one of the prompts")
        if not finite(row, GLOBAL_COLUMNS[3:]):
            problems.append(f"{where}: a position or size that is not a finite number")
        elif not math.isclose(float(row["size"]), math.prod(float(row[c]) for c in ("size_x", "size_y", "size_z")),
                              rel_tol=1e-3, abs_tol=1e-9):
            problems.append(f"{where}: size is not size_x * size_y * size_z")
    record = objects_csv.parent / "sheet.json"
    if not record.is_file():
        problems.append("no sheet.json")
    elif json.loads(record.read_text()).get("prompts") != list(prompts):
        problems.append(f"sheet.json lists other prompts: {json.loads(record.read_text()).get('prompts')}")
    return problems, counts


def pipeline_sheet(scene, question, prompts, **options):
    """(problems, rows per prompt, files) of the pipeline sheet of one question."""
    objects_csv, cameras_csv = question_scene_sheet.create_scene_sheet(scene, question, **options)
    problems, counts = check_pipeline_sheet(objects_csv, cameras_csv, prompts)
    return problems, counts, [objects_csv, cameras_csv]


def gold_sheet(scene, question, prompts, mode, work_dir=None, gold_dir=None, info=None):
    """(problems, rows per prompt, files) of the gold sheet of one question, with the size columns, in metres."""
    work = Path(work_dir) if work_dir else common.OUT_ROOT / scene.replace("/", "__") / "gold"
    out = (work / "sheets" / question_scene_sheet.folder_name(prompts)).resolve()
    with contextlib.redirect_stdout(sys.stderr):
        objects_csv, _ = sheet.build_gold_scene_sheet(scene, prompts, include_size=True, out_dir=out,
                                                      gold_dir=gold_dir)
    (out / "question.json").write_text(json.dumps(
        {"scene": scene, "question": question, "prompts": prompts, "prompt_mode": mode, "source": "gold"}, indent=1))
    if info is None:
        info = sheet.load_gold_scene(scene, gold_dir)[1]
    problems, counts = check_gold_sheet(objects_csv, prompts, info)
    return problems, counts, [objects_csv, out / "sheet.json"]


def report(scene, source="pipeline", data_root=None, questions=None, work_dir=None, gold_dir=None, types=None,
           limit=None, out=None, **sheet_options):
    """Print the report of a scene's questions to out (default stdout). Returns one result dict per question.

    sheet_options go to question_scene_sheet.create_scene_sheet (pipeline only); the command line passes none but
    --dense-dir, so that the sheets have the default settings.
    """
    out = out or sys.stdout
    scene, rows = load_questions(scene, source, data_root, questions, gold_dir)
    if types:
        rows = [r for r in rows if r["question_type"] in types]
    rows = rows[:limit] if limit else rows
    info = sheet.load_gold_scene(scene, gold_dir)[1] if source == "gold" else None
    what = "gold sheets (VSI-Bench's labels, metres, sizes)" if source == "gold" else "pipeline sheets (default settings)"
    print(f"Scene {scene}: {len(rows)} questions, {what}", file=out, flush=True)

    results = []
    for n, row in enumerate(rows, start=1):
        question = row["question"]
        print(f"\n[{n}/{len(rows)}] {row.get('question_type', '?')}, id {row.get('id', '?')}", file=out)
        print(f"  question: {question}", file=out, flush=True)
        result = {"id": row.get("id"), "question": question, "prompts": None, "problems": [], "files": []}
        try:
            prompts, mode = question_scene_sheet.extract_prompts(question)
            result.update(prompts=prompts, prompt_mode=mode)
            print(f"  prompts:  {', '.join(prompts) if prompts else '(none)'}  ({mode})", file=out, flush=True)
            if source == "gold":
                problems, counts, files = gold_sheet(scene, question, prompts, mode, work_dir, gold_dir, info)
            else:
                options = {"data_root": data_root, "work_dir": work_dir, **sheet_options}
                problems, counts, files = pipeline_sheet(scene, question, prompts, **options)
            result.update(problems=problems, counts=counts, files=[str(f) for f in files])
        except SystemExit as error:  # the steps stop with sys.exit("[question] ...") and their reason
            result["problems"] = [f"not made: {error.code}"]
        except Exception as error:  # noqa: BLE001 - one question's failure must not stop the report
            result["problems"] = [f"not made: {type(error).__name__}: {error}"]
        result["ok"] = not result["problems"]
        if result["ok"]:
            found = ", ".join(f"{p} {c}" for p, c in result["counts"].items())
            print(f"  sheet:    OK, {sum(result['counts'].values())} objects{f' ({found})' if found else ''}", file=out)
        else:
            print(f"  sheet:    FAILED: {result['problems'][0]}", file=out)
            for problem in result["problems"][1:]:
                print(f"            {problem}", file=out)
        empty = [p for p, c in result.get("counts", {}).items() if c == 0]
        if empty:
            why = "no gold class" if source == "gold" else "SAM 3 found nothing"
            print(f"  no rows:  {', '.join(empty)} ({why})", file=out)
        for i, path in enumerate(result["files"]):
            print(f"  {'files:   ' if i == 0 else '         '} {path}", file=out)
        out.flush()
        results.append(result)

    failed = sum(not r["ok"] for r in results)
    partial = sum(r["ok"] and any(c == 0 for c in r["counts"].values()) for r in results)
    print(f"\n{len(results)} questions: {len(results) - failed} sheets OK ({partial} with a prompt that has no rows), "
          f"{failed} failed", file=out)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", help="<dataset>/<scene>, e.g. scannet/scene0025_01, or only the scene's name")
    p.add_argument("--source", choices=["pipeline", "gold"], default="pipeline",
                   help="pipeline: SAM 3 and tracking (GPU); gold: VSI-Bench's ground truth (CPU)")
    p.add_argument("--questions", help="a .jsonl of questions in VSI-Bench's format (default: the export's)")
    p.add_argument("--types", nargs="+", help="only these question types, e.g. object_counting object_rel_distance")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--data-root", help="folder holding the exported scenes/ (default: as question_scene_sheet.py)")
    p.add_argument("--dense-dir", help="pipeline: folder with the scene's depth.npy and depth_conf.npy")
    p.add_argument("--work-dir", help="where geometry, masks, tracks and sheets are kept (default: "
                                      "tasks/openvocab_adapter/out/<dataset>__<scene>/all, or .../gold)")
    p.add_argument("--gold-dir", help="gold: folder with VSI-Bench's metadata (default: as scene_sheet_creator.py)")
    args = p.parse_args(argv)
    missing = [m for m in ("spacy", "nltk", "en_core_web_md") if importlib.util.find_spec(m) is None]
    if missing:  # fail here, not once per question
        p.error(f"the prompt extractor needs {', '.join(missing)} in this Python; see prompt_object_extractor.py")
    options = {"dense_dir": args.dense_dir} if args.source == "pipeline" else {}
    try:
        results = report(args.scene, args.source, args.data_root, args.questions, args.work_dir, args.gold_dir,
                         args.types, args.limit, **options)
    except (ValueError, RuntimeError) as error:
        p.error(str(error))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
