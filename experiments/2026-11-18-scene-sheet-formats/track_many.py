"""Run milestone4_video.py's track stage for several scenes in one process: Python, PyTorch and DINOv2 start once.

  python track_many.py <milestone4_video.py or its copy> <jobs.json> <results.json>

jobs.json is a list of argument lists, each as the tracker takes them, with --stages track. The tracker is imported
once (which parses the first job's arguments, as running it would). For each job, its module-level ARGS and OUT_DIR,
the only module state the track stage reads, are set from the job's arguments, and stage_track runs on frame_list(), as
main() runs it. torch.hub.load returns the model it loaded first for the same arguments instead of loading it again
(DINOv2, used in eval mode under no_grad, so never changed). A job's output goes to <out>/track.log; results.json gets
ok, error and seconds per job, written after each one, so the jobs done before a crash are known.
"""
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

script, jobs_file, results_file = sys.argv[1:4]
jobs = json.loads(Path(jobs_file).read_text())
sys.path.insert(0, str(Path(script).resolve().parent))  # as when the tracker runs as a script: saved_masks
sys.argv = [script, *jobs[0]]
spec = importlib.util.spec_from_file_location("milestone4_video_batch", script)
m4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m4)

torch = sys.modules.get("torch")  # imported by the tracker
if torch is not None:
    _load, _models = torch.hub.load, {}

    def cached_load(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in _models:
            _models[key] = _load(*args, **kwargs)
        return _models[key]

    torch.hub.load = cached_load

results = []
for argv in jobs:
    start = time.time()
    sys.argv = [script, *argv]
    args = m4.parse_args()
    record = {"out": args.out, "ok": False, "error": None}
    try:
        if [s.strip() for s in args.stages.split(",")] != ["track"]:
            raise ValueError("track_many.py runs --stages track only")
        m4.ARGS, m4.OUT_DIR = args, Path(args.out)
        m4.OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(m4.OUT_DIR / "track.log", "w") as log, contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log):
            m4.stage_track(m4.frame_list())
        record["ok"] = True
    except (Exception, SystemExit) as error:  # noqa: BLE001 - the tracker stops with sys.exit; the next job goes on
        record["error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = round(time.time() - start, 1)
    results.append(record)
    Path(results_file).write_text(json.dumps(results, indent=1))
    print(f"[track_many] {args.out}: {'ok' if record['ok'] else record['error']} in {record['seconds']} s", flush=True)
