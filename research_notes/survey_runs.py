"""Read-only survey of v5+ training runs: settings, what evals exist, and full-VSI overall scores."""
import glob, json, os, time

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
SKIP = {"wandb_entity", "wandb_project", "wandb_name", "no_wandb", "git_commit", "host", "gpu", "out", "workers", "world_size", "max_steps", "model",
        "clip", "mb_per_step", "max_q", "lora_r", "lora_alpha", "lora_dropout", "dev_every", "dev_per_task", "dev_videos", "nframes"}
runs = sorted((d for d in glob.glob(f"{R}/2B_v[5-9]*") + glob.glob(f"{R}/2B_xa*") + glob.glob(f"{R}/4B_v[4-9]*") if os.path.isdir(d)), key=os.path.getmtime)
for d in runs:
    name = os.path.basename(d)
    a = json.load(open(f"{d}/args.json")) if os.path.exists(f"{d}/args.json") else {}
    print(f"== {name} mtime {time.strftime('%m-%d %H:%M', time.gmtime(os.path.getmtime(d)))}")
    print("   ", {k: v for k, v in a.items() if k not in SKIP and v not in (None, False)})
    sums = sorted(glob.glob(f"{d}/**/*summary*.json", recursive=True))
    for s in sums[:14]:
        try:
            j = json.load(open(s))
        except Exception:
            continue
        print("    ", os.path.relpath(s, d), "n", j.get("n"), "frames", j.get("nframes"), "overall", round(j["overall"], 2) if "overall" in j else None)
    if len(sums) > 14:
        print("     ...", len(sums), "summaries")
