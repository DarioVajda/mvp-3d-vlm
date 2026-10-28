"""Pick a run's checkpoint with the highest dev accuracy (mvp.train.run --dev-videos > 0).

  python -m mvp.train.select_best results/train/<run>   -> prints the checkpoint path, writes <run>/best_epoch.json

Dev rows in log.jsonl carry dev_acc (mean over the six VLM-3R tasks) at step 0 (untrained) and at every checkpoint.
Newer runs name the checkpoint in the row ("ckpt", with "epoch_frac" for half-epoch points); older runs saved
ckpt_epoch<k>.pt after epoch index k-1. Step 0 is excluded and ties go to the later checkpoint.
"""
import json
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(out / "log.jsonl")]
    dev = [r for r in rows if "dev_acc" in r and r["step"] > 0]
    assert dev, f"no dev accuracy rows in {out / 'log.jsonl'}"
    for r in dev:
        r.setdefault("epoch_frac", r["epoch"] + 1)
        r.setdefault("ckpt", f"ckpt_epoch{r['epoch'] + 1}.pt")
    best = max(dev, key=lambda r: (r["dev_acc"], r["step"]))
    ckpt = out / best["ckpt"]
    assert ckpt.exists(), ckpt
    keep = ("epoch_frac", "step", "dev_acc", "dev_acc/all", "dev_acc/appearance_order", "dev_loss")
    (out / "best_epoch.json").write_text(json.dumps({
        "epoch": best["epoch_frac"], "step": best["step"], "ckpt": str(ckpt), "dev_acc": best["dev_acc"],
        "epochs": [{("epoch" if k == "epoch_frac" else k): r[k] for k in keep if k in r} for r in dev]}, indent=1))
    print(ckpt)


if __name__ == "__main__":
    main()
