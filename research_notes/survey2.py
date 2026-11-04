"""Read-only: branch layout of the v7/v8/v9 runs, every branch's VSI overall, the 128-frame eval list."""
import glob, json, os

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
print(open(f"{R}/2B_v9_fmix_additive_s0/branches.json").read()[:1500])
print(open(f"{R}/jobs_eval_f128.txt").read()[:1200])
for d in sorted(glob.glob(f"{R}/2B_v[789]*")):
    n = os.path.basename(d)
    a = json.load(open(f"{d}/args.json")) if os.path.exists(f"{d}/args.json") else {}
    rows = []
    for s in sorted(glob.glob(f"{d}/branches/*/eval/*summary.json")):
        j = json.load(open(s))
        rows.append((s.split("/branches/")[1].split("/eval/")[0], os.path.basename(s).replace("_summary.json", ""), j.get("n"), round(j["overall"], 2)))
    print(n, "arm", a.get("arm"), "pe", a.get("pe"), "after", a.get("fusion_after"), "lr", a.get("lr"), a.get("fusion_lr"), "cap", a.get("cap"), "mix", a.get("frame_mix"), "ep", a.get("epochs"))
    for r in rows:
        print("    ", r)
    print("     files in branch 5750:", os.listdir(f"{d}/branches/step5750") if os.path.isdir(f"{d}/branches/step5750") else None,
          os.listdir(f"{d}/branches/step5750/eval") if os.path.isdir(f"{d}/branches/step5750/eval") else None)
