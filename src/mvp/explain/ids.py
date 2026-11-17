"""Question-id lists used by the explainability runs (CPU only, light enough for the login node).

  python -m mvp.explain.ids v7       results/explain_v7/ids_200.txt        200 questions per type (seed 0), v7 ablations
  python -m mvp.explain.ids swap     results/ablation_v10/ids.txt          every relative direction / distance / route question
  python -m mvp.explain.ids screen   results/detail_v10/screen_ids.json    direction questions (landscape frames) that the
                                                                           v10 cross-attention arm and the additive control
                                                                           answer differently at step 5750, seed 0
"""
import json, os, sys

from mvp.benchmarks.vsibench import load_vsibench
from mvp.explain import ROOT

SWAP_TYPES = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
              "object_rel_distance", "route_planning"]


def v7(path=f"{ROOT}/results/explain_v7/ids_200.txt"):
    d = load_vsibench("full").groupby("question_type", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 200), random_state=0))
    open(path, "w").write("".join(f"{i}\n" for i in sorted(d["id"])))
    print(len(d), "questions ->", path)


def swap(path=f"{ROOT}/results/ablation_v10/ids.txt"):
    df = load_vsibench("full")
    d = df[df.question_type.isin(SWAP_TYPES)]
    open(path, "w").write("\n".join(map(str, d.id.tolist())) + "\n")
    print(len(d), "questions,", d.video_path.nunique(), "videos ->", path)


def screen(path=f"{ROOT}/results/detail_v10/screen_ids.json", step=5750):
    ld = lambda r: {x["id"]: x for x in map(json.loads, open(
        f"{ROOT}/results/train/{r}/branches/step{step}/eval/vsi_f32_full.jsonl"))}
    X, A = ld("2B_v10_xa3_qknu_s0"), ld("2B_v9_fmix_additive_s0")
    pick = lambda xs, as_: [i for i, x in X.items() if x["question_type"].startswith("object_rel_direction")
                            and x["frame_hw"] == [480, 640] and x["score"] == xs and A[i]["score"] == as_]
    json.dump({"xa": pick(1, 0), "add": pick(0, 1)}, open(path, "w"))
    print(path)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"v7": v7, "swap": swap, "screen": screen}.get(which)
    if fn is None:
        raise SystemExit(__doc__)
    fn()
