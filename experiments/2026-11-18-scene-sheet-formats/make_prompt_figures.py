"""The report's two example-prompt figures: one VSI-Bench question as Qwen3.5-2B receives it after the video, with
pipeline_v4's scene sheet and with the gold sheet and its relations, condition C of 2026-11-19-gold-sheet-relations.
The video is one marker line, and long parts are cut where a grey marker says so. Each figure is written three ways,
from the same lines:

- prompt_example_pipeline_v4.tex, prompt_example_gold.tex: a figure with a framed listing (needs listings, xcolor);
- figures/prompt_example_*.pdf (vector) and .png (300 dpi), drawn with matplotlib in DejaVu Sans Mono (embedded),
  as wide as a single-column A4 text block with 2 cm margins (17 cm);
- figures/prompt_figures.tex: the \\includegraphics figures with their captions and labels.

The text is the eval's own: pipeline_v4's sheet as final_test2 gave it (v4_eval.v4_sheet: b381205's renderer and the
v4 relations), the gold sheet as --sheet-source gold_relations renders it, both through
mvp.benchmarks.vsibench.build_prompt_parts. Every line shown is checked against the eval's whole prompt
(scene_sheet.CHAT), in order. The pipeline sheet's first_seen is in the video's seconds, so --vggt-root needs the
scene's meta.json with its video_fps (question 3159's video, scannet/scene0088_01: 1,066 frames at exactly 24 fps,
from its MP4 header).

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/make_prompt_figures.py \
      --v4-root data/scene_sheets_pipeline_v4 --gold-root tasks/openvocab_adapter/out_full \
      --vggt-root data/vggt --questions data/raw/VSI-Bench/test.jsonl [--question 3159]
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import v4_eval  # noqa: E402

from mvp.benchmarks import scene_sheet as ss  # noqa: E402
from mvp.benchmarks.vsibench import build_prompt_parts  # noqa: E402

PAD = 300  # video tokens per frame pair at 480 x 640: 480 * 640 / (32 * 32)
KEEP_RELATIONS = 3  # distance pairs shown before the cut
WIDTH_CM = 17.0  # A4 (21 cm) less two 2 cm margins
FONT = "DejaVu Sans Mono"  # ships with matplotlib, so the figures look the same on any machine
FONT_PT = 6.5
ADVANCE_EM = 1233 / 2048  # DejaVu Sans Mono's advance width, in ems
LINE = 1.25  # line height, in font sizes
INSET_PT = 5  # between the frame and the text
GREY = "#808080"
PREAMBLE = r"""% Needs \usepackage{{listings,xcolor}}. Written by make_prompt_figures.py from the eval's own rendering of VSI-Bench
% question {q}; grey text marks what is not shown.
\providecommand{{\promptnote}}[1]{{{{\color{{gray}}\itshape[#1]}}}}
\lstdefinestyle{{promptbox}}{{basicstyle=\ttfamily\scriptsize, frame=single, framesep=4pt, breaklines=true,
  breakatwhitespace=false, columns=fullflexible, keepspaces=true, escapeinside={{(*@}}{{@*)}},
  postbreak=\mbox{{\textcolor{{gray}}{{$\hookrightarrow$}}\space}}}}
"""


class Note(str):
    """A grey marker line, not part of the prompt; its text is plain ("…", "–")."""

    def tex(self):
        return ("(*@\\promptnote{" + self.replace("_", "\\_").replace("… ", "\\ldots\\ ").replace("–", "--")
                + "}@*)")


def shorten(post, keep_rows):
    """The text after the video with the sheet's rows passed through keep_rows and the distance pairs cut to the first
    KEEP_RELATIONS. Returns the lines, how many sheet rows and distance pairs there are."""
    lines = post.split("\n")
    i = 3  # "These are frames of a video.", the sheet preamble, the csv header
    rows = []
    while not lines[i].startswith("Relations between"):
        rows.append(lines[i])
        i += 1
    out = lines[:3] + keep_rows(rows) + lines[i:i + 2]  # the relations' introduction and header
    i += 2
    pairs = []
    while not lines[i].startswith("Distance from each candidate"):
        pairs.append(lines[i])
        i += 1
    out += pairs[:KEEP_RELATIONS]
    if len(pairs) > KEEP_RELATIONS:
        out.append(Note(f"… {len(pairs) - KEEP_RELATIONS} more pairs"))
    return out + lines[i:], len(rows), len(pairs)


def first_chairs(rows, keep=2):
    """The first `keep` chairs and every other row (a gold sheet lists many chairs)."""
    chairs = [r for r in rows if r.startswith("chair_")]
    if len(chairs) <= keep:
        return rows
    return (chairs[:keep] + [Note(f"… chair_{keep + 1} to chair_{len(chairs)}: {len(chairs) - keep} more chairs")]
            + [r for r in rows if not r.startswith("chair_")])


def check(lines, post, times):
    """Every line shown is in the eval's whole prompt, in order; marker lines are skipped."""
    video = "".join(f"<{t} seconds><|vision_start|><|video_pad|> x {PAD}<|vision_end|>" for t in times)
    full = ss.CHAT.format(system=ss.SYSTEM_PROMPT, pre="", video=video, post=post)
    pos = 0
    for line in lines:
        if isinstance(line, Note):
            continue
        k = full.find(line + ("\n" if line is not lines[-1] else ""), pos)
        if k < 0 or full[k - 1] not in "\n>":
            raise SystemExit(f"not in the prompt in this order: {line[:80]!r}")
        pos = k + len(line)


def tex_figure(q, lines, caption, label):
    body = "\n".join(line.tex() if isinstance(line, Note) else line for line in lines)
    return (PREAMBLE.format(q=q) + "\\begin{figure}[t]\n\\begin{lstlisting}[style=promptbox]\n" + body
            + "\n\\end{lstlisting}\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{figure}\n")


def draw(lines, stem):
    """stem.pdf and stem.png (300 dpi): the lines in a frame WIDTH_CM wide, wrapped at the frame, a wrapped line's
    continuation after a grey arrow, markers in grey italics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    matplotlib.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    upright = FontProperties(family=FONT, size=FONT_PT)
    italic = FontProperties(family=FONT, size=FONT_PT, style="oblique")
    width_pt = WIDTH_CM / 2.54 * 72
    char_pt = ADVANCE_EM * FONT_PT  # a monospace font: every character has the same advance
    columns = int((width_pt - 2 * INSET_PT) // char_pt) - 1  # one column spare, for rounding in the renderers
    rows = []  # (text, note, continuation)
    for line in lines:
        if isinstance(line, Note):
            rows.append(("[" + line + "]", True, False))
            continue
        # by words, each piece short enough to follow the continuation arrow (2 columns)
        pieces = textwrap.wrap(line, columns - 2, break_long_words=True, break_on_hyphens=False) or [""]
        rows += [(piece, False, k > 0) for k, piece in enumerate(pieces)]
    line_pt = FONT_PT * LINE
    height_pt = 2 * INSET_PT + len(rows) * line_pt
    fig = plt.figure(figsize=(width_pt / 72, height_pt / 72))
    fig.add_artist(Rectangle((0.5 / width_pt, 0.5 / height_pt), 1 - 1 / width_pt, 1 - 1 / height_pt,
                             transform=fig.transFigure, fill=False, linewidth=0.5, edgecolor="black"))
    for k, (text, note, cont) in enumerate(rows):
        y = 1 - (INSET_PT + (k + 0.8) * line_pt) / height_pt
        x = INSET_PT / width_pt
        if cont:
            fig.text(x, y, "↪", fontproperties=upright, color=GREY)
            x += 2 * char_pt / width_pt
        fig.text(x, y, text, fontproperties=italic if note else upright, color=GREY if note else "black")
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return len(rows)


def graphics_figure(name, caption, label):
    return ("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/" + name + ".pdf}\n"
            "\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{figure}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4-root", required=True, help="pipeline_v4's sheets (scene_sheets_pipeline_v4_batch2_A.zip)")
    ap.add_argument("--gold-root", required=True, help="the rebuilt gold sheets (build_gold_sheets.py)")
    ap.add_argument("--vggt-root", required=True, help="the scene's meta.json (video_fps)")
    ap.add_argument("--questions", required=True, help="VSI-Bench's test.jsonl")
    ap.add_argument("--question", type=int, default=3159)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    q = args.question
    doc = next(r for r in map(json.loads, open(args.questions)) if int(r["id"]) == q)
    scene = f"{doc['dataset']}/{doc['scene_name']}"
    v4_root, gold_root, vggt = Path(args.v4_root), Path(args.gold_root), Path(args.vggt_root)
    times = ss.frame_timestamps(v4_root / scene.replace("/", "__") / "subset-uniform16" / "scene", scene, vggt)[::ss.MERGE]
    marker = Note(f"16 video frames, {times[0]} s – {times[-1]} s")
    v4_eval.install()
    out, figures = Path(args.out), Path(args.out) / "figures"
    figures.mkdir(exist_ok=True)
    made = []

    _, post = build_prompt_parts(doc, "sit", v4_eval.v4_sheet(q, v4_root, vggt))
    lines, n_rows, n_pairs = shorten(post, lambda rows: rows)
    check(lines, post, times)
    cut = f"{n_pairs - KEEP_RELATIONS} of the {n_pairs} distance pairs" if n_pairs > KEEP_RELATIONS else "nothing"
    made.append(("prompt_example_pipeline_v4", [marker] + lines, (
        f"The text Qwen3.5-2B receives after the video for one VSI-Bench question ({q}, relative distance) with the "
        "pipeline\\_v4 scene sheet. Each sheet row is one tracked object: \\texttt{id} and \\texttt{label}; "
        "\\texttt{x, y, z}, its centre, and \\texttt{size\\_x, size\\_y, size\\_z}, its extent, in the coordinates of "
        "the first frame's camera (x right, y down, z forward), scaled to approximately metres from typical object "
        "sizes; \\texttt{first\\_seen}, the video timestamp at which it first appears; \\texttt{num\\_frames}, in how "
        "many of the 16 frames it is seen. For distance questions the sheet keeps one row per object the question "
        "names, then lists the closest-point distances between their boxes, and a relation line states each "
        f"candidate's distance to the reference object and which is closest. Grey text is not part of the prompt; "
        f"left out: {cut}."), "fig:prompt-pipeline-v4"))

    _, post = build_prompt_parts(doc, "sit", ss.render_scene(q, "gold_relations", "csv", False, gold_root))
    lines, n_rows, n_pairs = shorten(post, first_chairs)
    check(lines, post, times)
    shown = sum(1 for line in lines[3:] if not isinstance(line, Note) and "," in line and not line.startswith(
        ("Relations", "object_a", "Distance", "Measuring"))) - min(n_pairs, KEEP_RELATIONS)
    made.append(("prompt_example_gold", [marker] + lines, (
        f"The same question ({q}) with the gold scene sheet and its relations (condition C): VSI-Bench's labelled "
        "objects instead of the pipeline's. \\texttt{x, y, z} are box centres and \\texttt{size\\_x, size\\_y, "
        "size\\_z} box extents, in metres in the dataset's world frame (z up); gold sheets have no timestamps. The "
        "relation table gives the closest-point distance between every two objects of the named classes, and the "
        "relation line each candidate's distance to the nearest reference object. Grey text is not part of the "
        f"prompt; left out: {n_rows - shown} of the {n_rows} sheet rows, {n_pairs - KEEP_RELATIONS} of the {n_pairs} "
        "distance pairs."), "fig:prompt-gold"))

    snippets = []
    for name, shown_lines, caption, label in made:
        (out / f"{name}.tex").write_text(tex_figure(q, shown_lines, caption, label))
        n = draw(shown_lines, figures / name)
        snippets.append(graphics_figure(name, caption, label))
        print(f"{name}: .tex, figures/{name}.pdf and .png ({n} lines at {FONT_PT} pt, {WIDTH_CM} cm wide)")
    (figures / "prompt_figures.tex").write_text(
        f"% Written by make_prompt_figures.py (question {q}). The images are {WIDTH_CM:g} cm wide, a single-column A4 text "
        "block with 2 cm margins.\n" + "\n".join(snippets))
    print(f"figures/prompt_figures.tex: the \\includegraphics figures (video {times[0]} s to {times[-1]} s)")


if __name__ == "__main__":
    main()
