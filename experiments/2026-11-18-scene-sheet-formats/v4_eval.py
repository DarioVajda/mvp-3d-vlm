"""mvp.eval.vsibench with pipeline_v4's sheet exactly as final_test2.ipynb gave it, so that runs with other eval
options (e.g. --no-video, the sheet alone) compare with that run question by question.

final_test2 ran explore_sheets.py at b381205: the pipeline csv sheet as b381205's renderer wrote it (first_seen and
num_frames), with the preamble "approximately in metres" (explore_sheets.priors_preamble) and relations_v4.json's
relations appended (explore_sheets.relation_text, mode "v4"). The renderer has changed since (visible stretches
instead of first_seen), so the sheet is written here by scene_sheet_b381205.py, b381205's
src/mvp/benchmarks/scene_sheet.py unchanged, and put in place of scene_sheet.render_sheet for pipeline csv sheets.
For batch2_A's 798 questions the prompt with the video equals b381205's, and with --no-video it is that prompt less
its first line (checked 2026-11-24).

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/v4_eval.py --sheet-source pipeline \
      --sheet-format csv --sheets-root data/scene_sheets_pipeline_v4 ... [--no-video]
"""
import importlib.util
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import explore_sheets as es  # noqa: E402

from mvp.benchmarks import scene_sheet as ss  # noqa: E402

_spec = importlib.util.spec_from_file_location("scene_sheet_b381205", HERE / "scene_sheet_b381205.py")
b381205 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b381205)
B381205_PREAMBLE = b381205.PREAMBLES["pipeline"]


def v4_sheet(question_id, root, vggt_root):
    """pipeline_v4's sheet for the question, as final_test2 rendered it."""
    text = b381205.render_sheet(question_id, "pipeline", "csv", Path(root), vggt_root=Path(vggt_root))
    rel = es.relation_text(ss.load_index(Path(root))[int(question_id)], root, "pipeline", "v4")
    return text + ("\n" + rel if rel else "")


def install():
    """Pipeline csv sheets become pipeline_v4's; every other sheet is rendered as before. Calling it again changes
    nothing."""
    b381205.PREAMBLES["pipeline"] = es.priors_preamble(B381205_PREAMBLE)
    if getattr(ss.render_sheet, "pipeline_v4", False):
        return
    plain = ss.render_sheet
    signature = inspect.signature(plain)

    def render_sheet(*args, **kwargs):
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        a = call.arguments
        if a["source"] == "pipeline" and a["fmt"] == "csv":
            return v4_sheet(a["question_id"], a["root"], a["vggt_root"])
        return plain(*args, **kwargs)
    render_sheet.pipeline_v4 = True
    ss.render_sheet = render_sheet


def main():
    install()
    from mvp.eval import vsibench
    vsibench.main()


if __name__ == "__main__":
    main()
