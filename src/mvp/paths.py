"""Project paths. ROOT is the mvp/ directory; everything else hangs off it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
VSI_DIR = DATA / "raw" / "VSI-Bench"
MMSI_PATH = DATA / "raw" / "MMSI-Bench" / "MMSI_Bench.parquet"
VGGT_DIR = DATA / "vggt"
TRAIN_ANN = DATA / "raw" / "SpatialStackData"  # SpatialStack's training annotations
TRAIN_MEDIA = DATA / "train_media"  # their media, laid out as the annotations' paths expect
BENCH_MEDIA = DATA / "bench_media"  # videos of VSTI-Bench and Video-MME (slurm/prep_bench_media.sh)
SCENE_SHEETS = DATA / "scene_sheets_75"  # scene sheets of the 75 exported scenes (docs/scene_sheets_75.md)
RESULTS = ROOT / "results"
ZEROSHOT = RESULTS / "zeroshot"
FIGURES = RESULTS / "figures"
