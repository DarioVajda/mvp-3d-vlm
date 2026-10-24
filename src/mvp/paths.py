"""Project paths. ROOT is the mvp/ directory; everything else hangs off it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
VSI_DIR = DATA / "raw" / "VSI-Bench"
MMSI_PATH = DATA / "raw" / "MMSI-Bench" / "MMSI_Bench.parquet"
VGGT_DIR = DATA / "vggt"
TRAIN_ANN = DATA / "raw" / "SpatialStackData"  # SpatialStack's training annotations
TRAIN_MEDIA = DATA / "train_media"  # their media, laid out as the annotations' paths expect
RESULTS = ROOT / "results"
ZEROSHOT = RESULTS / "zeroshot"
FIGURES = RESULTS / "figures"
