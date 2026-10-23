"""Project paths. ROOT is the mvp/ directory; everything else hangs off it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
VSI_DIR = DATA / "raw" / "VSI-Bench"
MMSI_PATH = DATA / "raw" / "MMSI-Bench" / "MMSI_Bench.parquet"
VGGT_DIR = DATA / "vggt"
RESULTS = ROOT / "results"
ZEROSHOT = RESULTS / "zeroshot"
FIGURES = RESULTS / "figures"
