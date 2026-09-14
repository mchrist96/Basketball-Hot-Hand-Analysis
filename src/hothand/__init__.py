"""NBA play-by-play ingestion and sequential shot construction."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PBP_DIR = DATA_DIR / "raw" / "pbp"
PROCESSED_DIR = DATA_DIR / "processed"
