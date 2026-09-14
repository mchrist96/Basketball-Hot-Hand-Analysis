"""Ingest a season of NBA play-by-play into a sequential shot table.

Usage:
    python scripts/ingest.py --season 2023-24
    python scripts/ingest.py --season 2023-24 --limit 20      # first 20 games
    python scripts/ingest.py --season 2023-24 --free-throws  # include FTs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hothand import PROCESSED_DIR  # noqa: E402
from hothand.games import list_games  # noqa: E402
from hothand.pbp import fetch_pbp  # noqa: E402
from hothand.shots import build_shot_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", required=True, help='e.g. "2023-24"')
    ap.add_argument("--season-type", default="Regular Season", choices=["Regular Season", "Playoffs", "PlayIn"])
    ap.add_argument("--limit", type=int, default=None, help="only process the first N games")
    ap.add_argument("--free-throws", action="store_true", help="include free throws as shots")
    ap.add_argument("--force", action="store_true", help="re-download cached play-by-play")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("ingest")

    games = list_games(args.season, args.season_type)
    if args.limit:
        games = games.head(args.limit)
    log.info("%d games in %s %s", len(games), args.season, args.season_type)

    tables = []
    failed = []
    for row in tqdm(games.itertuples(index=False), total=len(games), unit="game"):
        try:
            pbp = fetch_pbp(row.game_id, force=args.force)
        except Exception as exc:  # noqa: BLE001
            log.error("game %s failed: %s", row.game_id, exc)
            failed.append(row.game_id)
            continue
        shots = build_shot_table(pbp, include_free_throws=args.free_throws)
        if shots.empty:
            continue
        shots.insert(1, "game_date", row.game_date)
        shots.insert(2, "home_team", row.home_team)
        shots.insert(3, "away_team", row.away_team)
        tables.append(shots)

    if not tables:
        log.error("no shot data produced")
        sys.exit(1)

    out = pd.concat(tables, ignore_index=True)
    out["season"] = args.season
    out["season_type"] = args.season_type

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.season_type.lower().replace(" ", "_")
    out_path = PROCESSED_DIR / f"shots_{args.season}_{tag}.parquet"
    out.to_parquet(out_path, index=False)

    log.info("wrote %d shots from %d games to %s", len(out), out["game_id"].nunique(), out_path)
    if failed:
        log.warning("%d games failed: %s", len(failed), ", ".join(failed))


if __name__ == "__main__":
    main()
