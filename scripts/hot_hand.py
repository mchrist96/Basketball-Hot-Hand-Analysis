"""Run the hot hand permutation test.

Usage:
    python scripts/hot_hand.py --player Jokic --k 3
    python scripts/hot_hand.py --player Jokic --k 3 --seasons 2023-24 2024-25
    python scripts/hot_hand.py --all --k 3 --min-shots 800 --out data/processed/hot_hand_k3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hothand.analysis import find_player, load_shots, permutation_test, run_all_players  # noqa: E402


def fmt_single(r) -> str:
    return f"""
{r.player_name} (player_id {r.player_id})  k = {r.k}
  shots: {r.n_shots:,} over {r.n_games} games   overall FG%: {r.fg_pct:.3f}

  after {r.k} straight MAKES   n = {r.n_after_makes:>5}
    observed rate          {r.rate_after_makes:.3f}
    permutation null mean  {r.null_mean_after_makes:.3f}  (sd {r.null_sd_after_makes:.3f})
    p(null >= observed)    {r.p_after_makes:.4f}

  after {r.k} straight MISSES  n = {r.n_after_misses:>5}
    observed rate          {r.rate_after_misses:.3f}
    permutation null mean  {r.null_mean_after_misses:.3f}
    p(null <= observed)    {r.p_after_misses:.4f}

  difference (makes - misses)
    observed               {r.diff:+.3f}
    permutation null mean  {r.null_mean_diff:+.3f}  (sd {r.null_sd_diff:.3f})
    p(null >= observed)    {r.p_diff:.4f}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", help="substring of player surname, e.g. Jokic")
    ap.add_argument("--all", action="store_true", help="run every player with >= --min-shots")
    ap.add_argument("--k", type=int, default=3, help="streak length to condition on")
    ap.add_argument("--seasons", nargs="*", default=None, help='e.g. 2023-24 2024-25 (default: all on disk)')
    ap.add_argument("--season-types", nargs="*", default=None, help='e.g. "Regular Season" (default: all)')
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--min-shots", type=int, default=800)
    ap.add_argument("--exclude-heaves", action="store_true", help="drop shots > 35 ft or < 3 s left in period")
    ap.add_argument("--out", help="CSV path for --all results")
    args = ap.parse_args()

    if not args.player and not args.all:
        ap.error("pass --player NAME or --all")

    shots = load_shots(args.seasons, args.season_types)
    if args.exclude_heaves:
        shots = shots[~((shots["shot_distance"] > 35) | (shots["seconds_remaining"] < 3))]
        # shot_index must be rebuilt after dropping rows
        shots = shots.sort_values(["game_id", "game_seconds", "action_number"])
        shots["shot_index"] = shots.groupby(["game_id", "player_id"]).cumcount() + 1

    if args.player:
        p = find_player(shots, args.player)
        print(fmt_single(permutation_test(p, k=args.k, n_perm=args.n_perm)))

    if args.all:
        res = run_all_players(shots, k=args.k, min_shots=args.min_shots, n_perm=args.n_perm)
        cols = ["player_name", "n_shots", "fg_pct", "rate_after_makes", "null_mean_after_makes",
                "rate_after_misses", "diff", "null_mean_diff", "p_diff"]
        print(res[cols].round(3).to_string(index=False))
        if args.out:
            res.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
