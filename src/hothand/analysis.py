"""Hot hand statistics with a within-game permutation null.

The core idea: for a player, compute the pooled hit rate on shots that follow
``k`` consecutive makes (or misses). Do not compare it to the player's overall
average, which is biased for finite sequences (Miller & Sanjurjo 2018). Instead
shuffle the player's outcomes *within each game*, recompute the same statistic
many times, and compare the real value to that null distribution.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import PROCESSED_DIR


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_shots(seasons: Iterable[str] | None = None, season_types: Iterable[str] | None = None) -> pd.DataFrame:
    """Concatenate processed shot tables. ``None`` means everything on disk."""
    files = sorted(PROCESSED_DIR.glob("shots_*.parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if seasons is not None and df["season"].iloc[0] not in set(seasons):
            continue
        if season_types is not None and df["season_type"].iloc[0] not in set(season_types):
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no matching shot tables in {PROCESSED_DIR}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["game_date", "game_id", "game_seconds", "action_number"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Core statistic
# --------------------------------------------------------------------------- #

def _prev_k_mask(made: np.ndarray, shot_index: np.ndarray, k: int, outcome: bool) -> np.ndarray:
    """True where the previous k shots *in the same game* all equal ``outcome``.

    Requires rows ordered by game then shot_index. ``shot_index`` is 1-based,
    so a shot has k in-game predecessors iff shot_index > k.
    """
    target = made if outcome else ~made
    cond = shot_index > k
    for j in range(1, k + 1):
        cond &= np.roll(target, j)
    return cond


def conditional_rate(made: np.ndarray, shot_index: np.ndarray, k: int, outcome: bool) -> tuple[float, int]:
    """Pooled hit rate on shots following k straight ``outcome``s. Returns (rate, n)."""
    m = _prev_k_mask(made, shot_index, k, outcome)
    n = int(m.sum())
    return (float(made[m].mean()) if n else np.nan), n


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #

@dataclass
class HotHandResult:
    player_id: int
    player_name: str
    k: int
    n_shots: int
    n_games: int
    fg_pct: float
    # after k makes
    rate_after_makes: float
    n_after_makes: int
    null_mean_after_makes: float
    null_sd_after_makes: float
    p_after_makes: float
    # after k misses
    rate_after_misses: float
    n_after_misses: int
    null_mean_after_misses: float
    p_after_misses: float
    # GVT-style difference: after makes minus after misses
    diff: float
    null_mean_diff: float
    null_sd_diff: float
    p_diff: float

    def to_dict(self) -> dict:
        return asdict(self)


def permutation_test(
    player_shots: pd.DataFrame,
    k: int = 3,
    n_perm: int = 5000,
    seed: int | None = 0,
) -> HotHandResult:
    """Within-game permutation test for one player's shot table.

    Shuffles outcomes inside each game (keeping each game's make count and
    the position of every shot), recomputes the conditional rates, and
    reports one-sided p-values: P(null >= observed) for the after-makes rate
    and the difference, P(null <= observed) for the after-misses rate.
    """
    df = player_shots.sort_values(["game_id", "shot_index"]).reset_index(drop=True)
    made = df["made"].to_numpy(dtype=bool)
    shot_index = df["shot_index"].to_numpy()
    game_codes = pd.factorize(df["game_id"])[0]

    obs_make, n_make = conditional_rate(made, shot_index, k, True)
    obs_miss, n_miss = conditional_rate(made, shot_index, k, False)
    obs_diff = obs_make - obs_miss

    rng = np.random.default_rng(seed)
    null_make = np.empty(n_perm)
    null_miss = np.empty(n_perm)
    for i in range(n_perm):
        # Random order within each game block; blocks stay in place.
        order = np.lexsort((rng.random(len(made)), game_codes))
        shuffled = made[order]
        null_make[i], _ = conditional_rate(shuffled, shot_index, k, True)
        null_miss[i], _ = conditional_rate(shuffled, shot_index, k, False)
    null_diff = null_make - null_miss

    def p_ge(null, obs):
        valid = null[~np.isnan(null)]
        return float((valid >= obs).mean()) if len(valid) and not np.isnan(obs) else np.nan

    def p_le(null, obs):
        valid = null[~np.isnan(null)]
        return float((valid <= obs).mean()) if len(valid) and not np.isnan(obs) else np.nan

    return HotHandResult(
        player_id=int(df["player_id"].iloc[0]),
        player_name=str(df["player_name"].iloc[0]),
        k=k,
        n_shots=len(df),
        n_games=int(df["game_id"].nunique()),
        fg_pct=float(made.mean()),
        rate_after_makes=obs_make,
        n_after_makes=n_make,
        null_mean_after_makes=float(np.nanmean(null_make)),
        null_sd_after_makes=float(np.nanstd(null_make)),
        p_after_makes=p_ge(null_make, obs_make),
        rate_after_misses=obs_miss,
        n_after_misses=n_miss,
        null_mean_after_misses=float(np.nanmean(null_miss)),
        p_after_misses=p_le(null_miss, obs_miss),
        diff=obs_diff,
        null_mean_diff=float(np.nanmean(null_diff)),
        null_sd_diff=float(np.nanstd(null_diff)),
        p_diff=p_ge(null_diff, obs_diff),
    )


def run_all_players(
    shots: pd.DataFrame,
    k: int = 3,
    min_shots: int = 500,
    n_perm: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation test for every player with at least ``min_shots`` attempts."""
    counts = shots.groupby("player_id").size()
    keep = counts[counts >= min_shots].index
    rows = []
    for pid, grp in shots[shots["player_id"].isin(keep)].groupby("player_id", sort=False):
        rows.append(permutation_test(grp, k=k, n_perm=n_perm, seed=seed).to_dict())
    out = pd.DataFrame(rows)
    return out.sort_values("p_diff").reset_index(drop=True)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def find_player(shots: pd.DataFrame, name: str) -> pd.DataFrame:
    """Case- and accent-insensitive substring match on player_name; errors if ambiguous."""
    names = shots[["player_id", "player_name"]].drop_duplicates()
    plain = names["player_name"].map(_strip_accents)
    hits = names[plain.str.contains(_strip_accents(name), case=False, regex=False)]
    ids = hits["player_id"].unique()
    if len(ids) == 0:
        raise ValueError(f"no player matching {name!r}")
    if len(ids) > 1:
        raise ValueError(f"ambiguous name {name!r}: {hits.to_dict('records')}")
    return shots[shots["player_id"] == ids[0]]
