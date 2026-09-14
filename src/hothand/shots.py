"""Turn raw play-by-play into a per-player sequential shot table."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CLOCK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")
# e.g. "(Murray 1 AST)" at the end of a made-shot description
_ASSIST_RE = re.compile(r"\(([^()]+?) \d+ AST\)")

REGULATION_PERIOD_SECONDS = 720
OVERTIME_PERIOD_SECONDS = 300


def parse_clock(clock: str) -> float:
    """Convert an ISO-8601 duration like 'PT11M32.00S' to seconds remaining."""
    m = _CLOCK_RE.fullmatch(str(clock))
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + float(m.group(2))


def period_length(period: int) -> int:
    return REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS


def game_seconds_elapsed(period: pd.Series, seconds_remaining: pd.Series) -> pd.Series:
    """Seconds elapsed since tip-off for each event."""
    prior = period.apply(lambda p: sum(period_length(q) for q in range(1, int(p))))
    length = period.apply(period_length)
    return prior + (length - seconds_remaining)


def extract_shots(pbp: pd.DataFrame, include_free_throws: bool = False) -> pd.DataFrame:
    """Filter raw PlayByPlayV3 rows down to shot attempts in a tidy schema."""
    if pbp.empty:
        return pd.DataFrame()

    pbp = pbp.sort_values("actionNumber").reset_index(drop=True)

    # Score is only stamped on scoring plays; carry it forward across every
    # event, then shift one row so each shot sees the score *before* it.
    home_score = pd.to_numeric(pbp["scoreHome"], errors="coerce").ffill().shift(1).fillna(0).astype(int)
    away_score = pd.to_numeric(pbp["scoreAway"], errors="coerce").ffill().shift(1).fillna(0).astype(int)

    # Home / away tricodes from the location flag ('h' / 'v') on team events.
    loc = pbp["location"].astype(str)
    tri = pbp["teamTricode"].astype(str)
    home_team = tri[(loc == "h") & (tri != "")].iloc[0] if ((loc == "h") & (tri != "")).any() else ""
    away_team = tri[(loc == "v") & (tri != "")].iloc[0] if ((loc == "v") & (tri != "")).any() else ""

    is_fg = pbp["isFieldGoal"].astype(int) == 1
    is_ft = pbp["actionType"].eq("Free Throw")
    mask = (is_fg | is_ft) if include_free_throws else is_fg
    df = pbp.loc[mask].copy()

    df["seconds_remaining"] = df["clock"].map(parse_clock)
    df["period"] = df["period"].astype(int)

    is_home = df["location"].astype(str).eq("h")
    own_score = home_score[mask].where(is_home, away_score[mask])
    opp_score = away_score[mask].where(is_home, home_score[mask])
    assister = df["description"].astype(str).str.extract(_ASSIST_RE, expand=False)

    shots = pd.DataFrame(
        {
            "game_id": df["gameId"].astype(str),
            "action_number": df["actionNumber"].astype(int),
            "period": df["period"],
            "seconds_remaining": df["seconds_remaining"],
            "game_seconds": game_seconds_elapsed(df["period"], df["seconds_remaining"]),
            "player_id": df["personId"].astype(int),
            "player_name": df["playerName"].astype(str),
            "team_id": df["teamId"].astype(int),
            "team": df["teamTricode"].astype(str),
            "is_home": is_home,
            "opponent": np.where(is_home, away_team, home_team),
            "score_before": own_score,
            "opp_score_before": opp_score,
            "score_diff_before": own_score - opp_score,
            "is_free_throw": df["actionType"].eq("Free Throw"),
            "shot_value": pd.to_numeric(df["shotValue"], errors="coerce").fillna(0).astype(int),
            "shot_distance": pd.to_numeric(df["shotDistance"], errors="coerce"),
            "x": pd.to_numeric(df["xLegacy"], errors="coerce"),
            "y": pd.to_numeric(df["yLegacy"], errors="coerce"),
            "sub_type": df["subType"].astype(str),
            "assisted": assister.notna(),
            "assister": assister,
            "description": df["description"].astype(str),
            "made": df["shotResult"].eq("Made"),
        }
    )
    return shots


def _signed_streak(made: pd.Series) -> pd.Series:
    """Run length entering each shot: +k after k makes, -k after k misses, 0 first."""
    out = np.zeros(len(made), dtype=int)
    run = 0
    for i, m in enumerate(made.to_numpy()):
        out[i] = run
        if m:
            run = run + 1 if run > 0 else 1
        else:
            run = run - 1 if run < 0 else -1
    return pd.Series(out, index=made.index)


def add_sequence_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Add per-player, per-game sequential context to each shot.

    Adds:
      shot_index      1-based index of this shot within the player's game
      prev_made       outcome of the player's previous shot (NA for first)
      prev2_made      outcome two shots back
      prev3_made      outcome three shots back
      makes_last3     makes among the previous 3 shots (NaN if fewer than 3)
      streak_before   signed run length entering this shot
      sec_since_prev  game seconds since the player's previous shot
    """
    if shots.empty:
        return shots

    shots = shots.sort_values(["game_id", "game_seconds", "action_number"]).reset_index(drop=True)
    grp = shots.groupby(["game_id", "player_id"], sort=False)

    shots["shot_index"] = grp.cumcount() + 1
    shots["prev_made"] = grp["made"].shift(1).astype("boolean")
    shots["prev2_made"] = grp["made"].shift(2).astype("boolean")
    shots["prev3_made"] = grp["made"].shift(3).astype("boolean")
    shots["makes_last3"] = grp["made"].transform(
        lambda s: s.astype(float).shift(1).rolling(3).sum()
    )
    shots["sec_since_prev"] = grp["game_seconds"].diff()
    shots["streak_before"] = grp["made"].transform(_signed_streak)
    return shots


def build_shot_table(pbp: pd.DataFrame, include_free_throws: bool = False) -> pd.DataFrame:
    return add_sequence_features(extract_shots(pbp, include_free_throws))
