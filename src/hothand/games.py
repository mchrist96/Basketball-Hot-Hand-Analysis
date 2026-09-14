"""Fetch the list of games for a season."""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from .pbp import call_with_retry


def list_games(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Return one row per game: game_id, game_date, home_team, away_team.

    ``season`` uses the NBA format, e.g. "2023-24".
    ``season_type`` is "Regular Season", "Playoffs", or "PlayIn".
    """
    finder = call_with_retry(
        leaguegamefinder.LeagueGameFinder,
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",  # NBA only (excludes G League / WNBA)
    )
    df = finder.get_data_frames()[0]

    # Each game appears twice (once per team). "vs." marks the home team.
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)
    home = df[df["is_home"]][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"]].rename(
        columns={"TEAM_ABBREVIATION": "home_team"}
    )
    away = df[~df["is_home"]][["GAME_ID", "TEAM_ABBREVIATION"]].rename(
        columns={"TEAM_ABBREVIATION": "away_team"}
    )
    games = home.merge(away, on="GAME_ID", how="inner")
    games = games.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = season
    games["season_type"] = season_type
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
