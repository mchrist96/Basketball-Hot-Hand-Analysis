"""Fetch and cache raw play-by-play data from stats.nba.com."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3
from requests.exceptions import RequestException

from . import RAW_PBP_DIR

log = logging.getLogger(__name__)

T = TypeVar("T")

# stats.nba.com throttles aggressively; be polite.
REQUEST_DELAY_SECONDS = 0.6
TIMEOUT_SECONDS = 30


def call_with_retry(fn: Callable[..., T], *args, retries: int = 4, **kwargs) -> T:
    """Call an nba_api endpoint constructor with exponential backoff."""
    kwargs.setdefault("timeout", TIMEOUT_SECONDS)
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            result = fn(*args, **kwargs)
            time.sleep(REQUEST_DELAY_SECONDS)
            return result
        except (RequestException, ValueError) as exc:  # ValueError: bad JSON
            if attempt == retries:
                raise
            log.warning("attempt %d/%d failed (%s); retrying in %.0fs", attempt, retries, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def raw_path(game_id: str) -> Path:
    return RAW_PBP_DIR / f"{game_id}.parquet"


def fetch_pbp(game_id: str, force: bool = False) -> pd.DataFrame:
    """Return raw play-by-play for one game, reading from cache when present."""
    path = raw_path(game_id)
    if path.exists() and not force:
        return pd.read_parquet(path)

    endpoint = call_with_retry(playbyplayv3.PlayByPlayV3, game_id=game_id)
    df = endpoint.get_data_frames()[0]
    if df.empty:
        log.warning("empty play-by-play for game %s", game_id)
        return df

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
