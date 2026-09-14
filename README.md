# Basketball Hot Hand Analysis

Analysis of the "hot hand" phenomenon in basketball shooting data, built on
NBA play-by-play from stats.nba.com via `nba_api`.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ingest a season

```
python scripts/ingest.py --season 2023-24
python scripts/ingest.py --season 2023-24 --limit 20       # first 20 games only
python scripts/ingest.py --season 2023-24 --free-throws    # include free throws
python scripts/ingest.py --season 2023-24 --season-type Playoffs
```

Raw play-by-play is cached per game in `data/raw/pbp/` so re-runs are free.
The processed table lands in `data/processed/shots_<season>_<type>.parquet`,
one row per shot attempt, ordered by game clock, with per-player sequential
context:

| column          | meaning                                                        |
|-----------------|----------------------------------------------------------------|
| `shot_index`    | 1-based index of the shot within the player's game             |
| `made`          | shot outcome                                                   |
| `prev_made`     | outcome of the player's previous shot (NA on the first)        |
| `prev2_made`, `prev3_made` | outcomes two and three shots back                   |
| `makes_last3`   | makes among the previous three shots (NaN if fewer than three) |
| `streak_before` | signed run entering the shot: +k after k makes, -k after k misses |
| `sec_since_prev`| game seconds since the player's previous attempt               |

Plus shot context: `period`, `seconds_remaining`, `game_seconds`, `shot_value`
(2 or 3), `shot_distance`, `x`/`y` court coordinates, `sub_type`, `description`,
and game context: `is_home`, `opponent`, `score_before` / `opp_score_before` /
`score_diff_before` (from the shooter's perspective, before the shot),
`assisted` and `assister` (parsed from the description), `season`, `season_type`.

## Data caveats

- **Assists exist only on made shots.** The NBA feed credits an assist inside the
  description of a made field goal, e.g. `(Murray 1 AST)`. Missed shots carry no
  assist information at all, so `assisted` is always `False` on a miss. Do not
  use `assisted` as a predictor of `made`; it can only describe the makes. It is
  still useful for characterizing shot selection within a streak (e.g. share of
  makes that were assisted vs. self-created).
- **Player names are surnames only** (`playerName` from the feed). Group by
  `player_id`, never by `player_name`.
- **Missing shot distance shows as 0**, not NaN. A `shot_value` of 3 with a
  distance of 0 is a missing value.
- **Same-clock shots** (a miss and an immediate putback) are ordered by
  `action_number` and have `sec_since_prev` of 0. Decide whether tip-ins count
  as independent attempts before analyzing streaks.
- **Score is forward-filled** from scoring plays, so `score_before` is the score
  entering the shot. It includes free throws even when free throws are excluded
  from the shot table.
- **Buzzer heaves** are not filtered. Flag shots with `shot_distance` above
  roughly 35 feet or `seconds_remaining` under about 3 seconds and test whether
  excluding them changes results.

## Analysis notes

Things that will decide whether a "hot hand" result is real or an artifact:

- **Streak selection bias.** Miller & Sanjurjo (2018) showed that conditioning
  on "made the last k shots" within a finite sequence biases the observed
  conditional hit rate *downward* even for a fair coin. Comparing the post-streak
  rate to the player's raw average therefore understates any hot hand. Use a
  permutation test: shuffle each player-game's outcomes, recompute the
  conditional rate, and compare the real value to that null distribution.
- **Shot difficulty is the main confound.** Players who feel hot take harder
  shots and defenses tighten. Control for `shot_distance`, `shot_value`,
  `sub_type`, and time/score context; otherwise "no hot hand" may really be
  "hot hand cancelled by worse shot selection", which is a different finding.
- **Multiple comparisons.** Testing hundreds of players across many conditions
  will yield false positives at p < 0.05. Prefer a hierarchical / mixed model or
  a false discovery rate correction over per-player tests.
- **Sample size.** A high-volume player takes ~1,500 FGA per season and enters a
  3-make streak perhaps 150 times. Pool several seasons before drawing
  conclusions about anyone but the highest-volume players.
- **Betting relevance.** Pregame prop lines already price season averages and
  matchup. Any edge from streak state can only be exploited live, in-game.

## Layout

- `src/hothand/games.py` — list games for a season
- `src/hothand/pbp.py` — fetch and cache raw play-by-play with retries
- `src/hothand/shots.py` — filter to shots and add sequence features
- `src/hothand/analysis.py` — pooled conditional rates + within-game permutation null
- `scripts/ingest.py` — ingestion CLI
- `scripts/hot_hand.py` — analysis CLI

## Run the hot hand test

```
python scripts/hot_hand.py --player Jokic --k 3
python scripts/hot_hand.py --player Jokic --k 3 --seasons 2023-24 2024-25 --exclude-heaves
python scripts/hot_hand.py --all --k 3 --min-shots 800 --out data/processed/hot_hand_k3.csv
```

For each player it reports the pooled hit rate after `k` straight makes and
after `k` straight misses, the mean of the same statistic under within-game
shuffling (the biased benchmark a no-hot-hand player would produce), and a
one-sided permutation p-value. `diff` is the GVT-style makes-minus-misses gap.
