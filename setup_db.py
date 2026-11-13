"""
Build energy.db from SMARD (Bundesnetzagentur) open electricity market data.

Downloads hourly data for Germany:
  - generation by source (solar, wind, lignite, gas, ...)  -> table `generation`
  - grid load and residual load                            -> table `consumption`
  - day-ahead wholesale price, DE-LU bidding zone          -> table `prices`

Usage:
  python setup_db.py              # last 52 weeks
  python setup_db.py --weeks 12   # faster, smaller

Data source: https://www.smard.de (licence: CC BY 4.0, Bundesnetzagentur | SMARD.de)
"""

import argparse
import json
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BASE = "https://www.smard.de/app/chart_data"
BERLIN = ZoneInfo("Europe/Berlin")

# filter_id: (source name, category)
GENERATION = {
    4068: ("solar", "renewable"),
    4067: ("wind_onshore", "renewable"),
    1225: ("wind_offshore", "renewable"),
    4066: ("biomass", "renewable"),
    1226: ("hydro", "renewable"),
    1228: ("other_renewable", "renewable"),
    1223: ("lignite", "conventional"),
    4069: ("hard_coal", "conventional"),
    4071: ("natural_gas", "conventional"),
    1224: ("nuclear", "conventional"),
    1227: ("other_conventional", "conventional"),
    4070: ("pumped_storage", "storage"),
}
GRID_LOAD, RESIDUAL_LOAD, PRICE = 410, 4359, 4169

SCHEMA = """
DROP TABLE IF EXISTS sources;
DROP TABLE IF EXISTS generation;
DROP TABLE IF EXISTS consumption;
DROP TABLE IF EXISTS prices;

CREATE TABLE sources (
    source          TEXT PRIMARY KEY,   -- e.g. 'solar', 'lignite'
    category        TEXT NOT NULL,      -- 'renewable' | 'conventional' | 'storage'
    smard_filter_id INTEGER NOT NULL
);
CREATE TABLE generation (
    ts_utc   TEXT NOT NULL,   -- start of the hour, UTC, 'YYYY-MM-DD HH:MM'
    ts_local TEXT NOT NULL,   -- same hour in Europe/Berlin local time
    source   TEXT NOT NULL REFERENCES sources(source),
    mwh      REAL NOT NULL,   -- energy generated in that hour (MWh)
    PRIMARY KEY (ts_utc, source)
);
CREATE TABLE consumption (
    ts_utc            TEXT PRIMARY KEY,
    ts_local          TEXT NOT NULL,
    grid_load_mwh     REAL,   -- total electricity consumption (Netzlast)
    residual_load_mwh REAL    -- grid load minus wind and solar (Residuallast)
);
CREATE TABLE prices (
    ts_utc      TEXT PRIMARY KEY,
    ts_local    TEXT NOT NULL,
    eur_per_mwh REAL NOT NULL  -- day-ahead price DE-LU; can be negative
);
CREATE INDEX idx_gen_local ON generation(ts_local);
"""


def fetch_json(url: str, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "energy-agent-portfolio/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_series(filter_id: int, region: str, weeks: int) -> list[tuple[int, float]]:
    """Return [(timestamp_ms, value)] for the last `weeks` weekly chunks, nulls dropped."""
    index = fetch_json(f"{BASE}/{filter_id}/{region}/index_hour.json")
    points = []
    for ts in index["timestamps"][-weeks:]:
        chunk = fetch_json(f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_hour_{ts}.json")
        points += [(t, v) for t, v in chunk["series"] if v is not None]
        time.sleep(0.1)  # be polite to a public service
    return points


def to_times(ts_ms: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M"), dt.astimezone(BERLIN).strftime("%Y-%m-%d %H:%M")


def build(conn: sqlite3.Connection, fetch, weeks: int) -> None:
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO sources VALUES (?, ?, ?)",
        [(name, cat, fid) for fid, (name, cat) in GENERATION.items()],
    )

    for fid, (name, _) in GENERATION.items():
        print(f"generation: {name}")
        conn.executemany(
            "INSERT OR REPLACE INTO generation VALUES (?, ?, ?, ?)",
            [(*to_times(t), name, v) for t, v in fetch(fid, "DE", weeks)],
        )

    print("consumption")
    load = dict(fetch(GRID_LOAD, "DE", weeks))
    resid = dict(fetch(RESIDUAL_LOAD, "DE", weeks))
    conn.executemany(
        "INSERT INTO consumption VALUES (?, ?, ?, ?)",
        [(*to_times(t), load.get(t), resid.get(t)) for t in sorted(load.keys() | resid.keys())],
    )

    print("prices")
    conn.executemany(
        "INSERT INTO prices VALUES (?, ?, ?)",
        [(*to_times(t), v) for t, v in fetch(PRICE, "DE-LU", weeks)],
    )
    conn.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weeks", type=int, default=52)
    p.add_argument("--db", default="energy.db")
    args = p.parse_args()
    with sqlite3.connect(args.db) as conn:
        build(conn, fetch_series, args.weeks)
        for table in ("generation", "consumption", "prices"):
            n, lo, hi = conn.execute(
                f"SELECT COUNT(*), MIN(ts_local), MAX(ts_local) FROM {table}"
            ).fetchone()
            print(f"{table:12} {n:>8} rows  {lo} -> {hi}")
    print(f"Saved {args.db}")


if __name__ == "__main__":
    main()
