#!/usr/bin/env python3
"""
Collect historical funding rate data from Drift Data API for HMM training.
Saves data to data/funding_rates_{symbol}.csv

Usage:
    python scripts/collect_training_data.py [--days 90] [--symbols SOL-PERP BTC-PERP ETH-PERP]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategy.signals.drift_data import DriftDataClient
from strategy.signals.funding_features import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["SOL-PERP", "BTC-PERP", "ETH-PERP"]
DEFAULT_DAYS = 90
DATA_DIR = Path(__file__).parent.parent / "data"


async def fetch_market_data(
    symbol: str,
    days: int,
    client: DriftDataClient,
) -> pd.DataFrame:
    """Fetch all available funding rate + candle data for a symbol."""
    logger.info("Fetching %d days of data for %s via pagination...", days, symbol)

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    min_ts = int(start_dt.timestamp())

    try:
        all_funding = await client.get_funding_rates_paginated(symbol, min_ts)
    except Exception as e:
        logger.warning("Paginated fetch failed for %s, falling back to daily: %s", symbol, e)
        all_funding = []
        current = start_dt
        while current <= end_dt:
            try:
                records = await client.get_funding_rates_date(
                    symbol, current.year, current.month, current.day
                )
                all_funding.extend(records)
            except Exception as ex:
                logger.warning("Failed to fetch %s for %s: %s", symbol, current.date(), ex)
            current += timedelta(days=1)
            await asyncio.sleep(0.1)

    if not all_funding:
        logger.warning("No data fetched for %s", symbol)
        return pd.DataFrame()

    rows = []
    for r in all_funding:
        rows.append({
            "ts": r.ts,
            "datetime": datetime.fromtimestamp(r.ts, tz=timezone.utc).isoformat(),
            "funding_rate": r.funding_rate,
            "funding_rate_long": r.funding_rate_long,
            "funding_rate_short": r.funding_rate_short,
            "mark_twap": r.mark_twap,
            "oracle_twap": r.oracle_twap,
            "basis_pct": r.basis_pct,
            "hourly_rate": r.hourly_rate,
            "apr": r.apr,
            "period_revenue": r.period_revenue,
        })

    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    logger.info("%s: collected %d records from %s to %s", symbol, len(df),
                df["datetime"].iloc[0], df["datetime"].iloc[-1])
    return df


async def main(symbols: list[str], days: int) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    summary = {}

    async with DriftDataClient() as client:
        for symbol in symbols:
            df = await fetch_market_data(symbol, days, client)
            if df.empty:
                logger.warning("Skipping %s — no data", symbol)
                continue

            # Build features
            enriched = build_features(df)

            # Save raw + enriched
            raw_path = DATA_DIR / f"funding_raw_{symbol.replace('-', '_').lower()}.csv"
            enriched_path = DATA_DIR / f"funding_features_{symbol.replace('-', '_').lower()}.csv"

            df.to_csv(raw_path, index=False)
            enriched.to_csv(enriched_path, index=False)

            summary[symbol] = {
                "records": len(df),
                "start": df["datetime"].iloc[0],
                "end": df["datetime"].iloc[-1],
                "mean_apr": enriched["apr"].mean(),
                "std_apr": enriched["apr"].std(),
                "positive_pct": (enriched["funding_rate"] > 0).mean(),
            }
            logger.info("Saved: %s (%d records)", raw_path, len(df))

    print("\n=== Data Collection Summary ===")
    for sym, stats in summary.items():
        print(f"\n{sym}:")
        print(f"  Records: {stats['records']}")
        print(f"  Period: {stats['start']} → {stats['end']}")
        print(f"  Mean APR: {stats['mean_apr']:.1f}%")
        print(f"  APR Std: {stats['std_apr']:.1f}%")
        print(f"  Positive funding %: {stats['positive_pct']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Drift funding rate data")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    args = parser.parse_args()
    asyncio.run(main(args.symbols, args.days))
