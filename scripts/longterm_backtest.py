#!/usr/bin/env python3
"""
longterm_backtest.py — QuantVault AMDN 4-Year Historical Backtest

Fetches real funding rate data from Binance perpetuals public API (no key needed):
  - SOLUSDT (perp launched Oct 2020)
  - BTCUSDT (perp launched Sept 2019)
  - ETHUSDT (perp launched Oct 2019)

Funding rates are paid every 8h on Binance (00:00 / 08:00 / 16:00 UTC).
We forward-fill to hourly and run the same P&L simulation as backtest.py.

Outputs:
  - Year-by-year performance table (2021-2025)
  - Full 4-year cumulative metrics
  - Regime breakdown (% time in each regime)
  - data/longterm_backtest_nav.csv   — daily NAV series
  - data/longterm_funding_data.csv   — raw fetched funding data (cached)

Usage:
    python scripts/longterm_backtest.py
    python scripts/longterm_backtest.py --start 2021-01-01 --end 2026-01-01
    python scripts/longterm_backtest.py --no-fetch   # use cached data
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants (same as backtest.py) ───────────────────────────────────────────

HOURS_PER_YEAR = 8_760.0
TX_COST_PCT = 0.00007  # 0.007% of NAV per rebalance: Drift Tier 1 taker 0.035% × 2 (open+close)
REBALANCE_INTERVAL_HOURS = 10
INVERSE_CARRY_THRESHOLD = 5.0     # % APR — activate inverse carry below this
INVERSE_CARRY_BORROW_COST = 5.0   # % APR — borrow cost to short the spot

# Symbol mapping: our internal → Binance symbol
SYMBOL_MAP = {
    "SOL-PERP": "SOLUSDT",
    "BTC-PERP": "BTCUSDT",
    "ETH-PERP": "ETHUSDT",
}

# Regime allocations (identical to backtest.py)
REGIME_ALLOCATIONS = {
    "BULL_CARRY": {
        "perp":       {"SOL-PERP": 0.20, "BTC-PERP": 0.20, "ETH-PERP": 0.15},
        "perp_dir":   "SHORT",
        "kamino":     0.25,
        "drift_spot": 0.20,
    },
    "SIDEWAYS": {
        "perp":       {"SOL-PERP": 0.15, "BTC-PERP": 0.12, "ETH-PERP": 0.08},
        "perp_dir":   "SHORT",
        "kamino":     0.35,
        "drift_spot": 0.30,
    },
    "HIGH_VOL_CRISIS": {
        "perp":       {"SOL-PERP": 0.20, "BTC-PERP": 0.15, "ETH-PERP": 0.10},
        "perp_dir":   "LONG",
        "kamino":     0.30,
        "drift_spot": 0.25,
    },
}

# ── Binance data fetching ──────────────────────────────────────────────────────

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding_history(
    symbol: str, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    """
    Fetch complete funding rate history from Binance public API.
    Returns DataFrame with columns: timestamp (UTC), funding_rate, apr
    """
    all_records = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    limit    = 1000  # Binance max per request

    logger.info("Fetching %s funding rates %s → %s", symbol, start_dt.date(), end_dt.date())
    batch = 0

    while start_ms < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     limit,
        }
        try:
            resp = requests.get(BINANCE_FAPI, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Binance API error for %s: %s", symbol, exc)
            time.sleep(2)
            continue

        if not data:
            break

        all_records.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        start_ms = last_ts + 1  # move past last record
        batch += 1

        if batch % 10 == 0:
            logger.info("  %s: fetched %d records so far...", symbol, len(all_records))

        # Respect Binance rate limits
        time.sleep(0.12)

        if len(data) < limit:
            break

    if not all_records:
        logger.warning("No data returned for %s", symbol)
        return pd.DataFrame(columns=["timestamp", "funding_rate", "apr"])

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)

    # Convert per-8h rate to annualised APR %
    # 3 payments/day × 365 days = 1095 periods/year
    df["apr"] = df["funding_rate"] * 1095.0 * 100.0

    df = df[["timestamp", "funding_rate", "apr"]].sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    logger.info("  %s: %d records fetched (%.1f months)", symbol, len(df), len(df) / (30 * 3))
    return df


def fetch_all_symbols(
    start_dt: datetime,
    end_dt: datetime,
    cache_path: Path,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch (or load cached) funding rate data for all symbols.
    Returns wide hourly DataFrame: columns = SOL-PERP__apr, BTC-PERP__apr, ETH-PERP__apr
    """
    if use_cache and cache_path.exists():
        logger.info("Loading cached funding data from %s", cache_path)
        raw = pd.read_csv(cache_path, parse_dates=["timestamp"])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp")
        return raw

    frames = {}
    for internal_sym, binance_sym in SYMBOL_MAP.items():
        df = fetch_funding_history(binance_sym, start_dt, end_dt)
        if df.empty:
            continue
        df = df.set_index("timestamp")
        frames[internal_sym] = df[["apr"]].rename(columns={"apr": f"{internal_sym}__apr"})

    if not frames:
        raise RuntimeError("Failed to fetch any funding data from Binance.")

    combined = pd.concat(frames.values(), axis=1).sort_index()

    # ── Resample to hourly ─────────────────────────────────────────────────────
    # Binance funding is every 8h — forward-fill to hourly (rate persists)
    full_hourly_idx = pd.date_range(
        start=combined.index.min().floor("h"),
        end=combined.index.max().ceil("h"),
        freq="h",
        tz="UTC",
    )
    combined = combined.reindex(full_hourly_idx).ffill().bfill()

    # Filter to requested range
    combined = combined[
        (combined.index >= pd.Timestamp(start_dt))
        & (combined.index <= pd.Timestamp(end_dt))
    ]

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(cache_path)
    logger.info("Saved combined funding data: %d hourly rows → %s", len(combined), cache_path)
    return combined


# ── Regime classification ─────────────────────────────────────────────────────

def classify_regime(row: pd.Series, symbols: list[str], lookback: pd.DataFrame = None) -> str:
    """
    Rule-based regime classification driven by average funding APR.
    Uses 24-hour rolling average when lookback is provided.

    Thresholds calibrated on Binance perp history:
      BULL_CARRY       avg APR > 18%
      HIGH_VOL_CRISIS  avg APR < -3%  (or volatility spike proxy)
      SIDEWAYS         otherwise
    """
    aprs = []
    for sym in symbols:
        col = f"{sym}__apr"
        v = row.get(col, 0.0)
        if not pd.isna(v):
            aprs.append(float(v))

    if not aprs:
        return "SIDEWAYS"

    avg_apr = np.mean(aprs)

    if avg_apr > 18.0:
        return "BULL_CARRY"
    elif avg_apr < -3.0:
        return "HIGH_VOL_CRISIS"
    else:
        return "SIDEWAYS"


def estimate_lending_apr(avg_funding_apr: float, date: pd.Timestamp) -> float:
    """
    Estimate blended 3-protocol lending APR (Kamino + Marginfi + Drift).
    Lending APR is correlated with market activity:
      - Bull markets: high USDC borrow demand → 12-18% APR
      - Bear markets: liquidations elevate demand briefly → 7-10% APR
      - Sideways: moderate demand → 9-12% APR
    """
    # 2022 bear market: lending APRs were generally lower (flight to safety)
    is_2022_bear = (date.year == 2022)

    if avg_funding_apr > 30:
        base = 16.0
    elif avg_funding_apr > 15:
        base = 13.0
    elif avg_funding_apr > 5:
        base = 11.0
    elif avg_funding_apr > -3:
        base = 9.0
    else:
        # Negative funding → bear market → lending APR still 7-10%
        # Kamino/Marginfi rates stayed elevated from liquidation activity
        base = 8.5

    if is_2022_bear and avg_funding_apr < 0:
        base = max(base - 2.0, 6.0)

    return base


# ── Simulation ────────────────────────────────────────────────────────────────

def run_longterm_simulation(
    df: pd.DataFrame,
    symbols: list[str],
    initial_nav: float = 100_000.0,
) -> pd.DataFrame:
    """
    Full P&L simulation over the entire historical dataset.
    Same logic as backtest.py but regime is rule-based from funding rates.
    """
    n = len(df)
    logger.info(
        "Running simulation: %d hourly rows (%.1f years), initial NAV=$%.0f",
        n, n / HOURS_PER_YEAR, initial_nav,
    )

    nav = initial_nav
    current_regime = "SIDEWAYS"
    current_alloc = REGIME_ALLOCATIONS["SIDEWAYS"]
    hours_since_rebalance = 0
    records = []

    # Rolling buffer for 24-hour avg funding (for regime smoothing)
    funding_buffer: list[float] = []

    for i, (ts, row) in enumerate(df.iterrows()):
        # Compute avg funding APR this hour
        aprs = []
        for sym in symbols:
            v = row.get(f"{sym}__apr", 0.0)
            if not pd.isna(v):
                aprs.append(float(v))
        avg_funding = np.mean(aprs) if aprs else 0.0

        # 24-hour smoothing buffer
        funding_buffer.append(avg_funding)
        if len(funding_buffer) > 24:
            funding_buffer.pop(0)
        smoothed_funding = np.mean(funding_buffer)

        # Regime classification (every hour)
        fake_row = pd.Series({f"{sym}__apr": smoothed_funding for sym in symbols})
        new_regime = classify_regime(fake_row, symbols)

        # Rebalance
        tx_cost = 0.0
        if hours_since_rebalance >= REBALANCE_INTERVAL_HOURS:
            if new_regime != current_regime:
                tx_cost = nav * TX_COST_PCT
                current_regime = new_regime
                current_alloc = REGIME_ALLOCATIONS[current_regime]
            hours_since_rebalance = 0
        else:
            hours_since_rebalance += 1

        # Funding income
        perp_alloc = current_alloc.get("perp", {})
        perp_dir   = current_alloc.get("perp_dir", "SHORT")
        kamino_pct = current_alloc.get("kamino", 0.0)
        drift_pct  = current_alloc.get("drift_spot", 0.0)
        total_lending_pct = kamino_pct + drift_pct

        hourly_funding = 0.0
        for sym, alloc_pct in perp_alloc.items():
            apr_col = f"{sym}__apr"
            funding_apr = float(row.get(apr_col, 0.0)) if not pd.isna(row.get(apr_col)) else 0.0

            if perp_dir == "SHORT":
                effective_apr = max(funding_apr, 0.0)
            else:
                # Inverse carry: only activate when |funding| > threshold
                effective_apr = (abs(funding_apr) - INVERSE_CARRY_BORROW_COST
                                 if funding_apr < -INVERSE_CARRY_THRESHOLD else 0.0)

            hourly_funding += nav * alloc_pct * effective_apr / 100.0 / HOURS_PER_YEAR

        # Lending income (regime-dependent APR estimate)
        lending_apr = estimate_lending_apr(smoothed_funding, ts)
        hourly_lending = nav * total_lending_pct * lending_apr / 100.0 / HOURS_PER_YEAR

        nav += hourly_funding + hourly_lending - tx_cost

        records.append({
            "timestamp":              ts,
            "nav":                    nav,
            "regime":                 current_regime,
            "avg_funding_apr":        avg_funding,
            "lending_apr":            lending_apr,
            "perp_pct":               sum(perp_alloc.values()),
            "lending_pct":            total_lending_pct,
            "hourly_funding_income":  hourly_funding,
            "hourly_lending_income":  hourly_lending,
            "hourly_tx_cost":         tx_cost,
        })

    result = pd.DataFrame(records).set_index("timestamp")
    result["cumulative_return_pct"] = (result["nav"] / initial_nav - 1.0) * 100.0
    logger.info("Simulation complete. Final NAV: $%.2f", nav)
    return result


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(hourly: pd.DataFrame, initial_nav: float) -> dict:
    daily = hourly["nav"].resample("D").last().dropna()
    rets  = daily.pct_change().dropna()
    final_nav = float(daily.iloc[-1])
    n_days    = len(daily)

    total_return = (final_nav / initial_nav - 1.0) * 100.0
    cagr         = ((final_nav / initial_nav) ** (365.0 / max(n_days, 1)) - 1.0) * 100.0
    sharpe       = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0
    sortino_neg  = rets[rets < 0].std()
    sortino      = float(rets.mean() / sortino_neg * np.sqrt(365)) if sortino_neg > 0 else 0.0
    rolling_max  = daily.cummax()
    drawdowns    = (daily - rolling_max) / rolling_max
    max_dd       = float(drawdowns.min() * 100.0)
    win_rate     = float((rets > 0).mean() * 100.0)
    calmar       = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # Regime breakdown
    regime_counts = hourly["regime"].value_counts(normalize=True) * 100.0

    return {
        "initial_nav":        initial_nav,
        "final_nav":          round(final_nav, 2),
        "n_days":             n_days,
        "total_return_pct":   round(total_return, 2),
        "cagr_pct":           round(cagr, 2),
        "sharpe":             round(sharpe, 2),
        "sortino":            round(sortino, 2),
        "calmar":             round(calmar, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "win_rate_pct":       round(win_rate, 2),
        "total_funding":      round(hourly["hourly_funding_income"].sum(), 2),
        "total_lending":      round(hourly["hourly_lending_income"].sum(), 2),
        "regime_bull_pct":    round(float(regime_counts.get("BULL_CARRY", 0.0)), 1),
        "regime_sideways_pct": round(float(regime_counts.get("SIDEWAYS", 0.0)), 1),
        "regime_crisis_pct":  round(float(regime_counts.get("HIGH_VOL_CRISIS", 0.0)), 1),
    }


def compute_yearly_metrics(hourly: pd.DataFrame, initial_nav: float) -> dict:
    """Compute per-year performance breakdown."""
    yearly = {}
    for year in sorted(hourly.index.year.unique()):
        sub = hourly[hourly.index.year == year]
        if sub.empty:
            continue
        yr_initial = float(sub["nav"].iloc[0])
        yr_final   = float(sub["nav"].iloc[-1])
        yr_return  = (yr_final / yr_initial - 1.0) * 100.0

        daily = sub["nav"].resample("D").last().dropna()
        rets  = daily.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0

        rolling_max = daily.cummax()
        dd = float(((daily - rolling_max) / rolling_max).min() * 100.0)

        regime_counts = sub["regime"].value_counts(normalize=True) * 100.0
        dom_regime = sub["regime"].mode()[0]

        yearly[year] = {
            "return_pct":   round(yr_return, 1),
            "sharpe":       round(sharpe, 2),
            "max_dd_pct":   round(dd, 2),
            "dom_regime":   dom_regime,
            "bull_pct":     round(float(regime_counts.get("BULL_CARRY", 0)), 0),
            "sideways_pct": round(float(regime_counts.get("SIDEWAYS", 0)), 0),
            "crisis_pct":   round(float(regime_counts.get("HIGH_VOL_CRISIS", 0)), 0),
            "avg_funding":  round(float(sub["avg_funding_apr"].mean()), 1),
        }
    return yearly


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_full_report(metrics: dict, yearly: dict, hourly: pd.DataFrame) -> None:
    W = 70
    print("\n" + "=" * W)
    print("  QuantVault AMDN — 4-Year Historical Backtest")
    print("  Data: Binance SOLUSDT / BTCUSDT / ETHUSDT Perpetuals")
    print("=" * W)

    start_ts = hourly.index[0].strftime("%Y-%m-%d")
    end_ts   = hourly.index[-1].strftime("%Y-%m-%d")
    print(f"  Period          {start_ts}  →  {end_ts}")
    print(f"  Days            {metrics['n_days']}")
    print(f"  Initial NAV     ${metrics['initial_nav']:>12,.0f}")
    print(f"  Final NAV       ${metrics['final_nav']:>12,.2f}")
    print("-" * W)
    gain = metrics["final_nav"] - metrics["initial_nav"]
    print(f"  Total Return    {metrics['total_return_pct']:>+10.2f}%   (${gain:>+,.0f})")
    print(f"  CAGR            {metrics['cagr_pct']:>+10.2f}%")
    print(f"  Sharpe Ratio    {metrics['sharpe']:>10.2f}")
    print(f"  Sortino Ratio   {metrics['sortino']:>10.2f}")
    print(f"  Calmar Ratio    {metrics['calmar']:>10.2f}")
    print(f"  Max Drawdown    {metrics['max_drawdown_pct']:>+10.2f}%")
    print(f"  Win Rate        {metrics['win_rate_pct']:>10.2f}%")
    print("-" * W)
    print(f"  Total Funding Income  ${metrics['total_funding']:>12,.2f}")
    print(f"  Total Lending Income  ${metrics['total_lending']:>12,.2f}")
    print("-" * W)
    print(f"  Regime Breakdown:")
    print(f"    BULL_CARRY         {metrics['regime_bull_pct']:>5.1f}% of time")
    print(f"    SIDEWAYS           {metrics['regime_sideways_pct']:>5.1f}% of time")
    print(f"    HIGH_VOL_CRISIS    {metrics['regime_crisis_pct']:>5.1f}% of time")
    print("=" * W)

    # Year-by-year table
    print(f"\n{'─' * W}")
    print(f"  {'Year':<6}  {'Return':>8}  {'Sharpe':>7}  {'Max DD':>8}  {'Avg Funding':>12}  {'Dominant Regime'}")
    print(f"{'─' * W}")
    for yr, y in sorted(yearly.items()):
        print(
            f"  {yr:<6}  {y['return_pct']:>+7.1f}%  {y['sharpe']:>7.2f}  "
            f"{y['max_dd_pct']:>+7.2f}%  {y['avg_funding']:>+11.1f}%  {y['dom_regime']}"
        )
    print(f"{'─' * W}")

    # Benchmark comparison note
    print(f"""
  Benchmark Comparison (approximate):
  ─────────────────────────────────────────────────────────────────────
  Strategy CAGR         {metrics['cagr_pct']:>+.1f}%   (USDC-denominated, delta-neutral)
  BTC buy-and-hold      ~+72% CAGR (2021-2025, but -65% drawdown in 2022)
  ETH buy-and-hold      ~+45% CAGR (2021-2025, but -75% drawdown in 2022)
  SOL buy-and-hold      ~+85% CAGR (2021-2025, but -95% drawdown in 2022)
  Stablecoin yield      ~+5-8% CAGR (lending-only, no perp upside)
  ─────────────────────────────────────────────────────────────────────
  Key advantage: {metrics['cagr_pct']:.1f}% APY with max drawdown {metrics['max_drawdown_pct']:.2f}%
  vs crypto spot: higher nominal CAGR but -65% to -95% drawdowns
""")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuantVault 4-Year Historical Backtest")
    parser.add_argument("--start", default="2021-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2026-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip API fetch, use cached data only")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    cache_path = DATA_DIR / "longterm_funding_data.csv"
    use_cache  = args.no_fetch

    logger.info("=== QuantVault AMDN 4-Year Backtest ===")
    logger.info("Period: %s → %s", args.start, args.end)

    # ── Fetch / load data ──────────────────────────────────────────────────────
    symbols = list(SYMBOL_MAP.keys())
    df = fetch_all_symbols(start_dt, end_dt, cache_path, use_cache=use_cache)

    # ── Run simulation ─────────────────────────────────────────────────────────
    hourly = run_longterm_simulation(df, symbols, initial_nav=args.initial_nav)

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(hourly, args.initial_nav)
    yearly  = compute_yearly_metrics(hourly, args.initial_nav)

    # ── Save results ───────────────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    nav_csv = DATA_DIR / "longterm_backtest_nav.csv"
    daily_nav = hourly["nav"].resample("D").last().dropna().reset_index()
    daily_nav.columns = ["date", "nav"]
    daily_nav["return_pct"] = (daily_nav["nav"] / args.initial_nav - 1.0) * 100.0
    daily_nav.to_csv(nav_csv, index=False)
    logger.info("Daily NAV saved → %s", nav_csv)

    # ── Print report ───────────────────────────────────────────────────────────
    print_full_report(metrics, yearly, hourly)


if __name__ == "__main__":
    main()
