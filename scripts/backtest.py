#!/usr/bin/env python3
"""
backtest.py — QuantVault AMDN Strategy Backtester

Simulates 90 days of strategy operation over historical data:
  - Hourly: update HMM features, predict regime
  - Every 10 hours: rebalance allocation (simulated)
  - Track PnL:
      funding_income  = sum(perp_allocation  * funding_apr / 8760) per hour
      lending_income  = sum(lending_allocation * lending_apr / 8760) per hour
      tx_costs        = 0.001% per rebalance

Outputs:
  - Summary statistics (total return, Sharpe, max drawdown, win rate, avg daily APR)
  - data/backtest_results.csv  with hourly/daily NAV series

Usage:
    python scripts/backtest.py [--days 90] [--initial-nav 100000]
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
HMM_MODEL_PATH = MODEL_DIR / "hmm_regime.pkl"
RESULTS_CSV = DATA_DIR / "backtest_results.csv"

# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOLS = ["SOL-PERP", "BTC-PERP", "ETH-PERP"]

# Allocation by regime (fractions of NAV, must sum ≤ 1)
REGIME_ALLOCATIONS: dict[str, dict] = {
    "BULL_CARRY": {
        "perp":    {"SOL-PERP": 0.20, "BTC-PERP": 0.20, "ETH-PERP": 0.15},
        "kamino":  0.25,
        "drift_spot": 0.20,
    },
    "SIDEWAYS": {
        "perp":    {"SOL-PERP": 0.10, "BTC-PERP": 0.10, "ETH-PERP": 0.05},
        "kamino":  0.35,
        "drift_spot": 0.25,
    },
    "HIGH_VOL_CRISIS": {
        "perp":    {},
        "kamino":  0.40,
        "drift_spot": 0.30,
    },
}

TX_COST_PCT = 0.00001          # 0.001% per rebalance
REBALANCE_INTERVAL_HOURS = 10
DEFAULT_LENDING_APR = 8.0      # fallback if no lending_apr column (% annual)
HOURS_PER_YEAR = 8_760.0

# ── Regime helpers ────────────────────────────────────────────────────────────

MOCK_REGIME_WEIGHTS = {
    "BULL_CARRY": 0.50,
    "SIDEWAYS": 0.35,
    "HIGH_VOL_CRISIS": 0.15,
}

REGIME_LABELS = list(MOCK_REGIME_WEIGHTS.keys())


def _mock_regime_sequence(n: int, seed: int = 42) -> list[str]:
    """
    Generate a plausible Markov-chain regime sequence for mocking when the
    HMM model is unavailable.
    """
    rng = np.random.default_rng(seed)
    # Simple transition matrix: regimes tend to persist
    transition = {
        "BULL_CARRY":      {"BULL_CARRY": 0.92, "SIDEWAYS": 0.07, "HIGH_VOL_CRISIS": 0.01},
        "SIDEWAYS":        {"BULL_CARRY": 0.10, "SIDEWAYS": 0.85, "HIGH_VOL_CRISIS": 0.05},
        "HIGH_VOL_CRISIS": {"BULL_CARRY": 0.05, "SIDEWAYS": 0.20, "HIGH_VOL_CRISIS": 0.75},
    }
    weights = list(MOCK_REGIME_WEIGHTS.values())
    state = rng.choice(REGIME_LABELS, p=weights)
    seq = [state]
    for _ in range(n - 1):
        t = transition[state]
        state = rng.choice(list(t.keys()), p=list(t.values()))
        seq.append(state)
    return seq


# ── Data loading ──────────────────────────────────────────────────────────────

def load_feature_data(symbols: list[str]) -> pd.DataFrame:
    """
    Load enriched funding-rate feature CSVs produced by collect_training_data.py.
    Returns a wide DataFrame indexed by timestamp (hourly), containing per-symbol
    funding APR and (optional) lending APR columns.
    """
    frames = {}
    for sym in symbols:
        slug = sym.replace("-", "_").lower()
        p = DATA_DIR / f"funding_features_{slug}.csv"
        if not p.exists():
            logger.warning("No feature file found for %s at %s", sym, p)
            continue
        df = pd.read_csv(p, parse_dates=["datetime"])
        df = df.set_index("datetime").sort_index()
        # Keep only columns we need
        keep = [c for c in ["apr", "funding_rate", "lending_apr"] if c in df.columns]
        frames[sym] = df[keep].add_prefix(f"{sym}__")

    if not frames:
        raise FileNotFoundError(
            f"No feature CSV files found in {DATA_DIR}. "
            "Run scripts/collect_training_data.py first."
        )

    combined = pd.concat(frames.values(), axis=1).sort_index().dropna(how="all")
    logger.info("Loaded combined data: %d hourly rows, %d columns", len(combined), combined.shape[1])
    return combined


# ── HMM regime prediction ─────────────────────────────────────────────────────

def load_hmm_model() -> Optional[object]:
    """Load a pickled HMM classifier if available."""
    if not HMM_MODEL_PATH.exists():
        logger.warning("HMM model not found at %s — using mock regime sequence", HMM_MODEL_PATH)
        return None
    try:
        with open(HMM_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("HMM model loaded from %s", HMM_MODEL_PATH)
        return model
    except Exception as exc:
        logger.warning("Failed to load HMM model: %s — using mock regime sequence", exc)
        return None


def build_hmm_features(row: pd.Series, symbols: list[str]) -> np.ndarray:
    """
    Extract the feature vector used by the HMM from a row of the combined df.
    Must match what was used during training (funding APR per symbol, basis).
    """
    features = []
    for sym in symbols:
        apr_col = f"{sym}__apr"
        features.append(float(row.get(apr_col, 0.0)) if not pd.isna(row.get(apr_col)) else 0.0)
    return np.array(features, dtype=np.float64).reshape(1, -1)


def predict_regime_hmm(model, features: np.ndarray) -> tuple[str, float]:
    """Return (regime_label, confidence) using the loaded HMM model."""
    try:
        result = model.predict(features)
        # Support both objects with .regime/.confidence and plain string returns
        if hasattr(result, "regime"):
            label = result.regime.name if hasattr(result.regime, "name") else str(result.regime)
            conf = float(result.confidence) if hasattr(result, "confidence") else 1.0
        else:
            label = str(result)
            conf = 1.0
        if label not in REGIME_LABELS:
            label = "SIDEWAYS"
        return label, conf
    except Exception as exc:
        logger.debug("HMM predict error: %s", exc)
        return "SIDEWAYS", 0.5


# ── Allocation helpers ────────────────────────────────────────────────────────

def get_allocation(regime: str) -> dict:
    return REGIME_ALLOCATIONS.get(regime, REGIME_ALLOCATIONS["SIDEWAYS"])


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(
    df: pd.DataFrame,
    hmm_model: Optional[object],
    days: int,
    initial_nav: float,
    symbols: list[str],
) -> pd.DataFrame:
    """
    Core simulation loop.

    Returns a DataFrame with columns:
      timestamp, nav, regime, perp_pct, lending_pct,
      hourly_funding_income, hourly_lending_income, hourly_tx_cost,
      cumulative_return_pct
    """
    # Restrict to the last `days` of data
    if len(df) == 0:
        raise ValueError("Combined feature DataFrame is empty after filtering")

    cutoff = df.index[-1] - pd.Timedelta(days=days)
    df_sim = df[df.index >= cutoff].copy()

    if len(df_sim) < 24:
        logger.warning(
            "Only %d rows in simulation window (need ≥24). Using all available data.",
            len(df_sim),
        )
        df_sim = df.copy()

    n_hours = len(df_sim)
    logger.info(
        "Simulating %d hours (%.1f days) of strategy operation, initial NAV=$%.0f",
        n_hours, n_hours / 24, initial_nav,
    )

    # If no HMM model, generate a mock regime sequence up front
    mock_regimes: Optional[list[str]] = None
    if hmm_model is None:
        mock_regimes = _mock_regime_sequence(n_hours)

    # ── State ─────────────────────────────────────────────────────────────────
    nav = initial_nav
    current_regime = "SIDEWAYS"
    current_alloc = get_allocation(current_regime)
    hours_since_rebalance = 0
    rebalance_count = 0

    records = []

    for i, (ts, row) in enumerate(df_sim.iterrows()):
        # ── Update regime every hour ─────────────────────────────────────────
        if hmm_model is not None:
            features = build_hmm_features(row, symbols)
            current_regime, _ = predict_regime_hmm(hmm_model, features)
        else:
            current_regime = mock_regimes[i]  # type: ignore[index]

        # ── Rebalance every REBALANCE_INTERVAL_HOURS ─────────────────────────
        tx_cost = 0.0
        if hours_since_rebalance >= REBALANCE_INTERVAL_HOURS:
            new_alloc = get_allocation(current_regime)
            if new_alloc != current_alloc:
                tx_cost = nav * TX_COST_PCT
                current_alloc = new_alloc
                rebalance_count += 1
                logger.debug("Rebalance #%d at %s — regime=%s", rebalance_count, ts, current_regime)
            hours_since_rebalance = 0
        else:
            hours_since_rebalance += 1

        # ── Compute hourly income ─────────────────────────────────────────────
        perp_alloc: dict[str, float] = current_alloc.get("perp", {})
        kamino_pct: float = current_alloc.get("kamino", 0.0)
        drift_spot_pct: float = current_alloc.get("drift_spot", 0.0)

        total_perp_pct = sum(perp_alloc.values())
        total_lending_pct = kamino_pct + drift_spot_pct

        # Funding income: short perp collects positive funding
        # funding_apr is in percentage units (e.g., 15.0 = 15%), so divide by 100
        hourly_funding = 0.0
        for sym, alloc_pct in perp_alloc.items():
            apr_col = f"{sym}__apr"
            funding_apr = float(row.get(apr_col, 0.0)) if not pd.isna(row.get(apr_col)) else 0.0
            hourly_funding += nav * alloc_pct * max(funding_apr, 0.0) / 100.0 / HOURS_PER_YEAR

        # Lending income: kamino + drift spot (use per-symbol lending APR if available)
        # DEFAULT_LENDING_APR is in percentage units (e.g., 8.0 = 8%), so divide by 100
        lending_apr_col = f"{symbols[0]}__lending_apr"
        lending_apr = (
            float(row.get(lending_apr_col, DEFAULT_LENDING_APR))
            if not pd.isna(row.get(lending_apr_col, np.nan))
            else DEFAULT_LENDING_APR
        )
        hourly_lending = nav * total_lending_pct * lending_apr / 100.0 / HOURS_PER_YEAR

        # ── Update NAV ────────────────────────────────────────────────────────
        nav += hourly_funding + hourly_lending - tx_cost

        records.append(
            {
                "timestamp": ts,
                "nav": nav,
                "regime": current_regime,
                "perp_pct": total_perp_pct,
                "lending_pct": total_lending_pct,
                "hourly_funding_income": hourly_funding,
                "hourly_lending_income": hourly_lending,
                "hourly_tx_cost": tx_cost,
            }
        )

    result = pd.DataFrame(records).set_index("timestamp")
    result["cumulative_return_pct"] = (result["nav"] / initial_nav - 1.0) * 100.0

    logger.info(
        "Simulation complete: %d hours, %d rebalances, final NAV=$%.2f",
        n_hours, rebalance_count, nav,
    )
    return result


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_daily_nav(hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly NAV to daily close."""
    daily = hourly["nav"].resample("D").last().dropna()
    return daily.to_frame(name="nav")


def compute_metrics(hourly: pd.DataFrame, initial_nav: float) -> dict:
    """Compute standard performance metrics from the hourly simulation result."""
    daily = compute_daily_nav(hourly)
    daily_returns = daily["nav"].pct_change().dropna()

    # Total return
    final_nav = daily["nav"].iloc[-1]
    total_return_pct = (final_nav / initial_nav - 1.0) * 100.0

    # Annualised Sharpe (assume 0 risk-free rate; daily returns)
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365.0)
    else:
        sharpe = 0.0

    # Maximum drawdown
    rolling_max = daily["nav"].cummax()
    drawdowns = (daily["nav"] - rolling_max) / rolling_max
    max_drawdown_pct = float(drawdowns.min() * 100.0)

    # Win rate: % of days with positive return
    win_rate_pct = float((daily_returns > 0).mean() * 100.0)

    # Average daily APR (annualise average daily return)
    avg_daily_apr = float(daily_returns.mean() * 365.0 * 100.0)

    # Funding vs lending breakdown
    total_funding = hourly["hourly_funding_income"].sum()
    total_lending = hourly["hourly_lending_income"].sum()
    total_tx_cost = hourly["hourly_tx_cost"].sum()

    return {
        "total_return_pct": round(total_return_pct, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "win_rate_pct": round(win_rate_pct, 2),
        "avg_daily_apr_pct": round(avg_daily_apr, 4),
        "final_nav": round(final_nav, 2),
        "initial_nav": round(initial_nav, 2),
        "total_funding_income": round(total_funding, 2),
        "total_lending_income": round(total_lending, 2),
        "total_tx_cost": round(total_tx_cost, 4),
        "n_days": len(daily),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 52


def print_summary(metrics: dict) -> None:
    """Print a formatted performance summary table."""
    print(f"\n{SEPARATOR}")
    print("  QuantVault AMDN Backtest Results")
    print(SEPARATOR)
    print(f"  {'Period':30s} {metrics['n_days']} days")
    print(f"  {'Initial NAV':30s} ${metrics['initial_nav']:>12,.2f}")
    print(f"  {'Final NAV':30s} ${metrics['final_nav']:>12,.2f}")
    print(SEPARATOR)
    print(f"  {'Total Return':30s} {metrics['total_return_pct']:>+10.2f}%")
    print(f"  {'Sharpe Ratio (ann.)':30s} {metrics['sharpe_ratio']:>11.4f}")
    print(f"  {'Max Drawdown':30s} {metrics['max_drawdown_pct']:>+10.2f}%")
    print(f"  {'Win Rate':30s} {metrics['win_rate_pct']:>10.2f}%")
    print(f"  {'Avg Daily APR':30s} {metrics['avg_daily_apr_pct']:>+10.2f}%")
    print(SEPARATOR)
    print(f"  {'Total Funding Income':30s} ${metrics['total_funding_income']:>11,.2f}")
    print(f"  {'Total Lending Income':30s} ${metrics['total_lending_income']:>11,.2f}")
    print(f"  {'Total Tx Costs':30s} ${metrics['total_tx_cost']:>11,.4f}")
    print(SEPARATOR)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="QuantVault AMDN Strategy Backtester")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of trailing days to simulate (default: 90)",
    )
    parser.add_argument(
        "--initial-nav",
        type=float,
        default=100_000.0,
        help="Starting vault NAV in USD (default: 100000)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Perp market symbols to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_CSV,
        help="Path for backtest results CSV (default: data/backtest_results.csv)",
    )
    args = parser.parse_args()

    logger.info("=== QuantVault AMDN Backtester ===")
    logger.info("Days: %d | Initial NAV: $%.0f | Symbols: %s", args.days, args.initial_nav, args.symbols)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_feature_data(args.symbols)

    # ── Load or mock HMM model ─────────────────────────────────────────────────
    hmm_model = load_hmm_model()

    # ── Run simulation ─────────────────────────────────────────────────────────
    hourly_result = run_simulation(
        df=df,
        hmm_model=hmm_model,
        days=args.days,
        initial_nav=args.initial_nav,
        symbols=args.symbols,
    )

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(hourly_result, initial_nav=args.initial_nav)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    daily_nav = compute_daily_nav(hourly_result).reset_index()
    daily_nav.columns = ["date", "nav"]
    daily_nav["daily_return_pct"] = daily_nav["nav"].pct_change() * 100.0
    daily_nav["cumulative_return_pct"] = (daily_nav["nav"] / args.initial_nav - 1.0) * 100.0

    daily_nav.to_csv(args.output, index=False)
    logger.info("Daily NAV series saved to %s", args.output)

    # ── Print summary ──────────────────────────────────────────────────────────
    print_summary(metrics)
    print(f"\n  Results CSV: {args.output}\n")


if __name__ == "__main__":
    main()
