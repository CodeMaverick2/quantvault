"""Feature engineering for funding rate regime detection and signal generation."""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw funding rate DataFrame.

    Expected columns: funding_rate, mark_twap, oracle_twap, basis_pct, apr

    Returns DataFrame with features suitable for HMM training and signal generation.
    """
    df = df.copy().sort_values("ts").reset_index(drop=True)

    fr = df["funding_rate"]
    basis = df["basis_pct"]

    # --- Funding rate features ---
    df["fr_log_diff"] = np.log(fr.abs() + 1e-9).diff()
    df["fr_sign"] = np.sign(fr)

    for h in [6, 24, 72, 168]:  # 6h, 1d, 3d, 7d rolling windows
        df[f"fr_mean_{h}h"] = fr.rolling(h).mean()
        df[f"fr_std_{h}h"] = fr.rolling(h).std()
        df[f"fr_z_{h}h"] = (fr - df[f"fr_mean_{h}h"]) / (df[f"fr_std_{h}h"] + 1e-12)

    # Funding rate momentum (rate-of-change)
    df["fr_mom_6h"] = fr.diff(6)
    df["fr_mom_24h"] = fr.diff(24)

    # Autocorrelation proxies (lag-1 and lag-24)
    df["fr_lag1"] = fr.shift(1)
    df["fr_lag24"] = fr.shift(24)

    # --- Basis features ---
    for h in [12, 48]:
        df[f"basis_mean_{h}h"] = basis.rolling(h).mean()
        df[f"basis_std_{h}h"] = basis.rolling(h).std()
        df[f"basis_z_{h}h"] = (basis - df[f"basis_mean_{h}h"]) / (
            df[f"basis_std_{h}h"] + 1e-12
        )

    # Basis momentum
    df["basis_mom_6h"] = basis.diff(6)

    # --- Price features (from oracle twap) ---
    price = df["oracle_twap"]
    df["price_return_1h"] = np.log(price / price.shift(1))
    df["price_return_4h"] = np.log(price / price.shift(4))
    df["price_return_24h"] = np.log(price / price.shift(24))

    # Normalized range (proxy for realized vol)
    if "high" in df.columns and "low" in df.columns:
        df["norm_range"] = (df["high"] - df["low"]) / (df["oracle_twap"] + 1e-9)
    else:
        df["norm_range"] = df["price_return_1h"].abs()

    df["vol_6h"] = df["price_return_1h"].rolling(6).std()
    df["vol_24h"] = df["price_return_1h"].rolling(24).std()

    # Vol ratio: short-term vs long-term (regime signal)
    df["vol_ratio"] = df["vol_6h"] / (df["vol_24h"] + 1e-12)

    return df


def get_hmm_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the feature columns used for HMM training.
    Returns array of shape (n_samples, n_features) with NaN rows dropped.
    """
    cols = [
        "fr_log_diff",
        "fr_z_24h",
        "fr_mom_24h",
        "basis_pct",
        "basis_z_48h",
        "price_return_1h",
        "price_return_4h",
        "vol_ratio",
        "norm_range",
    ]
    available = [c for c in cols if c in df.columns]
    sub = df[available].replace([np.inf, -np.inf], np.nan).dropna()
    return sub.values, sub.index


def get_lstm_feature_matrix(
    df: pd.DataFrame,
    sequence_length: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for LSTM training.

    X: (n_samples, sequence_length, n_features)
    y: (n_samples,) — 1 if next-period funding is positive, 0 otherwise
    """
    cols = [
        "funding_rate",
        "fr_z_6h",
        "fr_z_24h",
        "fr_mom_6h",
        "basis_pct",
        "price_return_1h",
        "price_return_4h",
        "vol_6h",
        "vol_ratio",
    ]
    available = [c for c in cols if c in df.columns]
    sub = df[available + ["funding_rate"]].replace([np.inf, -np.inf], np.nan).dropna()

    features = sub[available].values
    target = (sub["funding_rate"].shift(-1) > 0).astype(int).values

    X, y = [], []
    for i in range(sequence_length, len(features) - 1):
        X.append(features[i - sequence_length : i])
        y.append(target[i])

    if not X:
        return np.empty((0, sequence_length, len(available))), np.empty(0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def compute_funding_apr_composite(
    market_dfs: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Compute a composite funding APR across multiple markets (weighted average).
    Useful for multi-asset allocation decisions.
    """
    if weights is None:
        n = len(market_dfs)
        weights = {sym: 1.0 / n for sym in market_dfs}

    aligned: dict[str, pd.Series] = {}
    for sym, df in market_dfs.items():
        if df.empty or "apr" not in df.columns:
            continue
        s = df.set_index("ts")["apr"]
        aligned[sym] = s

    if not aligned:
        return pd.Series(dtype=float)

    combined = pd.DataFrame(aligned)
    composite = sum(combined[sym] * w for sym, w in weights.items() if sym in combined)
    return composite.rename("composite_apr")
