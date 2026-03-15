#!/usr/bin/env python3
"""
Train HMM regime classifier and LSTM+XGBoost funding predictor.

Usage:
    python scripts/train_models.py [--symbol SOL-PERP] [--validate]

Reads from data/ directory (run collect_training_data.py first).
Saves trained models to models/ directory.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategy.models.hmm_regime import HMMRegimeClassifier, MarketRegime
from strategy.models.lstm_signal import LSTMFundingPredictor
from strategy.signals.funding_features import (
    build_features,
    get_hmm_feature_matrix,
    get_lstm_feature_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"


def load_combined_data(symbols: list[str]) -> pd.DataFrame:
    """Load and combine feature data across all symbols."""
    dfs = []
    for sym in symbols:
        path = DATA_DIR / f"funding_features_{sym.replace('-', '_').lower()}.csv"
        if not path.exists():
            logger.warning("No data file found for %s at %s", sym, path)
            continue
        df = pd.read_csv(path)
        df["symbol"] = sym
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No data files found in {DATA_DIR}. Run collect_training_data.py first."
        )

    return pd.concat(dfs, ignore_index=True).sort_values("ts")


def train_hmm(df: pd.DataFrame, validate: bool = True) -> HMMRegimeClassifier:
    """Train and optionally validate the HMM regime classifier."""
    X, idx = get_hmm_feature_matrix(df)
    logger.info("Training HMM on %d samples, %d features", len(X), X.shape[1])

    if validate:
        # Walk-forward validation: train on 80%, validate on 20%
        split = int(len(X) * 0.80)
        X_train, X_val = X[:split], X[split:]
        clf = HMMRegimeClassifier(n_states=3, n_iter=1000)
        clf.fit(X_train)

        train_score = clf.score(X_train)
        val_score = clf.score(X_val)
        val_pred = clf.predict(X_val)

        logger.info("HMM validation:")
        logger.info("  Train log-likelihood: %.4f", train_score)
        logger.info("  Val log-likelihood:   %.4f", val_score)
        logger.info("  Predicted regime: %s (conf=%.2f)", val_pred.regime.name, val_pred.confidence)

    # Retrain on full data
    clf = HMMRegimeClassifier(
        n_states=3,
        n_iter=2000,
        model_path=MODEL_DIR / "hmm_regime.pkl",
    )
    clf.fit(X)

    # Print regime statistics
    regimes = clf.predict_sequence(X)
    for r in MarketRegime:
        count = sum(1 for reg in regimes if reg == r)
        pct = count / len(regimes) * 100
        logger.info("  Regime %-20s: %5d samples (%.1f%%)", r.name, count, pct)

    clf.save()
    logger.info("HMM model saved to %s", MODEL_DIR / "hmm_regime.pkl")
    return clf


def train_lstm(df: pd.DataFrame) -> LSTMFundingPredictor:
    """Train the LSTM+XGBoost funding direction predictor."""
    X_seq, y = get_lstm_feature_matrix(df, sequence_length=24)

    if len(X_seq) < 200:
        logger.warning(
            "Only %d samples available for LSTM training. Minimum is 200. "
            "Collecting more data is recommended.",
            len(X_seq)
        )
        if len(X_seq) < 50:
            logger.warning("Skipping LSTM training due to insufficient data.")
            return LSTMFundingPredictor(MODEL_DIR)

    logger.info("Training LSTM+XGBoost on %d samples", len(X_seq))
    logger.info("Class balance: positive=%.1f%%", y.mean() * 100)

    predictor = LSTMFundingPredictor(MODEL_DIR)
    predictor.fit(
        X_seq=X_seq,
        X_tab=np.empty((len(X_seq), 0)),  # no additional tabular features
        y=y,
        epochs=20,
        batch_size=64,
    )
    predictor.save()
    logger.info("LSTM+XGBoost model saved to %s", MODEL_DIR)

    # Quick validation on last 20%
    split = int(len(X_seq) * 0.80)
    signals = predictor.predict(X_seq[split:])
    accuracy = np.mean([
        (s.direction == "POSITIVE") == (y[split + i] == 1)
        for i, s in enumerate(signals)
        if s.direction != "UNCERTAIN"
    ])
    logger.info("LSTM validation accuracy (excluding uncertain): %.1f%%", accuracy * 100)

    return predictor


def main(symbols: list[str], validate: bool) -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    logger.info("Loading training data for: %s", symbols)
    df = load_combined_data(symbols)
    logger.info("Combined dataset: %d rows", len(df))

    # Apply feature engineering if not already done
    if "fr_z_24h" not in df.columns:
        logger.info("Applying feature engineering...")
        df = build_features(df)

    # Drop NaN rows
    df = df.dropna(subset=["funding_rate", "oracle_twap"]).reset_index(drop=True)
    logger.info("After cleaning: %d rows", len(df))

    logger.info("\n=== Training HMM Regime Classifier ===")
    train_hmm(df, validate=validate)

    logger.info("\n=== Training LSTM+XGBoost Funding Predictor ===")
    train_lstm(df)

    logger.info("\n=== Training Complete ===")
    logger.info("Models saved to: %s", MODEL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuantVault strategy models")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["SOL-PERP", "BTC-PERP", "ETH-PERP"]
    )
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()
    main(args.symbols, args.validate)
