"""
LSTM + XGBoost hybrid model for funding rate directional prediction.

Architecture:
  1. LSTM captures temporal dependencies in funding rate sequences
  2. XGBoost operates on LSTM hidden states + raw tabular features
  3. Output: P(funding_rate_next_period > 0)

This provides both temporal feature extraction and nonlinear interaction modeling.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingSignal:
    prob_positive: float     # P(next period funding > 0)
    direction: str           # "POSITIVE", "NEGATIVE", "UNCERTAIN"
    confidence: float        # abs(prob - 0.5) * 2 — higher = more certain
    raw_logit: float


class LSTMFundingPredictor:
    """
    Hybrid LSTM+XGBoost model predicting funding rate direction.

    The LSTM processes a sequence of market features, its hidden state
    is concatenated with current snapshot features, then passed to XGBoost.

    Falls back to a simple XGBoost-only model if torch is unavailable.
    """

    SEQUENCE_LENGTH = 24       # 24 hours of lookback
    LSTM_HIDDEN_SIZE = 128
    LSTM_LAYERS = 2
    DROPOUT = 0.2
    UNCERTAINTY_BAND = 0.60    # probs [0.40, 0.60] → UNCERTAIN

    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self._lstm: Optional[object] = None   # torch.nn.Module
        self._xgb: Optional[object] = None    # xgboost.XGBClassifier
        self._is_fitted: bool = False
        self._n_features: int = 0
        self._use_torch: bool = False

        # Try importing torch
        try:
            import torch
            self._use_torch = True
        except ImportError:
            logger.warning("PyTorch not available, will use XGBoost-only model")

    def fit(
        self,
        X_seq: np.ndarray,       # (n, seq_len, n_features)
        X_tab: np.ndarray,       # (n, n_tab_features) — current snapshot
        y: np.ndarray,           # (n,) binary labels
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> "LSTMFundingPredictor":
        """Train the LSTM+XGBoost pipeline."""
        if len(y) < 200:
            raise ValueError(f"Need ≥200 samples to train, got {len(y)}")

        self._n_features = X_seq.shape[2]

        if self._use_torch:
            lstm_features = self._train_lstm(X_seq, y, epochs, batch_size, lr)
        else:
            # Flatten sequence for XGBoost-only fallback
            lstm_features = X_seq.reshape(len(X_seq), -1)

        # Combine LSTM output with tabular features
        if X_tab.shape[0] == lstm_features.shape[0]:
            combined = np.hstack([lstm_features, X_tab])
        else:
            combined = lstm_features

        self._train_xgb(combined, y)
        self._is_fitted = True
        return self

    def _train_lstm(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> np.ndarray:
        """Train LSTM and return hidden state features."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        n, seq_len, n_feat = X_seq.shape

        class LSTMEncoder(nn.Module):
            def __init__(self, n_features, hidden_size, n_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features, hidden_size, n_layers,
                    batch_first=True, dropout=dropout if n_layers > 1 else 0.0
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                out, (h, _) = self.lstm(x)
                last_hidden = h[-1]
                logit = self.head(last_hidden).squeeze(-1)
                return logit, last_hidden

        model = LSTMEncoder(
            n_features=n_feat,
            hidden_size=self.LSTM_HIDDEN_SIZE,
            n_layers=self.LSTM_LAYERS,
            dropout=self.DROPOUT,
        )

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Class balance weight
        pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logit, _ = model(xb)
                loss = criterion(logit, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("LSTM epoch %d/%d — loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

        # Extract hidden states for XGBoost
        model.eval()
        with torch.no_grad():
            _, hidden = model(X_t)
            lstm_features = hidden.numpy()

        self._lstm = model
        return lstm_features  # (n, hidden_size)

    def _train_xgb(self, X: np.ndarray, y: np.ndarray) -> None:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        split = int(len(X) * 0.85)
        xgb.fit(
            X[:split], y[:split],
            eval_set=[(X[split:], y[split:])],
            verbose=False,
        )
        self._xgb = xgb
        logger.info("XGBoost trained. Best iteration: %d", xgb.best_iteration)

    def predict(
        self,
        X_seq: np.ndarray,    # (1 or n, seq_len, n_features)
        X_tab: np.ndarray | None = None,
    ) -> list[FundingSignal]:
        """Predict funding direction for one or more observations."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._use_torch and self._lstm is not None:
            import torch
            with torch.no_grad():
                X_t = torch.tensor(X_seq, dtype=torch.float32)
                _, hidden = self._lstm(X_t)
                lstm_feat = hidden.numpy()
        else:
            lstm_feat = X_seq.reshape(len(X_seq), -1)

        if X_tab is not None and X_tab.shape[0] == lstm_feat.shape[0]:
            combined = np.hstack([lstm_feat, X_tab])
        else:
            combined = lstm_feat

        probs = self._xgb.predict_proba(combined)[:, 1]  # P(positive)
        return [self._make_signal(float(p)) for p in probs]

    def _make_signal(self, prob: float) -> FundingSignal:
        if prob > self.UNCERTAINTY_BAND:
            direction = "POSITIVE"
        elif prob < (1.0 - self.UNCERTAINTY_BAND):
            direction = "NEGATIVE"
        else:
            direction = "UNCERTAIN"

        logit = float(np.log(prob / (1.0 - prob + 1e-9) + 1e-9))
        confidence = abs(prob - 0.5) * 2.0

        return FundingSignal(
            prob_positive=prob,
            direction=direction,
            confidence=confidence,
            raw_logit=logit,
        )

    def save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self._use_torch and self._lstm is not None:
            import torch
            torch.save(self._lstm.state_dict(), self.model_dir / "lstm_weights.pt")

        if self._xgb is not None:
            self._xgb.save_model(str(self.model_dir / "xgb_funding.json"))

        logger.info("Models saved to %s", self.model_dir)

    def load(self) -> "LSTMFundingPredictor":
        xgb_path = self.model_dir / "xgb_funding.json"
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")

        from xgboost import XGBClassifier
        self._xgb = XGBClassifier()
        self._xgb.load_model(str(xgb_path))
        self._is_fitted = True
        logger.info("Models loaded from %s", self.model_dir)
        return self
