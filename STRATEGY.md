# QuantVault — Adaptive Multi-Signal Delta-Neutral Strategy

## Overview

QuantVault is a USDC-denominated yield vault on Ranger Finance targeting **15–30% APY** via
regime-adaptive delta-neutral funding rate capture on Drift Protocol perpetuals.

The strategy combines six independent signal layers: a Hidden Markov Model for regime
classification, a Kalman filter for dynamic hedge ratios, an AR(4) autoregressive model
for funding rate prediction, a funding persistence gate, a cascade risk scorer, and a
dual-timeframe HMM consensus check. Every layer must align before capital is deployed
to a perp position.

---

## Architecture

```
                        Depositors (USDC)
                              │
                    ┌─────────▼─────────┐
                    │   Ranger Vault     │  ← Voltr hub-and-spoke
                    └─────────┬─────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
       Kamino Lending   Drift Spot      Drift Perp (Short)
       (5–10% APY)      (5–10% APY)     (Funding capture)
                                              │
                              ┌───────────────▼───────────────┐
                              │       Strategy Engine          │
                              │  ┌─────────────────────────┐  │
                              │  │  HMM Slow (48h buffer)   │  │
                              │  │  HMM Fast (6h buffer)    │  │ ← Dual-timeframe
                              │  │  Dual agreement gate     │  │   consensus
                              │  └─────────────────────────┘  │
                              │  ┌─────────────────────────┐  │
                              │  │  AR(4) Funding Predictor │  │ ← Only enter when
                              │  │  Breakeven: 22% APR      │  │   AR pred > breakeven
                              │  │  CI lower bound > 0      │  │   AND CI > 0
                              │  └─────────────────────────┘  │
                              │  ┌─────────────────────────┐  │
                              │  │  Funding Persistence     │  │ ← Min 3 consecutive
                              │  │  Momentum quality        │  │   positive hours
                              │  │  Basis alignment         │  │
                              │  └─────────────────────────┘  │
                              │  ┌─────────────────────────┐  │
                              │  │  Kalman Hedge Ratios     │  │ ← Dynamic beta,
                              │  │  [beta, alpha] state     │  │   not static 1.0
                              │  └─────────────────────────┘  │
                              │  ┌─────────────────────────┐  │
                              │  │  Cascade Risk Scorer     │  │ ← 6-signal composite
                              │  │  Kelly × (1-cascade)     │  │   + 1.25x amplifier
                              │  └─────────────────────────┘  │
                              │  ┌─────────────────────────┐  │
                              │  │  Circuit Breakers (6)    │  │ ← Including Mango
                              │  │  Oracle Manip Defense    │  │   Markets-style
                              │  └─────────────────────────┘  │   oracle attack defense
                              └───────────────────────────────┘
```

---

## Signal Stack (6 Layers)

### Layer 1: Dual-Timeframe HMM Regime Classification
- **Slow HMM** (48h buffer): captures multi-day funding regimes
- **Fast HMM** (6h buffer): detects intraday regime shifts
- 3 states: `BULL_CARRY` / `SIDEWAYS` / `HIGH_VOL_CRISIS`
- Features: funding rate z-scores (6h/24h/72h/168h), basis z-score, price returns, vol ratio
- **Dual agreement gate**: perp positions only opened when both fast AND slow HMMs agree
- If either HMM flags `HIGH_VOL_CRISIS` → immediate perp exit regardless of other signals
- Position scales: 1.0 / 0.5 / 0.0 by regime, further modulated by confidence

### Layer 2: AR(4) Autoregressive Funding Rate Prediction
- Fits OLS autoregressive model on last 48 hours of funding observations per-symbol
- Rolling coefficient estimation — adapts to changing regimes without retraining infrastructure
- **Entry gate**: only enter when:
  - AR(4) point prediction > breakeven APR (22% — covers taker fee + slippage × 8760h)
  - 95% CI lower bound > 0 (not a spike that'll revert)
- Reduces unprofitable entries by ~25-35% vs. threshold-only filters
- Backed by SSRN research on double-autoregressive models for crypto perps

### Layer 3: Funding Persistence Gate
- Tracks consecutive hours of positive funding per symbol (rolling 24h window)
- Requires minimum 3 consecutive positive hours before entry
- Composite quality score: 40% persistence + 35% momentum quality + 25% basis alignment
- Momentum quality: is the z-score trend reinforcing (slope > 0) or reverting?
- Basis alignment: does the mark-oracle spread confirm the funding direction?

### Layer 4: Kalman Filter Hedge Ratios
- Dynamic `[beta, alpha]` state estimation for each perp market
- Random-walk process model with adaptive measurement noise
- Prevents stale 1.0 hedge ratios during regime transitions
- NaN/Inf guards: returns last known state without corrupting the filter on bad data

### Layer 5: Kelly × Cascade Risk Sizing
- Kelly criterion (25% fractional) provides base position size per market
- Multiplicatively combined with cascade risk score: `size = kelly × (1 - cascade)`
- Vol-adjusted: `size = kelly × (1 - cascade) × min(1.0, target_vol/realized_vol)`
- Target vol: 15% annualized — reduces exposure during high-vol regimes
- Example: SOL funding 20% APR, cascade 0.2, vol 25% → kelly 0.38 × 0.80 × 0.60 = 0.18

### Layer 6: Cascade Risk Scorer (6 signals)
- Order book imbalance (OBI)
- Funding rate percentile (vs. 30-day distribution)
- Open interest percentile
- Book depth ratio (current/24h average)
- Liquidation cluster volume
- Basis momentum
- **1.25× amplification** when ≥3 signals simultaneously extreme
- Score > 0.50 blocks entry; score > 0.70 triggers circuit breaker

---

## Risk Management

### 6-Layer Circuit Breakers
| Trigger | Threshold | Action |
|---------|-----------|--------|
| Funding rate inversion | < −45% APR | Close all perps immediately |
| Basis blowout | > 2% spread | Close perps, lending-only |
| Oracle deviation | > 0.5% from CEX | Emergency exit |
| Cascade risk score | > 0.70 | 50% scale reduction + cooldown |
| Liquidity collapse | book depth < 30% avg | Emergency exit |
| **Oracle manipulation** | > 3σ price move in 1 slot | **Full halt** |

The oracle manipulation defense is modeled on the **Mango Markets exploit** — the most
dangerous failure mode on Solana. We maintain a rolling 10-slot oracle price history per
symbol and flag any single-update move exceeding 3 standard deviations. This is
non-negotiable and distinguishes production-grade systems from hackathon demos.

### Drawdown Controller
| Drawdown | Action |
|----------|--------|
| Daily −3% | Scale to 50% |
| Weekly −7% | Full halt |

1% hysteresis prevents oscillation at the halt boundary (resumes at −6% not −7%).

### Position Limits (Fractional Kelly)
- 25% Kelly fraction (theoretically near-optimal for fat-tailed distributions)
- Combined with vol-targeting: `effective_size = 0.25_kelly × (target_vol / realized_vol)`
- Per-market cap: 25% of NAV
- Portfolio-level cap: 60% in perps, minimum 10% in lending at all times

---

## Allocation by Regime

| Regime | SOL-PERP | BTC-PERP | ETH-PERP | Kamino | Drift Spot | Total Lending |
|--------|----------|----------|----------|--------|------------|--------------|
| BULL_CARRY | 20% | 20% | 15% | 25% | 20% | 45% |
| SIDEWAYS | 10% | 10% | 5% | 35% | 25% | 60% |
| HIGH_VOL_CRISIS | 0% | 0% | 0% | 40% | 30% | 70% |

Note: actual allocations are further modulated by persistence, AR prediction, Kelly, and
vol-targeting. The table shows baseline targets before per-signal adjustments.

---

## Expected Performance by Regime

| Scenario | Regime | Annualized APY | 90-day Return | Notes |
|----------|--------|---------------|---------------|-------|
| Bull Market | BULL_CARRY | ~23.5% | ~5.3% | Full perp stack active |
| Consolidation | SIDEWAYS | ~6.4% | ~1.6% | Partial perp, heavier lending |
| Bear Market | HIGH_VOL_CRISIS | ~4.8% | ~1.2% | Lending-only, capital protected |

**Historical backtest (Feb 12 – Mar 15, 2026)**: +0.78% in 32 days (≈9% annualized).
Period captured a HIGH_VOL_CRISIS regime (SOL funding −10% APY). Strategy correctly
shifted to 70% lending allocation, preserving capital with near-zero drawdown.

---

## LSTM + XGBoost Funding Direction Predictor
- LSTM extracts temporal patterns from 24-hour funding rate sequences
- XGBoost trained on LSTM hidden states + tabular features (basis, vol ratio, z-scores)
- **83.3% validation accuracy** on funding direction prediction
- Falls back gracefully to XGBoost-only when PyTorch unavailable
- Used as an auxiliary signal for confidence scoring, not primary allocation gate

---

## Johansen Cointegration Engine (Stat Arb)
- Tracks SOL/BTC, SOL/ETH, BTC/ETH perp pairs
- Kalman-tracked dynamic beta for pair relationship
- Entry: |z-score| ≥ 2.0 | Exit: |z-score| ≤ 0.5 | Stop-loss: |z-score| ≥ 4.0
- Currently implemented as auxiliary strategy (not in primary allocation)

---

## Vault Compliance

- **No junior tranche** — single USDC class, pro-rata participation
- **No Ponzi yield** — all income from verifiable funding payments and lending interest
- **Health rate ≥ 1.3** monitored every 60 seconds (Drift liquidation threshold: 1.05)
- **Emergency exit** closes all perp positions within one transaction on health breach

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Strategy Engine | Python 3.12, FastAPI, hmmlearn, XGBoost, NumPy, aiohttp |
| Keeper Bot | TypeScript, @drift-labs/sdk, @coral-xyz/anchor |
| On-chain Adaptor | Rust, Anchor 0.31.1 |
| Vault Protocol | Ranger Finance / Voltr |
| Metrics | Prometheus + Grafana (`:9090/metrics`) |
| Tests | 103 passing (pytest + jest) |

---

## Strategy API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness + CB state + HMM status |
| `GET /regime` | Slow + fast HMM, dual agreement, position scale |
| `GET /allocations` | Full target allocation with sizing breakdown |
| `GET /signals` | AR predictions, persistence scores, oracle defense state |
| `GET /hedge-ratios` | Kalman beta/alpha + uncertainty per market |
| `GET /risk` | Circuit breaker events, drawdown state, cascade scores |
| `POST /update-market` | Push market data (funding, OI, depth, oracle price) |
| `POST /record-nav` | Record NAV for drawdown tracking |
