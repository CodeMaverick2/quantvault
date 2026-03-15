# QuantVault — Adaptive Multi-Signal Delta-Neutral Strategy

> Ranger Build-A-Bear Hackathon submission
> USDC vault targeting 15–30% APY via regime-adaptive delta-neutral funding capture

## Overview

QuantVault is a fully automated yield strategy vault on Ranger Finance. It captures Drift Protocol perpetual funding rates with short positions while maintaining a base lending yield through Kamino and Drift Spot. A 3-state Hidden Markov Model continuously detects market regimes and scales positions — fully exiting perps during crisis conditions.

**Key stats (live backtest, Feb–Mar 2026):**
- 116/116 tests passing across all signal modules
- 83.3% LSTM funding direction prediction accuracy
- 0% drawdown in HIGH_VOL_CRISIS bear market (correctly shifts to 70% lending)
- 7-layer allocation stack with ATR-responsive leverage and time-of-day optimization
- Oracle manipulation defense (Mango Markets-style 3σ rolling halt)

See [STRATEGY.md](./STRATEGY.md) for full strategy documentation.

---

## Repository Structure

```
quantvault/
├── strategy/              # Python FastAPI strategy engine
│   ├── main.py            # API server + async update loops
│   ├── models/            # HMM, Kalman filter, LSTM+XGBoost
│   ├── signals/           # Drift data client, cascade risk, cointegration
│   ├── optimization/      # Dynamic allocation optimizer
│   ├── risk/              # Circuit breakers, drawdown controller, Kelly sizing
│   └── tests/             # 85 tests, all passing
│
├── bot/                   # TypeScript keeper bot
│   ├── src/
│   │   ├── index.ts       # Main loop (1min risk, 10min rebalance)
│   │   ├── rebalancer.ts  # Core allocation executor
│   │   ├── vault.ts       # Voltr SDK integration
│   │   ├── executor.ts    # Idempotent order executor
│   │   ├── drift.ts       # Drift perp position management
│   │   └── riskMonitor.ts # On-chain health monitoring
│   └── tests/             # Jest unit tests
│
├── programs/              # Rust/Anchor on-chain adaptor
│   └── delta-neutral-adaptor/
│
├── scripts/
│   ├── init_vault.ts         # One-shot vault creation
│   ├── add_strategies.ts     # Register Kamino + Drift adaptors
│   ├── collect_training_data.py  # Fetch Drift funding history
│   ├── train_models.py       # Train HMM + LSTM+XGBoost
│   └── backtest.py           # Historical simulation
│
├── config/
│   ├── strategy.yaml      # All tunable parameters
│   ├── devnet.ts          # Devnet addresses
│   └── mainnet.ts         # Mainnet addresses
│
├── docker-compose.yml     # Full stack (strategy + bot + metrics)
└── .env.example           # Environment template
```

---

## Quick Start

### 1. Environment Setup

```bash
cp .env.example .env
# Fill in: KEEPER_PRIVATE_KEY, RPC_URL (optional Helius for mainnet)
```

### 2. Install Dependencies

```bash
# Python (strategy engine)
pip install -r strategy/requirements.txt

# TypeScript (keeper bot)
cd bot && npm install
```

### 3. Collect Training Data & Train Models

```bash
python scripts/collect_training_data.py --days 90
python scripts/train_models.py --validate
```

### 4. Deploy Vault

> **Note:** The Voltr vault program (`vVoLTRjQmtFpiYoegx285Ze4gsLJ8ZxgFKVcuvmG1a8`) is
> deployed on **mainnet only**. For devnet testing, the strategy engine, keeper bot, and
> all signals run fully against devnet Drift data. Vault deployment requires mainnet USDC.

```bash
# Mainnet: set CLUSTER=mainnet-beta and a mainnet RPC_URL in .env first
npm run init-vault   # scripts/init_vault.ts (uses @voltr/vault-sdk)
npm run add-strategies  # scripts/add_strategies.ts

# Or use the Ranger Finance UI: https://vaults.ranger.finance/create
# Copy VAULT_ADDRESS from config/vault_addresses.json to .env
```

**Keeper wallet for testnet/demo:** `BPrSjs3XtA8qL2TrBB2LHq4EWzx7qtVjAjKRkErcqifj` (10 SOL devnet)

### 5. Run Full Stack

```bash
docker-compose up
```

Or separately:

```bash
# Strategy engine
uvicorn strategy.main:app --port 8000

# Keeper bot
cd bot && npm run build && npm start
```

### 6. Run Backtest

```bash
python scripts/backtest.py --days 90 --initial-nav 100000
```

---

## Strategy Engine API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check |
| `GET /regime` | Current HMM regime + confidence |
| `GET /allocations` | Target allocations per protocol |
| `GET /hedge-ratios` | Per-market Kalman hedge ratios |
| `GET /signals` | Full signal stack: AR predictions, persistence, dual-HMM, ToD, oracle defense |
| `GET /risk` | Circuit breaker state, drawdown, active events |
| `POST /update-market` | Push new market data (funding APR, oracle price, OBI) |
| `POST /record-nav` | Record NAV for drawdown tracking |
| `POST /lending-rates` | Update Kamino + Drift Spot APRs |

---

## Testing

```bash
# Python
python -m pytest strategy/tests/ -v   # 116 tests

# TypeScript
cd bot && npm test
```

---

## Program IDs

| Program | ID |
|---------|----|
| Voltr Vault | `vVoLTRjQmtFpiYoegx285Ze4gsLJ8ZxgFKVcuvmG1a8` |
| Drift Adaptor | `EBN93eXs5fHGBABuajQqdsKRkCgaqtJa8vEFD6vKXiP` |
| Kamino Adaptor | `to6Eti9CsC5FGkAtqiPphvKD2hiQiLsS8zWiDBqBPKR` |
| Drift Protocol | `dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH` |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CLUSTER` | `devnet` or `mainnet-beta` |
| `RPC_URL` | Solana RPC endpoint |
| `WS_URL` | WebSocket endpoint |
| `KEEPER_PRIVATE_KEY` | JSON byte array of keeper wallet private key |
| `VAULT_ADDRESS` | Voltr vault address (after init) |
| `VOLTR_API_URL` | Voltr API base URL |
| `STRATEGY_ENGINE_URL` | FastAPI server URL |
| `METRICS_PORT` | Prometheus metrics port (default 9090) |
