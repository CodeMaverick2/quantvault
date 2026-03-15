#!/usr/bin/env bash
# QuantVault — Mainnet startup script with pre-flight safety checks
# Usage: bash scripts/start_mainnet.sh
set -e

BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

echo -e "\n${RED}════════════════════════════════════════${NC}"
echo -e "${RED}  QuantVault — MAINNET Startup${NC}"
echo -e "${RED}  REAL FUNDS — READ EVERY CHECK BELOW${NC}"
echo -e "${RED}════════════════════════════════════════${NC}\n"

# ── Load env ──────────────────────────────────────────────────────────────────
[ -f .env ] || error ".env not found. Copy .env.example and configure for mainnet."
set -a; source .env; set +a

# ── Safety gate: must be on mainnet ──────────────────────────────────────────
if [ "$CLUSTER" != "mainnet-beta" ]; then
  error "CLUSTER=$CLUSTER — set CLUSTER=mainnet-beta in .env for mainnet"
fi
success "Cluster: mainnet-beta"

# ── RPC check: reject public RPC ─────────────────────────────────────────────
if echo "$RPC_URL" | grep -q "api.mainnet-beta.solana.com"; then
  error "Public RPC detected. Use a private RPC (Helius/Triton/QuickNode) for mainnet."
fi
success "RPC: $RPC_URL"

# ── Check keeper wallet is set ────────────────────────────────────────────────
if [ -z "$KEEPER_PRIVATE_KEY" ] || echo "$KEEPER_PRIVATE_KEY" | grep -q "0,0,0,0,0,0,0,0,0"; then
  error "KEEPER_PRIVATE_KEY not set or still default. Set real keypair."
fi
success "Keeper wallet configured"

# ── Check vault address ───────────────────────────────────────────────────────
if [ -z "$VAULT_ADDRESS" ]; then
  warn "VAULT_ADDRESS is empty."
  info "Run: npx ts-node scripts/init_vault.ts"
  error "Set VAULT_ADDRESS in .env before starting mainnet"
fi
success "Vault: $VAULT_ADDRESS"

# ── Check pre-trained models ──────────────────────────────────────────────────
[ -f models/hmm_regime.pkl ] || error "No trained HMM model. Run: python3 -m scripts.train_models"
success "HMM model present"

# ── Build check ───────────────────────────────────────────────────────────────
[ -f bot/dist/index.js ] || error "Bot not built. Run: cd bot && npm run build"
success "Bot binary present"

# ── Solana RPC connectivity ────────────────────────────────────────────────────
info "Testing RPC connectivity..."
node -e "
const { Connection } = require('@solana/web3.js');
const c = new Connection('$RPC_URL', 'confirmed');
c.getSlot().then(s => { console.log('Slot:', s); process.exit(0); }).catch(e => { console.error(e.message); process.exit(1); });
" 2>/dev/null && success "RPC reachable" || error "RPC unreachable. Check RPC_URL."

# ── Manual confirmation ────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}You are about to start QuantVault on MAINNET.${NC}"
echo -e "  Cluster:    mainnet-beta"
echo -e "  RPC:        $RPC_URL"
echo -e "  Vault:      $VAULT_ADDRESS"
echo ""
read -rp "Type 'yes I understand' to continue: " confirm
[ "$confirm" = "yes I understand" ] || error "Aborted."

# ── Python deps ───────────────────────────────────────────────────────────────
info "Installing Python deps..."
cd strategy && pip install -q -r requirements.txt && cd ..
success "Python deps OK"

# ── Start strategy engine ──────────────────────────────────────────────────────
echo ""
info "Starting strategy engine..."
PYTHONPATH="$(pwd)" python3 -m uvicorn strategy.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --log-level "${LOG_LEVEL:-info}" &
STRATEGY_PID=$!

for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT:-8000}/health" > /dev/null 2>&1; then
    success "Strategy engine ready"
    break
  fi
  sleep 2
  [ $i -eq 30 ] && error "Strategy engine failed to start"
done

# ── Start keeper bot ──────────────────────────────────────────────────────────
echo ""
info "Starting keeper bot (mainnet)..."
STRATEGY_ENGINE_URL="http://localhost:${PORT:-8000}" node bot/dist/index.js &
BOT_PID=$!

echo ""
success "QuantVault LIVE on mainnet!"
echo -e "  Strategy engine: http://localhost:${PORT:-8000}"
echo -e "  Signals:         http://localhost:${PORT:-8000}/signals"
echo -e "  Risk:            http://localhost:${PORT:-8000}/risk"
echo -e "  Metrics:         http://localhost:${METRICS_PORT:-9090}/metrics"
echo ""
warn "Monitor metrics closely for first 30 minutes."
echo ""

trap "echo 'Shutting down...'; kill $STRATEGY_PID $BOT_PID 2>/dev/null; echo 'Stopped.'" SIGINT SIGTERM
wait
