#!/usr/bin/env bash
# QuantVault — Devnet startup script
# Usage: bash scripts/start_devnet.sh
set -e

BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  QuantVault — Devnet Startup${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}\n"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Running pre-flight checks..."

# Check .env exists
[ -f .env ] || error ".env file not found. Copy .env.example and fill in your values."

# Load env
set -a; source .env; set +a

# Check cluster is devnet
if [ "$CLUSTER" != "devnet" ]; then
  warn "CLUSTER=$CLUSTER — this script is for devnet. Use scripts/start_mainnet.sh for mainnet."
fi

# Check Python
python3 --version > /dev/null 2>&1 || error "python3 not found"
success "Python: $(python3 --version)"

# Check Node
node --version > /dev/null 2>&1 || error "node not found"
success "Node: $(node --version)"

# Check pip dependencies
info "Checking Python dependencies..."
cd strategy
pip install -q -r requirements.txt
cd ..
success "Python deps OK"

# Check npm dependencies
info "Checking Node dependencies..."
cd bot
npm install --silent
cd ..
success "Node deps OK"

# ── Train models if not present ───────────────────────────────────────────────
if [ ! -f models/hmm_regime.pkl ]; then
  warn "No trained HMM model found."
  info "Fetching training data and fitting models (takes ~2 min)..."
  python3 -m scripts.collect_training_data 2>/dev/null || true
  python3 -m scripts.train_models 2>/dev/null || true
  if [ -f models/hmm_regime.pkl ]; then
    success "Models trained and saved to models/"
  else
    warn "Model training failed or skipped — engine will train on first data fetch"
  fi
else
  success "Pre-trained HMM model found"
fi

# ── Build TypeScript ──────────────────────────────────────────────────────────
info "Building TypeScript keeper bot..."
cd bot && npm run build && cd ..
success "TypeScript build OK"

# ── Start services ────────────────────────────────────────────────────────────
echo ""
info "Starting strategy engine on port ${PORT:-8000}..."
PYTHONPATH="$(pwd)" python3 -m uvicorn strategy.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --log-level "${LOG_LEVEL:-info}" &
STRATEGY_PID=$!

# Wait for strategy engine to be ready
info "Waiting for strategy engine..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT:-8000}/health" > /dev/null 2>&1; then
    success "Strategy engine ready"
    break
  fi
  sleep 2
  if [ $i -eq 30 ]; then
    error "Strategy engine failed to start. Check logs."
  fi
done

echo ""
info "Starting keeper bot (devnet)..."
STRATEGY_ENGINE_URL="http://localhost:${PORT:-8000}" node bot/dist/index.js &
BOT_PID=$!

echo ""
success "QuantVault running on devnet!"
echo -e "  Strategy engine: http://localhost:${PORT:-8000}"
echo -e "  Metrics:         http://localhost:${METRICS_PORT:-9090}/metrics"
echo -e "  Strategy PID:    $STRATEGY_PID"
echo -e "  Bot PID:         $BOT_PID"
echo -e "\nPress Ctrl+C to stop both processes\n"

# Trap Ctrl+C to kill both
trap "kill $STRATEGY_PID $BOT_PID 2>/dev/null; echo 'Stopped.'" SIGINT SIGTERM
wait
