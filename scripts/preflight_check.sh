#!/usr/bin/env bash
# QuantVault pre-flight check — run before starting on any network
# Usage: bash scripts/preflight_check.sh
set -e

BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; ERRORS=$((ERRORS+1)); }
warn() { echo -e "${YELLOW}!${NC} $*"; }

ERRORS=0

echo -e "\n${BLUE}QuantVault Pre-flight Check${NC}\n"

# .env
[ -f .env ] && ok ".env exists" || fail ".env not found — copy .env.example"
[ -f .env ] && source .env 2>/dev/null || true

# Dependencies
python3 --version > /dev/null 2>&1 && ok "python3 installed" || fail "python3 not found"
node --version > /dev/null 2>&1    && ok "node installed"    || fail "node not found"
pip3 --version > /dev/null 2>&1    && ok "pip installed"     || fail "pip not found"

# Python deps
python3 -c "import fastapi, hmmlearn, numpy, pandas, aiohttp, yaml" 2>/dev/null \
  && ok "Python packages installed" \
  || warn "Some Python packages missing — run: pip install -r strategy/requirements.txt"

# Node deps
[ -d bot/node_modules ] && ok "Node modules installed" || warn "Run: cd bot && npm install"

# Built bot
[ -f bot/dist/index.js ] && ok "TypeScript compiled" || warn "Run: cd bot && npm run build"

# Models
[ -f models/hmm_regime.pkl ] && ok "HMM model present" || warn "No trained model — will train on first data fetch"
[ -f models/hmm_fast.pkl ]   && ok "Fast HMM model present" || warn "Fast HMM will train on first data fetch"

# Config
[ -f config/strategy.yaml ] && ok "strategy.yaml present" || fail "config/strategy.yaml missing"

# Env vars
[ -n "$CLUSTER" ]           && ok "CLUSTER=$CLUSTER" || fail "CLUSTER not set"
[ -n "$RPC_URL" ]           && ok "RPC_URL set"       || fail "RPC_URL not set"
[ -n "$KEEPER_PRIVATE_KEY" ] && ok "KEEPER_PRIVATE_KEY set" || fail "KEEPER_PRIVATE_KEY not set"
[ -n "$VAULT_ADDRESS" ]     && ok "VAULT_ADDRESS set"  || warn "VAULT_ADDRESS empty — set after init_vault.ts"

# Mainnet-specific
if [ "$CLUSTER" = "mainnet-beta" ]; then
  echo ""
  echo -e "${YELLOW}── Mainnet-specific checks ──${NC}"
  echo "$RPC_URL" | grep -q "api.mainnet-beta.solana.com" \
    && fail "Using public mainnet RPC — switch to private RPC" \
    || ok "Private RPC configured"
fi

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
  echo -e "${GREEN}All checks passed. Ready to launch.${NC}\n"
else
  echo -e "${RED}$ERRORS check(s) failed. Fix above issues before starting.${NC}\n"
  exit 1
fi
