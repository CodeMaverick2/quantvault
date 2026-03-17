"""
QuantVault Strategy Backtester
==============================
Fetches real historical funding rate data from Binance (public API, no key).
Simulates QuantVault strategy on 24-36 months of SOL/BTC/ETH data.

Usage:
  python3 backtest/backtest.py                          # 36mo, $1000, conservative
  python3 backtest/backtest.py --aggressive             # 36mo, $1000, high-risk/high-return
  python3 backtest/backtest.py --capital 500 --months 24
  python3 backtest/backtest.py --capital-analysis       # fee drag by capital size
  python3 backtest/backtest.py --clear-cache            # delete cached data and re-fetch
  python3 backtest/backtest.py --compare                # run both modes side-by-side
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
CACHE_DIR = ROOT / "backtest" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

from strategy.models.hmm_regime import MarketRegime, HMMRegimeClassifier

SYMBOLS    = ["SOL", "BTC", "ETH"]
BINANCE_MAP = {"SOL": "SOLUSDT", "BTC": "BTCUSDT", "ETH": "ETHUSDT"}

# ── Conservative mode parameters ─────────────────────────────────────────────
CONSERVATIVE = dict(
    FUNDING_THRESHOLD     = 8.0,    # % APR min to enter carry
    FUNDING_THRESHOLD_HI  = 5.2,    # when BULL_CARRY ≥90% conf
    FUNDING_THRESHOLD_SID = 10.0,   # SIDEWAYS — more selective
    MIN_PERSISTENCE       = 1,      # hours of +ve funding before entry
    MIN_EDGE_OVER_LENDING = 1.5,    # perp must beat lending by this %
    KELLY_FRACTION        = 0.35,
    MAX_SINGLE_PERP       = 0.25,   # max 25% NAV per market
    MAX_PERP_TOTAL        = 0.60,   # max 60% NAV total carry
    DIRECTIONAL           = False,  # no directional bets
    MAX_DIRECTIONAL       = 0.00,
    DRY_POWDER            = 0.05,
)

# ── Aggressive mode parameters ────────────────────────────────────────────────
# Delta-neutral carry PLUS momentum directional longs during bull runs.
# Higher returns during bull markets, significant drawdowns in bear markets.
AGGRESSIVE = dict(
    FUNDING_THRESHOLD     = 5.0,    # lower gate — enter more often
    FUNDING_THRESHOLD_HI  = 3.5,
    FUNDING_THRESHOLD_SID = 8.0,
    MIN_PERSISTENCE       = 1,
    MIN_EDGE_OVER_LENDING = 0.5,    # willing to earn just above lending
    KELLY_FRACTION        = 0.70,   # 70% Kelly — more aggressive sizing
    MAX_SINGLE_PERP       = 0.40,
    MAX_PERP_TOTAL        = 1.20,   # 120% NAV → implicit 1.2x leverage
    DIRECTIONAL           = True,   # add LONG momentum trades
    MAX_DIRECTIONAL       = 0.50,   # up to 50% NAV in directional longs
    DRY_POWDER            = 0.02,
)

# ── Shared parameters ─────────────────────────────────────────────────────────
KAMINO_APR     = 6.5
DRIFT_SPOT_APR = 5.5
MAKER_FEE      = 0.0001   # 0.01%
SOL_GAS_PER_TX = 0.002

CB_NEGATIVE_THRESHOLD = -10.0
CB_COOLDOWN_HOURS     = 0.5

HMM_FIT_WIN    = 720    # 30 days — sufficient for EM convergence
HMM_REFIT_FREQ = 168    # refit weekly
HMM_PREDICT_WIN = 48

# Momentum thresholds for directional long entry (aggressive mode)
MOM_24H_ENTRY  = 0.03   # 3%+ in last 24h
MOM_7D_ENTRY   = 0.06   # 6%+ in last 7 days
MOM_24H_EXIT   = -0.04  # exit on -4% reversal in 24h


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_funding_rates(months: int) -> dict[str, pd.DataFrame]:
    result = {}
    for sym in SYMBOLS:
        cache_file = CACHE_DIR / f"funding_{sym}_{months}mo.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col="ts", parse_dates=True)
            print(f"  {sym}: loaded {len(df)} rows from cache")
            result[sym] = df
            continue

        url   = "https://fapi.binance.com/fapi/v1/fundingRate"
        b_sym = BINANCE_MAP[sym]
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - months * 30 * 24 * 3600 * 1000
        rows, batch = [], start_ms
        print(f"  {sym}: fetching...", end="", flush=True)
        while batch < end_ms:
            try:
                r = requests.get(url, params={
                    "symbol": b_sym, "startTime": batch,
                    "endTime": min(batch + 1000*8*3600*1000, end_ms), "limit": 1000,
                }, timeout=20)
                r.raise_for_status()
                data = r.json()
                if not data: break
                rows.extend(data)
                batch = data[-1]["fundingTime"] + 1
                print(".", end="", flush=True)
                time.sleep(0.05)
            except Exception as e:
                print(f"\n  {sym} fetch error: {e}"); break

        if not rows:
            raise RuntimeError(f"No data for {sym}")

        df = pd.DataFrame(rows)
        df["ts"]  = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fr8h"] = df["fundingRate"].astype(float)
        df["apr"]  = df["fr8h"] * 3 * 365.25 * 100
        df = df[["ts","fr8h","apr"]].set_index("ts").sort_index()
        df = df.resample("1h").ffill()
        df["hourly_rate"] = df["fr8h"] / 8.0
        df.to_csv(cache_file)
        print(f" {len(df)} rows → cached")
        result[sym] = df
    return result


def fetch_prices(months: int) -> dict[str, pd.Series]:
    result = {}
    for sym in SYMBOLS:
        cache_file = CACHE_DIR / f"prices_{sym}_{months}mo.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col="ts", parse_dates=True)
            print(f"  {sym} prices: loaded {len(df)} rows from cache")
            result[sym] = df["close"]
            continue

        url   = "https://fapi.binance.com/fapi/v1/klines"
        b_sym = BINANCE_MAP[sym]
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - months * 30 * 24 * 3600 * 1000
        rows, batch = [], start_ms
        print(f"  {sym} prices: fetching...", end="", flush=True)
        while batch < end_ms:
            try:
                r = requests.get(url, params={
                    "symbol": b_sym, "interval": "1h",
                    "startTime": batch, "endTime": min(batch + 500*3600*1000, end_ms),
                    "limit": 500,
                }, timeout=20)
                r.raise_for_status()
                data = r.json()
                if not data: break
                rows.extend(data)
                batch = data[-1][6] + 1
                print(".", end="", flush=True)
                time.sleep(0.05)
            except Exception as e:
                print(f"\n  {sym} prices fetch error: {e}"); break

        df = pd.DataFrame(rows, columns=[
            "ot","open","high","low","close","vol","ct","qvol","n","tb","tq","_"])
        df["ts"]    = pd.to_datetime(df["ot"], unit="ms", utc=True)
        df["close"] = df["close"].astype(float)
        df = df[["ts","close"]].set_index("ts").sort_index()
        df.to_csv(cache_file)
        print(f" {len(df)} rows → cached")
        result[sym] = df["close"]
    return result


# ── Core simulation ───────────────────────────────────────────────────────────

def run_backtest(capital: float, months: int, aggressive: bool = False) -> dict:
    mode   = "AGGRESSIVE" if aggressive else "CONSERVATIVE"
    params = AGGRESSIVE if aggressive else CONSERVATIVE
    print(f"\n{'='*60}")
    print(f"QuantVault Backtest  |  ${capital:,.0f}  |  {months}mo  |  {mode}")
    print('='*60)

    print("\nMarket data:")
    funding = fetch_funding_rates(months)
    print("Prices:")
    prices  = fetch_prices(months)

    idx = funding["SOL"].index
    for sym in SYMBOLS:
        idx = idx.intersection(funding[sym].index).intersection(prices[sym].index)
    idx = idx.sort_values()
    N   = len(idx)
    print(f"\nData: {N} hours  ({idx[0].date()} → {idx[-1].date()})")

    fr  = {s: funding[s]["apr"].reindex(idx).ffill().values          for s in SYMBOLS}
    hr  = {s: funding[s]["hourly_rate"].reindex(idx).ffill().values   for s in SYMBOLS}
    px  = {s: prices[s].reindex(idx).ffill().values                  for s in SYMBOLS}

    # Realized vol (annualized)
    vol = {}
    for s in SYMBOLS:
        log_ret     = np.diff(np.log(np.where(px[s]>0, px[s], 1.0)), prepend=np.nan)
        rolling_std = pd.Series(log_ret).rolling(24, min_periods=8).std().values
        vol[s]      = np.where(np.isnan(rolling_std), 0.25, rolling_std * np.sqrt(24*365.25))

    avg_apr = np.array([(fr["SOL"][i]+fr["BTC"][i]+fr["ETH"][i])/3 for i in range(N)])
    avg_vol = np.array([(vol["SOL"][i]+vol["BTC"][i]+vol["ETH"][i])/3 for i in range(N)])

    # ── HMM regimes ──────────────────────────────────────────────────────────
    print(f"Computing HMM regimes (fit_win={HMM_FIT_WIN}h, refit every {HMM_REFIT_FREQ}h)...")
    regime_arr     = np.full(N, MarketRegime.SIDEWAYS, dtype=int)
    confidence_arr = np.full(N, 0.5)
    hmm = HMMRegimeClassifier(n_states=3)
    last_fit_i = -1

    for i in range(HMM_FIT_WIN, N):
        if (i - last_fit_i) >= HMM_REFIT_FREQ or last_fit_i < 0:
            X_fit = np.column_stack([avg_apr[max(0,i-HMM_FIT_WIN):i],
                                     avg_vol[max(0,i-HMM_FIT_WIN):i]])
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hmm.fit(X_fit)
                last_fit_i = i
            except Exception:
                pass

        if hmm.is_fitted:
            X_pred = np.column_stack([avg_apr[max(0,i-HMM_PREDICT_WIN):i+1],
                                      avg_vol[max(0,i-HMM_PREDICT_WIN):i+1]])
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = hmm.predict(X_pred)
                regime_arr[i]     = int(pred.regime)
                confidence_arr[i] = pred.confidence
            except Exception:
                pass

    print("Regime computation complete.")
    unique, counts = np.unique(regime_arr[HMM_FIT_WIN:], return_counts=True)
    total_l = counts.sum()
    for u, c in zip(unique, counts):
        print(f"  {MarketRegime(u).name:<22} {c:5d}  ({c/total_l*100:.1f}%)")

    # ── Pre-compute price momentum arrays ─────────────────────────────────────
    mom_24h = {}
    mom_7d  = {}
    for s in SYMBOLS:
        p = px[s]
        m24 = np.where(np.arange(N) >= 24,
                       (p - np.roll(p, 24)) / np.where(np.roll(p, 24) > 0, np.roll(p, 24), 1.0),
                       0.0)
        m7d = np.where(np.arange(N) >= 168,
                       (p - np.roll(p, 168)) / np.where(np.roll(p, 168) > 0, np.roll(p, 168), 1.0),
                       0.0)
        m24[:24]  = 0.0
        m7d[:168] = 0.0
        mom_24h[s] = m24
        mom_7d[s]  = m7d

    # ── Simulation ────────────────────────────────────────────────────────────
    nav          = capital
    # Carry positions: SHORT perps, collect positive funding
    carry_pos    = {s: 0.0 for s in SYMBOLS}
    # Directional positions: LONG perps, gain from price appreciation
    dir_pos      = {s: 0.0 for s in SYMBOLS}
    consec_pos   = {s: 0   for s in SYMBOLS}
    cb_cooldown  = 0.0
    dd_hwm       = capital

    nav_series       = np.zeros(N)
    funding_inc      = np.zeros(N)
    dir_pnl_arr      = np.zeros(N)
    lending_inc      = np.zeros(N)
    fee_costs        = np.zeros(N)
    regime_series    = np.empty(N, dtype=object)
    active_pos_count = np.zeros(N, dtype=int)

    best_lending = max(KAMINO_APR, DRIFT_SPOT_APR)
    WARMUP = HMM_FIT_WIN

    for i in range(N):
        regime_i = MarketRegime(regime_arr[i])
        conf_i   = confidence_arr[i]

        # Track consecutive positive funding
        for s in SYMBOLS:
            consec_pos[s] = consec_pos[s] + 1 if fr[s][i] > 0 else 0

        # ── Collect carry funding income (SHORT positions) ─────────────────────
        hour_carry = 0.0
        for s in SYMBOLS:
            if carry_pos[s] > 0:
                pnl = carry_pos[s] * hr[s][i]
                hour_carry += pnl
                nav += pnl
        funding_inc[i] = hour_carry

        # ── Directional P&L: LONG perps gain from price + pay funding ─────────
        hour_dir = 0.0
        if i > 0:
            for s in SYMBOLS:
                if dir_pos[s] > 0:
                    price_ret    = (px[s][i] - px[s][i-1]) / max(px[s][i-1], 1.0)
                    price_pnl    = dir_pos[s] * price_ret
                    funding_paid = dir_pos[s] * hr[s][i]   # longs PAY funding
                    net          = price_pnl - funding_paid
                    hour_dir    += net
                    nav         += net
        dir_pnl_arr[i] = hour_dir

        # ── Lending income on idle capital ────────────────────────────────────
        total_deployed = sum(carry_pos.values()) + sum(dir_pos.values())
        idle           = max(0.0, nav - total_deployed)
        lend_h         = idle * (best_lending / 100.0) / (24 * 365.25)
        nav           += lend_h
        lending_inc[i] = lend_h

        # ── CB cooldown ───────────────────────────────────────────────────────
        if cb_cooldown > 0:
            cb_cooldown -= 1.0
        if avg_apr[i] / 100.0 < CB_NEGATIVE_THRESHOLD:
            cb_cooldown = CB_COOLDOWN_HOURS
        cb_scale = (min(1.0, 1.0 - cb_cooldown/CB_COOLDOWN_HOURS)
                    if cb_cooldown > 0 else 1.0)

        # ── Drawdown control ──────────────────────────────────────────────────
        if nav > dd_hwm:
            dd_hwm = nav
        drawdown = (dd_hwm - nav) / dd_hwm if dd_hwm > 0 else 0.0
        if drawdown > 0.20:          # 20% max drawdown → flat
            dd_scale = 0.0
        elif drawdown > 0.10:
            dd_scale = 0.3
        elif drawdown > 0.05:
            dd_scale = max(0.0, 1.0 - drawdown * 8)
        else:
            dd_scale = 1.0

        # ── Compute target allocations ────────────────────────────────────────
        carry_target = {s: 0.0 for s in SYMBOLS}
        dir_target   = {s: 0.0 for s in SYMBOLS}
        fees_h       = 0.0

        if i >= WARMUP and cb_scale > 0 and dd_scale > 0:
            eff = dd_scale * cb_scale

            # Regime-adaptive carry threshold
            if regime_i == MarketRegime.BULL_CARRY and conf_i >= 0.90:
                thresh = params["FUNDING_THRESHOLD_HI"]
            elif regime_i == MarketRegime.BULL_CARRY and conf_i >= 0.70:
                thresh = max(params["FUNDING_THRESHOLD_HI"] + 1.0,
                             params["FUNDING_THRESHOLD"] * 0.80)
            elif regime_i == MarketRegime.SIDEWAYS:
                thresh = params["FUNDING_THRESHOLD_SID"]
            else:
                thresh = params["FUNDING_THRESHOLD"]

            # Regime-adaptive carry scale
            carry_scale = eff
            if regime_i == MarketRegime.HIGH_VOL_CRISIS:
                carry_scale = 0.0 if avg_apr[i] <= 0 else eff * 0.30
            elif regime_i == MarketRegime.BULL_CARRY:
                carry_scale = eff * min(1.3, 0.7 + conf_i)
            else:
                carry_scale = eff * max(0.3, conf_i * 0.8)

            # Carry (SHORT) positions: collect positive funding
            eligible = [
                s for s in SYMBOLS
                if (fr[s][i] >= thresh
                    and consec_pos[s] >= params["MIN_PERSISTENCE"]
                    and (fr[s][i] - best_lending) >= params["MIN_EDGE_OVER_LENDING"])
            ]

            if eligible:
                raw_kelly = {}
                for s in eligible:
                    vol_s      = max(vol[s][i], 0.10)
                    period_ret = (fr[s][i] / 100.0) / (24 * 365.25)
                    period_var = (vol_s / np.sqrt(24 * 365.25)) ** 2
                    k          = params["KELLY_FRACTION"] * period_ret / max(period_var, 1e-10)
                    raw_kelly[s] = min(max(k, 0.0), params["MAX_SINGLE_PERP"])

                total_k = sum(raw_kelly.values())
                if total_k > 0:
                    perp_budget = params["MAX_PERP_TOTAL"] * carry_scale * (1 - params["DRY_POWDER"])
                    for s in eligible:
                        frac = raw_kelly[s] / total_k
                        carry_target[s] = min(frac * perp_budget, params["MAX_SINGLE_PERP"])

            # ── Directional (LONG) positions: momentum-driven ─────────────────
            if params["DIRECTIONAL"] and regime_i == MarketRegime.BULL_CARRY and conf_i >= 0.75:
                dir_budget = params["MAX_DIRECTIONAL"] * eff
                for s in SYMBOLS:
                    m24 = mom_24h[s][i]
                    m7d = mom_7d[s][i]
                    if m24 >= MOM_24H_ENTRY and m7d >= MOM_7D_ENTRY:
                        # Kelly on momentum: expected return = m24 over next period
                        vol_s = max(vol[s][i], 0.15)
                        # Scale by momentum strength (stronger momentum → bigger bet)
                        mom_strength = min(m7d / 0.15, 2.0)   # cap at 2× boost
                        k_dir = params["KELLY_FRACTION"] * m24 / max(vol_s**2 / (24*365.25), 1e-8)
                        k_dir = min(max(k_dir, 0.0), params["MAX_DIRECTIONAL"])
                        dir_target[s] = min(k_dir * mom_strength, dir_budget)

        # ── Execute rebalance every 4 hours ───────────────────────────────────
        if i % 4 == 0:
            for s in SYMBOLS:
                # Carry positions (shorts)
                c_target = nav * carry_target[s]
                c_delta  = c_target - carry_pos[s]
                if abs(c_delta) >= nav * 0.04:
                    fee = abs(c_delta) * MAKER_FEE
                    fees_h     += fee
                    nav        -= fee
                    carry_pos[s] = max(0.0, c_target)

                # Directional positions (longs)
                d_target = nav * dir_target[s]
                d_delta  = d_target - dir_pos[s]

                # Force-close directional on momentum reversal
                if dir_pos[s] > 0 and mom_24h[s][i] < MOM_24H_EXIT:
                    d_target = 0.0
                    d_delta  = -dir_pos[s]

                if abs(d_delta) >= nav * 0.04:
                    fee = abs(d_delta) * MAKER_FEE
                    fees_h    += fee
                    nav       -= fee
                    dir_pos[s] = max(0.0, d_target)

        fee_costs[i]        = fees_h
        nav_series[i]       = nav
        regime_series[i]    = regime_i.name
        active_pos_count[i] = sum(
            1 for s in SYMBOLS if carry_pos[s] + dir_pos[s] > nav * 0.01
        )

    # ── Metrics ───────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "nav":              nav_series,
        "funding":          funding_inc,
        "directional":      dir_pnl_arr,
        "lending":          lending_inc,
        "fees":             fee_costs,
        "regime":           regime_series,
        "active_positions": active_pos_count,
    }, index=idx)

    df_daily      = df.resample("1D").last()
    daily_navs    = df_daily["nav"].values
    daily_returns = np.diff(daily_navs) / np.where(daily_navs[:-1]>0, daily_navs[:-1], 1.0)

    total_return  = (nav - capital) / capital * 100
    ann_return    = ((nav / capital) ** (365.0 / max(1, len(df_daily))) - 1) * 100
    max_dd        = _max_drawdown(daily_navs) * 100
    sharpe        = _sharpe(daily_returns)

    df_monthly    = df_daily["nav"].resample("ME").last()
    monthly_ret   = df_monthly.pct_change().dropna() * 100

    df_daily["year"] = df_daily.index.year
    yearly_nav  = df_daily.groupby("year")["nav"].agg(["first","last"])
    yearly_ret  = ((yearly_nav["last"] / yearly_nav["first"]) - 1) * 100

    return {
        "mode":              mode,
        "capital":           capital,
        "months":            months,
        "start":             str(idx[0].date()),
        "end":               str(idx[-1].date()),
        "final_nav":         nav,
        "total_return_pct":  total_return,
        "ann_return_pct":    ann_return,
        "max_drawdown_pct":  max_dd,
        "sharpe_ratio":      sharpe,
        "win_days":          int((daily_returns > 0).sum()),
        "lose_days":         int((daily_returns <= 0).sum()),
        "win_rate":          float((daily_returns > 0).mean() * 100),
        "total_funding":     float(funding_inc.sum()),
        "total_directional": float(dir_pnl_arr.sum()),
        "total_lending":     float(lending_inc.sum()),
        "total_fees":        float(fee_costs.sum()),
        "avg_monthly_pct":   float(monthly_ret.mean()),
        "best_month_pct":    float(monthly_ret.max()),
        "worst_month_pct":   float(monthly_ret.min()),
        "monthly_returns":   monthly_ret.to_dict(),
        "yearly_returns":    yearly_ret.to_dict(),
        "df":                df_daily,
        "regime_dist":       df["regime"].value_counts().to_dict(),
    }


def _max_drawdown(navs):
    peak = navs[0]; max_dd = 0.0
    for v in navs:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd
    return max_dd


def _sharpe(daily_rets, rf=0.065/365):
    excess = daily_rets - rf
    return float(np.sqrt(365) * excess.mean() / max(excess.std(), 1e-10))


def print_results(r):
    if not r: return
    print(f"\n{'='*60}")
    print(f"RESULTS [{r['mode']}]  {r['start']} → {r['end']}")
    print('='*60)
    print(f"  Capital:             ${r['capital']:>10,.2f}")
    print(f"  Final NAV:           ${r['final_nav']:>10,.2f}  (${r['final_nav']-r['capital']:+,.2f})")
    print(f"  Total return:        {r['total_return_pct']:>+10.2f}%")
    print(f"  Annualized return:   {r['ann_return_pct']:>+10.2f}%")
    print(f"  Max drawdown:        {r['max_drawdown_pct']:>10.2f}%")
    print(f"  Sharpe ratio:        {r['sharpe_ratio']:>10.2f}")
    print(f"  Win rate (daily):    {r['win_rate']:>10.1f}%  ({r['win_days']}W / {r['lose_days']}L)")
    print(f"\n  Income breakdown:")
    print(f"    Carry funding:    ${r['total_funding']:>10,.2f}")
    if r['total_directional'] != 0:
        print(f"    Directional P&L: ${r['total_directional']:>10,.2f}  (LONG momentum)")
    print(f"    Lending income:   ${r['total_lending']:>10,.2f}")
    print(f"    Total fees paid:  ${r['total_fees']:>10,.2f}")

    print(f"\n  Per-year returns:")
    for yr, ret in sorted(r["yearly_returns"].items()):
        bar  = "█" * int(min(abs(ret), 40) / 0.8)
        neg  = "▼" if ret < 0 else "▲"
        print(f"    {yr}  {neg} {ret:+8.2f}%  {bar}")

    print(f"\n  Monthly performance:")
    for ts, ret in sorted(r["monthly_returns"].items()):
        bar  = "█" * int(min(abs(ret), 25) / 0.4)
        sign = "▼" if ret < 0 else "▲"
        color = " ◄ LOSS" if ret < -2 else ""
        print(f"    {str(ts)[:7]}  {sign} {ret:+6.2f}%  {bar}{color}")

    print(f"\n  Avg monthly: {r['avg_monthly_pct']:+.2f}%  "
          f"Best: {r['best_month_pct']:+.2f}%  Worst: {r['worst_month_pct']:+.2f}%")

    print(f"\n  Regime distribution:")
    for reg, cnt in r["regime_dist"].items():
        pct = cnt / sum(r["regime_dist"].values()) * 100
        print(f"    {reg:<24} {pct:5.1f}%")

    lend_bm = r['capital'] * (max(KAMINO_APR,DRIFT_SPOT_APR)/100) * r['months']/12
    print(f"\n  vs lending-only benchmark: ${lend_bm:,.2f} ({lend_bm/r['capital']*100:.1f}% over {r['months']}mo)")
    print(f"  Outperformance:            ${r['final_nav']-r['capital']-lend_bm:+,.2f}")


def compare_modes(capital: float, months: int):
    print("\nRunning CONSERVATIVE mode...")
    r_c = run_backtest(capital, months, aggressive=False)
    print("\nRunning AGGRESSIVE mode...")
    r_a = run_backtest(capital, months, aggressive=True)

    print_results(r_c)
    print_results(r_a)

    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE COMPARISON")
    print('='*60)
    fmt = "  {:<28} {:>15} {:>15}"
    print(fmt.format("Metric", "Conservative", "Aggressive"))
    print("  " + "-"*56)
    print(fmt.format("Final NAV",
                     f"${r_c['final_nav']:,.2f}", f"${r_a['final_nav']:,.2f}"))
    print(fmt.format("Annualized return",
                     f"{r_c['ann_return_pct']:+.2f}%", f"{r_a['ann_return_pct']:+.2f}%"))
    print(fmt.format("Max drawdown",
                     f"{r_c['max_drawdown_pct']:.2f}%", f"{r_a['max_drawdown_pct']:.2f}%"))
    print(fmt.format("Sharpe ratio",
                     f"{r_c['sharpe_ratio']:.2f}", f"{r_a['sharpe_ratio']:.2f}"))
    print(fmt.format("Best month",
                     f"{r_c['best_month_pct']:+.2f}%", f"{r_a['best_month_pct']:+.2f}%"))
    print(fmt.format("Worst month",
                     f"{r_c['worst_month_pct']:+.2f}%", f"{r_a['worst_month_pct']:+.2f}%"))
    print(fmt.format("Win rate",
                     f"{r_c['win_rate']:.1f}%", f"{r_a['win_rate']:.1f}%"))
    print(fmt.format("Carry income",
                     f"${r_c['total_funding']:,.2f}", f"${r_a['total_funding']:,.2f}"))
    print(fmt.format("Directional P&L",
                     f"${r_c['total_directional']:,.2f}", f"${r_a['total_directional']:,.2f}"))


def capital_analysis():
    print(f"\n{'='*60}")
    print("MINIMUM VIABLE CAPITAL ANALYSIS")
    print('='*60)
    sol_price  = 130.0
    usd_per_tx = SOL_GAS_PER_TX * sol_price
    print(f"\nGas per tx: ${usd_per_tx:.3f}  Maker fee: {MAKER_FEE*100:.2f}%/side\n")
    print(f"{'Capital':>10}  {'Gas/mo':>8}  {'Fees/mo':>8}  {'Gross/mo':>10}  {'Net/mo':>10}  {'Net APY':>8}  {'Viable':>8}")
    print("-"*70)
    for cap in [50, 100, 200, 300, 500, 1000, 2000, 5000]:
        trades_pm  = 8
        gas_cost   = trades_pm * usd_per_tx
        trade_fee  = cap * MAKER_FEE * trades_pm * 2
        total_cost = gas_cost + trade_fee
        gross_mo   = cap * 0.20 / 12
        net_mo     = gross_mo - total_cost
        net_apy    = (net_mo / cap) * 12 * 100 if cap > 0 else 0
        viable     = "YES ✓" if net_apy > 8 else ("MARGINAL" if net_apy > 2 else "NO ✗")
        print(f"  ${cap:>8,.0f}  ${gas_cost:>7.2f}  ${trade_fee:>7.3f}  ${gross_mo:>9.2f}  ${net_mo:>9.2f}  {max(net_apy,0):>7.1f}%  {viable}")
    print(f"\n  Recommendation: $300+ for meaningful returns, $500+ for fee-efficiency")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital",         type=float, default=1000.0)
    parser.add_argument("--months",          type=int,   default=36)
    parser.add_argument("--aggressive",      action="store_true",
                        help="High-risk mode: momentum longs + higher leverage")
    parser.add_argument("--compare",         action="store_true",
                        help="Run both conservative and aggressive and compare")
    parser.add_argument("--capital-analysis", action="store_true")
    parser.add_argument("--clear-cache",     action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.csv"):
            f.unlink()
        print(f"Cleared cache in {CACHE_DIR}")

    if args.capital_analysis:
        capital_analysis()
    elif args.compare:
        compare_modes(args.capital, args.months)
    else:
        r = run_backtest(args.capital, args.months, aggressive=args.aggressive)
        print_results(r)
