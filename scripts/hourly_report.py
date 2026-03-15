#!/usr/bin/env python3
"""
QuantVault Hourly Performance Report

Pulls data from:
  - Prometheus metrics (localhost:9090)
  - Strategy engine (localhost:8000)

Sends an HTML email with full system analysis.

Usage:
  python3 hourly_report.py

Config (env vars in .env or environment):
  REPORT_EMAIL_TO      = recipient email
  REPORT_EMAIL_FROM    = sender email (Gmail address)
  REPORT_EMAIL_PASS    = Gmail App Password (https://myaccount.google.com/apppasswords)
  REPORT_SMTP_HOST     = smtp.gmail.com (default)
  REPORT_SMTP_PORT     = 587 (default)
  STRATEGY_ENGINE_URL  = http://localhost:8000 (default)
  METRICS_URL          = http://localhost:9090 (default)
"""

import os
import sys
import json
import smtplib
import datetime
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent.parent / "bot" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

STRATEGY_URL = os.getenv("STRATEGY_ENGINE_URL", "http://localhost:8000")
METRICS_URL  = os.getenv("METRICS_URL", "http://localhost:9090")
TO_EMAIL     = os.getenv("REPORT_EMAIL_TO")
FROM_EMAIL   = os.getenv("REPORT_EMAIL_FROM")
EMAIL_PASS   = os.getenv("REPORT_EMAIL_PASS")
SMTP_HOST    = os.getenv("REPORT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("REPORT_SMTP_PORT", "587"))

# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

def fetch_prometheus(metric_name):
    """Query Prometheus for a specific metric value."""
    try:
        url = f"{METRICS_URL}/metrics"
        r = requests.get(url, timeout=5)
        for line in r.text.splitlines():
            if line.startswith(metric_name + " ") or line.startswith(metric_name + "{"):
                # Parse simple gauge: metric_name 1.23
                if "{" not in line:
                    return float(line.split()[-1])
        return None
    except:
        return None

def fetch_prometheus_all():
    """Return all Prometheus metrics as a dict."""
    result = {}
    try:
        r = requests.get(f"{METRICS_URL}/metrics", timeout=5)
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    result[parts[0]] = float(parts[-1])
                except:
                    pass
    except:
        pass
    return result

# ── Report builder ─────────────────────────────────────────────────────────────

def build_report():
    now = datetime.datetime.utcnow()
    ts = now.strftime("%Y-%m-%d %H:%M UTC")

    # Fetch all data
    health    = fetch(f"{STRATEGY_URL}/health")
    regime    = fetch(f"{STRATEGY_URL}/regime")
    allocs    = fetch(f"{STRATEGY_URL}/allocations")
    risk      = fetch(f"{STRATEGY_URL}/risk")
    signals   = fetch(f"{STRATEGY_URL}/signals")
    prom      = fetch_prometheus_all()

    # Helper to safely get prom value
    def p(key, default="N/A"):
        v = prom.get(key)
        return f"{v:.4f}" if v is not None else default

    def pf(key, fmt=".2f", default="N/A"):
        v = prom.get(key)
        return format(v, fmt) if v is not None else default

    # ── Regime info ──────────────────────────────────────────────────────────
    regime_name  = regime.get("regime", "UNKNOWN") if "_error" not in regime else "OFFLINE"
    regime_conf  = regime.get("confidence", 0)
    regime_probs = regime.get("probabilities", {})
    pos_scale    = regime.get("position_scale", 0)

    REGIME_EMOJI = {
        "BULL_CARRY": "🟢",
        "SIDEWAYS": "🟡",
        "HIGH_VOL_CRISIS": "🔴",
    }
    r_emoji = REGIME_EMOJI.get(regime_name, "⚪")

    # ── Allocation info ───────────────────────────────────────────────────────
    exp_apr      = allocs.get("expected_blended_apr", 0) if "_error" not in allocs else 0
    perp_allocs  = allocs.get("perp_allocations", {}) if "_error" not in allocs else {}
    k_pct        = allocs.get("kamino_lending_pct", 0) if "_error" not in allocs else 0
    d_pct        = allocs.get("drift_spot_lending_pct", 0) if "_error" not in allocs else 0

    # ── Risk info ─────────────────────────────────────────────────────────────
    cb_state     = risk.get("circuit_breaker_state", "UNKNOWN") if "_error" not in risk else "OFFLINE"
    dd_halted    = risk.get("drawdown_halted", False) if "_error" not in risk else False
    cb_events    = risk.get("active_circuit_breaker_events", []) if "_error" not in risk else []
    cascade_risk = risk.get("market_cascade_risks", {}) if "_error" not in risk else {}

    # ── Prometheus metrics ────────────────────────────────────────────────────
    nav_usd         = prom.get("quantvault_nav_usd", 0)
    rebal_total     = prom.get("quantvault_rebalances_total", 0)
    rebal_errors    = prom.get("quantvault_rebalance_errors_total", 0)
    drawdown_halted = prom.get("quantvault_drawdown_halted", 0)

    # ── Funding APRs ──────────────────────────────────────────────────────────
    funding_rates = {}
    for k, v in prom.items():
        if k.startswith('quantvault_funding_apr{'):
            sym = k.split('symbol="')[1].rstrip('"}')
            funding_rates[sym] = v

    # ── Predictive signals ────────────────────────────────────────────────────
    pred = signals.get("predictive_signals", {}) if "_error" not in signals else {}
    leading = signals.get("leading_indicators", {}) if "_error" not in signals else {}
    transition = signals.get("regime_transition", {}) if "_error" not in signals else {}

    # ── Status summary ────────────────────────────────────────────────────────
    system_ok   = "_error" not in health and health.get("status") == "ok"
    overall_status = "✅ OPERATIONAL" if system_ok and cb_state == "NORMAL" and not dd_halted else \
                     "⚠️ DEGRADED" if system_ok else "🔴 OFFLINE"

    error_rate = (rebal_errors / rebal_total * 100) if rebal_total > 0 else 0

    # ── Build HTML ────────────────────────────────────────────────────────────
    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; background: #0a0a0f; color: #e0e0e0; margin: 0; padding: 0; }}
  .container {{ max-width: 700px; margin: 0 auto; padding: 20px; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a4a; }}
  .header h1 {{ margin: 0; font-size: 22px; color: #a78bfa; }}
  .header .sub {{ color: #6b7280; font-size: 13px; margin-top: 4px; }}
  .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; }}
  .ok {{ background: #052e16; color: #4ade80; border: 1px solid #166534; }}
  .warn {{ background: #431407; color: #fb923c; border: 1px solid #9a3412; }}
  .danger {{ background: #3b0000; color: #f87171; border: 1px solid #991b1b; }}
  .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 18px; margin-bottom: 16px; }}
  .card h2 {{ margin: 0 0 14px 0; font-size: 15px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .metric {{ background: #0f172a; border-radius: 8px; padding: 12px; }}
  .metric .label {{ font-size: 11px; color: #6b7280; margin-bottom: 4px; }}
  .metric .value {{ font-size: 20px; font-weight: bold; color: #f3f4f6; }}
  .metric .sub {{ font-size: 11px; color: #6b7280; margin-top: 2px; }}
  .green {{ color: #4ade80 !important; }}
  .yellow {{ color: #fbbf24 !important; }}
  .red {{ color: #f87171 !important; }}
  .signal-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1f2937; font-size: 13px; }}
  .signal-row:last-child {{ border-bottom: none; }}
  .signal-label {{ color: #9ca3af; }}
  .signal-value {{ color: #f3f4f6; font-weight: 500; }}
  .alloc-bar {{ background: #1f2937; border-radius: 4px; height: 8px; margin-top: 6px; overflow: hidden; }}
  .alloc-fill {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, #7c3aed, #a78bfa); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: #6b7280; font-weight: normal; padding: 6px 8px; border-bottom: 1px solid #1f2937; }}
  td {{ padding: 8px 8px; border-bottom: 1px solid #111827; }}
  .footer {{ text-align: center; color: #4b5563; font-size: 11px; padding: 16px; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>⚡ QuantVault System Report</h1>
    <div class="sub">{ts} &nbsp;|&nbsp; {overall_status}</div>
  </div>

  <!-- Overview Metrics -->
  <div class="card">
    <h2>Portfolio Overview</h2>
    <div class="metric-grid">
      <div class="metric">
        <div class="label">VAULT NAV</div>
        <div class="value {'green' if nav_usd > 0 else ''}">${nav_usd:,.2f}</div>
        <div class="sub">Total assets in Drift</div>
      </div>
      <div class="metric">
        <div class="label">EXPECTED APR</div>
        <div class="value {'green' if exp_apr > 5 else 'yellow'}">{exp_apr:.1f}%</div>
        <div class="sub">Blended funding + lending</div>
      </div>
      <div class="metric">
        <div class="label">REBALANCES</div>
        <div class="value">{int(rebal_total)}</div>
        <div class="sub">Error rate: {error_rate:.1f}%</div>
      </div>
      <div class="metric">
        <div class="label">CIRCUIT BREAKER</div>
        <div class="value {'green' if cb_state == 'NORMAL' else 'red'}">{cb_state}</div>
        <div class="sub">Drawdown halt: {'YES' if dd_halted else 'NO'}</div>
      </div>
    </div>
  </div>

  <!-- Regime -->
  <div class="card">
    <h2>Market Regime</h2>
    <div class="signal-row">
      <span class="signal-label">Active Regime</span>
      <span class="signal-value">{r_emoji} {regime_name} ({regime_conf*100:.0f}% confidence)</span>
    </div>
    <div class="signal-row">
      <span class="signal-label">Position Scale</span>
      <span class="signal-value">{pos_scale:.2f}x</span>
    </div>
    {"".join(f'<div class="signal-row"><span class="signal-label">P({r})</span><span class="signal-value">{v*100:.1f}%</span></div>' for r, v in regime_probs.items())}
    {f'<div class="signal-row"><span class="signal-label">Transition Warning</span><span class="signal-value {'red' if transition.get('warning') != 'NO_WARNING' else 'green'}">{transition.get("warning", "N/A")}</span></div>' if transition else ''}
    {f'<div class="signal-row"><span class="signal-label">Crisis Prob (24h)</span><span class="signal-value">{transition.get("crisis_approach_24h", 0)*100:.1f}%</span></div>' if transition else ''}
  </div>

  <!-- Funding Rates -->
  <div class="card">
    <h2>Live Funding Rates</h2>
    <table>
      <tr><th>Market</th><th>Funding APR</th><th>Cascade Risk</th><th>Target Alloc</th></tr>
      {"".join(f'''<tr>
        <td>{sym}</td>
        <td class="{'green' if apr > 0.05 else 'yellow'}">{apr*100:.2f}%</td>
        <td class="{'red' if prom.get(f'quantvault_cascade_risk_score{{symbol="{sym}"}}', 0) > 0.7 else 'yellow' if prom.get(f'quantvault_cascade_risk_score{{symbol="{sym}"}}', 0) > 0.5 else 'green'}">{prom.get(f'quantvault_cascade_risk_score{{symbol="{sym}"}}', 0):.2f}</td>
        <td>{perp_allocs.get(sym, 0)*100:.1f}%</td>
      </tr>''' for sym, apr in sorted(funding_rates.items(), key=lambda x: -x[1]))}
    </table>
  </div>

  <!-- Predictive Signals -->
  {"" if not pred else f'''
  <div class="card">
    <h2>Predictive Signals (AR4 + Multi-Horizon)</h2>
    <table>
      <tr><th>Market</th><th>Trajectory</th><th>1h Forecast</th><th>4h Forecast</th><th>Action</th></tr>
      {"".join(f"""<tr>
        <td>{sym}</td>
        <td>{sig.get('trajectory', 'N/A')}</td>
        <td>{sig.get('forecast_1h_apr', 0)*100:.2f}%</td>
        <td>{sig.get('forecast_4h_apr', 0)*100:.2f}%</td>
        <td class="{'green' if sig.get('pre_position') else 'yellow' if sig.get('exit_signal') else ''}">{
          '🟢 PRE-POSITION' if sig.get('pre_position') else
          '🔴 EXIT' if sig.get('exit_signal') else
          '⚪ HOLD'
        }</td>
      </tr>""" for sym, sig in pred.items())}
    </table>
  </div>'''}

  <!-- Leading Indicators -->
  {"" if not leading else f'''
  <div class="card">
    <h2>Leading Indicators</h2>
    {"".join(f"""
    <div style="margin-bottom: 12px;">
      <div style="font-size: 13px; color: #9ca3af; margin-bottom: 6px;">{sym}</div>
      <div class="signal-row">
        <span class="signal-label">Liquidation Pressure</span>
        <span class="signal-value {'red' if ind.get('liq_pressure_score', 0) > 0.6 else 'yellow' if ind.get('liq_pressure_score', 0) > 0.3 else 'green'}">{ind.get('liq_pressure_score', 0):.2f}</span>
      </div>
      <div class="signal-row">
        <span class="signal-label">Cascade Warning</span>
        <span class="signal-value">{ind.get('cascade_warning', False)}</span>
      </div>
    </div>""" for sym, ind in leading.items())}
  </div>'''}

  <!-- Active Risk Events -->
  {"" if not cb_events else f'''
  <div class="card" style="border-color: #7f1d1d;">
    <h2 style="color: #f87171;">⚠️ Active Circuit Breaker Events</h2>
    {"".join(f'<div class="signal-row"><span class="signal-label">{e.get("trigger","?")}</span><span class="signal-value red">{e.get("value",0):.4f} (threshold: {e.get("threshold",0):.4f})</span></div>' for e in cb_events)}
  </div>'''}

  <!-- Allocations -->
  <div class="card">
    <h2>Current Allocation</h2>
    <div class="signal-row">
      <span class="signal-label">Kamino Lending</span>
      <span class="signal-value">{k_pct*100:.1f}%</span>
    </div>
    <div class="alloc-bar"><div class="alloc-fill" style="width: {k_pct*100:.1f}%"></div></div>
    <br/>
    <div class="signal-row">
      <span class="signal-label">Drift Spot Lending</span>
      <span class="signal-value">{d_pct*100:.1f}%</span>
    </div>
    <div class="alloc-bar"><div class="alloc-fill" style="width: {d_pct*100:.1f}%"></div></div>
    <br/>
    {"".join(f'''
    <div class="signal-row">
      <span class="signal-label">{sym} Perp (short)</span>
      <span class="signal-value">{pct*100:.1f}%</span>
    </div>
    <div class="alloc-bar"><div class="alloc-fill" style="width: {pct*100:.1f}%"></div></div><br/>
    ''' for sym, pct in perp_allocs.items())}
  </div>

  <div class="footer">
    QuantVault Keeper Bot &nbsp;|&nbsp; Devnet &nbsp;|&nbsp; Generated {ts}<br/>
    Next report in ~1 hour
  </div>

</div>
</body>
</html>
"""

    # Plain text fallback
    text = f"""QuantVault Report — {ts}

STATUS: {overall_status}
NAV: ${nav_usd:,.2f} | Expected APR: {exp_apr:.1f}%
Regime: {regime_name} ({regime_conf*100:.0f}%) | Scale: {pos_scale:.2f}x
Circuit Breaker: {cb_state} | Drawdown Halt: {dd_halted}
Rebalances: {int(rebal_total)} total, {error_rate:.1f}% error rate

FUNDING RATES:
{chr(10).join(f"  {sym}: {apr*100:.2f}% APR" for sym, apr in sorted(funding_rates.items(), key=lambda x: -x[1]))}

ALLOCATIONS:
  Kamino Lending: {k_pct*100:.1f}%
  Drift Spot:     {d_pct*100:.1f}%
{chr(10).join(f"  {sym} Perp: {pct*100:.1f}%" for sym, pct in perp_allocs.items())}
"""

    return html, text, ts


def send_email(html, text, ts):
    if not all([TO_EMAIL, FROM_EMAIL, EMAIL_PASS]):
        print("Email not configured — printing report to stdout instead.")
        print(text)
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"QuantVault Report {ts}"
    msg["From"]    = FROM_EMAIL
    msg["To"]      = TO_EMAIL

    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(FROM_EMAIL, EMAIL_PASS)
            server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
        print(f"✓ Report sent to {TO_EMAIL}")
        return True
    except Exception as e:
        print(f"✗ Email failed: {e}", file=sys.stderr)
        print(text)  # fallback to stdout
        return False


if __name__ == "__main__":
    print("Generating QuantVault hourly report...")
    html, text, ts = build_report()
    send_email(html, text, ts)
