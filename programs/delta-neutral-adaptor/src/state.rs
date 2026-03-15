use anchor_lang::prelude::*;

/// Strategy state account: tracks all position data for the delta-neutral strategy.
/// One instance per vault × adaptor combination.
#[account]
#[derive(Default)]
pub struct StrategyState {
    /// The vault that owns this strategy
    pub vault: Pubkey,

    /// The vault's strategy authority PDA (signer for CPI calls)
    pub vault_strategy_auth: Pubkey,

    /// Total USDC allocated to this strategy (6 decimals)
    pub allocated_amount: u64,

    /// Current perp position notional in USD (6 decimals)
    pub perp_notional_usd: u64,

    /// Weighted average entry price of the current short position (6 decimals)
    pub entry_price: u64,

    /// Cumulative realized + unrealized funding P&L (signed, 6 decimals)
    pub funding_pnl: i64,

    /// Health rate in basis points (10000 = 1.00)
    pub health_rate_bps: u16,

    /// Timestamp of last position update (Unix seconds)
    pub last_update_ts: i64,

    /// Bump seed for the strategy PDA
    pub bump: u8,

    /// Reserved for future use
    pub _reserved: [u8; 64],
}

impl StrategyState {
    pub const SIZE: usize = 8          // discriminator
        + 32                           // vault
        + 32                           // vault_strategy_auth
        + 8                            // allocated_amount
        + 8                            // perp_notional_usd
        + 8                            // entry_price
        + 8                            // funding_pnl
        + 2                            // health_rate_bps
        + 8                            // last_update_ts
        + 1                            // bump
        + 64;                          // _reserved

    /// Estimate current position value:
    /// allocated_amount + funding_pnl (as unrealized yield)
    pub fn position_value(&self) -> u64 {
        let pnl_adjusted = self.allocated_amount as i64 + self.funding_pnl;
        if pnl_adjusted < 0 { 0u64 } else { pnl_adjusted as u64 }
    }

    /// Health rate as a floating-point multiplier (e.g., 1.30 = 130%)
    pub fn health_rate_f64(&self) -> f64 {
        self.health_rate_bps as f64 / 10_000.0
    }

    /// Whether the position is within safe leverage limits (health rate > 1.05)
    pub fn is_healthy(&self) -> bool {
        self.health_rate_bps >= 10_500  // 1.05 × 10000
    }
}
