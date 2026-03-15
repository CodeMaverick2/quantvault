use anchor_lang::prelude::*;
use crate::state::StrategyState;
use crate::errors::AdaptorError;

#[derive(Accounts)]
pub struct UpdatePosition<'info> {
    /// Only the vault strategy authority can update position data
    #[account(signer)]
    pub vault_strategy_auth: AccountInfo<'info>,

    /// The strategy state account
    #[account(
        mut,
        seeds = [
            b"strategy",
            vault_strategy_auth.key().as_ref(),
        ],
        bump = strategy.bump,
        has_one = vault_strategy_auth @ AdaptorError::Unauthorized,
    )]
    pub strategy: Account<'info, StrategyState>,
}

/// Update position snapshot after each rebalance cycle.
///
/// The keeper bot calls this after executing perp orders to keep the
/// on-chain state in sync with the actual Drift position.
///
/// Args:
///   perp_notional_usd: current total short notional (USDC, 6 decimals)
///   entry_price: weighted average entry price (6 decimals)
///   funding_pnl: cumulative funding P&L (signed, 6 decimals)
///   health_rate_bps: health rate × 10000 (e.g., 13000 = 1.30)
pub fn handler(
    ctx: Context<UpdatePosition>,
    perp_notional_usd: u64,
    entry_price: u64,
    funding_pnl: i64,
    health_rate_bps: u16,
) -> Result<()> {
    require!(
        health_rate_bps >= 10_500,
        AdaptorError::HealthRateTooLow
    );

    let strategy = &mut ctx.accounts.strategy;

    strategy.perp_notional_usd = perp_notional_usd;
    strategy.entry_price = entry_price;
    strategy.funding_pnl = funding_pnl;
    strategy.health_rate_bps = health_rate_bps;
    strategy.last_update_ts = Clock::get()?.unix_timestamp;

    msg!(
        "Position updated: notional={} entry={} funding_pnl={} health={}bps",
        perp_notional_usd,
        entry_price,
        funding_pnl,
        health_rate_bps
    );

    Ok(())
}
