use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Mint};

use crate::state::StrategyState;
use crate::errors::AdaptorError;

#[derive(Accounts)]
pub struct Withdraw<'info> {
    /// The vault's strategy authority PDA
    #[account(signer)]
    pub vault_strategy_auth: AccountInfo<'info>,

    /// The strategy state account
    #[account(
        mut,
        seeds = [
            b"strategy",
            vault_strategy_auth.key().as_ref(),
            vault_asset_mint.key().as_ref(),
        ],
        bump = strategy.bump,
        has_one = vault_strategy_auth @ AdaptorError::Unauthorized,
    )]
    pub strategy: Account<'info, StrategyState>,

    /// The vault's base asset mint (USDC)
    pub vault_asset_mint: Account<'info, Mint>,

    /// The vault's token account — destination for withdrawn funds
    #[account(mut)]
    pub vault_strategy_asset_ata: Account<'info, TokenAccount>,

    pub asset_token_program: Program<'info, Token>,
}

/// Records a capital withdrawal from the delta-neutral strategy.
///
/// Returns the remaining position value after withdrawal (USDC, 6 decimals).
/// The keeper bot is responsible for actually closing perp positions before
/// this instruction is called (so funds are available in the token account).
pub fn handler(ctx: Context<Withdraw>, amount: u64) -> Result<u64> {
    require!(amount > 0, AdaptorError::ZeroAmount);

    let strategy = &mut ctx.accounts.strategy;

    // Enforce minimum health rate before allowing withdrawal
    require!(strategy.is_healthy(), AdaptorError::HealthRateTooLow);

    require!(
        strategy.allocated_amount >= amount,
        AdaptorError::InsufficientAllocation
    );

    strategy.allocated_amount = strategy
        .allocated_amount
        .checked_sub(amount)
        .ok_or(AdaptorError::Overflow)?;

    // Scale down perp notional proportionally
    if strategy.allocated_amount == 0 {
        strategy.perp_notional_usd = 0;
        strategy.entry_price = 0;
        strategy.funding_pnl = 0;
    }

    strategy.last_update_ts = Clock::get()?.unix_timestamp;

    let remaining_value = strategy.position_value();

    msg!(
        "Withdraw: amount={} remaining_allocated={} remaining_value={}",
        amount,
        strategy.allocated_amount,
        remaining_value
    );

    Ok(remaining_value)
}
