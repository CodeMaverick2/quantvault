use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Mint};

use crate::state::StrategyState;
use crate::errors::AdaptorError;

#[derive(Accounts)]
pub struct Deposit<'info> {
    /// The vault's strategy authority PDA — must sign to authorize the deposit
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

    /// The vault's token account — source of the deposited funds
    #[account(mut)]
    pub vault_strategy_asset_ata: Account<'info, TokenAccount>,

    pub asset_token_program: Program<'info, Token>,
}

/// Records a capital deposit into the delta-neutral strategy.
///
/// Returns the current estimated total position value in USDC (6 decimals).
/// The vault uses this value to track strategy share of total AUM.
pub fn handler(ctx: Context<Deposit>, amount: u64) -> Result<u64> {
    require!(amount > 0, AdaptorError::ZeroAmount);

    let strategy = &mut ctx.accounts.strategy;

    strategy.allocated_amount = strategy
        .allocated_amount
        .checked_add(amount)
        .ok_or(AdaptorError::Overflow)?;

    strategy.last_update_ts = Clock::get()?.unix_timestamp;

    let position_value = strategy.position_value();

    msg!(
        "Deposit: amount={} allocated_total={} position_value={}",
        amount,
        strategy.allocated_amount,
        position_value
    );

    Ok(position_value)
}
