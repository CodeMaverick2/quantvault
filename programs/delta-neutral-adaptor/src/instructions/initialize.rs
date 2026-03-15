use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Mint};

use crate::state::StrategyState;
use crate::errors::AdaptorError;

#[derive(Accounts)]
pub struct Initialize<'info> {
    /// The vault's strategy authority PDA — signs on behalf of the vault.
    /// This is the only account authorized to call deposit/withdraw.
    #[account(signer)]
    pub vault_strategy_auth: AccountInfo<'info>,

    /// The strategy state account — PDA derived from vault + adaptor.
    #[account(
        init,
        payer = payer,
        space = StrategyState::SIZE,
        seeds = [
            b"strategy",
            vault_strategy_auth.key().as_ref(),
            vault_asset_mint.key().as_ref(),
        ],
        bump,
    )]
    pub strategy: Account<'info, StrategyState>,

    /// The vault's base asset mint (USDC)
    pub vault_asset_mint: Account<'info, Mint>,

    /// The vault's token account for the base asset
    #[account(
        associated_token::mint = vault_asset_mint,
        associated_token::authority = vault_strategy_auth,
    )]
    pub vault_strategy_asset_ata: Account<'info, TokenAccount>,

    /// Payer for account rent (typically the vault admin)
    #[account(mut)]
    pub payer: Signer<'info>,

    pub system_program: Program<'info, System>,
    pub token_program: Program<'info, Token>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handler(ctx: Context<Initialize>) -> Result<()> {
    let strategy = &mut ctx.accounts.strategy;

    strategy.vault = ctx.accounts.vault_strategy_auth.key();
    strategy.vault_strategy_auth = ctx.accounts.vault_strategy_auth.key();
    strategy.allocated_amount = 0;
    strategy.perp_notional_usd = 0;
    strategy.entry_price = 0;
    strategy.funding_pnl = 0;
    strategy.health_rate_bps = 13_000; // 1.30 initial (conservative)
    strategy.last_update_ts = Clock::get()?.unix_timestamp;
    strategy.bump = ctx.bumps.strategy;
    strategy._reserved = [0u8; 64];

    msg!(
        "QuantVault delta-neutral adaptor initialized for vault_auth: {}",
        ctx.accounts.vault_strategy_auth.key()
    );

    Ok(())
}
