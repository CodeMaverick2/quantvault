/*!
 * QuantVault Delta-Neutral Adaptor
 *
 * A Voltr-compatible adaptor that tracks the delta-neutral perp position state
 * and reports position value back to the vault.
 *
 * This adaptor works alongside the existing Voltr Drift Adaptor (EBN93eXs5fHGBABuajQqdsKRkCgaqtJa8vEFD6vKXiP)
 * to provide strategy-level accounting: it records the notional position sizes,
 * entry prices, funding P&L, and health rate metrics that the keeper bot uses
 * for rebalancing decisions.
 *
 * The three required Voltr adaptor instructions are implemented:
 *   - initialize: set up the strategy state account
 *   - deposit:    record capital allocation, return position value
 *   - withdraw:   reduce allocation, return remaining value
 */

use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Mint};

pub mod errors;
pub mod instructions;
pub mod state;

use instructions::*;

declare_id!("DNAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

#[program]
pub mod delta_neutral_adaptor {
    use super::*;

    /// Initialize the strategy state account for this adaptor instance.
    /// Called once by the vault admin after registering the adaptor.
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        instructions::initialize::handler(ctx)
    }

    /// Record capital being deposited into the delta-neutral strategy.
    /// Returns the current estimated position value in USDC (u64, 6 decimals).
    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<u64> {
        instructions::deposit::handler(ctx, amount)
    }

    /// Record capital being withdrawn from the delta-neutral strategy.
    /// Returns the remaining position value after withdrawal (u64, 6 decimals).
    pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<u64> {
        instructions::withdraw::handler(ctx, amount)
    }

    /// Update position snapshot (called by keeper bot after each rebalance).
    /// Stores perp notional, entry price, and funding P&L for NAV calculation.
    pub fn update_position(
        ctx: Context<UpdatePosition>,
        perp_notional_usd: u64,
        entry_price: u64,
        funding_pnl: i64,
        health_rate_bps: u16,
    ) -> Result<()> {
        instructions::update_position::handler(
            ctx,
            perp_notional_usd,
            entry_price,
            funding_pnl,
            health_rate_bps,
        )
    }
}
