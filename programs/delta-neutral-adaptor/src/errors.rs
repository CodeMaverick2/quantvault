use anchor_lang::prelude::*;

#[error_code]
pub enum AdaptorError {
    #[msg("Amount is zero")]
    ZeroAmount,

    #[msg("Insufficient allocated amount for withdrawal")]
    InsufficientAllocation,

    #[msg("Unauthorized: caller is not the vault strategy authority")]
    Unauthorized,

    #[msg("Strategy is in an unhealthy state — withdrawal blocked")]
    UnhealthyPosition,

    #[msg("Arithmetic overflow")]
    Overflow,

    #[msg("Health rate below minimum required (1.05)")]
    HealthRateTooLow,
}
