export const DEVNET_CONFIG = {
  cluster: "devnet" as const,
  rpcUrl: process.env.DEVNET_RPC_URL || "https://api.devnet.solana.com",
  wsUrl: process.env.DEVNET_WS_URL || "wss://api.devnet.solana.com",

  programs: {
    voltrVault: "vVoLTRjQmtFpiYoegx285Ze4gsLJ8ZxgFKVcuvmG1a8",
    voltrLendingAdaptor: "aVoLTRCRt3NnnchvLYH6rMYehJHwM5m45RmLBZq7PGz",
    voltrDriftAdaptor: "EBN93eXs5fHGBABuajQqdsKRkCgaqtJa8vEFD6vKXiP",
    voltrKaminoAdaptor: "to6Eti9CsC5FGkAtqiPphvKD2hiQiLsS8zWiDBqBPKR",
    voltrJupiterAdaptor: "EW35URAx3LiM13fFK3QxAXfGemHso9HWPixrv7YDY4AM",
    drift: "dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH",
  },

  tokens: {
    usdcMint: "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU", // devnet USDC
    solMint: "So11111111111111111111111111111111111111112",
  },

  drift: {
    perpMarkets: {
      SOL: 0,
      BTC: 1,
      ETH: 2,
    },
    spotMarkets: {
      USDC: 0,
      SOL: 1,
    },
  },

  // Filled by init_vault.ts
  vault: {
    address: process.env.VAULT_ADDRESS || "",
    lookupTable: process.env.LOOKUP_TABLE_ADDRESS || "",
  },

  strategyEngine: {
    url: process.env.STRATEGY_ENGINE_URL || "http://localhost:8000",
  },

  metrics: {
    port: 9090,
  },
};
