export const MAINNET_CONFIG = {
  cluster: "mainnet-beta" as const,
  rpcUrl: process.env.MAINNET_RPC_URL || "https://mainnet.helius-rpc.com/?api-key=YOUR_HELIUS_KEY",
  wsUrl: process.env.MAINNET_WS_URL || "wss://mainnet.helius-rpc.com/?api-key=YOUR_HELIUS_KEY",

  programs: {
    voltrVault: "vVoLTRjQmtFpiYoegx285Ze4gsLJ8ZxgFKVcuvmG1a8",
    voltrLendingAdaptor: "aVoLTRCRt3NnnchvLYH6rMYehJHwM5m45RmLBZq7PGz",
    voltrDriftAdaptor: "EBN93eXs5fHGBABuajQqdsKRkCgaqtJa8vEFD6vKXiP",
    voltrKaminoAdaptor: "to6Eti9CsC5FGkAtqiPphvKD2hiQiLsS8zWiDBqBPKR",
    voltrJupiterAdaptor: "EW35URAx3LiM13fFK3QxAXfGemHso9HWPixrv7YDY4AM",
    drift: "dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH",
  },

  tokens: {
    usdcMint: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
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
