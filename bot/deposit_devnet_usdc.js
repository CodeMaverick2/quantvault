/**
 * Deposit wSOL into the Drift user account (devnet).
 *
 * We have 10 devnet SOL. We deposit 5 SOL as collateral into Drift spot market 1 (SOL/wSOL).
 * The Drift SDK handles wrapping SOL → wSOL automatically.
 *
 * Usage: cd /home/tejas/quantvault/bot && node deposit_devnet_usdc.js
 */
require("dotenv").config();

const { Connection, Keypair, LAMPORTS_PER_SOL } = require("@solana/web3.js");
const {
  DriftClient,
  Wallet,
  initialize,
  BN,
  BASE_PRECISION,
} = require("@drift-labs/sdk");

const DEPOSIT_SOL = 5; // deposit 5 SOL (keep 5 for fees)

async function main() {
  const rpcUrl = process.env.RPC_URL ?? "https://api.devnet.solana.com";
  const key = process.env.KEEPER_PRIVATE_KEY;
  if (!key) throw new Error("KEEPER_PRIVATE_KEY not set");

  const keypair = Keypair.fromSecretKey(Uint8Array.from(JSON.parse(key)));
  const connection = new Connection(rpcUrl, { commitment: "confirmed" });
  const wallet = new Wallet(keypair);

  console.log(`Wallet:  ${keypair.publicKey.toBase58()}`);
  console.log(`Cluster: devnet`);
  console.log(`RPC:     ${rpcUrl}`);

  // Check current SOL balance
  const lamports = await connection.getBalance(keypair.publicKey);
  console.log(`SOL balance: ${(lamports / LAMPORTS_PER_SOL).toFixed(4)} SOL`);

  if (lamports < DEPOSIT_SOL * LAMPORTS_PER_SOL + 0.1 * LAMPORTS_PER_SOL) {
    throw new Error(`Insufficient SOL. Need ${DEPOSIT_SOL + 0.1} SOL, have ${lamports / LAMPORTS_PER_SOL}`);
  }

  // ── Connect to Drift ─────────────────────────────────────────────────────────
  console.log(`\n[1/2] Connecting to Drift...`);
  const sdkConfig = initialize({ env: "devnet" });

  const driftClient = new DriftClient({
    connection,
    wallet,
    env: "devnet",
    accountSubscription: { type: "websocket" },
    perpMarketIndexes: [0, 1, 2],
    spotMarketIndexes: [0, 1],
    oracleInfos: sdkConfig.PERP_MARKETS.slice(0, 3).map((m) => ({
      publicKey: m.oracle,
      source: m.oracleSource,
    })),
  });

  await driftClient.subscribe();
  console.log("  ✓ Drift subscribed");

  // ── Deposit SOL ──────────────────────────────────────────────────────────────
  // Drift spot market 1 = SOL (wSOL). SDK wraps SOL automatically.
  console.log(`\n[2/2] Depositing ${DEPOSIT_SOL} SOL into Drift (spot market 1)...`);

  // SOL precision is 1e9 (same as lamports)
  const depositAmt = new BN(DEPOSIT_SOL * LAMPORTS_PER_SOL);

  // For native SOL deposits, pass the wallet public key as the token account
  // The SDK wraps SOL → wSOL automatically when it sees the wallet's pubkey
  const depositTx = await driftClient.deposit(
    depositAmt,
    1,                    // spot market index 1 = SOL/wSOL
    keypair.publicKey,    // wallet pubkey = signal to wrap native SOL
    0,                    // sub-account id
    false                 // reduce only
  );

  console.log(`  ✓ Deposited ${DEPOSIT_SOL} SOL! Tx: ${depositTx}`);
  console.log(`    https://explorer.solana.com/tx/${depositTx}?cluster=devnet`);

  // ── Final NAV check ──────────────────────────────────────────────────────────
  await new Promise(r => setTimeout(r, 2000)); // wait for subscription update
  const user = driftClient.getUser();
  const nav = user.getTotalAssetValue();
  const navUsdc = nav.toNumber() / 1e6;
  console.log(`\n✅ Drift account total asset value: ~$${navUsdc.toFixed(2)}`);

  await driftClient.unsubscribe();
  console.log("Done.");
}

main().catch((err) => {
  console.error("Error:", err.stack ?? err);
  process.exit(1);
});
