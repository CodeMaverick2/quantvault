/**
 * Initialize a Drift user account for the keeper wallet on devnet.
 * Run once before starting the keeper bot.
 *
 * Usage: cd /home/tejas/quantvault/bot && node init_drift_user.js
 */
require("dotenv").config();

const { Connection, Keypair } = require("@solana/web3.js");
const { DriftClient, Wallet, initialize, getUserAccountPublicKeySync } = require("@drift-labs/sdk");

async function main() {
  const clusterEnv = process.env.CLUSTER ?? "devnet";
  const rpcUrl = process.env.RPC_URL ?? "https://api.devnet.solana.com";

  const key = process.env.KEEPER_PRIVATE_KEY;
  if (!key) throw new Error("KEEPER_PRIVATE_KEY not set");
  const keypair = Keypair.fromSecretKey(Uint8Array.from(JSON.parse(key)));

  console.log(`Wallet:  ${keypair.publicKey.toBase58()}`);
  console.log(`Cluster: ${clusterEnv}`);
  console.log(`RPC:     ${rpcUrl}`);

  const connection = new Connection(rpcUrl, { commitment: "confirmed" });
  const wallet = new Wallet(keypair);

  const sdkConfig = initialize({ env: clusterEnv });
  const driftClient = new DriftClient({
    connection,
    wallet,
    env: clusterEnv,
    accountSubscription: { type: "websocket" },
    perpMarketIndexes: [0, 1, 2],
    spotMarketIndexes: [0, 1],
    oracleInfos: sdkConfig.PERP_MARKETS.slice(0, 3).map((m) => ({
      publicKey: m.oracle,
      source: m.oracleSource,
    })),
  });

  await driftClient.subscribe();
  console.log("Drift client subscribed");

  // Derive user account address without requiring the account to exist
  const userAccountPublicKey = getUserAccountPublicKeySync(
    driftClient.program.programId,
    keypair.publicKey,
    0
  );
  console.log(`User account address: ${userAccountPublicKey.toBase58()}`);

  const accountInfo = await connection.getAccountInfo(userAccountPublicKey);
  if (accountInfo) {
    console.log("✓ Drift user account already exists — no action needed");
    await driftClient.unsubscribe();
    return;
  }

  console.log("Drift user account not found — initializing...");
  const [txSig] = await driftClient.initializeUserAccount();
  console.log(`✓ User account initialized! Tx: ${txSig}`);
  console.log(`  Explorer: https://explorer.solana.com/tx/${txSig}?cluster=devnet`);

  await driftClient.unsubscribe();
  console.log("Done.");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
