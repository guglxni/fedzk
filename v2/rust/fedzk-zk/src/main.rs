//! fedzk-zk — Phase 3 verify-first sidecar.
//!
//! Wire format: `fedzk.proof.v1` (snarkjs-shaped Groth16 JSON on BN254).
//! Cryptographic verify uses arkworks `ark-groth16` against snarkjs vkeys.

mod snarkjs_ark;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "fedzk-zk", version, about = "FEDzk ZK verify sidecar (Phase 3)")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Liveness for engine=auto health probes
    Health,
    /// Verify snarkjs-shaped Groth16 proof with arkworks (exit 0 = valid)
    Verify {
        #[arg(long)]
        vkey: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        public: PathBuf,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("fedzk-zk error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Health => {
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "service": "fedzk-zk",
                    "phase": "arkworks_verify",
                    "verify": "ark_groth16",
                    "wire": "fedzk.proof.v1",
                })
            );
            Ok(ExitCode::SUCCESS)
        }
        Commands::Verify { vkey, proof, public } => {
            let vkey_raw =
                fs::read_to_string(&vkey).with_context(|| format!("read vkey {}", vkey.display()))?;
            let proof_raw = fs::read_to_string(&proof)
                .with_context(|| format!("read proof {}", proof.display()))?;
            let public_raw = fs::read_to_string(&public)
                .with_context(|| format!("read public {}", public.display()))?;

            let vkey_json: Value = serde_json::from_str(&vkey_raw).context("parse vkey JSON")?;
            let proof_json: Value = serde_json::from_str(&proof_raw).context("parse proof JSON")?;
            let public_json: Value =
                serde_json::from_str(&public_raw).context("parse public JSON")?;

            if !public_json.is_array() {
                bail!("public inputs must be a JSON array");
            }

            let ok = snarkjs_ark::verify_snarkjs(&vkey_json, &proof_json, &public_json)
                .context("arkworks verify")?;
            if ok {
                println!("{{\"ok\":true,\"engine\":\"arkworks\"}}");
                Ok(ExitCode::SUCCESS)
            } else {
                eprintln!("fedzk-zk: proof INVALID");
                Ok(ExitCode::from(2))
            }
        }
    }
}
