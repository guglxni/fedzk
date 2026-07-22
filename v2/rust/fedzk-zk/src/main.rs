//! fedzk-zk — Phase 3 verify-first sidecar (scaffold).
//!
//! Wire format: `fedzk.proof.v1` (snarkjs-shaped Groth16 JSON).
//! Crypto verify (arkworks) lands in a follow-up bolt; this binary is the
//! fail-closed surface so Python `FEDZK_ZK_BACKEND=rust` has something to call.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "fedzk-zk", version, about = "FEDzk ZK verify sidecar (Phase 3 scaffold)")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Liveness for engine=auto health probes
    Health,
    /// Validate proof envelope shape; arkworks pairing verify TBD (exit 3 = not wired)
    Verify {
        #[arg(long)]
        vkey: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        public: PathBuf,
    },
}

#[derive(Debug, Deserialize)]
struct SnarkProof {
    protocol: Option<String>,
    curve: Option<String>,
    pi_a: Option<Value>,
    pi_b: Option<Value>,
    pi_c: Option<Value>,
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
                    "phase": "scaffold",
                    "verify": "envelope_only",
                    "wire": "fedzk.proof.v1",
                })
            );
            Ok(ExitCode::SUCCESS)
        }
        Commands::Verify { vkey, proof, public } => verify_envelope(vkey, proof, public),
    }
}

fn verify_envelope(vkey: PathBuf, proof: PathBuf, public: PathBuf) -> Result<ExitCode> {
    let vkey_raw = fs::read_to_string(&vkey).with_context(|| format!("read vkey {}", vkey.display()))?;
    let proof_raw =
        fs::read_to_string(&proof).with_context(|| format!("read proof {}", proof.display()))?;
    let public_raw =
        fs::read_to_string(&public).with_context(|| format!("read public {}", public.display()))?;

    let vkey_json: Value = serde_json::from_str(&vkey_raw).context("parse vkey JSON")?;
    let proof: SnarkProof = serde_json::from_str(&proof_raw).context("parse proof JSON")?;
    let public: Value = serde_json::from_str(&public_raw).context("parse public JSON")?;

    if vkey_json.get("protocol").and_then(|p| p.as_str()) != Some("groth16")
        && vkey_json.get("vk_alpha_1").is_none()
    {
        bail!("vkey missing groth16 fields (expected snarkjs verification_key.json)");
    }
    if proof.pi_a.is_none() || proof.pi_b.is_none() || proof.pi_c.is_none() {
        bail!("proof missing pi_a/pi_b/pi_c");
    }
    if !public.is_array() {
        bail!("public inputs must be a JSON array");
    }

    // Fail-closed for cryptographic verify until arkworks bolt lands.
    eprintln!(
        "fedzk-zk: envelope OK (protocol={:?} curve={:?} publics={}); arkworks pairing verify not wired yet",
        proof.protocol.as_deref().unwrap_or("?"),
        proof.curve.as_deref().unwrap_or("?"),
        public.as_array().map(|a| a.len()).unwrap_or(0)
    );
    Ok(ExitCode::from(3))
}
