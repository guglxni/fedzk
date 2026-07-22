//! fedzk-zk — Phase 3 verify-first sidecar (+ optional Axum HTTP).

mod snarkjs_ark;

use anyhow::{bail, Context, Result};
use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::net::SocketAddr;
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
    /// HTTP sidecar: GET /healthz, POST /verify
    Serve {
        #[arg(long, default_value = "127.0.0.1:8787")]
        bind: String,
    },
}

#[derive(Debug, Deserialize)]
struct VerifyBody {
    vkey: Value,
    proof: Value,
    public: Value,
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
                json!({
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
        Commands::Serve { bind } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(serve(bind))?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

async fn serve(bind: String) -> Result<()> {
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/verify", post(verify_http));
    let addr: SocketAddr = bind.parse().context("parse bind addr")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("fedzk-zk listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn healthz() -> impl IntoResponse {
    Json(json!({
        "ok": true,
        "service": "fedzk-zk",
        "wire": "fedzk.proof.v1",
        "verify": "ark_groth16",
    }))
}

async fn verify_http(Json(body): Json<VerifyBody>) -> impl IntoResponse {
    if !body.public.is_array() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"ok": false, "error": "public must be array"})),
        );
    }
    match snarkjs_ark::verify_snarkjs(&body.vkey, &body.proof, &body.public) {
        Ok(true) => (StatusCode::OK, Json(json!({"ok": true, "engine": "arkworks"}))),
        Ok(false) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"ok": false, "error": "invalid_proof"})),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"ok": false, "error": format!("{e:#}")})),
        ),
    }
}
