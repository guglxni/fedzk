# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Command Line Interface for FedZK.

This module provides a CLI for interacting with the FedZK system, 
allowing users to generate and verify zero-knowledge proofs.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver

# Create Typer app
app = typer.Typer(help="FedZK: Zero-Knowledge Proofs for Federated Learning")
client_app = typer.Typer(help="Client commands for local training and proof generation")
mpc_app = typer.Typer(help="MPC proof server commands")
benchmark_app = typer.Typer(help="Benchmark commands for FedZK")

# Register sub-apps
app.add_typer(client_app, name="client")
app.add_typer(mpc_app, name="mpc")
app.add_typer(benchmark_app, name="benchmark")


def load_gradient_data(input_path):
    """Load gradient data from a file (npz or JSON)."""
    input_path = Path(input_path)

    if input_path.suffix == ".npz":
        # Load from numpy .npz file
        np_data = np.load(input_path)
        gradient_dict = {}

        for key in np_data.files:
            gradient_dict[key] = torch.tensor(np_data[key])

        return gradient_dict

    elif input_path.suffix == ".json":
        # Load from JSON file
        with open(input_path, "r") as f:
            data = json.load(f)

        gradient_dict = {}
        for key, value in data.items():
            gradient_dict[key] = torch.tensor(value)

        return gradient_dict

    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


@app.command("setup")
def setup_command():
    """Setup ZK circuits and keys."""
    # This is just a wrapper for setup_zk.sh
    typer.echo("Setting up ZK circuits and keys...")

    # Check if setup_zk.sh exists and is executable
    setup_script = Path(__file__).parent / "scripts" / "setup_zk.sh"
    if not setup_script.exists():
        typer.echo(f"Error: Setup script not found at {setup_script}")
        raise typer.Exit(code=1)

    import subprocess
    result = subprocess.run([str(setup_script)], check=False)

    if result.returncode != 0:
        typer.echo("Error: Setup failed")
        raise typer.Exit(code=result.returncode)

    typer.echo("Setup completed successfully")


@app.command("generate")
def generate_command(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input file with gradient tensors"),
    output: str = typer.Option("proof_output.json", "--output", "-o", help="Path to output proof file"),
    secure: bool = typer.Option(False, "--secure", "-s", help="Use secure circuit with constraints"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Use batch processing for large gradients"),
    chunk_size: int = typer.Option(4, "--chunk-size", "-c", help="Chunk size for batch processing"),
    max_norm: float = typer.Option(100.0, "--max-norm", "-m", help="Maximum L2 norm squared (for secure circuit)"),
    min_active: int = typer.Option(1, "--min-active", "-a", help="Minimum non-zero elements (for secure circuit)"),
    mpc_server: Optional[str] = typer.Option(None, "--mpc-server", help="URL of MPC proof server to offload proof generation"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for authenticating with the MPC proof server"),
    fallback_disabled: bool = typer.Option(False, "--fallback-disabled", help="Disable fallback to local proof when MPC server fails"),
    fallback_mode: str = typer.Option(os.getenv("FALLBACK_MODE", "silent"), "--fallback-mode", help="Fallback mode for MPC generate_proof")
):
    """Generate ZK proof for gradients."""
    typer.echo(f"Generating proof from {input_file}...")

    try:
        gradient_dict = load_gradient_data(input_file)
    except Exception as e:
        typer.echo(f"Error loading gradient data: {e}")
        raise typer.Exit(code=1)

    if secure:
        typer.echo("Using secure circuit with constraints")

    if batch:
        typer.echo(f"Using batch processing with chunk size {chunk_size}")

    # Initialize the prover
    if batch:
        from fedzk.prover.batch_zkgenerator import BatchZKProver
        prover = BatchZKProver(
            chunk_size=chunk_size,
            secure=secure,
            max_norm_squared=max_norm,
            min_active=min_active
        )
    else:
        prover = ZKProver(
            secure=secure,
            max_norm_squared=max_norm,
            min_active=min_active
        )

    # Generate the proof
    try:
        if mpc_server:
            typer.echo(f"Using MPC server at {mpc_server} for proof generation")
            from fedzk.mpc.client import MPCClient

            mpc_client = MPCClient(
                server_url=mpc_server,
                api_key=api_key,
                fallback_disabled=fallback_disabled,
                fallback_mode=fallback_mode
            )

            result = mpc_client.generate_proof(
                gradient_dict,
                secure=secure,
                batch=batch,
                chunk_size=chunk_size if batch else None,
                max_norm_squared=max_norm if secure else None,
                min_active=min_active if secure else None
            )
        else:
            # Generate proof locally
            result = prover.generate_proof(gradient_dict)

        # Save the proof to file
        with open(output, "w") as f:
            json.dump(result, f, indent=2)

        typer.echo(f"Proof saved to {output}")

    except Exception as e:
        typer.echo(f"Error generating proof: {e}")
        raise typer.Exit(code=1)


@app.command("verify")
def verify_command(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to proof file to verify"),
    secure: bool = typer.Option(False, "--secure", "-s", help="Use secure circuit verification"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Verify a batch proof"),
    mpc_server: Optional[str] = typer.Option(None, "--mpc-server", help="URL of MPC proof server"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the MPC server")
):
    """Verify ZK proof."""
    typer.echo(f"Verifying proof from {input_file}...")

    try:
        with open(input_file, "r") as f:
            proof_data = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading proof data: {e}")
        raise typer.Exit(code=1)

    if secure:
        typer.echo("Using secure circuit verification")

    if batch:
        typer.echo("Verifying batch proof")

    # Initialize the verifier
    if batch:
        from fedzk.prover.batch_zkgenerator import BatchZKVerifier
        verifier = BatchZKVerifier(secure=secure)
    else:
        # Determine the verification key path based on the secure flag
        from pathlib import Path
        ASSET_DIR = Path(__file__).resolve().parent / "zk"
        vkey_path = str(ASSET_DIR / "verification_key_secure.json" if secure else "verification_key.json")
        verifier = ZKVerifier(verification_key_path=vkey_path)

    # Verify the proof
    try:
        if mpc_server:
            typer.echo(f"Using MPC server at {mpc_server} for proof verification")
            from fedzk.mpc.client import MPCClient

            mpc_client = MPCClient(
                server_url=mpc_server,
                api_key=api_key
            )

            is_valid = mpc_client.verify_proof(
                proof_data,
                secure=secure,
                batch=batch
            )
        else:
            # Verify proof locally
            if batch:
                is_valid = verifier.verify_proof(proof_data)
            else:
                # The ZKVerifier from verifier.py takes proof and public_inputs separately
                if isinstance(proof_data, list) and len(proof_data) >= 2:
                    # Handle tuple-like format: [proof_dict, public_inputs_list]
                    proof, public_inputs = proof_data[0], proof_data[1]
                    is_valid = verifier.verify_real_proof(proof, public_inputs)
                elif isinstance(proof_data, dict) and "proof" in proof_data and "public_inputs" in proof_data:
                    # Handle object format: {"proof": {...}, "public_inputs": [...]}
                    is_valid = verifier.verify_real_proof(proof_data["proof"], proof_data["public_inputs"])
                else:
                    typer.echo("Error: Unrecognized proof data format")
                    raise typer.Exit(code=1)

        if is_valid:
            typer.echo("✅ Proof verification succeeded!")
        else:
            typer.echo("❌ Proof verification failed!")
            raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"Error verifying proof: {e}")
        raise typer.Exit(code=1)


@mpc_app.command("serve")
def serve_mpc_command(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to serve on"),
    port: int = typer.Option(9000, "--port", "-p", help="Port to serve on")
):
    """Serve the MPC proof HTTP API."""
    typer.echo(f"Starting MPC server on {host}:{port}...")

    try:
        from fedzk.mpc.server import run_server
        run_server(host=host, port=port)
    except ImportError:
        typer.echo("Error: FastAPI not installed. Install with 'pip install -e \".[all]\"'")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error starting MPC server: {e}")
        raise typer.Exit(code=1)


@benchmark_app.command("run")
def benchmark_run_command(
    clients: int = typer.Option(5, "--clients", "-c", help="Number of clients to simulate"),
    secure: bool = typer.Option(False, "--secure", "-s", help="Use secure ZK circuits with constraints"),
    mpc_server: Optional[str] = typer.Option(None, "--mpc-server", help="URL of MPC proof server"),
    output: str = typer.Option("benchmark_report.json", "--output", "-o", help="Output JSON report path"),
    csv: Optional[str] = typer.Option(None, "--csv", help="Output CSV report path"),
    report_url: Optional[str] = typer.Option(None, "--report-url", help="URL to post benchmark report"),
    fallback_mode: str = typer.Option("warn", "--fallback-mode", help="MPC fallback mode if server is unavailable"),
    input_size: int = typer.Option(10, "--input-size", help="Size of gradient tensor for benchmarking"),
    coordinator_host: str = typer.Option("127.0.0.1", "--coordinator-host", help="Hostname for coordinator server"),
    coordinator_port: int = typer.Option(8000, "--coordinator-port", help="Port for coordinator server")
):
    """Run end-to-end benchmarks."""
    typer.echo(f"Running benchmark with {clients} clients...")

    try:
        from fedzk.benchmark.end_to_end import run_benchmark
        run_benchmark(
            num_clients=clients,
            secure=secure,
            mpc_server=mpc_server,
            output_json=output,
            output_csv=csv,
            report_url=report_url,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
            fallback_mode=fallback_mode,
            input_size=input_size
        )
        typer.echo(f"Benchmark report saved to {output}")
        if csv:
            typer.echo(f"Benchmark CSV report saved to {csv}")

    except ImportError as e:
        typer.echo(f"Error importing benchmark runner: {e}")
        typer.echo("Ensure benchmark dependencies are installed correctly.")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error running benchmark: {e}")
        raise typer.Exit(code=1)


@client_app.command("train")
def client_train_command():
    """Train a model locally."""
    typer.echo("Training model locally...")
    # Placeholder for actual training logic
    typer.echo("Model training completed.")


@client_app.command("prove")
def client_prove_command():
    """Generate zero-knowledge proof for model updates."""
    typer.echo("Generating zero-knowledge proof for model updates...")
    # Placeholder for proof generation logic
    typer.echo("Proof generation completed.")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
