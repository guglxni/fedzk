# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Command Line Interface for FedZK.

This module provides a CLI for interacting with the FedZK system, 
allowing users to generate and verify zero-knowledge proofs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver


def setup_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="FedZK: Zero-Knowledge Proofs for Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command")

    # Setup command - just a wrapper for setup_zk.sh
    setup_parser = subparsers.add_parser(
        "setup",
        help="Setup ZK circuits and keys"
    )

    # Generate proof command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate ZK proof for gradients"
    )
    gen_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input file with gradient tensors (numpy .npz or JSON)"
    )
    gen_parser.add_argument(
        "--output", "-o",
        default="proof_output.json",
        help="Path to output proof file"
    )
    gen_parser.add_argument(
        "--secure", "-s",
        action="store_true",
        help="Use secure circuit with constraints"
    )
    gen_parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Use batch processing for large gradients"
    )
    gen_parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=4,
        help="Chunk size for batch processing"
    )
    gen_parser.add_argument(
        "--max-norm", "-m",
        type=float,
        default=100.0,
        help="Maximum L2 norm squared (for secure circuit)"
    )
    gen_parser.add_argument(
        "--min-active", "-a",
        type=int,
        default=1,
        help="Minimum non-zero elements (for secure circuit)"
    )
    gen_parser.add_argument(
        "--mpc-server", type=str, default=None,
        help="URL of MPC proof server to offload proof generation"
    )
    gen_parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for authenticating with the MPC proof server"
    )
    gen_parser.add_argument(
        "--fallback-disabled", action="store_true",
        help="Disable fallback to local proof when MPC server fails"
    )
    fallback_mode_default = os.getenv("FALLBACK_MODE", "silent")
    gen_parser.add_argument(
         "--fallback-mode", type=str, choices=["strict", "warn", "silent"],
         default=fallback_mode_default,
         help="Fallback mode for MPC generate_proof: strict raises error (no fallback), warn logs warning and falls back, silent falls back silently. Can also be set via FALLBACK_MODE env variable."
    )

    # Verify proof command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify ZK proof"
    )
    verify_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to proof file to verify"
    )
    verify_parser.add_argument(
        "--secure", "-s",
        action="store_true",
        help="Use secure circuit verification"
    )
    verify_parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Verify a batch proof"
    )
    verify_parser.add_argument(
        "--mpc-server", type=str, default=None,
        help="URL of MPC proof server to offload proof verification"
    )
    verify_parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for authenticating with the MPC proof server"
    )

    # MPC server commands
    mpc_parser = subparsers.add_parser("mpc", help="MPC proof server commands")
    mpc_subparsers = mpc_parser.add_subparsers(dest="mpc_command")

    serve_parser = mpc_subparsers.add_parser("serve", help="Serve the MPC proof HTTP API")
    serve_parser.add_argument("--host", "-H", default="127.0.0.1", help="Host to serve on")
    serve_parser.add_argument("--port", "-p", type=int, default=9000, help="Port to serve on")

    # Benchmark commands
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark commands for FedZK")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command")

    run_parser = benchmark_subparsers.add_parser("run", help="Run end-to-end benchmark")
    run_parser.add_argument("--clients", "-c", type=int, default=5,
                            help="Number of clients to simulate")
    run_parser.add_argument("--secure", "-s", action="store_true",
                            help="Use secure circuit generation and verification")
    run_parser.add_argument("--mpc-server", type=str, default=None,
                            help="URL of MPC proof server")
    run_parser.add_argument("--output", "-o", type=str, default="benchmark_report.json",
                            help="Output JSON report path")
    run_parser.add_argument("--csv", type=str, default=None,
                            help="Output CSV report path")
    run_parser.add_argument(
        "--report-url", type=str, default=None,
        help="URL to post benchmark report"
    )
    run_parser.add_argument(
        "--fallback-mode", type=str, choices=["strict", "warn", "silent"], default="warn",
        help="MPC fallback mode if server is unavailable"
    )
    run_parser.add_argument(
        "--input-size", type=int, default=10,
        help="Size of gradient tensor for benchmarking"
    )
    run_parser.add_argument(
        "--coordinator-host", type=str, default="127.0.0.1",
        help="Hostname for coordinator server"
    )
    run_parser.add_argument(
        "--coordinator-port", type=int, default=8000,
        help="Port for coordinator server"
    )

    return parser


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


def setup_command():
    """Run the setup_zk.sh script."""
    setup_script = Path(__file__).parent.parent / "scripts" / "setup_zk.sh"

    if not setup_script.exists():
        print(f"Error: Setup script not found at {setup_script}")
        return 1

    import subprocess
    print(f"Running ZK setup script: {setup_script}")
    result = subprocess.run(["bash", str(setup_script)], check=False)

    if result.returncode != 0:
        print("Setup failed. See error messages above.")
        return 1

    print("Setup completed successfully!")
    return 0


def generate_command(args):
    """Generate a ZK proof for the given gradients."""
    try:
        # Load gradient data
        print(f"Loading gradient data from {args.input}")
        gradient_dict = load_gradient_data(args.input)

        # Remote MPC generation with fallback support
        if args.mpc_server:
            if not args.api_key:
                print("Error: --api-key is required when using --mpc-server")
                return 1
            import requests
            # Prepare payload for remote MPC server
            if args.batch:
                # Load batch gradients from input file (expects JSON with gradient_batches)
                with open(args.input, "r") as f:
                    data = json.load(f)
                gradient_batches = data.get("gradient_batches")
                if not gradient_batches:
                    print("Error: Input JSON must contain 'gradient_batches' for batch mode")
                    return 1
                payload = {"batch": True, "gradient_batches": gradient_batches, "secure": args.secure}
            else:
                # Flatten single gradient dict into list
                grad_list = []
                for tensor in gradient_dict.values():
                    grad_list.extend(tensor.flatten().tolist())
                payload = {"gradients": grad_list, "secure": args.secure}
            headers = {"x-api-key": args.api_key}
            try:
                resp = requests.post(f"{args.mpc_server}/generate_proof", json=payload, headers=headers)
                resp.raise_for_status()
                proof_result = resp.json()
                # Save proof to file
                with open(args.output, "w") as f:
                    json.dump(proof_result, f, indent=2)
                print(f"Proof saved to {args.output}")
                return 0
            except Exception as e:
                # Fallback logic based on fallback_mode
                if args.fallback_disabled or args.fallback_mode == "strict":
                    print(f"Error: MPC generate_proof failed: {e}")
                    return 1

                from datetime import datetime
                fallback_info = {
                    "timestamp": datetime.now().isoformat(),
                    "exception": type(e).__name__,
                    "cli_command": " ".join(sys.argv) if sys.argv else "N/A",
                    "circuit": "secure" if args.secure else "standard"
                }
                logging.info(f"Fallback event: {json.dumps(fallback_info)}")

                if args.fallback_mode == "warn":
                    try:
                        from rich import print as rprint
                        rprint("[yellow]⚠️ MPC server failed, falling back to local ZK proof[/yellow]")
                    except ImportError:
                        print("⚠️ MPC server failed, falling back to local ZK proof")
                # Silent mode: no message printed
                # Continue to local proof generation

        # Setup paths
        zk_dir = Path("zk")
        if not zk_dir.exists():
            print("Error: ZK directory not found. Run 'fedzk setup' first.")
            return 1

        if args.secure:
            circuit_path = zk_dir / "model_update_secure.wasm"
            proving_key_path = zk_dir / "proving_key_secure.zkey"
        else:
            circuit_path = zk_dir / "model_update.wasm"
            proving_key_path = zk_dir / "proving_key.zkey"

        if not circuit_path.exists() or not proving_key_path.exists():
            print("Error: Required ZK files not found. Run 'fedzk setup' first.")
            return 1

        # Create prover
        print(f"Initializing ZKProver with {'secure' if args.secure else 'standard'} circuit")
        prover = ZKProver(str(circuit_path), str(proving_key_path))

        # Generate proof
        print("Generating proof...")
        if args.batch:
            if not args.secure:
                print("Error: Batch processing is only supported with secure circuit")
                return 1

            proof_result = prover.batch_generate_proof_secure(
                gradient_dict,
                chunk_size=args.chunk_size,
                max_norm=args.max_norm,
                min_active=args.min_active
            )

            print(f"Generated batch proof with {proof_result['metadata']['total_chunks']} chunks")

        elif args.secure:
            proof, public_inputs = prover.generate_real_proof_secure(
                gradient_dict,
                max_norm=args.max_norm,
                min_active=args.min_active
            )

            # Format for output
            proof_result = {
                "proof": proof,
                "public_inputs": public_inputs,
                "is_batch": False,
                "is_secure": True
            }

            print("Generated secure proof successfully")

        else:
            proof, public_inputs = prover.generate_real_proof(gradient_dict)

            # Format for output
            proof_result = {
                "proof": proof,
                "public_inputs": public_inputs,
                "is_batch": False,
                "is_secure": False
            }

            print("Generated standard proof successfully")

        # Save proof to file
        with open(args.output, "w") as f:
            json.dump(proof_result, f, indent=2)

        print(f"Proof saved to {args.output}")
        return 0

    except Exception as e:
        print(f"Error generating proof: {e}")
        return 1


def verify_command(args):
    """Verify a ZK proof."""
    try:
        # Remote MPC verification
        if args.mpc_server:
            if not args.api_key:
                print("Error: --api-key is required when using --mpc-server")
                return 1
            import requests
            # Load proof data
            with open(args.input, "r") as f:
                proof_result = json.load(f)
            payload = {
                "proof": proof_result.get("proof"),
                "public_inputs": proof_result.get("public_inputs"),
                "secure": args.secure or proof_result.get("is_secure", False)
            }
            headers = {"x-api-key": args.api_key}
            try:
                resp = requests.post(f"{args.mpc_server}/verify_proof", json=payload, headers=headers)
                if resp.status_code == 401:
                    print("Unauthorized: Invalid or missing API key")
                    return 1
                resp.raise_for_status()
            except Exception as e:
                print(f"Error: MPC verify_proof failed: {e}")
                return 1
            valid = resp.json().get("valid", False)
            if valid:
                print("✅ Proof VERIFIED successfully via MPC server!")
                return 0
            else:
                print("❌ Proof verification FAILED via MPC server!")
                return 1
        # Load proof data
        print(f"Loading proof from {args.input}")
        with open(args.input, "r") as f:
            proof_result = json.load(f)

        # Setup paths
        zk_dir = Path("zk")
        if not zk_dir.exists():
            print("Error: ZK directory not found. Run 'fedzk setup' first.")
            return 1

        if args.secure or proof_result.get("is_secure", False):
            verification_key_path = zk_dir / "verification_key_secure.json"
        else:
            verification_key_path = zk_dir / "verification_key.json"

        if not verification_key_path.exists():
            print("Error: Verification key not found. Run 'fedzk setup' first.")
            return 1

        # Create verifier
        print(f"Initializing ZKVerifier with {'secure' if args.secure else 'standard'} verification key")
        verifier = ZKVerifier(str(verification_key_path))

        # Verify proof
        print("Verifying proof...")
        if args.batch or proof_result.get("is_batch", False):
            verified, details = verifier.verify_batch_proof_secure(proof_result)

            if verified:
                print("✅ Batch proof VERIFIED successfully!")
                print(f"   - Total chunks: {details['total_chunks']}")
                print(f"   - Verified chunks: {details['verified_chunks']}")
                print(f"   - Merkle root verified: {details['merkle_root_verified']}")
            else:
                print("❌ Batch proof verification FAILED!")
                if details["merkle_root_verified"]:
                    print("   - Merkle root verified: Yes")
                    print("   - Some chunk proofs failed verification")
                else:
                    print("   - Merkle root verification failed")

                # Show details of failed chunks
                failed_chunks = [r for r in details["chunk_results"] if not r["verified"]]
                if failed_chunks:
                    print(f"   - Failed chunks: {len(failed_chunks)}")
                    for chunk in failed_chunks[:3]:
                        print(f"     - Chunk {chunk['chunk_index']}: {chunk['message']}")
                    if len(failed_chunks) > 3:
                        print(f"     - ... and {len(failed_chunks) - 3} more failed chunks")

        elif args.secure or proof_result.get("is_secure", False):
            verified = verifier.verify_real_proof_secure(
                proof_result["proof"],
                proof_result["public_inputs"]["public_inputs"]
            )

            if verified:
                print("✅ Secure proof VERIFIED successfully!")
            else:
                print("❌ Secure proof verification FAILED!")

        else:
            verified = verifier.verify_real_proof(
                proof_result["proof"],
                proof_result["public_inputs"]
            )

            if verified:
                print("✅ Standard proof VERIFIED successfully!")
            else:
                print("❌ Standard proof verification FAILED!")

        return 0 if verified else 1

    except Exception as e:
        print(f"Error verifying proof: {e}")
        return 1


def serve_mpc_command(args):
    """Serve the MPC proof HTTP API using Uvicorn."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required to serve MPC server. Install with `pip install uvicorn`.")
        return 1
    print(f"Starting MPC server at {args.host}:{args.port}")
    uvicorn.run("fedzk.mpc.server:app", host=args.host, port=args.port)
    return 0


def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "setup":
        return setup_command()
    elif args.command == "generate":
        return generate_command(args)
    elif args.command == "verify":
        return verify_command(args)
    elif args.command == "mpc":
        if args.mpc_command == "serve":
            return serve_mpc_command(args)
        else:
            parser.print_help()
            return 1
    elif args.command == "benchmark":
        if args.benchmark_command == "run":
            # Bare 'benchmark run' without flags: show help and exit
            import sys
            if len(sys.argv) <= 3:
                parser.print_help()
                return 1
            from fedzk.benchmark.end_to_end import run_benchmark
            run_benchmark(
                num_clients=args.clients,
                secure=args.secure,
                mpc_server=args.mpc_server,
                output_json=args.output,
                output_csv=args.csv,
                report_url=args.report_url,
                coordinator_host=args.coordinator_host,
                coordinator_port=args.coordinator_port,
                fallback_mode=args.fallback_mode,
                input_size=args.input_size
            )
            return 0
        # Unknown benchmark subcommand
        parser.print_help()
        return 1
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
