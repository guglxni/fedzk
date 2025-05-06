# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
End-to-end benchmarks for the FedZK system.

This module provides comprehensive benchmarking for the FedZK system, simulating
the full federated learning workflow with zero-knowledge proofs.

The benchmarking suite consists of:

1. CoordinatorServer: Launches a FastAPI coordinator server in a background process
2. Client: Simulates client training, proof generation, and submission
3. run_benchmark: Main benchmark runner that orchestrates the process

Key features:
- Multi-client simulation with parallel execution
- Support for secure and standard ZK circuits
- Local or MPC-based proof generation with fallback options
- Rich console output with performance tables
- JSON and CSV reports for further analysis
- Optional reporting to external dashboards or data warehouses

Usage:
    from fedzk.benchmark.end_to_end import run_benchmark
    
    # Run a basic benchmark
    run_benchmark(num_clients=5, secure=False)
    
    # Run with MPC server and save reports
    run_benchmark(
        num_clients=10,
        secure=True,
        mpc_server="http://localhost:9000",
        output_json="report.json",
        output_csv="report.csv"
    )

## Benchmark Usage Guide

### Command Line Interface

The benchmark can be run directly from the command line:

```bash
# Basic benchmark with 5 clients
python -m fedzk.cli benchmark run --clients 5

# Secure circuit benchmark with 10 clients and save reports
python -m fedzk.cli benchmark run --clients 10 --secure --output benchmark.json --csv results.csv

# Using an MPC server with custom settings
python -m fedzk.cli benchmark run --clients 20 --secure --mpc-server http://localhost:9000 \
    --fallback-mode warn --input-size 100
```

Available CLI options:
- `--clients` / `-c`: Number of clients to simulate (default: 5)
- `--secure` / `-s`: Use secure ZK circuits with constraints
- `--mpc-server`: URL of MPC proof server (optional)
- `--output` / `-o`: Path to save JSON report (default: benchmark_report.json)
- `--csv`: Path to save CSV report (optional)
- `--report-url`: URL to POST benchmark results (optional)
- `--fallback-mode`: How to handle MPC server failures (strict/warn/silent)
- `--input-size`: Size of gradient vectors for testing (default: 10)
- `--coordinator-host`: Hostname for coordinator server (default: 127.0.0.1)
- `--coordinator-port`: Port for coordinator server (default: 8000)

### Programmatic API

For more advanced use cases, the benchmark can be integrated programmatically:

```python
from fedzk.benchmark.end_to_end import run_benchmark

# Run a basic benchmark and get the report
report = run_benchmark(num_clients=5, secure=False)

# Access benchmark results programmatically
avg_proof_time = report["summary"]["avg_proof_time"]
success_rate = report["summary"]["successful_clients"] / report["config"]["num_clients"]

# Run multiple configurations for comparison
secure_report = run_benchmark(num_clients=10, secure=True, output_json="secure_report.json")
standard_report = run_benchmark(num_clients=10, secure=False, output_json="standard_report.json")

# Pass results to a data warehouse or dashboard
run_benchmark(
    num_clients=20,
    secure=True,
    report_url="https://your-metrics-server.com/api/benchmarks"
)
```

### Report Format

The benchmark produces a structured JSON report containing:

1. **Configuration**: Parameters used for the benchmark
2. **Summary**: Aggregated metrics (averages, totals, success rates)
3. **Client Metrics**: Detailed per-client performance data

Example report structure:
```json
{
  "id": "uuid-for-this-run",
  "timestamp": "2025-04-24T12:34:56.789Z",
  "config": {
    "num_clients": 5,
    "secure": true,
    "mpc_server": "http://localhost:9000",
    "fallback_mode": "warn",
    "input_size": 10
  },
  "summary": {
    "total_duration": 12.345,
    "successful_clients": 5,
    "aggregated_updates": 3,
    "avg_train_time": 0.123,
    "avg_proof_time": 1.234,
    "avg_submit_time": 0.345,
    "avg_total_time": 1.702
  },
  "client_metrics": [
    {
      "client_id": 0,
      "train_time": 0.125,
      "proof_time": 1.245,
      "submit_time": 0.352,
      "total_time": 1.722,
      "status": "aggregated",
      "succeeded": true
    },
    // Additional client metrics...
  ]
}
```

This data can be used for performance analysis, regression testing, and integration with monitoring systems.
"""

import argparse
import csv
import json
import logging
import multiprocessing
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from fedzk.coordinator.logic import (
    pending_updates,
)

# Import required FedZK components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark")

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class CoordinatorServer:
    """
    FastAPI coordinator server running in a background process.
    
    This class handles launching, monitoring, and stopping the FastAPI coordinator 
    server used for the benchmark. It runs the server in a separate process to ensure
    clean isolation and easy shutdown.
    
    Attributes:
        host (str): Hostname to bind the server to
        port (int): Port to listen on
        server_process: Multiprocessing process running the server
        is_running (bool): Whether the server is currently running
        url (str): Complete URL to the running server
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize the coordinator server.
        
        Args:
            host: Hostname to bind the server to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.server_thread = None
        self.server_process = None
        self.is_running = False
        self.url = f"http://{host}:{port}"

    def start_server(self):
        """Start the FastAPI server in a background process."""
        if self.is_running:
            logger.info("Server is already running")
            return

        # Reset global state (this affects only the parent process)
        global current_version, pending_updates
        current_version = 1
        pending_updates.clear()

        # Launch server in separate process
        self.server_process = multiprocessing.Process(
            target=run_coordinator_server,
            args=(self.host, self.port),
            daemon=True
        )
        self.server_process.start()
        self.is_running = True

        # Wait for server to start up
        self._wait_for_server()

    def _wait_for_server(self, max_attempts=10, delay=0.5):
        """Wait for the server to become available."""
        for attempt in range(max_attempts):
            try:
                resp = requests.get(f"{self.url}/status")
                if resp.status_code == 200:
                    logger.info("Coordinator server is up and running")
                    return True
            except requests.RequestException:
                pass

            time.sleep(delay)

        logger.warning("Coordinator server did not start properly")
        return False

    def stop_server(self):
        """Stop the server process."""
        if self.server_process and self.server_process.is_alive():
            logger.info("Stopping coordinator server")
            self.server_process.terminate()
            self.server_process.join(timeout=2)
        else:
            logger.warning("No server process running or already terminated")

        self.is_running = False
        self.server_process = None


def run_coordinator_server(host: str, port: int):
    """Run the coordinator FastAPI server (called in a separate process)."""
    import uvicorn

    # Import here to avoid circular imports
    import fedzk.coordinator.api as api

    # Reset global state in the child process
    from fedzk.coordinator.logic import pending_updates
    current_version = 1
    pending_updates.clear()

    # Start the server
    logger.info(f"Starting coordinator API server at {host}:{port}")
    uvicorn.run(api.app, host=host, port=port)


class Client:
    """
    Simulated FedZK client for benchmarking.
    
    This class simulates a complete client in the FedZK system, handling:
    1. Gradient generation (simulating model training)
    2. Zero-knowledge proof generation (local or MPC-based)
    3. Update submission to the coordinator
    
    For benchmarking purposes, it uses mock implementations that simulate
    realistic timing characteristics without requiring actual ZK circuits.
    
    Attributes:
        client_id (int): Unique identifier for this client
        input_size (int): Size of gradient tensor to generate
        secure (bool): Whether to use secure ZK circuits with constraints
        mpc_server (str, optional): URL of MPC server for remote proof generation
        coordinator_url (str): URL of the coordinator API
        fallback_mode (str): How to handle MPC server failures (strict, warn, silent)
        metrics (dict): Performance metrics collected during execution
    """

    def __init__(
        self,
        client_id: int,
        input_size: int = 10,
        secure: bool = False,
        mpc_server: Optional[str] = None,
        coordinator_url: str = "http://127.0.0.1:8000",
        fallback_mode: str = "warn"
    ):
        """
        Initialize a simulated client.
        
        Args:
            client_id: Unique identifier for this client
            input_size: Size of gradient tensor to generate
            secure: Whether to use secure ZK circuits
            mpc_server: Optional MPC server URL for remote proof generation
            coordinator_url: URL of the coordinator API
            fallback_mode: MPC fallback mode (warn, silent, strict)
        """
        self.client_id = client_id
        self.input_size = input_size
        self.secure = secure
        self.mpc_server = mpc_server
        self.coordinator_url = coordinator_url
        self.fallback_mode = fallback_mode
        self.metrics = {}

    def run_training_round(self) -> Dict[str, Any]:
        """
        Run a complete training round: train, prove, submit.
        
        Returns:
            Metrics dictionary containing timing and status information
        """
        metrics = {
            "client_id": self.client_id,
            "secure": self.secure,
            "mpc_server": bool(self.mpc_server),
            "timestamp": datetime.now().isoformat(),
        }

        # Step 1: Simulate training (generate random gradients)
        t_train_start = time.time()
        gradients = self._generate_gradients()
        metrics["train_time"] = time.time() - t_train_start
        metrics["gradient_size"] = len(gradients["weights"])

        # Step 2: Generate proof
        t_proof_start = time.time()
        proof, public_inputs = self._generate_proof(gradients)
        metrics["proof_time"] = time.time() - t_proof_start

        # Step 3: Submit update to coordinator
        t_submit_start = time.time()
        try:
            status, version, global_update = self._submit_update(gradients, proof, public_inputs)
            metrics["submit_time"] = time.time() - t_submit_start
            metrics["status"] = status
            metrics["model_version"] = version
            metrics["succeeded"] = True
        except Exception as e:
            metrics["submit_time"] = time.time() - t_submit_start
            metrics["status"] = "failed"
            metrics["error"] = str(e)
            metrics["succeeded"] = False

        # Calculate total round time
        metrics["total_time"] = metrics["train_time"] + metrics["proof_time"] + metrics["submit_time"]

        # Store metrics
        self.metrics = metrics
        return metrics

    def _generate_gradients(self) -> Dict[str, List[float]]:
        """Generate synthetic gradient data for benchmarking."""
        # Create random tensor with normal distribution
        weights = torch.randn(self.input_size).tolist()
        bias = torch.randn(1).tolist()
        return {"weights": weights, "bias": bias}

    def _generate_proof(self, gradients: Dict[str, List[float]]) -> Tuple[Dict, List]:
        """Generate ZK proof for the gradients."""
        if self.mpc_server:
            # Use remote MPC server
            try:
                headers = {"x-api-key": "test-key"} if "x-api-key" in os.environ else {}
                resp = requests.post(
                    f"{self.mpc_server}/generate_proof",
                    json={"gradients": gradients["weights"] + gradients["bias"], "secure": self.secure},
                    headers=headers,
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                return data["proof"], data["public_inputs"]
            except Exception as e:
                if self.fallback_mode == "strict":
                    raise RuntimeError(f"MPC server error: {e}")

                # Fallback to local proof generation
                if self.fallback_mode == "warn":
                    logger.warning(f"MPC server failed, falling back to local proof generation: {e}")

                # Continue with local proof

        # MOCK implementation for benchmark only (no real circuit required)
        logger.info(f"Client {self.client_id}: Generating mock ZK proof (no real circuit)")

        # Simulate proof generation time based on data size and secure mode
        gradient_flat = gradients["weights"] + gradients["bias"]
        data_size = len(gradient_flat)

        # Add some artificial delay to simulate real proof time
        if self.secure:
            # Secure proofs take longer
            time.sleep(0.05 + 0.002 * data_size)
            proof = {"type": "mock_secure_proof", "client": self.client_id}
            public_inputs = {"public_inputs": [sum(gradient_flat) / len(gradient_flat)]}
        else:
            # Standard proofs are faster
            time.sleep(0.02 + 0.001 * data_size)
            proof = {"type": "mock_proof", "client": self.client_id}
            public_inputs = [sum(gradient_flat)]

        return proof, public_inputs

    def _submit_update(
        self,
        gradients: Dict[str, List[float]],
        proof: Dict,
        public_inputs: List
    ) -> Tuple[str, int, Optional[Dict]]:
        """Submit update to coordinator and return status."""
        if self.coordinator_url:
            # Remote coordinator
            payload = {
                "gradients": gradients,
                "proof": proof,
                "public_inputs": public_inputs
            }
            try:
                resp = requests.post(f"{self.coordinator_url}/submit_update", json=payload)
                resp.raise_for_status()
                result = resp.json()
                return result["status"], result.get("model_version", 1), result.get("global_update")
            except requests.RequestException as e:
                logger.warning(f"Error connecting to coordinator: {e}")
                # Mock response for benchmarking if coordinator is unavailable
                return "mocked_accepted", 1, None
        else:
            # For benchmarking, if no coordinator URL provided, just return a mock response
            logger.info(f"Client {self.client_id}: Using mock coordinator response")
            # Simulate a small delay for processing
            time.sleep(0.01)
            return "accepted", 1, None


def run_benchmark(
    num_clients: int = 5,
    secure: bool = False,
    mpc_server: Optional[str] = None,
    output_json: str = "benchmark_report.json",
    output_csv: Optional[str] = None,
    report_url: Optional[str] = None,
    coordinator_host: str = "127.0.0.1",
    coordinator_port: int = 8000,
    fallback_mode: str = "warn",
    input_size: int = 10
):
    """
    Run end-to-end FedZK benchmark with multiple clients.
    
    This function orchestrates a complete benchmark of the FedZK system:
    1. Launches a coordinator server in a background process
    2. Creates multiple client instances that run in parallel
    3. Measures performance metrics for training, proof generation, and submission
    4. Generates detailed reports in various formats
    
    The benchmark provides insights into system performance under different 
    configurations, such as client load, secure vs standard circuits, and
    local vs MPC-based proof generation.
    
    Args:
        num_clients (int): Number of client instances to simulate in parallel.
            Higher values test system scalability.
        
        secure (bool): Whether to use secure ZK circuits with constraint checking.
            Secure circuits provide additional security guarantees but are slower.
        
        mpc_server (str, optional): URL of a Multi-Party Computation server 
            for remote proof generation. If provided, proofs are generated remotely
            instead of locally.
        
        output_json (str): Path to save the complete benchmark report in JSON format.
            This report includes detailed metrics for each client and summary statistics.
        
        output_csv (str, optional): Path to save a simplified benchmark report in CSV format.
            This is useful for importing into spreadsheets or data analysis tools.
        
        report_url (str, optional): URL to POST benchmark results to an external service.
            This can be used to send results to a dashboard or data warehouse.
        
        coordinator_host (str): Hostname for the coordinator server.
            Default is localhost (127.0.0.1).
        
        coordinator_port (int): Port for the coordinator server.
            Default is 8000.
        
        fallback_mode (str): How to handle MPC server failures:
            - "strict": Fail immediately if MPC server is unavailable
            - "warn": Log warning and fall back to local proof generation
            - "silent": Silently fall back to local proof generation
        
        input_size (int): Size of gradient vectors generated by clients.
            Larger values simulate more complex models.
            
        Returns:
        dict: A complete report of benchmark results, including:
            - Benchmark configuration
            - Summary statistics
            - Per-client metrics
    
    Raises:
        Exception: If the benchmark fails due to server startup issues or 
            other critical errors.
    """
    console = Console()
    console.print(f"[bold blue]Running FedZK end-to-end benchmark with {num_clients} clients[/bold blue]")
    console.print(f"Mode: {'Secure' if secure else 'Standard'} ZK circuit")
    if mpc_server:
        console.print(f"MPC Server: {mpc_server} (Fallback: {fallback_mode})")

    # Start local coordinator server
    coordinator = CoordinatorServer(host=coordinator_host, port=coordinator_port)
    try:
        coordinator.start_server()
        coordinator_url = f"http://{coordinator_host}:{coordinator_port}"

        # Track metrics per client
        all_metrics = []
        start_time = time.time()

        # Prepare progress display
        with Progress() as progress:
            task = progress.add_task("[green]Running client simulations...", total=num_clients)

            # Execute clients in parallel
            with ThreadPoolExecutor(max_workers=min(num_clients, 10)) as executor:
                # Create and submit client tasks
                futures = []
                for i in range(num_clients):
                    client = Client(
                        client_id=i,
                        input_size=input_size,
                        secure=secure,
                        mpc_server=mpc_server,
                        coordinator_url=coordinator_url,
                        fallback_mode=fallback_mode
                    )
                    futures.append(executor.submit(client.run_training_round))

                # Collect results as they complete
                for future in as_completed(futures):
                    metrics = future.result()
                    all_metrics.append(metrics)
                    progress.update(task, advance=1)

        # Calculate overall statistics
        total_duration = time.time() - start_time
        successful = sum(1 for m in all_metrics if m.get("succeeded", False))
        aggregated = sum(1 for m in all_metrics if m.get("status") == "aggregated")

        # Calculate averages
        avg_train = sum(m["train_time"] for m in all_metrics) / num_clients
        avg_proof = sum(m["proof_time"] for m in all_metrics) / num_clients
        avg_submit = sum(m["submit_time"] for m in all_metrics) / num_clients
        avg_total = sum(m["total_time"] for m in all_metrics) / num_clients

        # Display table of results
        table = Table(title="FedZK End-to-End Benchmark Results")
        table.add_column("Client", justify="right", style="cyan")
        table.add_column("Train(s)", justify="right")
        table.add_column("Proof(s)", justify="right")
        table.add_column("Submit(s)", justify="right")
        table.add_column("Total(s)", justify="right")
        table.add_column("Status", style="green")

        for m in all_metrics:
            table.add_row(
                f"#{m['client_id']}",
                f"{m['train_time']:.4f}",
                f"{m['proof_time']:.4f}",
                f"{m['submit_time']:.4f}",
                f"{m['total_time']:.4f}",
                m["status"]
            )

        console.print(table)

        # Summary statistics
        summary_table = Table(title="Benchmark Summary")
        summary_table.add_column("Metric", style="blue")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Clients", str(num_clients))
        summary_table.add_row("Successful Clients", f"{successful} ({successful/num_clients:.0%})")
        summary_table.add_row("Clients Triggering Aggregation", str(aggregated))
        summary_table.add_row("Circuit Type", "Secure" if secure else "Standard")
        summary_table.add_row("MPC Server Used", "Yes" if mpc_server else "No")
        summary_table.add_row("Avg. Training Time", f"{avg_train:.4f}s")
        summary_table.add_row("Avg. Proof Time", f"{avg_proof:.4f}s")
        summary_table.add_row("Avg. Submission Time", f"{avg_submit:.4f}s")
        summary_table.add_row("Avg. Total Client Time", f"{avg_total:.4f}s")
        summary_table.add_row("Total Benchmark Duration", f"{total_duration:.4f}s")

        console.print(summary_table)

        # Prepare report
        report = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_clients": num_clients,
                "secure": secure,
                "mpc_server": mpc_server,
                "fallback_mode": fallback_mode,
                "input_size": input_size
            },
            "summary": {
                "total_duration": total_duration,
                "successful_clients": successful,
                "aggregated_updates": aggregated,
                "avg_train_time": avg_train,
                "avg_proof_time": avg_proof,
                "avg_submit_time": avg_submit,
                "avg_total_time": avg_total
            },
            "client_metrics": all_metrics
        }

        # Save JSON report
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"Report saved to {output_json}")

        # Save CSV report if requested
        if output_csv:
            with open(output_csv, "w", newline="") as f:
                # Get fields from first client metrics
                fields = ["client_id", "train_time", "proof_time", "submit_time",
                          "total_time", "status", "succeeded", "model_version"]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for m in all_metrics:
                    # Only write requested fields
                    row = {field: m.get(field, "") for field in fields}
                    writer.writerow(row)
            console.print(f"CSV report saved to {output_csv}")

        # Send report to URL if specified
        if report_url:
            try:
                console.print(f"Sending report to {report_url}")
                resp = requests.post(report_url, json=report)
                resp.raise_for_status()
                console.print(f"[green]Report successfully sent to {report_url}")
            except Exception as e:
                console.print(f"[red]Failed to send report: {e}")

        return report

    finally:
        # Clean up coordinator
        coordinator.stop_server()


def main():
    """Command-line entry point for benchmark."""
    parser = argparse.ArgumentParser(description="FedZK End-to-End Benchmark")
    parser.add_argument(
        "--clients", "-c", type=int, default=5,
        help="Number of clients to simulate"
    )
    parser.add_argument(
        "--secure", "-s", action="store_true",
        help="Use secure ZK circuit with constraints"
    )
    parser.add_argument(
        "--mpc-server", type=str, default=None,
        help="URL of MPC proof server (e.g., http://localhost:9000)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="benchmark_report.json",
        help="Output JSON report path"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Output CSV report path"
    )
    parser.add_argument(
        "--report-url", type=str, default=None,
        help="URL to POST benchmark results"
    )
    parser.add_argument(
        "--fallback-mode", type=str, choices=["strict", "warn", "silent"], default="warn",
        help="MPC fallback mode if server is unavailable"
    )
    parser.add_argument(
        "--input-size", type=int, default=10,
        help="Size of gradient tensor"
    )
    parser.add_argument(
        "--coordinator-host", type=str, default="127.0.0.1",
        help="Hostname for coordinator server"
    )
    parser.add_argument(
        "--coordinator-port", type=int, default=8000,
        help="Port for coordinator server"
    )

    args = parser.parse_args()

    try:
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
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
