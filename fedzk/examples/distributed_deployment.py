#!/usr/bin/env python3
"""
Distributed Deployment Example for FedZK

This example demonstrates how to:
1. Configure and launch the MPC server with API key authentication
2. Configure and launch the Coordinator server
3. Connect a client to the deployed servers
4. Set up secure communication between components

Note: In a real deployment, these components would run on separate
      machines or containers. This example simulates the setup.
"""

import argparse
import asyncio
import os
import threading
import time

from dotenv import load_dotenv

from fedzk.coordinator import CoordinatorAPI
from fedzk.mpc.server import MPCServer
from fedzk.zk import ZKGenerator


def run_mpc_server(host="0.0.0.0", port=8001, api_keys=None):
    """
    Run the MPC server with API key authentication
    """
    # Set environment variable for API keys if provided
    if api_keys:
        os.environ["MPC_API_KEYS"] = ",".join(api_keys)

    # Create and start MPC server
    server = MPCServer()

    # Run in a thread
    thread = threading.Thread(
        target=server.start_server,
        args=(host, port),
        daemon=True
    )
    thread.start()

    print(f"MPC Server running at http://{host}:{port}")
    print(f"API Key authentication: {'Enabled' if api_keys else 'Disabled'}")

    return f"http://{host}:{port}", thread


def run_coordinator(host="0.0.0.0", port=8000, mpc_server_url=None, mpc_api_key=None):
    """
    Run the coordinator server
    """
    # Create coordinator with MPC server config if provided
    coordinator = CoordinatorAPI(
        mpc_server_url=mpc_server_url,
        mpc_api_key=mpc_api_key
    )

    # Run in a thread
    thread = threading.Thread(
        target=coordinator.start_server,
        args=(host, port),
        daemon=True
    )
    thread.start()

    print(f"Coordinator running at http://{host}:{port}")
    if mpc_server_url:
        print(f"Connected to MPC server at {mpc_server_url}")

    return f"http://{host}:{port}", thread


async def simulate_client(coordinator_url, mpc_server_url=None, api_key=None):
    """
    Simulate a client connecting to the deployed servers
    """
    print("\n==== Client Simulation ====")
    print(f"Connecting to coordinator at {coordinator_url}")

    # In a real deployment, you would:
    # 1. Train a model on local data
    # 2. Generate a proof (locally or via MPC server)
    # 3. Submit the update to the coordinator

    # Simulate sending a model update
    print("Simulating model update submission...")

    # For this example, we'll just create a dummy payload
    dummy_gradients = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Use the MPC server if URL is provided
    if mpc_server_url:
        print(f"Generating proof via MPC server at {mpc_server_url}")
        # In a real implementation, you would:
        # 1. Send gradients to MPC server
        # 2. Receive proof
        # 3. Submit proof to coordinator

        # Simulating this process
        generator = ZKGenerator()
        proof, public_inputs = generator.generate_proof(dummy_gradients)

        print("Proof generated successfully")
        print("Submitting update to coordinator...")
    else:
        # Generate proof locally
        print("Generating proof locally")
        generator = ZKGenerator()
        proof, public_inputs = generator.generate_proof(dummy_gradients)

        print("Proof generated successfully")
        print("Submitting update to coordinator...")

    # In a real implementation, this would be an HTTP request to the coordinator
    print("Update submitted successfully")

    return True


def setup_environment():
    """Set up the environment with required configurations"""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Example .env file contents:
    # MPC_API_KEYS=key1,key2,key3
    # COORDINATOR_HOST=0.0.0.0
    # COORDINATOR_PORT=8000
    # MPC_SERVER_HOST=0.0.0.0
    # MPC_SERVER_PORT=8001

    return {
        "coordinator_host": os.getenv("COORDINATOR_HOST", "0.0.0.0"),
        "coordinator_port": int(os.getenv("COORDINATOR_PORT", "8000")),
        "mpc_server_host": os.getenv("MPC_SERVER_HOST", "0.0.0.0"),
        "mpc_server_port": int(os.getenv("MPC_SERVER_PORT", "8001")),
        "mpc_api_keys": os.getenv("MPC_API_KEYS", "test_key").split(","),
    }


async def run_distributed_deployment(simulate_client_connection=True):
    """
    Run a simulated distributed deployment of FedZK components
    """
    # Setup environment and configuration
    config = setup_environment()

    print("==== FedZK Distributed Deployment ====")

    # Start MPC server
    mpc_url, mpc_thread = run_mpc_server(
        host=config["mpc_server_host"],
        port=config["mpc_server_port"],
        api_keys=config["mpc_api_keys"]
    )

    # Wait for MPC server to start
    time.sleep(1)

    # Start coordinator server
    coordinator_url, coord_thread = run_coordinator(
        host=config["coordinator_host"],
        port=config["coordinator_port"],
        mpc_server_url=mpc_url,
        mpc_api_key=config["mpc_api_keys"][0]  # Use first key for coordinator
    )

    # Wait for coordinator to start
    time.sleep(1)

    # Simulate a client if requested
    if simulate_client_connection:
        success = await simulate_client(
            coordinator_url=coordinator_url,
            mpc_server_url=mpc_url,
            api_key=config["mpc_api_keys"][0]
        )
        print(f"Client simulation {'successful' if success else 'failed'}")

    print("\n==== Deployment Running ====")
    print("Press Ctrl+C to stop the servers")

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="FedZK Distributed Deployment Example")
    parser.add_argument("--no-client", action="store_true", help="Don't simulate a client connection")

    args = parser.parse_args()

    # Run the distributed deployment
    asyncio.run(run_distributed_deployment(
        simulate_client_connection=not args.no_client
    ))


if __name__ == "__main__":
    main()
