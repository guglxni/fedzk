# FedZK Service Scripts

This directory contains scripts for running FedZK services.

## Available Scripts

- `run-mpc-server.sh`: Start the MPC server for generating and verifying ZK proofs
- `run-coordinator.sh`: Start the Coordinator server for federated learning
- `run-services.sh`: Run both the MPC server and Coordinator together

## Usage

### Running the MPC Server

```bash
./run-mpc-server.sh [options]
```

Options:
- `--port PORT`: Port to run the server on (default: 8081)
- `--host HOST`: Host to bind to (default: 127.0.0.1)
- `--debug`: Enable debug mode with hot reloading
- `--log-level LEVEL`: Set logging level (default: info)
- `--api-keys KEYS`: Comma-separated list of allowed API keys

### Running the Coordinator

```bash
./run-coordinator.sh [options]
```

Options:
- `--port PORT`: Port to run the server on (default: 8000)
- `--host HOST`: Host to bind to (default: 127.0.0.1)
- `--debug`: Enable debug mode with hot reloading
- `--log-level LEVEL`: Set logging level (default: info)

### Running Both Services

```bash
./run-services.sh [options]
```

Options:
- `--mpc-port PORT`: Port for MPC server (default: 8081)
- `--coord-port PORT`: Port for Coordinator server (default: 8000)
- `--mpc-host HOST`: Host for MPC server (default: 127.0.0.1)
- `--coord-host HOST`: Host for Coordinator server (default: 127.0.0.1)
- `--mpc-debug`: Enable debug mode for MPC server
- `--coord-debug`: Enable debug mode for Coordinator server
- `--log-level LEVEL`: Log level (default: info)
- `--api-keys KEYS`: Comma-separated list of allowed API keys for MPC server

## Example

Start both services with API key authentication:

```bash
./run-services.sh --api-keys "key1,key2,key3" --mpc-debug --coord-debug
``` 