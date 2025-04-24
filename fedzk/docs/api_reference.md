*Last updated: April 24, 2025*

# FedZK API Reference

This document provides an overview of the main modules, classes, and functions available in the FedZK framework.

## Table of Contents

- [Client Module](#client-module)
- [Zero-Knowledge Module](#zero-knowledge-module)
- [MPC Server](#mpc-server)
- [Coordinator](#coordinator)
- [CLI](#cli)
- [Benchmark](#benchmark)
- [Utilities](#utilities)

## Client Module

### `fedzk.client.trainer`

#### `LocalTrainer`

A client-side trainer for federated learning.

```python
from fedzk.client.trainer import LocalTrainer

trainer = LocalTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    dataset=dataset,
    batch_size=32,
    secure=True,
    max_norm=10.0,
    min_active=0.2
)
```

**Methods:**

- `train(epochs=1)`: Train the model for the specified number of epochs
- `generate_proof()`: Generate a ZK proof for the model gradients
- `save_gradients(filepath)`: Save gradients to a file (.npz or .json)
- `load_gradients(filepath)`: Load gradients from a file
- `submit_update(coordinator_url)`: Submit the proof and gradients to the coordinator

## Zero-Knowledge Module

### `fedzk.zk.generator`

#### `ZKGenerator`

Generates zero-knowledge proofs for model gradients.

```python
from fedzk.zk.generator import ZKGenerator

generator = ZKGenerator(secure=True, batch=False, chunk_size=1000)
proof, public_inputs = generator.generate_proof(gradients)
```

**Methods:**

- `generate_proof(gradients)`: Generate a proof for the given gradients
- `batch_generate_proof(gradients, chunk_size=1000)`: Generate proofs for large gradients in batches
- `load_circuit(circuit_path)`: Load a custom circuit for proof generation

### `fedzk.zk.verifier`

#### `ZKVerifier`

Verifies zero-knowledge proofs of model gradients.

```python
from fedzk.zk.verifier import ZKVerifier

verifier = ZKVerifier(secure=True)
is_valid = verifier.verify_proof(proof, public_inputs)
```

**Methods:**

- `verify_proof(proof, public_inputs)`: Verify a proof with the given public inputs
- `batch_verify_proof(proofs, public_inputs_list)`: Verify multiple proofs in batch mode
- `load_verification_key(key_path)`: Load a custom verification key

## MPC Server

### `fedzk.mpc.server`

#### `MPCServer`

A FastAPI server for remote proof generation and verification.

```python
from fedzk.mpc.server import create_app

app = create_app(secure=True, batch_enabled=True)
```

**REST API Endpoints:**

- `POST /generate_proof`: Generate a proof for provided gradients
  - Headers: `x-api-key: <api_key>`
  - Body: `{ "gradients": [...], "secure": true, "batch": false }`
  - Response: `{ "proof": "...", "public_inputs": [...] }`

- `POST /verify_proof`: Verify a proof with public inputs
  - Headers: `x-api-key: <api_key>`
  - Body: `{ "proof": "...", "public_inputs": [...], "secure": true }`
  - Response: `{ "valid": true|false }`

- `POST /health`: Check server health
  - Response: `{ "status": "ok" }`

## Coordinator

### `fedzk.coordinator.api`

#### `CoordinatorAPI`

A FastAPI server for federated learning coordination.

```python
from fedzk.coordinator.api import create_app

app = create_app()
```

**REST API Endpoints:**

- `POST /register`: Register a new client
  - Body: `{ "client_id": "..." }`
  - Response: `{ "client_id": "...", "round": 1 }`

- `POST /submit_update`: Submit a model update with proof
  - Body: `{ "client_id": "...", "round": 1, "gradients": [...], "proof": "...", "public_inputs": [...] }`
  - Response: `{ "status": "accepted"|"rejected", "message": "..." }`

- `GET /get_model`: Get the latest global model
  - Response: `{ "model_weights": [...], "round": 1 }`

- `GET /status`: Get coordinator status
  - Response: `{ "round": 1, "clients": 5, "active_clients": 3 }`

## CLI

### `fedzk.cli`

Command-line interface for FedZK operations.

```python
from fedzk.cli import main

main()
```

**Commands:**

- `setup`: Set up ZK circuits and keys
- `generate`: Generate ZK proof for gradients
- `verify`: Verify ZK proof
- `benchmark run`: Run end-to-end benchmark
- `client train`: Train a model locally
- `client prove`: Generate proof for trained model
- `client submit`: Submit model update to coordinator
- `coordinator start`: Start coordinator server
- `mpc start`: Start MPC server

## Benchmark

### `fedzk.benchmark.end_to_end`

#### `FedZKBenchmark`

Runs end-to-end benchmarks for the FedZK framework.

```python
from fedzk.benchmark.end_to_end import FedZKBenchmark

benchmark = FedZKBenchmark(
    clients=5,
    secure=True,
    mpc_server=None,
    output="results.json",
    csv_output="results.csv",
    report_url=None
)

results = benchmark.run()
```

**Methods:**

- `run()`: Run the benchmark and return results
- `save_results(filepath)`: Save results to a file (JSON or CSV)
- `load_results(filepath)`: Load results from a file
- `report_to_endpoint(url, api_key=None)`: Send results to a reporting endpoint

## Utilities

### `fedzk.utils.tensor`

Utilities for tensor operations.

```python
from fedzk.utils.tensor import flatten_tensors, unflatten_tensors

flat_gradients = flatten_tensors(gradients)
original_format = unflatten_tensors(flat_gradients, shapes)
```

**Functions:**

- `flatten_tensors(tensors)`: Flatten a dictionary of tensors into a single list
- `unflatten_tensors(flat_tensors, shapes)`: Restore flattened tensors to original format
- `compute_l2_norm(tensors)`: Compute the L2 norm of tensors
- `count_non_zeros(tensors)`: Count non-zero elements in tensors

### `fedzk.utils.io`

Utilities for file I/O operations.

```python
from fedzk.utils.io import save_gradients, load_gradients

save_gradients(gradients, "gradients.npz")
loaded_gradients = load_gradients("gradients.npz")
```

**Functions:**

- `save_gradients(gradients, filepath)`: Save gradients to file (supports .npz and .json)
- `load_gradients(filepath)`: Load gradients from file
- `save_proof(proof, public_inputs, filepath)`: Save proof and public inputs to file
- `load_proof(filepath)`: Load proof and public inputs from file 