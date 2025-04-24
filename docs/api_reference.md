# API Reference

This document provides detailed information about FedZK's API, including classes, methods, and parameters.

## Client Module

### `Trainer` Class

The `Trainer` class is responsible for local model training and proof generation.

```python
from fedzk.client import Trainer

trainer = Trainer(model_config={...})
```

#### Methods

##### `train(data, epochs=10, batch_size=32, learning_rate=0.01)`

Trains the model locally on the provided data.

- **Parameters**:
  - `data`: Dataset to train on
  - `epochs`: Number of training epochs
  - `batch_size`: Batch size for training
  - `learning_rate`: Learning rate for optimization
- **Returns**: Model updates (weights and gradients)

##### `generate_proof(updates, public_params=None)`

Generates a zero-knowledge proof for model updates.

- **Parameters**:
  - `updates`: Model updates to prove
  - `public_params`: Public parameters for the proof (optional)
- **Returns**: A proof object that can be verified

##### `set_circuit(circuit_path)`

Sets a custom verification circuit.

- **Parameters**:
  - `circuit_path`: Path to the custom circuit file

## Coordinator Module

### `Aggregator` Class

The `Aggregator` class coordinates the federated learning process.

```python
from fedzk.coordinator import Aggregator

aggregator = Aggregator(config={...})
```

#### Methods

##### `add_client(client_id, metadata=None)`

Registers a client with the coordinator.

- **Parameters**:
  - `client_id`: Unique identifier for the client
  - `metadata`: Additional client information (optional)

##### `start()`

Starts the coordination process.

##### `submit_update(client_id, updates, proof)`

Submits model updates with a proof from a client.

- **Parameters**:
  - `client_id`: Client identifier
  - `updates`: Model updates (weights/gradients)
  - `proof`: Zero-knowledge proof for the updates
- **Returns**: Success status

##### `aggregate(strategy='fedavg')`

Aggregates verified model updates.

- **Parameters**:
  - `strategy`: Aggregation strategy to use
- **Returns**: Aggregated global model

## Prover Module

### `Verifier` Class

The `Verifier` class handles proof verification.

```python
from fedzk.prover import Verifier

verifier = Verifier()
```

#### Methods

##### `verify(proof, public_inputs)`

Verifies a zero-knowledge proof.

- **Parameters**:
  - `proof`: The proof to verify
  - `public_inputs`: Public inputs for verification
- **Returns**: Boolean indicating if the proof is valid

### `CircuitBuilder` Class

The `CircuitBuilder` class helps create custom verification circuits.

```python
from fedzk.prover import CircuitBuilder

builder = CircuitBuilder()
```

#### Methods

##### `add_constraint(constraint)`

Adds a constraint to the circuit.

- **Parameters**:
  - `constraint`: The constraint to add

##### `compile(circuit_name)`

Compiles the circuit with all added constraints.

- **Parameters**:
  - `circuit_name`: Name for the compiled circuit
- **Returns**: Path to the compiled circuit

## MPC Module

### `SecureAggregator` Class

The `SecureAggregator` class provides secure multi-party computation for aggregation.

```python
from fedzk.mpc import SecureAggregator

secure_agg = SecureAggregator(privacy_budget=0.1)
```

#### Methods

##### `aggregate(updates, weights=None)`

Securely aggregates updates without revealing individual contributions.

- **Parameters**:
  - `updates`: List of updates to aggregate
  - `weights`: Optional weights for each update
- **Returns**: Securely aggregated result

## CLI Module

The command-line interface provides easy access to FedZK's functionality.

### Client Commands

- `fedzk client train`: Trains a model locally
- `fedzk client prove`: Generates a proof for model updates

### Benchmark Commands

- `fedzk benchmark run`: Runs benchmarks
  - `--dataset`: Dataset to use for benchmarking
  - `--iterations`: Number of iterations to run

For more details on each module, see the source code documentation. 