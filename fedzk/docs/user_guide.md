*Last updated: April 24, 2025*

# FedZK User Guide

This guide provides instructions on how to use the FedZK framework for secure federated learning with zero-knowledge proofs.

## Installation

### Prerequisites

Before installing FedZK, ensure you have:

- Python 3.8+
- Node.js 14+
- npm 6+

### From GitHub

```bash
# Clone the repository
git clone https://github.com/aaryanguglani/fedzk.git
cd fedzk

# Install the package
pip install -e .

# Set up ZK circuits
python -m fedzk.cli setup
```

## Using the Command Line Interface

FedZK provides a command-line interface for common operations.

### Getting Help

To see the available commands:

```bash
python -m fedzk.cli --help
```

For help on a specific command:

```bash
python -m fedzk.cli <command> --help
```

### Setting Up Circuits

To set up the ZK circuits:

```bash
python -m fedzk.cli setup
```

Options:
- `--output-dir`: Directory to store generated keys (default: `.zk`)
- `--force`: Overwrite existing keys

### Generating Proofs

To generate a ZK proof for a gradient file:

```bash
python -m fedzk.cli generate -i gradients.npz -o proof.json
```

Options:
- `--input, -i`: Path to input file with gradient tensors
- `--output, -o`: Path to output proof file
- `--secure, -s`: Use secure circuit with constraints
- `--batch, -b`: Enable batch processing for large gradients
- `--chunk-size, -c`: Chunk size for batch processing
- `--max-norm, -m`: Maximum L2 norm squared for secure circuit
- `--min-active, -a`: Minimum non-zero elements for secure circuit
- `--mpc-server`: URL of MPC server for remote proof generation
- `--api-key`: API key for MPC server authentication

### Verifying Proofs

To verify a proof:

```bash
python -m fedzk.cli verify -i proof.json
```

Options:
- `--input, -i`: Path to input proof file
- `--secure, -s`: Verify using secure circuit constraints
- `--mpc-server`: URL of MPC server for remote verification
- `--api-key`: API key for MPC server authentication

### Running Benchmarks

To run benchmarks:

```bash
python -m fedzk.cli benchmark run --clients 5
```

Options:
- `--clients`: Number of clients to simulate
- `--secure`: Use secure circuit with constraints
- `--mpc-server`: URL of MPC server for remote operations
- `--output`: Path to save JSON report
- `--csv`: Path to save CSV report
- `--report-url`: URL to send benchmark report

## Client Training

### Local Training

To train a model locally and generate a proof:

```python
from fedzk.client import LocalTrainer
import torch
import torch.nn as nn
import torch.optim as optim

# Define your model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Create trainer
trainer = LocalTrainer(
    model=model,
    optimizer=optim.SGD(model.parameters(), lr=0.01),
    loss_fn=nn.MSELoss()
)

# Load your data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Train the model and generate proof
trainer.train(X, y, epochs=5)
gradients = trainer.extract_gradients()
trainer.save_gradients('gradients.npz', gradients)

# Generate proof
from fedzk.zk import ZKGenerator

generator = ZKGenerator(secure=True, batch=True)
proof, public_inputs = generator.generate_proof(gradients)

# Save proof
import json
with open('proof.json', 'w') as f:
    json.dump({'proof': proof, 'public_inputs': public_inputs}, f)
```

### Using the MPC Server

For clients with limited computational resources, you can use the MPC server:

```python
from fedzk.client import MPCClient
import requests

# Extract gradients as before
# ...

# Use MPC server for proof generation
client = MPCClient(server_url="https://your-mpc-server.com", api_key="your-api-key")
proof_data = client.generate_proof(gradients, secure=True)

# Submit proof to coordinator
response = requests.post(
    "https://your-coordinator.com/submit_proof",
    json=proof_data
)
```

## Coordinator Server

### Starting the Coordinator

To start the coordinator server:

```python
from fedzk.coordinator import CoordinatorServer
import torch.nn as nn

# Define your global model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Start coordinator
coordinator = CoordinatorServer(
    model=model,
    host="0.0.0.0",
    port=8000,
    min_clients=5,
    rounds=10,
    secure=True
)

coordinator.start()
```

### REST API Endpoints

The coordinator exposes the following REST API endpoints:

- `GET /model`: Get the current global model
- `POST /register`: Register a new client
- `POST /submit_proof`: Submit a proof and gradient update
- `GET /status`: Get the current training status

## MPC Server

### Starting the MPC Server

To start the MPC server:

```python
from fedzk.mpc import MPCServer
import os

# Set API keys in environment
os.environ["MPC_API_KEYS"] = "key1,key2,key3"

# Start MPC server
server = MPCServer(host="0.0.0.0", port=8001)
server.start()
```

### API Endpoints

The MPC server exposes the following API endpoints:

- `POST /generate_proof`: Generate a ZK proof (requires API key)
- `POST /verify_proof`: Verify a ZK proof (requires API key)

Authentication is done via the `x-api-key` header.

## Input/Output Formats

### Gradient Format

Gradients can be saved in NPZ (NumPy) or JSON format:

NPZ format:
```python
# Save
np.savez('gradients.npz', **gradients)

# Load
loaded = np.load('gradients.npz')
gradients = {name: loaded[name] for name in loaded.files}
```

JSON format:
```python
# Save
with open('gradients.json', 'w') as f:
    json.dump({k: v.tolist() for k, v in gradients.items()}, f)

# Load
with open('gradients.json', 'r') as f:
    data = json.load(f)
    gradients = {k: np.array(v) for k, v in data.items()}
```

### Proof Format

Proofs are stored in JSON format:

```json
{
  "proof": {
    "pi_a": [...],
    "pi_b": [...],
    "pi_c": [...]
  },
  "public_inputs": [...]
}
```

For batch proofs:

```json
{
  "proofs": [
    {
      "proof": {...},
      "public_inputs": [...]
    },
    ...
  ],
  "is_batch": true
}
```

## Benchmarking

### Running End-to-End Benchmarks

To run end-to-end benchmarks:

```python
from fedzk.benchmark import FedZKBenchmark

benchmark = FedZKBenchmark(
    num_clients=5,
    secure=True,
    mpc_server_url=None  # Set to URL if using MPC
)

results = benchmark.run()
benchmark.save_results('benchmark_report.json')
benchmark.save_csv('benchmark_report.csv')
```

### Analyzing Results

The benchmark generates a report with the following metrics:

- Training time per client
- Proof generation time per client
- Verification time per proof
- Communication overhead (proof size, model size)
- End-to-end latency
- Success/failure status

## Troubleshooting

### Common Issues

1. **Circuit Compilation Errors**:
   - Ensure Node.js and npm are installed
   - Check that circom and snarkjs are accessible

2. **Proof Generation Failures**:
   - Verify gradient format is correct
   - For secure circuits, check if gradients meet constraints
   - For batch processing, try reducing chunk size

3. **Verification Failures**:
   - Ensure the proof and public inputs match
   - Check if the verification key corresponds to the proving key used

4. **MPC Server Connection Issues**:
   - Verify the server URL is correct
   - Check that your API key is valid and included in the request

### Logs

To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Logs are saved to `fedzk.log` by default.

## Advanced Usage

### Custom Circuits

You can create custom circuits by:

1. Creating a new `.circom` file in the `circuits/` directory
2. Implementing your custom logic
3. Setting up the circuit with:
   ```bash
   python -m fedzk.cli setup --circuit your_circuit_name
   ```

### Custom Aggregation Strategies

To implement a custom aggregation strategy:

```python
from fedzk.coordinator import AggregationStrategy

class MyAggregationStrategy(AggregationStrategy):
    def aggregate(self, updates):
        # Custom aggregation logic
        return aggregated_update

# Use in coordinator
coordinator = CoordinatorServer(
    # ... other parameters
    aggregation_strategy=MyAggregationStrategy()
)
```

### Custom Data Loaders

You can implement custom data loaders for federated learning:

```python
from fedzk.client import DataLoader

class MyDataLoader(DataLoader):
    def load(self, client_id):
        # Custom data loading logic
        return X, y

# Use in benchmark
benchmark = FedZKBenchmark(
    # ... other parameters
    data_loader=MyDataLoader()
)
```

## Examples

See the `examples/` directory for complete examples:

- `simple_linear.py`: Basic training with a linear model
- `secure_cnn.py`: Secure training with a CNN model
- `distributed_setup.py`: Setting up a distributed system
- `benchmark_comparison.py`: Comparing different configurations 