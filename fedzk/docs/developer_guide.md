# Developer Guide

This guide is intended for advanced users, researchers, and maintainers who want to understand and extend the FedZK framework.

## Architecture Overview

FedZK implements a federated learning system with zero-knowledge proofs using a client-server architecture:

```
┌───────────┐     ┌─────────────┐     ┌────────────────┐
│           │     │             │     │                │
│  Client   ├────►│  MPC Server ├────►│  Coordinator   │
│           │     │             │     │                │
└───────────┘     └─────────────┘     └────────────────┘
      │                                       │
      │                                       │
      └───────────────────────────────────────┘
          (Direct connection also supported)
```

### Key Components

1. **Client**: Trains models locally and generates proofs for gradient updates
2. **MPC Server**: Optional component that provides remote proof generation 
3. **Coordinator**: Verifies proofs and aggregates model updates

### Data Flow

1. Clients train models locally on private data
2. Gradients are extracted from the trained model
3. Zero-knowledge proofs are generated (locally or via MPC server)
4. Proofs and public inputs are submitted to the coordinator
5. Coordinator verifies proofs and aggregates valid updates
6. Updated global model is distributed to clients

## Directory Structure

```
fedzk/
├── benchmark/         # Benchmarking tools
│   ├── __init__.py
│   └── end_to_end.py  # End-to-end benchmark implementation
├── circuits/          # Circom circuit definitions
│   ├── model_update.circom
│   └── model_update_secure.circom
├── client/            # Client implementation
│   ├── __init__.py
│   └── trainer.py     # Local model training
├── coordinator/       # Coordinator implementation
│   ├── __init__.py
│   └── server.py      # FastAPI server implementation
├── mpc/               # MPC server implementation
│   ├── __init__.py
│   └── server.py      # Remote proof generation and verification
├── zk/                # Zero-knowledge proof system
│   ├── __init__.py
│   ├── generator.py   # Proof generation
│   ├── verifier.py    # Proof verification
│   └── batch_generator.py # Batch processing for large models
├── utils/             # Utility functions
│   ├── __init__.py
│   └── tensor_utils.py # Tensor manipulation utilities
├── cli.py             # Command-line interface
└── fallback.py        # Fallback mode implementation
```

## CLI Implementation

The FedZK CLI is built using the [Typer](https://typer.tiangolo.com/) framework. The main CLI structure is defined in `fedzk/cli.py`.

### CLI Structure

```python
import typer

app = typer.Typer()

@app.command()
def setup(
    output_dir: str = typer.Option(".zk", "--output-dir", "-o"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Setup ZK circuits and keys."""
    # Implementation

@app.command()
def generate(
    input_file: str = typer.Option(None, "--input", "-i"),
    output_file: str = typer.Option("proof.json", "--output", "-o"),
    secure: bool = typer.Option(False, "--secure", "-s"),
    # Other options...
):
    """Generate ZK proof for gradients."""
    # Implementation

# More commands...

if __name__ == "__main__":
    app()
```

### Command Groups

For more complex functionality, FedZK uses command groups:

```python
benchmark_app = typer.Typer()
app.add_typer(benchmark_app, name="benchmark")

@benchmark_app.command("run")
def run_benchmark(
    clients: int = typer.Option(5, "--clients", "-c"),
    # Other options...
):
    """Run benchmark with simulated clients."""
    # Implementation
```

## Extending FedZK

### Adding New CLI Commands

To add a new command:

1. Define a new function in `fedzk/cli.py` decorated with `@app.command()` or as part of a command group
2. Implement the command's functionality
3. Add tests in `tests/test_cli.py`

Example:

```python
@app.command()
def new_command(
    param1: str = typer.Argument(..., help="Description of param1"),
    param2: int = typer.Option(42, "--param2", "-p", help="Description of param2"),
):
    """Description of the new command."""
    # Implementation
    pass
```

### Adding New Circuits

To add a new circuit:

1. Create a new `.circom` file in the `circuits/` directory
2. Update the setup command in `fedzk/zk/setup.py` to compile the new circuit
3. Implement generator and verifier classes in `fedzk/zk/`
4. Update the CLI to use the new circuit

Example circuit file `circuits/new_circuit.circom`:

```circom
pragma circom 2.0.0;

template NewCircuit(n) {
    signal input values[n];
    signal output result;
    
    var sum = 0;
    for (var i = 0; i < n; i++) {
        sum += values[i];
    }
    
    result <== sum;
}

component main = NewCircuit(10);
```

### Adding New Benchmark Workloads

To add a new benchmark workload:

1. Add a new benchmark class in `fedzk/benchmark/`
2. Update the CLI to expose the new benchmark
3. Add tests for the new benchmark

Example:

```python
class NewBenchmark:
    def __init__(self, params):
        self.params = params
    
    def run(self):
        # Implementation
        results = {...}
        return results
    
    def save_results(self, output_path):
        # Save results to file
        pass
```

## Fallback Mode

FedZK includes a fallback mode that bypasses the full zero-knowledge proof system for debugging and development purposes.

### How Fallback Mode Works

The fallback mode implements a simplified proof generation and verification process that:

1. Generates deterministic "proof" signatures based on the input gradients
2. Verifies these signatures without using the full ZK machinery
3. Still enforces the same constraints (norm bounds, etc.)

This is useful for:
- Development and testing without ZK circuit compilation
- Debugging issues in the constraint logic
- Performance comparisons

### Implementation

The fallback mode is implemented in `fedzk/fallback.py`:

```python
class FallbackGenerator:
    def __init__(self, secure=False, max_norm=100.0, min_active=10):
        self.secure = secure
        self.max_norm = max_norm
        self.min_active = min_active
    
    def generate_proof(self, gradients):
        # Check constraints if secure mode is enabled
        if self.secure:
            self._check_constraints(gradients)
        
        # Generate deterministic "proof"
        flattened = flatten_gradients(gradients)
        proof = {
            "signature": hash(tuple(flattened.tolist())),
            "metadata": {
                "secure": self.secure,
                "norm": compute_l2_norm_squared(flattened),
                "active": count_non_zero(flattened)
            }
        }
        
        return proof, [sum(flattened)]

class FallbackVerifier:
    # Similar implementation for verification
```

### Using Fallback Mode

Fallback mode can be enabled via the CLI:

```bash
python -m fedzk.cli generate --input gradients.json --output proof.json --fallback
python -m fedzk.cli verify --input proof.json --fallback
```

Or programmatically:

```python
from fedzk.fallback import FallbackGenerator

generator = FallbackGenerator(secure=True)
proof, inputs = generator.generate_proof(gradients)
```

## Advanced Topics

### Custom Gradient Constraints

You can implement custom constraints by creating a new secure circuit and updating the constraint checking logic in the generator:

1. Create a new circuit file with the custom constraints
2. Update `ZKGenerator._check_constraints()` to enforce the constraints
3. Add tests for the new constraints

### Custom Aggregation Strategies

To implement a custom aggregation strategy:

1. Create a class that inherits from `AggregationStrategy` in `fedzk/coordinator/aggregation.py`
2. Implement the `aggregate()` method
3. Pass your strategy to the `CoordinatorServer` constructor

### Integrating with External Systems

FedZK can integrate with external federated learning frameworks:

1. Use the `LocalTrainer` API to extract gradients from external models
2. Generate proofs for these gradients
3. Use the coordinator's REST API to submit proofs and retrieve model updates 