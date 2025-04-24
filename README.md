# FedZK: Federated Learning with Zero-Knowledge Proofs

A framework for secure federated learning using zero-knowledge proofs to ensure model update integrity.

## Features

- Secure federated learning with privacy guarantees
- Zero-knowledge proof generation and verification
- Distributed model training coordination
- Benchmarking tools for performance evaluation
- MPC server for secure aggregation

## Installation

```bash
# Clone the repository
git clone https://github.com/guglxni/fedzk.git
cd fedzk

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

```python
from fedzk.client import Trainer
from fedzk.coordinator import Aggregator

# Initialize a trainer
trainer = Trainer(model_config={...})

# Train locally
updates = trainer.train(data)

# Generate zero-knowledge proof
proof = trainer.generate_proof(updates)

# Submit updates with proof to coordinator
coordinator = Aggregator()
coordinator.submit_update(updates, proof)
```

## Documentation

For more detailed documentation, examples, and API references, please refer to the [docs](/fedzk/docs) directory.

## Examples

The [examples](/fedzk/examples) directory contains sample code and deployment configurations for various use cases.

## License

This project is licensed under the terms found in the [LICENSE](/fedzk/LICENSE) file. 