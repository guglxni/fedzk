# FEDzk: Secure Federated Learning with Zero-Knowledge Proofs

**FEDzk** is a Python framework for building privacy-preserving federated learning systems using zero-knowledge proofs (ZKPs). It provides a complete end-to-end workflow for training, proving, and verifying model updates in a distributed environment.

## Key Features

- **Provable Security**: Unlike conventional federated learning frameworks, FEDzk provides mathematical guarantees for the integrity of model updates
- **Scalability**: Built with performance in mind, our framework can handle large-scale federated learning tasks with minimal overhead
- **Flexibility**: FEDzk supports multiple ZK backends and can be easily integrated with existing machine learning pipelines
- **Ease of Use**: With a simple and intuitive API, developers can quickly get started with building secure and private ML systems

## Quick Start

### Installation

```bash
pip install fedzk
```

### Basic Usage

```python
from fedzk.client import Trainer
from fedzk.coordinator import Aggregator

# Initialize a trainer with your model configuration
trainer = Trainer(model_config={
    'architecture': 'mlp',
    'layers': [784, 128, 10],
    'activation': 'relu'
})

# Train locally on your data
updates = trainer.train(data, epochs=5)

# Generate zero-knowledge proof for model updates
proof = trainer.generate_proof(updates)

# Submit updates with proof to coordinator
coordinator = Aggregator()
coordinator.submit_update(updates, proof)
```

## Documentation

- [Getting Started](getting_started.md) - Detailed installation and setup guide
- [Contributing](CONTRIBUTING.md) - Guidelines for contributing to FEDzk
- [Security](SECURITY.md) - Security policy and vulnerability reporting

## License

This project is licensed under the Functional Source License 1.1 with Apache 2.0 Future Grant (FSL-1.1-Apache-2.0).

## GitHub Repository

Visit our [GitHub repository](https://github.com/guglxni/fedzk) for the latest source code, issues, and releases.
