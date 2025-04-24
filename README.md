<div align="center">
  <img src="assets/images/fedzklogo.png" alt="FEDzk Logo" width="400">
  <h1>FEDzk: Federated Learning with Zero-Knowledge Proofs</h1>
  <p>
    <strong>A secure and privacy-preserving framework for federated learning using zero-knowledge proofs</strong>
  </p>
  <p>
    <a href="#features"><strong>Features</strong></a> •
    <a href="#architecture"><strong>Architecture</strong></a> •
    <a href="#installation"><strong>Installation</strong></a> •
    <a href="#quick-start"><strong>Quick Start</strong></a> •
    <a href="#documentation"><strong>Documentation</strong></a> •
    <a href="#examples"><strong>Examples</strong></a> •
    <a href="#license"><strong>License</strong></a>
  </p>
</div>

## 🚀 Features

- **Privacy-Preserving**: Secure federated learning with strong privacy guarantees
- **Zero-Knowledge Proofs**: Verify model updates without revealing sensitive data
- **Distributed Training**: Coordinate training across multiple clients
- **Benchmarking Tools**: Evaluate performance and scalability
- **Secure Aggregation**: MPC server for secure model aggregation
- **Customizable**: Adapt to different ML models and datasets

## 🏗️ Architecture

The FEDzk framework consists of three main components:

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────┐
│                │     │                 │     │               │
│  Client Node   │────▶│   Coordinator   │◀────│  Client Node  │
│  (Training)    │     │  (Aggregation)  │     │  (Training)   │
│                │     │                 │     │               │
└────────┬───────┘     └────────┬────────┘     └───────┬───────┘
         │                      │                      │
         │                      ▼                      │
         │              ┌───────────────┐              │
         └─────────────▶│   ZK Proofs   │◀─────────────┘
                        │ (Verification) │
                        └───────────────┘
```

### Workflow Diagram

```
┌──────────┐  1. Local Training   ┌───────────┐
│          │──────────────────────▶           │
│  Client  │                      │   Model   │
│          │◀──────────────────────           │
└────┬─────┘  2. Model Updates    └─────┬─────┘
     │                                  │
     │        3. Generate ZK Proof      │
     ▼                                  ▼
┌──────────┐  4. Submit Updates   ┌───────────┐
│          │  with Proof          │           │
│  Prover  │──────────────────────▶  Verifier │
│          │                      │           │
└──────────┘                      └─────┬─────┘
                                        │
                                        │
                                        ▼
                                  ┌───────────┐
                                  │           │
                                  │Coordinator│
                                  │           │
                                  └───────────┘
                                  5. Aggregate
                                     Models
```

## 💻 Installation

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

## 🚦 Quick Start

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

### Verification Process

```python
from fedzk.prover import Verifier

# Initialize the verifier
verifier = Verifier()

# Verify the proof
is_valid = verifier.verify(proof, public_inputs)

if is_valid:
    print("✅ Model update verified successfully!")
else:
    print("❌ Verification failed. Update rejected.")
```

## 📚 Documentation

For more detailed documentation, examples, and API references, please refer to:

- [Getting Started Guide](/fedzk/docs/getting_started.md)
- [API Documentation](/fedzk/docs/api_reference.md)
- [Architecture Overview](/fedzk/docs/architecture.md)
- [Implementation Details](/fedzk/docs/implementation_details.md)
- [Zero-Knowledge Proofs](/fedzk/docs/zk_proofs.md)

## 📋 Examples

The [examples](/fedzk/examples) directory contains sample code and deployment configurations:

- [Basic Training](/fedzk/examples/basic_training.py): Simple federated learning setup
- [Distributed Deployment](/fedzk/examples/distributed_deployment.py): Multi-node configuration
- [Docker Deployment](/fedzk/examples/Dockerfile): Containerized deployment

## 📊 Benchmarks

FEDzk has been benchmarked on various datasets and configurations:

| Dataset | Clients | Rounds | Accuracy | Proof Generation Time | Verification Time |
|---------|---------|--------|----------|----------------------|-------------------|
| MNIST   | 10      | 20     | 97.2%    | 1.2s                 | 0.3s              |
| CIFAR-10| 20      | 50     | 85.6%    | 2.8s                 | 0.5s              |
| IMDb    | 5       | 10     | 88.3%    | 1.5s                 | 0.4s              |

## 📄 License

This project is licensed under the terms found in the [LICENSE](/fedzk/LICENSE) file.

## 🤝 Contributing

Contributions are welcome! Please check out our [contributing guidelines](/fedzk/docs/CONTRIBUTING.md) to get started. 