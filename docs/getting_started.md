# Getting Started with FedZK

This guide will help you set up and start using FedZK for your federated learning projects.

## Installation

### Prerequisites

Before installing FedZK, ensure you have:

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (optional but recommended)

### Installation Steps

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

For development or with additional features:

```bash
# Install with all optional dependencies
pip install -e ".[all]"

# Install with specific feature sets
pip install -e ".[benchmark]"  # For benchmarking tools
pip install -e ".[dev]"        # For development tools
pip install -e ".[docs]"       # For documentation generation
```

## Basic Usage

### Client-Side Training

```python
from fedzk.client import Trainer

# Initialize a trainer with your model configuration
trainer = Trainer(model_config={
    'architecture': 'mlp',
    'layers': [784, 128, 10],
    'activation': 'relu'
})

# Load your data
data = load_your_data()  # Replace with your data loading function

# Train locally on your data
updates = trainer.train(data, epochs=5)

# Generate zero-knowledge proof for model updates
proof = trainer.generate_proof(updates)
```

### Using the Coordinator

```python
from fedzk.coordinator import Aggregator

# Initialize an aggregator
coordinator = Aggregator()

# Add clients
coordinator.add_client("client_1")
coordinator.add_client("client_2")

# Start the federation process
coordinator.start()

# Submit updates with proof
coordinator.submit_update(client_id="client_1", updates=updates, proof=proof)

# Aggregate when enough updates are received
global_model = coordinator.aggregate()
```

## Command-Line Interface

FedZK comes with a command-line interface for common operations:

```bash
# Train a model locally
fedzk client train

# Generate a proof for model updates
fedzk client prove

# Run benchmarks
fedzk benchmark run --iterations 5
```

## Next Steps

- Explore the [API Documentation](api_reference.md) for detailed information about FedZK's classes and methods
- Check out the [Examples](../examples) directory for sample use cases
- Read the [Architecture Overview](architecture.md) to understand FedZK's design

## Troubleshooting

If you encounter issues during installation or usage:

1. Ensure you have the correct Python version (3.8+)
2. Check that all dependencies are installed correctly
3. Consult the [Troubleshooting Guide](troubleshooting.md)

If problems persist, please [file an issue](https://github.com/guglxni/fedzk/issues) on GitHub. 