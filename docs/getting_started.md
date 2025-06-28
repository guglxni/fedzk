# Getting Started

## Prerequisites

- Python 3.9+
- Pip
- Git

## Installation

```bash
pip install fedzk
```

For more advanced use cases, you can install optional dependencies:

```bash
pip install fedzk[all]     # All dependencies
pip install fedzk[dev]     # Development tools
```

## Quick Start

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

For more detailed examples, see the [examples](../examples) directory.
