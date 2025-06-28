# Getting Started with FEDzk

## Prerequisites

- Python 3.9+
- Node.js 16+ (required for SNARKjs)
- Rust 1.70+ (required for Circom)
- Git

## Installation and Setup

FEDzk requires a complete ZK infrastructure setup before use. **This is not optional** - the framework only works with real zero-knowledge proofs.

### Step 1: Clone Repository and Setup ZK Infrastructure

```bash
# Clone the repository
git clone https://github.com/yourusername/fedzk.git
cd fedzk

# Run the comprehensive ZK setup script
chmod +x scripts/setup_zk.sh
./scripts/setup_zk.sh
```

The setup script will:
- Install Circom compiler
- Install SNARKjs library
- Compile all circuits
- Generate proving and verification keys
- Set up the trusted setup ceremony artifacts

### Step 2: Install Python Package

```bash
pip install -e .
```

For development with additional tools:

```bash
pip install -e .[dev]
```

## Verification of Setup

Verify your installation works correctly:

```python
from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.verifier import ZKVerifier

# This will fail with clear error messages if setup is incomplete
prover = ZKProver()
verifier = ZKVerifier()

# Test proof generation (requires compiled circuits)
test_proof = prover.generate_proof([1.0, 2.0, 3.0])
is_valid = verifier.verify_proof(test_proof)
print(f"Setup verification: {'SUCCESS' if is_valid else 'FAILED'}")
```

## Quick Start Example

```python
from fedzk.client import Trainer
from fedzk.coordinator import Aggregator

# Initialize trainer (requires ZK setup completed)
trainer = Trainer(model_config={
    'architecture': 'mlp',
    'layers': [784, 128, 10],
    'activation': 'relu'
})

# Train locally on your data
updates = trainer.train(data, epochs=5)

# Generate REAL zero-knowledge proof
proof = trainer.generate_proof(updates)

# Verify proof cryptographically
is_valid = trainer.verify_proof(proof)

# Submit to coordinator if valid
if is_valid:
    coordinator = Aggregator()
    coordinator.submit_update(updates, proof)
```

## Performance Expectations

With proper ZK setup, expect these performance characteristics:
- **Proof Generation**: ~2.3s for 1000-parameter models
- **Proof Verification**: ~0.8s
- **Circuit Compilation**: One-time ~30s setup per circuit
- **Memory Usage**: ~512MB during proof generation

## Troubleshooting

If you encounter issues:

1. **"Circuit not found" errors**: Run `./scripts/setup_zk.sh` again
2. **SNARKjs errors**: Ensure Node.js 16+ is installed
3. **Circom errors**: Ensure Rust is properly installed
4. **Performance issues**: Ensure you have at least 4GB RAM available

For more detailed examples, see the [examples](../examples) directory.
