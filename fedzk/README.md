# FedZK: Secure Federated Learning with Zero-Knowledge Proofs

[![Build Status](https://github.com/aaryanguglani/fedzk/workflows/ci/badge.svg)](https://github.com/aaryanguglani/fedzk/actions)
[![License](https://img.shields.io/github/license/aaryanguglani/fedzk)](https://github.com/aaryanguglani/fedzk/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/aaryanguglani/fedzk)](https://github.com/aaryanguglani/fedzk/releases)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

## Project Structure

```
fedzk/
│
├── src/                    # Source code
│   └── fedzk/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── client/         # Client-side training and operations
│       ├── coordinator/    # Coordinator server logic
│       ├── mpc/            # Multi-Party Computation server
│       ├── prover/         # Zero-knowledge proof generation
│       ├── utils/          # Utility functions
│       └── benchmark/      # Benchmarking tools
│
├── tests/                  # Unit and integration tests
│
├── docs/                   # Documentation
│   ├── legal/              # Licenses and legal documents
│   ├── user_guide.md
│   ├── api_reference.md
│   └── contributions.md
│
├── examples/               # Example scripts and use cases
│
├── scripts/                # Utility scripts
│   ├── ci/                 # Continuous Integration scripts
│   ├── deployment/         # Deployment scripts
│   └── hooks/              # Git hooks and other utility hooks
│
├── build/                  # Build-related files
│   ├── docker/             # Dockerfiles and docker-compose
│   └── packaging/          # Packaging configuration
│
├── artifacts/              # Generated files during testing/benchmarking
│   ├── benchmarks/         # Benchmark results
│   ├── tests/              # Test artifacts
│   └── proofs/             # Generated proofs
│
├── zk/                     # Zero-Knowledge related files
│   └── circuits/           # ZK circuit definitions
│
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── CHANGELOG.md            # Version history
└── LICENSE                 # Project licensing
```

## Development Setup

1. **Prerequisites**:
   - Python 3.8+
   - Node.js 14+
   - npm 6+

2. **Installation**:
   ```bash
   # Clone the repository
   git clone https://github.com/aaryanguglani/fedzk.git
   cd fedzk

   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in editable mode with development dependencies
   pip install -e ".[dev]"
   ```

3. **Running Tests**:
   ```bash
   pytest tests/
   ```

4. **Code Formatting**:
   ```bash
   black src/
   ```

## Quick Start

```bash
# Run a benchmark
python -m fedzk.cli benchmark run --clients 5 --secure

# Generate a proof
python -m fedzk.cli generate -i gradients.npz -o proof.json --secure
```

## Contributing

Please read our [Contribution Guide](docs/contributions.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use FedZK in your research, please cite:

```
@software{fedzk2025,
  author = {Guglani, Aaryan},
  title = {FedZK: Secure Federated Learning with Zero-Knowledge Proofs},
  year = {2025},
  url = {https://github.com/aaryanguglani/fedzk}
}
```

## Topics

- `federated-learning`
- `zero-knowledge-proofs`
- `secure-machine-learning`
- `privacy-preserving-ai`
- `cryptography`
- `python`

## Release Notes

### v0.1.0 (Initial Release)

Our first release provides the core functionality for secure federated learning with zero-knowledge proofs:

- Complete client training cycle with ZK proof generation
- MPC server for remote proof generation and verification
- Coordinator API for model aggregation
- Batch processing support for efficient proof generation
- End-to-end benchmarking suite
- API key authentication for secure MPC server access
- Command-line interface for all operations

[Full Changelog](CHANGELOG.md)