# FedZK Project Structure

This document explains the organization of the FedZK project codebase.

## Overview

FedZK is structured as a Python package with the following high-level organization:

```
fedzk/
├── src/
│   └── fedzk/           # Main Python package
├── tests/               # Testing directory
├── docs/                # Documentation
├── examples/            # Example applications 
├── scripts/             # Utility scripts
└── ...                  # Other project files
```

## Core Directories

### src/fedzk/

This is the main Python package. It contains:

- `__init__.py` - Package initialization and version info
- `cli.py` - Command-line interface implementation
- `client/` - Client implementation for federated learning
- `coordinator/` - Coordinator code for orchestrating training
- `prover/` - Zero-knowledge proof generation
- `mpc/` - Secure multi-party computation components
- `utils/` - Utility functions and helpers
- `benchmark/` - Performance benchmarking tools

### tests/

Contains all test files organized in a structure mirroring the main package.

### docs/

Contains all documentation:

- `getting_started.md` - Beginner's guide
- `api_reference.md` - API documentation
- `architecture.md` - System architecture overview
- `deployment_guide.md` - Deployment instructions
- `troubleshooting.md` - Troubleshooting information
- `legal/` - Legal documents (Code of Conduct, Security Policy)

### examples/

Complete examples showing how to use FedZK for different use cases.

### scripts/

Utility scripts for development, deployment, and maintenance.

## Build and Packaging

The project uses `pyproject.toml` for build configuration, which is located at the project root. This file defines dependencies, metadata, and build settings.

## Version Control

- `.gitignore` - Specifies files to ignore in version control
- `.github/` - GitHub-specific configurations, including CI/CD workflows

## Development Tools

- `.pre-commit-config.yaml` - Pre-commit hook configuration
- `.bumpversion.cfg` - Version bumping configuration

## Notes for Contributors

1. Always refer to the official Python package in `src/fedzk/`
2. Follow the existing directory structure when adding new components
3. Keep related files together in their respective modules
4. Update documentation when making significant structural changes
5. Add tests that mirror the structure of your implementation

## Legacy Structure Note

As of April 24, 2025, a duplicate structure in the root `fedzk/` directory has been deprecated. Always use the `src/fedzk/` package for development. 