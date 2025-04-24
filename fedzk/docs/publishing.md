# Publishing FedZK to PyPI

*Last updated: April 24, 2025*

This guide explains how to package and publish FedZK to PyPI, the Python Package Index.

## Overview

FedZK uses modern Python packaging with `pyproject.toml` and can be published to both TestPyPI (for testing) and PyPI (for production). The process is largely automated through GitHub Actions and helper scripts.

## Prerequisites

- Python 3.8 or higher
- The `build` and `twine` packages
- Access to the FedZK GitHub repository
- PyPI and TestPyPI accounts with access tokens

## Manual Publishing

### Building the Package

1. Ensure you're in the project root directory.

2. Install build tools:
   ```bash
   pip install build twine
   ```

3. Build the package:
   ```bash
   python -m build
   ```
   
   This will create the distribution packages in the `dist/` directory.

4. Validate the package:
   ```bash
   twine check dist/*
   ```

### Publishing to TestPyPI

1. Upload to TestPyPI:
   ```bash
   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

2. Test the installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fedzk
   ```

### Publishing to PyPI

After testing the package on TestPyPI:

```bash
twine upload dist/*
```

## Automated Publishing

### Using the Helper Scripts

The `scripts/deployment/` directory contains helper scripts:

- `build_and_publish.sh`: Builds and publishes to TestPyPI
- `verify_installation.sh`: Verifies the published package installation
- `bump_version.sh`: Bumps the version number

Example usage:

```bash
# Bump version (patch, minor, or major)
./scripts/deployment/bump_version.sh minor

# Build and publish
./scripts/deployment/build_and_publish.sh

# Verify installation
./scripts/deployment/verify_installation.sh
```

### GitHub Actions

The repository includes a GitHub Actions workflow that automatically:

1. Runs tests
2. Builds the package
3. Publishes to TestPyPI on any push to GitHub
4. Publishes to PyPI when a version tag is pushed

To trigger this workflow:

1. Bump the version:
   ```bash
   ./scripts/deployment/bump_version.sh patch
   ```

2. Push the changes and tag:
   ```bash
   git push && git push --tags
   ```

## Version Management

The version is defined in `pyproject.toml`. Follow semantic versioning guidelines:

- **Major version (X.0.0)**: Incompatible API changes
- **Minor version (0.X.0)**: Add functionality in a backward-compatible manner
- **Patch version (0.0.X)**: Backward-compatible bug fixes

## Installation Instructions for Users

The FedZK package can be installed directly from PyPI:

```bash
pip install fedzk
```

After installation, users can verify it works with:

```bash
fedzk --version
fedzk --help
``` 