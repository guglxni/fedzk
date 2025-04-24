# FedZK Testing Guide

This guide explains how to run existing tests and write new tests for the FedZK framework.

## Prerequisites

Before running tests, make sure you have:

1. Installed all dependencies: `pip install -e ".[dev]"`
2. Compiled the Circom circuits: `python -m fedzk.cli setup`
3. Set up a testing environment with pytest: `pip install pytest pytest-cov`

## Running Tests

### Running All Tests

To run the entire test suite:

```bash
python -m pytest
```

### Running Specific Tests

To run tests from a specific file:

```bash
python -m pytest tests/test_batch_zkgenerator.py
```

To run a specific test:

```bash
python -m pytest tests/test_batch_zkgenerator.py::test_batch_proof_single_tensor
```

### Running Tests with Coverage

To generate a coverage report:

```bash
python -m pytest --cov=fedzk
```

For a detailed HTML coverage report:

```bash
python -m pytest --cov=fedzk --cov-report=html
```

## Test Directory Structure

- `tests/`: Main test directory
  - `test_zkgenerator.py`: Tests for the base ZK proof generation
  - `test_batch_zkgenerator.py`: Tests for batch proof generation
  - `test_cli.py`: Tests for the command-line interface
  - `test_cli_benchmark.py`: Tests for the benchmark CLI commands
  - `test_mpc_server.py`: Tests for the MPC server endpoints
  - `test_coordinator.py`: Tests for the coordinator API

## Writing New Tests

### Test Naming Convention

- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

### Basic Test Structure

```python
import pytest
from fedzk.zkgenerator import ZKGenerator

def test_proof_generation():
    # Arrange
    generator = ZKGenerator()
    gradient = [1, 2, 3]
    
    # Act
    proof, public_inputs = generator.generate_proof(gradient)
    
    # Assert
    assert proof is not None
    assert public_inputs is not None
    assert generator.verify_proof(proof, public_inputs)
```

### Using Fixtures

Create fixtures for common test setup:

```python
@pytest.fixture
def zk_generator():
    generator = ZKGenerator()
    generator.setup()
    return generator

def test_with_fixture(zk_generator):
    # Test using the fixture
    assert zk_generator.is_setup
```

### Testing Batch Processing

When testing batch functionality:

```python
def test_batch_processing():
    generator = ZKGenerator(batch_mode=True, chunk_size=10)
    large_gradient = [i for i in range(100)]
    
    proof, public_inputs = generator.generate_proof(large_gradient)
    
    assert generator.verify_proof(proof, public_inputs)
```

### Testing API Endpoints

For testing FastAPI endpoints:

```python
from fastapi.testclient import TestClient
from fedzk.mpc.server import app

client = TestClient(app)

def test_generate_proof_endpoint():
    response = client.post(
        "/generate_proof",
        headers={"x-api-key": "test-key"},
        json={"gradients": [1, 2, 3]}
    )
    assert response.status_code == 200
    assert "proof" in response.json()
```

### Testing CLI Commands

For testing CLI commands:

```python
from fedzk.cli import main
import sys
from unittest.mock import patch

def test_cli_help():
    with patch.object(sys, 'argv', ['fedzk', '--help']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
```

## Test Data

For consistent testing, use the test data generators:

```python
def generate_test_gradients(size=10):
    return [i + 0.1 for i in range(size)]
```

Store large test datasets in the `tests/data/` directory.

## Mocking

Use mocking to isolate components:

```python
from unittest.mock import patch, MagicMock

def test_with_mock():
    with patch('fedzk.zkgenerator.ZKGenerator.generate_proof') as mock_generate:
        mock_generate.return_value = ("fake_proof", "fake_inputs")
        
        # Test code that uses generate_proof()
        # ...
        
        mock_generate.assert_called_once()
```

## Continuous Integration

All tests are run in GitHub Actions CI. The CI pipeline:

1. Sets up Python
2. Installs dependencies
3. Runs pytest
4. Reports coverage

## Troubleshooting Tests

### Common Issues

1. **Circuit compilation failures**: Ensure Circom is installed and circuits are compiled
2. **Timeouts**: Large tests may time out; consider reducing test data size
3. **Random failures**: Use fixed random seeds for deterministic tests

### Debugging

For detailed logs during test runs:

```bash
python -m pytest -v --log-cli-level=DEBUG
```

For stepping through tests:

```bash
python -m pytest --pdb
```

## Security Testing

When testing ZK proof security properties:

1. Verify proofs are deterministic
2. Test constraint violation detection
3. Ensure proof verification fails with tampered data

```python
def test_tampered_proof_fails():
    generator = ZKGenerator()
    gradients = [1, 2, 3]
    
    proof, public_inputs = generator.generate_proof(gradients)
    
    # Tamper with public inputs
    tampered_inputs = public_inputs.copy()
    tampered_inputs[0] += 1
    
    assert not generator.verify_proof(proof, tampered_inputs)
``` 