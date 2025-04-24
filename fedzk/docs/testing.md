# Testing FedZK

This guide provides instructions on how to test the FedZK framework.

## Prerequisites

Before running tests, ensure you have:

- Installed FedZK with development dependencies: `pip install -e ".[dev]"`
- Set up ZK circuits: `python -m fedzk.cli setup`

## Running Tests

### Using pytest

To run all tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=fedzk
```

### Test Categories

Tests are organized into several categories:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Test the complete workflow
4. **Benchmark Tests**: Test performance metrics

## Unit Tests

### Client Tests

Tests for the client module:

```bash
pytest tests/test_client/
```

### ZK Tests

Tests for zero-knowledge proof generation and verification:

```bash
pytest tests/test_zk/
```

### Coordinator Tests

Tests for the coordinator server:

```bash
pytest tests/test_coordinator/
```

### MPC Tests

Tests for the Multi-Party Computation server:

```bash
pytest tests/test_mpc/
```

## Integration Tests

### Client-Coordinator Integration

Tests for client and coordinator interactions:

```bash
pytest tests/test_integration/test_client_coordinator.py
```

### ZK-MPC Integration

Tests for interaction between ZK proof generation and MPC server:

```bash
pytest tests/test_integration/test_zk_mpc.py
```

## End-to-End Tests

Tests that simulate a complete federated learning workflow:

```bash
pytest tests/test_e2e/
```

## Benchmark Tests

Tests that measure performance:

```bash
pytest tests/test_benchmarks/
```

## Test Configuration

Tests can be configured using environment variables or pytest configuration:

```bash
# Set environment variables for testing
export FEDZK_TEST_MPC_URL=http://localhost:8001
export FEDZK_TEST_COORD_URL=http://localhost:8000

# Run tests with specific configuration
pytest --fedzk-secure-mode
```

## Writing Tests

### Test Structure

Tests should follow this structure:

```python
def test_my_feature():
    # Arrange: Set up test conditions
    ...
    
    # Act: Execute the functionality being tested
    ...
    
    # Assert: Verify the expected outcome
    assert result == expected
```

### Test Fixtures

Use pytest fixtures for common test setup:

```python
@pytest.fixture
def model():
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

def test_trainer_with_model(model):
    trainer = LocalTrainer(model)
    # Test functionality
```

### Mocking

Use `unittest.mock` or `pytest-mock` for mocking dependencies:

```python
def test_coordinator_verify(mocker):
    # Mock the ZKVerifier
    mock_verifier = mocker.patch('fedzk.zk.ZKVerifier')
    mock_verifier.return_value.verify_proof.return_value = True
    
    # Test coordinator's verification logic
    coordinator = Coordinator()
    result = coordinator.verify_proof(proof, public_inputs)
    assert result is True
```

## Continuous Integration

FedZK uses GitHub Actions for continuous integration. The CI pipeline runs:

1. Code style checks with `black` and `flake8`
2. Unit and integration tests
3. End-to-end tests
4. Benchmark tests (on schedule)

## Troubleshooting

### Common Test Issues

1. **Circuit Compilation Failures**: Ensure you've run `python -m fedzk.cli setup` before tests
2. **Resource Constraints**: Some tests might take longer on machines with limited resources
3. **Random Failures**: Some tests might fail occasionally due to randomness in the algorithms

### Debug Mode

To run tests in debug mode:

```bash
pytest --log-cli-level=DEBUG
```

## Best Practices

1. **Test Each Feature**: Ensure every feature has corresponding tests
2. **Test Edge Cases**: Test boundary conditions and error handling
3. **Keep Tests Fast**: Optimize tests for quick execution
4. **Test Independence**: Tests should not depend on each other
5. **Clean Up**: Tests should clean up after themselves 