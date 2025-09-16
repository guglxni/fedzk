# FEDzk Test Suite

Comprehensive test suite for FEDzk framework.

## Test Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interaction

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=src/fedzk tests/
```
