# Contributing to FEDzk

We welcome contributions from the community! Please follow these guidelines when contributing to FEDzk.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/fedzk.git
   cd fedzk
   ```
3. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

## Code Style

- Use Black for code formatting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings for all public functions

## Testing

Run tests before submitting:
```bash
python -m pytest tests/
```

## License

By contributing to FEDzk, you agree that your contributions will be licensed under the same license as the project (FSL-1.1-Apache-2.0).
