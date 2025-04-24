# Contributing to FedZK

Thank you for your interest in contributing to FedZK! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](legal/CODE_OF_CONDUCT.md).

## How to Contribute

There are many ways to contribute to FedZK:

- Reporting bugs
- Suggesting enhancements
- Writing documentation
- Submitting code changes
- Helping with code reviews
- Answering questions from other users

### Reporting Bugs

If you encounter a bug in FedZK, please [file an issue](https://github.com/guglxni/fedzk/issues/new) with the following information:

- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the problem
- Expected behavior vs. actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please [file an issue](https://github.com/guglxni/fedzk/issues/new) with:

- A clear, descriptive title
- A detailed description of the proposed feature
- Any relevant examples or mockups
- An explanation of why this enhancement would be useful

### Pull Requests

1. **Fork the repository**
2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests**:
   ```bash
   pytest
   ```
5. **Ensure code quality**:
   ```bash
   black .
   flake8
   mypy
   ```
6. **Commit your changes**:
   ```bash
   git commit -m "Add feature: your feature description"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/guglxni/fedzk.git
   cd fedzk
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=fedzk

# Run a specific test file
pytest tests/test_trainer.py

# Run a specific test
pytest tests/test_trainer.py::test_train_method
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- Line length: 100 characters
- Use Black for formatting
- Use type hints

### Documentation

- All public functions, classes, and methods should have docstrings
- Follow the [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- Update documentation when changing functionality

### Testing

- Write tests for all new features
- Maintain or improve test coverage
- Ensure all tests pass before submitting a pull request

## Git Workflow

### Branching Model

- `main`: Stable release version
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation
- `hotfix/*`: Emergency fixes for production

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Changes that don't affect the code's meaning
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes to the build system
- `ci`: Changes to CI configuration
- `chore`: Other changes that don't modify src or test files

Example:
```
feat(trainer): add support for differential privacy

This adds the ability to apply differential privacy to model updates
during training to provide additional privacy guarantees.
```

## Release Process

1. Update the version number in `pyproject.toml` and `__init__.py`
2. Update the `CHANGELOG.md`
3. Create a release branch: `release/vX.Y.Z`
4. Submit a pull request to `main`
5. After approval, merge the pull request
6. Create a tag and GitHub release
7. The CI/CD pipeline will automatically publish to PyPI

## License

By contributing to FedZK, you agree that your contributions will be licensed under the project's [MIT License](../LICENSE).

## Questions?

Feel free to [open an issue](https://github.com/guglxni/fedzk/issues/new) for any questions or join our [community channels](../README.md#community--support) for real-time discussions. 