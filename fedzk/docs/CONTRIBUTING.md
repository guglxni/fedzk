# Contributing to FedZK

Thank you for your interest in contributing to FedZK! This document provides guidelines and instructions for contributing to the project.

## Setting Up Development Environment

### Prerequisites

- Python 3.9 or later
- Node.js 16.0 or later (for circom and snarkjs)
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/fedzk.git
cd fedzk
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install in Development Mode

```bash
pip install -e ".[dev]"  # Installs the package in editable mode with development dependencies
```

### Install Circuit Dependencies

```bash
npm install -g snarkjs circom
```

### Set up Pre-commit Hooks

```bash
pre-commit install
```

## Development Workflow

### Running Tests

Run the full test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=fedzk --cov-report=term --cov-report=html
```

Run specific test files or modules:

```bash
pytest tests/test_zk_generator.py
pytest tests/test_mpc_server.py::test_generate_proof
```

Run specific benchmark tests:

```bash
pytest tests/test_benchmark.py -v
```

Run tests for fallback mode:

```bash
pytest tests/test_fallback.py
```

### Code Style and Linting

FedZK follows PEP 8 style guidelines and uses Black for code formatting and flake8 for linting.

Format code with Black:

```bash
black fedzk tests
```

Run linting with flake8:

```bash
flake8 fedzk tests
```

The pre-commit hooks will automatically check formatting and linting when you commit changes.

### Documentation

Update documentation when adding new features:

1. Add docstrings to new functions and classes
2. Update the API reference if necessary
3. Add examples to the user guide for new functionality

Build the documentation locally:

```bash
mkdocs serve
```

Then visit `http://localhost:8000` to view the documentation.

## Pull Request Process

1. **Create a Branch**: Create a branch from `main` with a descriptive name
   ```bash
   git checkout -b feature/add-new-circuit
   ```

2. **Make Your Changes**: Implement your changes with appropriate tests

3. **Write Good Commit Messages**: Follow these guidelines:
   - Use the present tense ("Add feature" not "Added feature")
   - Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
   - First line should be 50 characters or less
   - Reference issues and pull requests where appropriate
   - Consider using the conventional commits format:
     ```
     feat: add new secure circuit implementation
     fix: resolve batch processing bug
     docs: update API reference
     test: add tests for MPC server
     ```

4. **Run Tests**: Ensure all tests pass
   ```bash
   pytest
   ```

5. **Submit Pull Request**: Push your branch and create a pull request
   ```bash
   git push origin feature/add-new-circuit
   ```

6. **Code Review**: Address reviewer comments and update your PR as needed

## Project Structure

When adding new features, here's where different components belong:

### Adding New CLI Commands

1. Edit `fedzk/cli.py` to add new command functions
2. Use the Typer framework pattern for consistent CLI design
3. Add help text and type annotations for parameters
4. Add unit tests in `tests/test_cli.py`

Example:
```python
@app.command()
def new_command(
    param1: str = typer.Argument(..., help="Description of parameter 1"),
    flag1: bool = typer.Option(False, "--flag", "-f", help="Description of flag"),
):
    """Description of what the command does."""
    # Implementation
```

### Adding New ZK Circuits

1. Add circuit files to `circuits/` directory
2. Update `fedzk/zk/setup.py` to include new circuit compilation
3. Add generator and verifier code in `fedzk/zk/`
4. Add unit tests in `tests/test_zk_generator.py` and `tests/test_zk_verifier.py`

### Adding New Modules

1. Create a new directory under `fedzk/` for your module
2. Add `__init__.py` to expose public interfaces
3. Update API documentation
4. Add unit tests in the `tests/` directory

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release PR
4. After merging, create a GitHub release and tag

## Getting Help

If you have questions or need help:

1. Open an issue with the "question" label
2. For code-related questions, use inline comments in the PR
3. Reach out to project maintainers through GitHub issues

Thank you for contributing to FedZK! 