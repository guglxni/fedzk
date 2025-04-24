*Last updated: April 24, 2025*

# Contributing to FedZK

Thank you for your interest in contributing to FedZK! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](../CODE_OF_CONDUCT.md). Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/fedzk.git
   cd fedzk
   ```
3. **Set up the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Branch Naming Convention

- `feature/` - For new features
- `bugfix/` - For bug fixes
- `docs/` - For documentation changes
- `refactor/` - For code refactoring
- `test/` - For adding or modifying tests

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Where `<type>` is one of:
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code changes that neither fix bugs nor add features
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI configuration

Example:
```
feat(client): add support for batch processing
```

### Testing

Run tests before submitting a pull request:

```bash
pytest tests/
```

For code coverage:

```bash
pytest --cov=fedzk tests/
```

### Code Style

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines. Use the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Format your code with Black:

```bash
black fedzk/ tests/
```

## Pull Request Process

1. **Update documentation** if necessary
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update the README.md** with details of significant changes
5. **Submit your pull request** with a clear description of the changes

## Feature Requests and Bug Reports

- Use the GitHub issue tracker to report bugs or request features
- For bug reports, include:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Environment details (OS, Python version, etc.)

## Adding New Components

### Adding a New ZK Circuit

1. Create a new circuit definition in the `circuits/` directory
2. Implement the corresponding Python interface in `fedzk/circuits/`
3. Add tests in the `tests/` directory
4. Update documentation in the `docs/` directory

### Adding a New CLI Command

1. Define the command in `fedzk/cli.py`
2. Implement the command handler
3. Add tests for the command
4. Update the user guide documentation

## Documentation

- **Code comments**: Document complex code sections with detailed comments
- **Docstrings**: Use Google-style docstrings for all functions, classes, and modules
- **Documentation files**: Update Markdown files in the `docs/` directory

Example docstring:
```python
def generate_proof(gradients, circuit_params):
    """Generate a zero-knowledge proof for gradients.
    
    Args:
        gradients (List[float]): The gradient values to generate a proof for.
        circuit_params (Dict): Parameters for the proof circuit.
        
    Returns:
        Dict: A dictionary containing the proof and public inputs.
        
    Raises:
        ValueError: If the gradients are invalid or circuit compilation fails.
    """
```

## Communication

- **GitHub Issues**: For bug reports, feature requests, and discussions
- **Pull Requests**: For code contributions and reviews
- **Discussions**: For general questions and community interaction

## Release Process

1. Version numbers follow [Semantic Versioning](https://semver.org/)
2. Update `CHANGELOG.md` with the new version and changes
3. Tag the release with the version number
4. Build and publish the package to PyPI (maintainers only)

## Continuous Integration

- GitHub Actions are used for CI/CD
- Every pull request triggers a workflow that:
  - Runs tests
  - Checks code style
  - Generates code coverage reports

## Advanced Development Topics

### Working with ZK Circuits

When modifying or creating ZK circuits:

1. Understand the underlying cryptographic principles
2. Test with various inputs, including edge cases
3. Benchmark performance with large inputs
4. Consider security implications

### Performance Optimization

- Profile code with tools like `cProfile` or `py-spy`
- Use batch processing for large datasets
- Implement caching where appropriate
- Consider parallel processing for time-consuming operations

## Reviewer Guidelines

If you're reviewing contributions:

1. Verify that all tests pass
2. Check code style consistency
3. Ensure proper documentation
4. Consider security implications
5. Verify performance for large input sizes
6. Test edge cases

## Questions?

If you have any questions about contributing, feel free to open a discussion or reach out to the maintainers. 