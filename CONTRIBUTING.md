# 🤝 Contributing to FEDZK

Thank you for your interest in contributing to FEDZK! We welcome contributions from the community and are grateful for your help in making FEDZK better.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Reporting Issues](#reporting-issues)
7. [Security](#security)

## Code of Conduct

This project adheres to a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- **Be respectful** and inclusive in all interactions
- **Focus on constructive feedback** and collaborative problem-solving
- **Respect differing viewpoints** and experiences
- **Show empathy** towards other community members
- **Gracefully accept responsibility** for mistakes
- **Focus on what is best for the community**

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)
- Make (optional, for using make commands)

### Fork and Clone

1. Fork the FEDZK repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/fedzk.git
cd fedzk
```

3. Set up the upstream remote:

```bash
git remote add upstream https://github.com/fedzk/fedzk.git
git fetch upstream
```

## Development Setup

### Local Development Environment

1. **Create a virtual environment:**

```bash
# Using venv (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fedzk python=3.9
conda activate fedzk
```

2. **Install dependencies:**

```bash
# Install in development mode with all optional dependencies
pip install -e .[dev,docs,gpu]

# Or install specific groups
pip install -e .[dev]      # Development tools only
pip install -e .[gpu]      # GPU support
pip install -e .[docs]     # Documentation tools
```

3. **Verify installation:**

```bash
# Check that FEDZK is properly installed
python -c "import fedzk; print(f'FEDZK version: {fedzk.__version__}')"

# Run basic health check
python -c "import fedzk; fedzk.health_check()"
```

### Using Docker for Development

```bash
# Build development container
docker build -t fedzk-dev -f Dockerfile.dev .

# Run development container
docker run -it --rm \
  -v $(pwd):/app \
  -p 8000:8000 \
  fedzk-dev

# Or use docker-compose for full development environment
docker-compose -f docker-compose.dev.yml up
```

### IDE Setup

We recommend using VS Code with the following extensions:

- Python (Microsoft)
- Pylance
- Python Docstring Generator
- GitLens
- Docker
- Remote-SSH (for remote development)

### Pre-commit Hooks

Install pre-commit hooks to automatically run linting and formatting:

```bash
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Contributing Guidelines

### Code Style

FEDZK follows these coding standards:

#### Python Code Style

- **PEP 8** compliance with some modifications
- **Black** for code formatting (line length: 88)
- **isort** for import sorting
- **Google-style** docstrings
- **Type hints** for all public APIs

```python
# Good example
def create_federation(
    name: str,
    model_config: ModelConfig,
    privacy_config: Optional[PrivacyConfig] = None
) -> Federation:
    """
    Create a new federated learning federation.

    Args:
        name: Human-readable name for the federation
        model_config: Configuration for the ML model
        privacy_config: Optional privacy and security settings

    Returns:
        The created Federation instance

    Raises:
        ValueError: If name is empty or model_config is invalid
    """
    if not name:
        raise ValueError("Federation name cannot be empty")

    # Implementation here
    pass
```

#### Naming Conventions

- **Classes**: `PascalCase` (e.g., `FederatedLearning`, `ZeroKnowledgeProof`)
- **Functions/Methods**: `snake_case` (e.g., `create_federation`, `validate_proof`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_EPSILON`, `MAX_BATCH_SIZE`)
- **Private attributes**: `_single_leading_underscore` (e.g., `_zk_circuit`)
- **Module-level variables**: `_double_leading_underscore` (rarely used)

#### File Organization

```
fedzk/
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── federation.py       # Federation management
│   └── participant.py      # Participant handling
├── zk/                     # Zero-knowledge proofs
│   ├── __init__.py
│   ├── circuit.py          # ZK circuit definitions
│   └── prover.py           # Proof generation
├── privacy/                # Privacy mechanisms
│   ├── __init__.py
│   ├── differential_privacy.py
│   └── encryption.py
├── compliance/             # Compliance frameworks
│   ├── __init__.py
│   ├── gdpr.py
│   └── audit.py
├── monitoring/             # Monitoring and metrics
│   ├── __init__.py
│   ├── metrics.py
│   └── health.py
└── utils/                  # Utilities
    ├── __init__.py
    ├── config.py
    └── logging.py
```

### Testing

#### Test Structure

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Test scalability and efficiency

```python
# Example test structure
import unittest
from fedzk.core import FederatedLearning

class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.federation = FederatedLearning.create_federation(
            name="Test Federation",
            model_config=ModelConfig(...),
            privacy_config=PrivacyConfig(...)
        )

    def test_federation_creation(self):
        """Test federation creation."""
        self.assertIsNotNone(self.federation.id)
        self.assertEqual(self.federation.name, "Test Federation")

    def test_participant_join(self):
        """Test participant joining federation."""
        participant = self.federation.join(
            participant_id="test_participant",
            public_key="test_key"
        )
        self.assertIsNotNone(participant)
        self.assertEqual(participant.id, "test_participant")
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_federation.py

# Run with coverage
pytest --cov=fedzk --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run tests in parallel
pytest -n auto
```

#### Test Coverage Requirements

- **Minimum coverage**: 85% overall
- **Core modules**: 90% minimum
- **Critical security components**: 95% minimum

### Documentation

#### Docstring Standards

All public APIs must have comprehensive docstrings:

```python
def train_model(
    self,
    training_data: TrainingData,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> TrainingResult:
    """
    Train the federated model on provided data.

    This method implements the federated learning training loop with
    zero-knowledge proofs for privacy preservation.

    Args:
        training_data: Dataset to train on
        epochs: Number of training epochs
        batch_size: Size of training batches
        learning_rate: Learning rate for optimization

    Returns:
        TrainingResult containing metrics and model updates

    Raises:
        ValueError: If training_data is invalid
        RuntimeError: If training fails

    Example:
        >>> result = federation.train_model(data, epochs=50)
        >>> print(f"Final accuracy: {result.accuracy}")
    """
```

#### Documentation Updates

When making changes:

1. **Update docstrings** for any modified public APIs
2. **Add examples** for new features
3. **Update guides** if user-facing behavior changes
4. **Update changelog** with breaking changes

### Security Considerations

#### Secure Coding Practices

- **Input validation**: Always validate and sanitize inputs
- **Secure defaults**: Use secure default configurations
- **Cryptographic practices**: Use well-vetted cryptographic libraries
- **Access control**: Implement proper authorization checks
- **Logging**: Never log sensitive information

```python
# Good: Secure input validation
def validate_participant_id(participant_id: str) -> bool:
    """Validate participant ID format and security."""
    if not participant_id or len(participant_id) > 100:
        return False

    # Only allow alphanumeric characters, hyphens, and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', participant_id):
        return False

    return True

# Bad: No validation
def process_participant(self, participant_id):
    self.participants[participant_id] = Participant(participant_id)
```

#### Cryptographic Security

- Use **well-vetted libraries** (cryptography, hashlib)
- Implement **proper key management**
- Use **secure random number generation**
- Implement **perfect forward secrecy** where applicable

## Pull Request Process

### Creating a Pull Request

1. **Create a feature branch:**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

2. **Make your changes:**

```bash
# Make changes to code
# Add tests for new functionality
# Update documentation
# Run tests locally
pytest
```

3. **Commit your changes:**

```bash
git add .
git commit -m "feat: add new federated learning algorithm

- Implements new FedProx algorithm
- Adds configurable proximal term
- Includes comprehensive tests
- Updates documentation

Closes #123"
```

4. **Push to your fork:**

```bash
git push origin feature/your-feature-name
```

5. **Create Pull Request:**

- Go to GitHub and create a PR from your branch
- Fill out the PR template completely
- Reference any related issues
- Request review from maintainers

### PR Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security enhancement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass locally

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] Breaking changes documented
- [ ] Security implications reviewed

## Additional Notes
Any additional information or context.
```

### Review Process

1. **Automated Checks:**
   - CI/CD pipeline runs tests
   - Code quality checks pass
   - Security scanning completes

2. **Code Review:**
   - At least one maintainer reviews the code
   - Review focuses on:
     - Code quality and style
     - Test coverage
     - Security implications
     - Documentation
     - Performance impact

3. **Approval and Merge:**
   - All required checks pass
   - At least one approval from maintainer
   - Squash and merge (preferred) or rebase merge

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment**
- FEDZK version: [e.g., 1.0.0]
- Python version: [e.g., 3.9.0]
- OS: [e.g., Ubuntu 20.04]
- Hardware: [e.g., CPU/GPU specs]

**Additional context**
Add any other context about the problem here, including logs, error messages, etc.
```

### Feature Requests

For feature requests, please provide:

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Security

### Reporting Security Vulnerabilities

If you discover a security vulnerability, please:

1. **DO NOT** create a public issue
2. Email security@fedzk.io with details
3. Allow time for the team to investigate and fix
4. Receive credit for responsible disclosure

### Security Best Practices for Contributors

- Never commit sensitive information (API keys, passwords, etc.)
- Use secure coding practices
- Validate all inputs and outputs
- Implement proper error handling
- Follow the principle of least privilege

## Recognition

Contributors are recognized through:

- **GitHub contributor statistics**
- **Changelog entries**
- **Author credits in documentation**
- **Community recognition**

## Getting Help

- **Documentation**: [docs.fedzk.io](https://docs.fedzk.io)
- **Discussions**: [GitHub Discussions](https://github.com/fedzk/fedzk/discussions)
- **Community Forum**: [community.fedzk.io](https://community.fedzk.io)
- **Discord**: [discord.gg/fedzk](https://discord.gg/fedzk)

---

Thank you for contributing to FEDZK! Your efforts help make privacy-preserving federated learning accessible to everyone. 🚀
