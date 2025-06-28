# Contributing to FEDzk

We welcome contributions from the research and development community! FEDzk is a production-grade framework, so we maintain high standards for code quality, cryptographic correctness, and documentation.

## Development Setup

### Prerequisites
- Python 3.9+
- Node.js 16+ (for SNARKjs)
- Rust 1.70+ (for Circom)
- Git

### Setup Process

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/fedzk.git
   cd fedzk
   ```
3. Set up the complete ZK infrastructure:
   ```bash
   chmod +x scripts/setup_zk.sh
   ./scripts/setup_zk.sh
   ```
4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

## Code Standards

### Python Code Quality
- **Formatting**: Use Black for code formatting (`black src/`)
- **Linting**: Follow PEP 8 with flake8 (`flake8 src/`)
- **Type Hints**: Mandatory for all public functions
- **Docstrings**: Google-style docstrings for all public APIs
- **Testing**: Minimum 90% code coverage required

### Cryptographic Code Requirements
- **No Simulation Logic**: All ZK operations must use real cryptographic primitives
- **Error Handling**: Clear error messages for missing ZK infrastructure
- **Performance**: Document computational complexity and benchmark results
- **Security Review**: Cryptographic changes require security expert review

### Circuit Development
- **Circom Best Practices**: Follow official Circom style guidelines
- **Formal Verification**: All circuits must pass formal verification tests
- **Documentation**: Comprehensive comments explaining circuit logic
- **Testing**: Both unit tests and integration tests with real proofs

## Testing Requirements

### Before Submitting
```bash
# Run all tests
python -m pytest tests/ -v

# Check code formatting
black --check src/

# Verify ZK infrastructure
python -c "from fedzk.prover.zkgenerator import ZKProver; ZKProver()"

# Run benchmarks
python scripts/run_benchmarks.py
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end workflows with real ZK proofs
- **Performance Tests**: Benchmark proof generation and verification times
- **Security Tests**: Verify cryptographic properties and error handling

## Contribution Types

### High-Priority Areas
- **Circuit Optimization**: Improve proof generation performance
- **Scalability**: Support for larger model architectures
- **Security Analysis**: Formal verification and security proofs
- **Documentation**: Academic papers, tutorials, and examples

### Research Contributions
- Novel ZK circuit designs for federated learning primitives
- Formal security analysis and proofs
- Performance optimizations and benchmarks
- Integration with other cryptographic protocols

## Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
2. **Implement Changes**: Follow code standards and add comprehensive tests
3. **Verify ZK Integration**: Ensure all ZK operations work with real infrastructure
4. **Update Documentation**: Include docstrings, README updates, and examples
5. **Run Full Test Suite**: All tests must pass before submission
6. **Submit PR**: Provide detailed description and benchmark results

### PR Review Criteria
- ✅ All tests pass with real ZK infrastructure
- ✅ Code follows style guidelines and best practices
- ✅ Cryptographic correctness verified
- ✅ Performance benchmarks included
- ✅ Documentation updated
- ✅ Security implications assessed

## Reporting Issues

### Bug Reports
- Include full error messages and stack traces
- Specify ZK infrastructure setup details
- Provide minimal reproduction steps
- Include performance context if relevant

### Security Issues
- **DO NOT** report security vulnerabilities in public issues
- Email security@fedzkproject.org for responsible disclosure
- Include proof-of-concept code if applicable

## License

By contributing to FEDzk, you agree that your contributions will be licensed under the FSL-1.1-Apache-2.0 license. All contributions must include appropriate license headers.
