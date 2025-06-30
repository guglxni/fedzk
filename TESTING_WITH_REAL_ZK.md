# Testing FEDzk with Real Zero-Knowledge Proofs

This guide explains how to set up and run tests with real, production-grade ZK proofs instead of mocks or simulations, following Ian Sommerville's software engineering best practices.

## Comprehensive Testing Strategy

We follow Ian Sommerville's recommended testing levels to ensure thorough coverage while using real ZK infrastructure:

1. **Unit Testing**: Testing individual ZK components with real cryptographic operations
2. **Integration Testing**: Testing interactions between components using real ZK proofs
3. **System Testing**: Testing complete workflows with end-to-end real ZK infrastructure
4. **Acceptance Testing**: Validating against user requirements with real cryptography
5. **Performance Testing**: Measuring real ZK proof generation/verification times
6. **Security Testing**: Validating security features with cryptographic guarantees 
7. **Regression Testing**: Ensuring changes don't break existing functionalities
8. **Real Environment Testing**: Using actual production-grade ZK infrastructure

## Key Features of Our Testing Approach

- **No Mocking or Simulation**: All tests use real ZK infrastructure
- **Real Cryptographic Operations**: Actual Groth16 proof generation and verification
- **End-to-End Workflows**: Complete federated learning with real ZK proofs
- **Production-Grade Circuits**: All 6 circuits tested with real constraints
- **Performance Benchmarking**: Real-world performance metrics

## Prerequisites

1. Node.js and npm (required for Circom and SNARKjs)
2. Python 3.8+ with pip
3. FEDzk repository cloned locally

## Setup Instructions

### 1. Install the ZK Toolchain

First, run the setup script to install Circom and SNARKjs globally:

```bash
# Make the script executable
chmod +x scripts/setup_zk.sh

# Run the setup script
./scripts/setup_zk.sh
```

This script:
- Installs Node.js and npm if not already installed
- Installs Circom and SNARKjs globally
- Sets up the ZK directories
- Compiles all Circom circuits
- Generates trusted setup files
- Creates proving and verification keys
- Tests the setup

### 2. Prepare the Test Environment

After installing the ZK toolchain, prepare the test environment:

```bash
# Make the script executable
chmod +x scripts/prepare_test_environment.sh

# Run the test environment preparation script
./scripts/prepare_test_environment.sh
```

This script:
- Verifies that Circom and SNARKjs are installed
- Checks circuit artifacts and copies them to the right locations if needed
- Creates test files for verification
- Sets up environment variables for testing
- Runs a verification test to confirm everything works

### 3. Set Environment Variables

Before running tests, source the environment variables:

```bash
source test_env.sh
```

This sets:
- `FEDZK_ZK_VERIFIED=true` - Marks ZK as verified
- `FEDZK_TEST_MODE=true` - Enables test mode for better error handling
- Paths to WASM files, proving keys, and verification keys

### 4. Run Tests

Now you can run the tests with real ZK proofs:

```bash
# Option 1: Use the automated test runner (recommended)
./scripts/run_real_zk_tests.sh

# Option 2: Run specific test module manually
python -m pytest -xvs src/fedzk/tests/test_mpc_server.py

# Option 3: Run all tests
python -m pytest -xvs
```

## Troubleshooting

### Missing ZK Files

If tests fail with errors about missing ZK files:

1. Make sure you've run both scripts:
   ```bash
   ./scripts/setup_zk.sh
   ./scripts/prepare_test_environment.sh
   ```

2. Check that the environment variables are set:
   ```bash
   source test_env.sh
   ```

3. Verify file paths in the test environment:
   ```bash
   ls -l src/fedzk/zk/
   ```

### ZK Toolchain Not Found

If tests fail with "ZK toolchain unavailable":

1. Verify Circom and SNARKjs are installed globally:
   ```bash
   circom --version
   snarkjs --version
   ```

2. Make sure they're in your PATH:
   ```bash
   which circom
   which snarkjs
   ```

3. Try installing them again:
   ```bash
   npm install -g circom snarkjs
   ```

## Key Test Files

We've completed a comprehensive audit of all test files to ensure they use real ZK infrastructure:

1. **test_mpc_server.py**: MPC server endpoints with real proof verification
2. **test_aggregator.py**: Federation logic with real gradient aggregation
3. **test_zkgenerator.py**: ZK proof generation with real Circom circuits
4. **test_coordinator_api.py**: API testing with real ZK workflows
5. **test_zkgenerator_real.py**: Specialized real ZK testing suite

## Comprehensive Testing Suite

For running the full test suite following Ian Sommerville's approach:

```bash
# Run the comprehensive test suite
./scripts/comprehensive_testing_suite.sh
```

This script implements all testing levels:

1. Unit tests for individual components
2. Integration tests for component interactions
3. System tests for complete workflows
4. Performance benchmarks with real ZK proofs
5. Security tests for authentication and validation
6. Regression tests across the entire codebase

## ZK Circuit Validation

All tests validate the full ZK circuit stack:

- **Model Update**: Basic gradient verification
- **Secure Model Update**: Enhanced verification with constraints
- **Batch Verification**: Multi-gradient processing
- **Differential Privacy**: Privacy budget tracking
- **Sparse Gradients**: Optimization verification
- **Custom Constraints**: Domain-specific rules

## References

- [Circom Documentation](https://docs.circom.io/)
- [SNARKjs GitHub Repository](https://github.com/iden3/snarkjs)
- [Sommerville's Software Engineering](https://www.pearson.com/en-us/subject-catalog/p/software-engineering/P200000003315)
- [Zero-Knowledge Proofs: An illustrated primer](https://blog.cryptographyengineering.com/2014/11/27/zero-knowledge-proofs-illustrated-primer/)
