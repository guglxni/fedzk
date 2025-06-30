# Testing FEDzk with Real Zero-Knowledge Proofs

This guide explains how to set up and run tests with real, production-grade ZK proofs instead of mocks or simulations.

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

## References

- [Circom Documentation](https://docs.circom.io/)
- [SNARKjs GitHub Repository](https://github.com/iden3/snarkjs)
