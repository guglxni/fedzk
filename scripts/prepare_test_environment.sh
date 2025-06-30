#!/bin/bash

# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

# FEDzk Test Environment Preparation Script
# This script prepares the test environment for running tests with real ZK proofs

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_ZK_DIR="$PROJECT_ROOT/src/fedzk/zk"
SETUP_ARTIFACTS_DIR="$PROJECT_ROOT/setup_artifacts"

print_info "Preparing test environment for FEDzk with real ZK proofs..."

# Check if the main setup script was run
if ! command -v circom &> /dev/null || ! command -v snarkjs &> /dev/null; then
    print_warning "Circom and/or SNARKjs not found. Running setup_zk.sh first..."
    bash "$SCRIPT_DIR/setup_zk.sh"
else
    print_status "ZK toolchain found: $(circom --version) and SNARKjs"
fi

# Make sure all circuit files exist and are compiled
print_info "Checking circuit files..."

# Function to check if all circuit artifacts exist for a given circuit
check_circuit_artifacts() {
    local CIRCUIT_NAME=$1
    local ALL_PRESENT=true
    
    # Required files for each circuit
    local REQUIRED_FILES=(
        "$SRC_ZK_DIR/circuits/build/${CIRCUIT_NAME}.r1cs"
        "$SRC_ZK_DIR/circuits/build/${CIRCUIT_NAME}_js/${CIRCUIT_NAME}.wasm"
        "$SRC_ZK_DIR/${CIRCUIT_NAME}.wasm"
        "$SRC_ZK_DIR/proving_key_${CIRCUIT_NAME}.zkey"
        "$SRC_ZK_DIR/verification_key_${CIRCUIT_NAME}.json"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_warning "Missing artifact: $file"
            ALL_PRESENT=false
        fi
    done
    
    # Special case for the main model_update circuit
    if [[ "$CIRCUIT_NAME" == "model_update" ]]; then
        if [[ ! -f "$SRC_ZK_DIR/proving_key.zkey" ]]; then
            print_warning "Missing artifact: $SRC_ZK_DIR/proving_key.zkey"
            ALL_PRESENT=false
        fi
        if [[ ! -f "$SRC_ZK_DIR/verification_key.json" ]]; then
            print_warning "Missing artifact: $SRC_ZK_DIR/verification_key.json"
            ALL_PRESENT=false
        fi
    fi
    
    echo "$ALL_PRESENT"
}

# Ensure all WASM files are copied to the main ZK directory
copy_wasm_files() {
    print_info "Copying WASM files to main ZK directory..."
    
    for circuit in model_update model_update_secure batch_verification differential_privacy sparse_gradients custom_constraints; do
        local src="$SRC_ZK_DIR/circuits/build/${circuit}_js/${circuit}.wasm"
        local dst="$SRC_ZK_DIR/${circuit}.wasm"
        
        if [[ -f "$src" && ! -f "$dst" ]]; then
            cp "$src" "$dst"
            print_status "Copied $circuit.wasm"
        elif [[ -f "$dst" ]]; then
            print_status "$circuit.wasm already exists"
        else
            print_warning "Source WASM file not found: $src"
        fi
    done
}

# Copy verification and proving keys from setup_artifacts if needed
copy_keys() {
    print_info "Checking and copying keys from setup_artifacts if needed..."
    
    # Core model update circuit
    if [[ ! -f "$SRC_ZK_DIR/proving_key.zkey" && -f "$SETUP_ARTIFACTS_DIR/model_update_0001.zkey" ]]; then
        cp "$SETUP_ARTIFACTS_DIR/model_update_0001.zkey" "$SRC_ZK_DIR/proving_key.zkey"
        print_status "Copied model_update proving key"
    fi
    
    if [[ ! -f "$SRC_ZK_DIR/verification_key.json" && -f "$SETUP_ARTIFACTS_DIR/model_update_verification_key.json" ]]; then
        cp "$SETUP_ARTIFACTS_DIR/model_update_verification_key.json" "$SRC_ZK_DIR/verification_key.json"
        print_status "Copied model_update verification key"
    fi
    
    # Model update secure circuit
    if [[ ! -f "$SRC_ZK_DIR/proving_key_secure.zkey" && -f "$SETUP_ARTIFACTS_DIR/model_update_secure_0001.zkey" ]]; then
        cp "$SETUP_ARTIFACTS_DIR/model_update_secure_0001.zkey" "$SRC_ZK_DIR/proving_key_secure.zkey"
        print_status "Copied model_update_secure proving key"
    fi
    
    if [[ ! -f "$SRC_ZK_DIR/verification_key_secure.json" && -f "$SETUP_ARTIFACTS_DIR/model_update_secure_verification_key.json" ]]; then
        cp "$SETUP_ARTIFACTS_DIR/model_update_secure_verification_key.json" "$SRC_ZK_DIR/verification_key_secure.json"
        print_status "Copied model_update_secure verification key"
    fi
    
    # Additional circuits
    for circuit in batch_verification differential_privacy sparse_gradients custom_constraints; do
        if [[ ! -f "$SRC_ZK_DIR/proving_key_${circuit}.zkey" && -f "$SETUP_ARTIFACTS_DIR/${circuit}_0001.zkey" ]]; then
            cp "$SETUP_ARTIFACTS_DIR/${circuit}_0001.zkey" "$SRC_ZK_DIR/proving_key_${circuit}.zkey"
            print_status "Copied ${circuit} proving key"
        fi
        
        if [[ ! -f "$SRC_ZK_DIR/verification_key_${circuit}.json" && -f "$SETUP_ARTIFACTS_DIR/${circuit}_verification_key.json" ]]; then
            cp "$SETUP_ARTIFACTS_DIR/${circuit}_verification_key.json" "$SRC_ZK_DIR/verification_key_${circuit}.json"
            print_status "Copied ${circuit} verification key"
        fi
    done
}

# Create necessary test files
create_test_files() {
    print_info "Creating test files and directories..."
    
    # Create test input file for simple verification
    TEST_INPUT='{
        "gradients": ["1", "2", "3", "4"]
    }'
    
    echo "$TEST_INPUT" > "$SRC_ZK_DIR/test_input.json"
    print_status "Created test input file"
}

# Main execution
copy_wasm_files
copy_keys
create_test_files

# Run a verification test
print_info "Running verification test..."
cd "$SRC_ZK_DIR"

if [[ -f "model_update.wasm" && -f "proving_key.zkey" && -f "test_input.json" ]]; then
    if snarkjs wtns calculate model_update.wasm test_input.json test_witness.wtns &> /dev/null && \
       snarkjs groth16 prove proving_key.zkey test_witness.wtns test_proof.json test_public.json &> /dev/null && \
       snarkjs groth16 verify verification_key.json test_public.json test_proof.json &> /dev/null; then
        print_status "ZK verification test passed"
    else
        print_error "ZK verification test failed"
    fi
else
    print_warning "Missing files for verification test"
fi

# Verify availability for test environment
print_info "Checking environment variables for tests..."

# Set environment variables for tests if not already set
if [[ -z "$FEDZK_ZK_VERIFIED" ]]; then
    echo "# Add these environment variables to your test environment:" > "$PROJECT_ROOT/test_env.sh"
    echo "export FEDZK_ZK_VERIFIED=true" >> "$PROJECT_ROOT/test_env.sh"
    echo "export FEDZK_TEST_MODE=true" >> "$PROJECT_ROOT/test_env.sh"
    
    # Set paths to WASM files and keys
    echo "export MPC_STD_WASM_PATH=\"$SRC_ZK_DIR/model_update.wasm\"" >> "$PROJECT_ROOT/test_env.sh"
    echo "export MPC_STD_ZKEY_PATH=\"$SRC_ZK_DIR/proving_key.zkey\"" >> "$PROJECT_ROOT/test_env.sh"
    echo "export MPC_STD_VER_KEY_PATH=\"$SRC_ZK_DIR/verification_key.json\"" >> "$PROJECT_ROOT/test_env.sh"
    echo "export MPC_SEC_WASM_PATH=\"$SRC_ZK_DIR/model_update_secure.wasm\"" >> "$PROJECT_ROOT/test_env.sh"
    echo "export MPC_SEC_ZKEY_PATH=\"$SRC_ZK_DIR/proving_key_secure.zkey\"" >> "$PROJECT_ROOT/test_env.sh"
    echo "export MPC_SEC_VER_KEY_PATH=\"$SRC_ZK_DIR/verification_key_secure.json\"" >> "$PROJECT_ROOT/test_env.sh"
    
    print_status "Created test_env.sh file with environment variables"
    print_info "Run 'source $PROJECT_ROOT/test_env.sh' before running tests"
else
    print_status "Environment variables already set"
fi

echo
echo "🎉 Test environment preparation complete!"
echo
print_status "You can now run tests with real ZK proofs:"
echo "  source $PROJECT_ROOT/test_env.sh"
echo "  python -m pytest -xvs src/fedzk/tests/test_mpc_server.py"
echo
