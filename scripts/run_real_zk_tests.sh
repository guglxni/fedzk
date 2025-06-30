#!/bin/bash

# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

# FEDzk Test Runner with Real Zero-Knowledge Proofs
# This script runs all tests with real ZK proofs enabled

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

print_info "FEDzk Test Runner with Real Zero-Knowledge Proofs"
print_info "================================================="
echo

# Check if prepare_test_environment.sh exists
if [[ ! -f "scripts/prepare_test_environment.sh" ]]; then
    print_error "Test environment preparation script not found!"
    exit 1
fi

# Check if test_env.sh exists, if not run the preparation script
if [[ ! -f "test_env.sh" ]]; then
    print_info "Test environment not set up, running preparation script..."
    ./scripts/prepare_test_environment.sh
else
    print_status "Test environment file found"
fi

# Source the environment variables
print_info "Sourcing test environment variables..."
source test_env.sh

# Run the tests
print_info "Running MPC server tests with real ZK proofs..."
python3 -m pytest -xvs src/fedzk/tests/test_mpc_server.py

print_info "Test run complete!"
echo
print_info "To run all tests, use:"
echo "python3 -m pytest -xvs"
