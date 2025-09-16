#!/bin/bash

# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

# FEDzk Zero-Knowledge Infrastructure Setup Script
# This script installs and configures the complete ZK toolchain for production use

set -e  # Exit on any error

echo "🔧 Setting up FEDzk Zero-Knowledge Infrastructure..."
echo

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

# Check if running on supported OS
check_os() {
    print_info "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Install Node.js and npm if not present
install_nodejs() {
    print_info "Checking Node.js installation..."
    
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_status "Node.js $NODE_VERSION and npm $NPM_VERSION already installed"
        return
    fi
    
    print_info "Installing Node.js and npm..."
    
    if [[ "$OS" == "linux" ]]; then
        # Install Node.js on Linux
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OS" == "macos" ]]; then
        # Install Node.js on macOS
        if command -v brew &> /dev/null; then
            brew install node
        else
            print_error "Homebrew not found. Please install Node.js manually from https://nodejs.org/"
            exit 1
        fi
    fi
    
    # Verify installation
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_status "Successfully installed Node.js $NODE_VERSION and npm $NPM_VERSION"
    else
        print_error "Failed to install Node.js and npm"
        exit 1
    fi
}

# Install Circom
install_circom() {
    print_info "Installing Circom..."
    
    if command -v circom &> /dev/null; then
        CIRCOM_VERSION=$(circom --version)
        print_status "Circom already installed: $CIRCOM_VERSION"
        return
    fi
    
    # Install Circom globally
    npm install -g circom
    
    # Verify installation
    if command -v circom &> /dev/null; then
        CIRCOM_VERSION=$(circom --version)
        print_status "Successfully installed Circom: $CIRCOM_VERSION"
    else
        print_error "Failed to install Circom"
        exit 1
    fi
}

# Install SNARKjs
install_snarkjs() {
    print_info "Installing SNARKjs..."
    
    if command -v snarkjs &> /dev/null; then
        print_status "SNARKjs already installed"
        return
    fi
    
    # Install SNARKjs globally
    npm install -g snarkjs
    
    # Verify installation
    if command -v snarkjs &> /dev/null; then
        print_status "Successfully installed SNARKjs"
    else
        print_error "Failed to install SNARKjs"
        exit 1
    fi
}

# Setup project directories
setup_directories() {
    print_info "Setting up ZK directories..."
    
    # Get script directory and project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    ZK_DIR="$PROJECT_ROOT/src/fedzk/zk"
    CIRCUITS_DIR="$ZK_DIR/circuits"
    BUILD_DIR="$CIRCUITS_DIR/build"
    
    # Create build directory if it doesn't exist
    mkdir -p "$BUILD_DIR"
    
    print_status "ZK directories ready at $ZK_DIR"
}

# Compile Circom circuits
compile_circuits() {
    print_info "Compiling Circom circuits..."
    
    cd "$CIRCUITS_DIR"
    
    # Compile basic model update circuit
    print_info "Compiling model_update.circom..."
    circom model_update.circom --r1cs --wasm --sym --c -o build/
    
    # Compile secure model update circuit
    print_info "Compiling model_update_secure.circom..."
    circom model_update_secure.circom --r1cs --wasm --sym --c -o build/
    
    print_status "Circuits compiled successfully"
}

# Generate trusted setup (Powers of Tau ceremony)
generate_trusted_setup() {
    print_info "Generating trusted setup for circuits..."
    
    cd "$BUILD_DIR"
    
    # Download or generate Powers of Tau file
    POT_FILE="pot12_final.ptau"
    
    if [[ ! -f "$POT_FILE" ]]; then
        print_info "Downloading Powers of Tau ceremony file..."
        # Use existing ceremony file for efficiency (2^12 constraints)
        wget -q https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_12.ptau -O "$POT_FILE"
        print_status "Downloaded Powers of Tau ceremony file"
    else
        print_status "Powers of Tau file already exists"
    fi
}

# Generate proving and verification keys
generate_keys() {
    print_info "Generating proving and verification keys..."
    
    cd "$BUILD_DIR"
    
    # Generate keys for basic circuit
    if [[ ! -f "../proving_key.zkey" ]]; then
        print_info "Generating keys for model_update circuit..."
        snarkjs groth16 setup model_update.r1cs pot12_final.ptau model_update_0000.zkey
        snarkjs zkey contribute model_update_0000.zkey model_update_0001.zkey --name="First contribution" -v -e="random text"
        snarkjs zkey export verificationkey model_update_0001.zkey ../verification_key.json
        cp model_update_0001.zkey ../proving_key.zkey
        print_status "Generated keys for model_update circuit"
    fi
    
    # Generate keys for secure circuit
    if [[ ! -f "../proving_key_secure.zkey" ]]; then
        print_info "Generating keys for model_update_secure circuit..."
        snarkjs groth16 setup model_update_secure.r1cs pot12_final.ptau model_update_secure_0000.zkey
        snarkjs zkey contribute model_update_secure_0000.zkey model_update_secure_0001.zkey --name="First contribution" -v -e="random text"
        snarkjs zkey export verificationkey model_update_secure_0001.zkey ../verification_key_secure.json
        cp model_update_secure_0001.zkey ../proving_key_secure.zkey
        print_status "Generated keys for model_update_secure circuit"
    fi
}

# Copy WASM files to correct locations
setup_wasm_files() {
    print_info "Setting up WASM files..."
    
    cd "$BUILD_DIR"
    
    # Copy WASM files to parent directory for easier access
    if [[ -f "model_update_js/model_update.wasm" ]]; then
        cp model_update_js/model_update.wasm ../model_update.wasm
        print_status "Copied model_update.wasm"
    fi
    
    if [[ -f "model_update_secure_js/model_update_secure.wasm" ]]; then
        cp model_update_secure_js/model_update_secure.wasm ../model_update_secure.wasm
        print_status "Copied model_update_secure.wasm"
    fi
}

# Test the setup
test_setup() {
    print_info "Testing ZK setup..."
    
    cd "$ZK_DIR"
    
    # Create a test input
    cat > test_input.json << EOF
{
    "gradients": ["1", "2", "3", "4"]
}
EOF
    
    # Test basic circuit
    if [[ -f "model_update.wasm" && -f "proving_key.zkey" ]]; then
        print_info "Testing basic circuit..."
        snarkjs wtns calculate model_update.wasm test_input.json test_witness.wtns
        snarkjs groth16 prove proving_key.zkey test_witness.wtns test_proof.json test_public.json
        snarkjs groth16 verify verification_key.json test_public.json test_proof.json
        
        # Clean up test files
        rm -f test_input.json test_witness.wtns test_proof.json test_public.json
        
        print_status "Basic circuit test passed"
    else
        print_warning "Basic circuit files not found, skipping test"
    fi
}

# Verify installation
verify_installation() {
    print_info "Verifying complete installation..."
    
    # Check all required files exist
    REQUIRED_FILES=(
        "$ZK_DIR/model_update.wasm"
        "$ZK_DIR/proving_key.zkey"
        "$ZK_DIR/verification_key.json"
        "$ZK_DIR/model_update_secure.wasm"
        "$ZK_DIR/proving_key_secure.zkey"
        "$ZK_DIR/verification_key_secure.json"
    )
    
    ALL_PRESENT=true
    for file in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Missing required file: $file"
            ALL_PRESENT=false
        fi
    done
    
    if [[ "$ALL_PRESENT" == true ]]; then
        print_status "All ZK infrastructure files are present"
    else
        print_error "Some required files are missing"
        exit 1
    fi
}

# Main execution
main() {
    echo "🚀 FEDzk Zero-Knowledge Infrastructure Setup"
    echo "==========================================="
    echo
    
    check_os
    install_nodejs
    install_circom
    install_snarkjs
    setup_directories
    compile_circuits
    generate_trusted_setup
    generate_keys
    setup_wasm_files
    test_setup
    verify_installation
    
    echo
    echo "🎉 ZK Infrastructure Setup Complete!"
    echo
    print_status "FEDzk is now ready for production zero-knowledge proof generation"
    print_info "You can now run FEDzk with real ZK proofs enabled"
    echo
    print_info "To test the installation, run:"
    echo "  python -c \"from fedzk.prover import ZKProver; print('ZK setup verified!')\""
    echo
}

# Run the setup
main "$@"