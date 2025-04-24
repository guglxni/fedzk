#!/bin/bash
# setup_zk.sh
# Script to compile Circom circuits and setup snarkjs for FedZK

set -e

# Define paths
CIRCUIT_PATH="circuits/minimal.circom"
SECURE_CIRCUIT_PATH="circuits/minimal.circom"
BUILD_DIR="build/zk"
PTAU_PATH="$BUILD_DIR/pot12_final.ptau"
CIRCUIT_NAME="model_update"
SECURE_CIRCUIT_NAME="model_update_secure"

# Create build directory
mkdir -p $BUILD_DIR

echo "=== FedZK ZK Circuit Setup ==="

# Check if circom is installed
if ! command -v circom &> /dev/null; then
    echo "Error: circom is not installed. Please install circom first."
    echo "Installation instructions: https://docs.circom.io/getting-started/installation/"
    exit 1
fi

# Check if snarkjs is installed
if ! command -v snarkjs &> /dev/null; then
    echo "Error: snarkjs is not installed. Please install snarkjs first."
    echo "Run: npm install -g snarkjs"
    exit 1
fi

echo "Step 1: Compiling standard circuit..."
circom $CIRCUIT_PATH --r1cs --wasm --sym -o $BUILD_DIR

echo "Step 1b: Compiling secure circuit..."
circom -l circuits/vendor $SECURE_CIRCUIT_PATH --r1cs --wasm --sym -o $BUILD_DIR

# Generate a small Powers of Tau file (sufficient for our test circuits)
echo "Generating Powers of Tau file..."
mkdir -p $(dirname "$PTAU_PATH")
rm -f $PTAU_PATH  # Remove any potentially corrupted file

echo "Step 1.5: Creating a small Powers of Tau file for our circuits..."
snarkjs powersoftau new bn128 12 $PTAU_PATH -v

echo "Step 1.6: Contributing to the ceremony..."
echo "FedZK Contribution" | snarkjs powersoftau contribute $PTAU_PATH $PTAU_PATH.tmp -v -n="FedZK"
mv $PTAU_PATH.tmp $PTAU_PATH

echo "Step 1.7: Preparing for phase 2..."
snarkjs powersoftau prepare phase2 $PTAU_PATH $PTAU_PATH.p2 -v
mv $PTAU_PATH.p2 $PTAU_PATH

echo "Step 2: Generating proving key and verification key for standard circuit..."
# Setup phase 1
snarkjs groth16 setup $BUILD_DIR/$CIRCUIT_NAME.r1cs $PTAU_PATH $BUILD_DIR/circuit_0000.zkey

# Contribute to the phase 2 ceremony
echo "FedZK" | snarkjs zkey contribute $BUILD_DIR/circuit_0000.zkey $BUILD_DIR/proving_key.zkey

# Export verification key
echo "Step 3: Exporting verification key for standard circuit..."
snarkjs zkey export verificationkey $BUILD_DIR/proving_key.zkey $BUILD_DIR/verification_key.json

echo "Step 4: Generating proving key and verification key for secure circuit..."
# Setup phase 1 for secure circuit
snarkjs groth16 setup $BUILD_DIR/$SECURE_CIRCUIT_NAME.r1cs $PTAU_PATH $BUILD_DIR/secure_circuit_0000.zkey

# Contribute to the phase 2 ceremony for secure circuit
echo "FedZK-Secure" | snarkjs zkey contribute $BUILD_DIR/secure_circuit_0000.zkey $BUILD_DIR/proving_key_secure.zkey

# Export verification key for secure circuit
echo "Step 5: Exporting verification key for secure circuit..."
snarkjs zkey export verificationkey $BUILD_DIR/proving_key_secure.zkey $BUILD_DIR/verification_key_secure.json

# Create a dummy input for testing standard circuit
echo "Step 6: Testing standard circuit with sample input..."
echo '{"gradients": [1, 2, 3, 4]}' > $BUILD_DIR/input.json

# Generate a sample witness for standard circuit
snarkjs wtns calculate $BUILD_DIR/${CIRCUIT_NAME}_js/${CIRCUIT_NAME}.wasm $BUILD_DIR/input.json $BUILD_DIR/witness.wtns

# Generate a proof for standard circuit
snarkjs groth16 prove $BUILD_DIR/proving_key.zkey $BUILD_DIR/witness.wtns $BUILD_DIR/proof.json $BUILD_DIR/public.json

# Verify the proof for standard circuit
echo "Step 7: Verifying standard circuit proof..."
snarkjs groth16 verify $BUILD_DIR/verification_key.json $BUILD_DIR/public.json $BUILD_DIR/proof.json

# Create a dummy input for testing secure circuit
echo "Step 8: Testing secure circuit with sample input..."
echo '{"gradients": [1, 2, 3, 4], "maxNorm": 100, "minNonZero": 3}' > $BUILD_DIR/secure_input.json

# Generate a sample witness for secure circuit
snarkjs wtns calculate $BUILD_DIR/${SECURE_CIRCUIT_NAME}_js/${SECURE_CIRCUIT_NAME}.wasm $BUILD_DIR/secure_input.json $BUILD_DIR/secure_witness.wtns

# Generate a proof for secure circuit
snarkjs groth16 prove $BUILD_DIR/proving_key_secure.zkey $BUILD_DIR/secure_witness.wtns $BUILD_DIR/secure_proof.json $BUILD_DIR/secure_public.json

# Verify the proof for secure circuit
echo "Step 9: Verifying secure circuit proof..."
snarkjs groth16 verify $BUILD_DIR/verification_key_secure.json $BUILD_DIR/secure_public.json $BUILD_DIR/secure_proof.json

# Copy outputs to accessible location
echo "Step 10: Copying outputs to 'zk' directory..."
mkdir -p zk
cp $BUILD_DIR/${CIRCUIT_NAME}_js/${CIRCUIT_NAME}.wasm zk/
cp $BUILD_DIR/proving_key.zkey zk/
cp $BUILD_DIR/verification_key.json zk/

cp $BUILD_DIR/${SECURE_CIRCUIT_NAME}_js/${SECURE_CIRCUIT_NAME}.wasm zk/
cp $BUILD_DIR/proving_key_secure.zkey zk/
cp $BUILD_DIR/verification_key_secure.json zk/

echo "=== Setup Complete ==="
echo "- Standard circuit files:"
echo "  - Proving key: zk/proving_key.zkey"
echo "  - Verification key: zk/verification_key.json"
echo "  - Circuit WASM: zk/$CIRCUIT_NAME.wasm"
echo ""
echo "- Secure circuit files:"
echo "  - Proving key: zk/proving_key_secure.zkey"
echo "  - Verification key: zk/verification_key_secure.json"
echo "  - Circuit WASM: zk/$SECURE_CIRCUIT_NAME.wasm"
echo ""
echo "Use these files with ZKProver and ZKVerifier in FedZK"