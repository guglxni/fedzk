#!/bin/bash
set -e
echo "Compiling circuits manually..."
mkdir -p circuits/build

cd circuits/
echo 'pragma circom 2.0.0;' > minimal.circom
echo '' >> minimal.circom
echo 'template MinimalCircuit() {' >> minimal.circom
echo '    signal output out;' >> minimal.circom
echo '    out <== 1;' >> minimal.circom
echo '}' >> minimal.circom
echo '' >> minimal.circom
echo 'component main = MinimalCircuit();' >> minimal.circom

circom minimal.circom --r1cs --wasm --sym -o build/

echo "✅ Minimal circuit compilation test complete" 