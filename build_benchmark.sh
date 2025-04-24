#!/bin/bash

# Setup basic directory structure
mkdir -p zk build/minimal_js

# Create placeholder WASM file
echo "Creating placeholder WASM file..."
cat > build/minimal_js/minimal.wasm << EOL
(module
  (type (;0;) (func (param i32 i32) (result i32)))
  (func (;0;) (type 0) (param i32 i32) (result i32)
    i32.const 1)
  (export "main" (func 0))
)
EOL

# Copy necessary files to zk directory
cp build/minimal_js/minimal.wasm zk/model_update.wasm

# Create dummy keys
echo '{"dummy": "key"}' > zk/proving_key.zkey
echo '{"dummy": "verification"}' > zk/verification_key.json

# Create secure version copies
cp zk/model_update.wasm zk/model_update_secure.wasm
cp zk/proving_key.zkey zk/proving_key_secure.zkey
cp zk/verification_key.json zk/verification_key_secure.json

echo "ZK setup complete for benchmarking" 