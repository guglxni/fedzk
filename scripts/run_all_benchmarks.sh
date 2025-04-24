#!/bin/bash

# Make the scripts directory
mkdir -p scripts

# Ensure the script is executable
chmod +x scripts/run_benchmarks.py

# Install required dependencies
pip install numpy pandas torch torchvision sklearn

# Check if ZK infrastructure is available
echo "Checking for ZK infrastructure..."
ZK_DIR="./zk"
if [ -f "$ZK_DIR/model_update.wasm" ] && [ -f "$ZK_DIR/proving_key.zkey" ] && [ -f "$ZK_DIR/verification_key.json" ]; then
    REAL_ZK=true
    echo "ZK infrastructure found. Will use real ZK proofs."
else
    REAL_ZK=false
    echo "ZK infrastructure not found. Will use mock ZK implementation."
    echo "To use real ZK proofs, run: './fedzk/scripts/setup_zk.sh'"
fi

# Run benchmarks for each dataset
echo "Running benchmarks for all datasets..."

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    DEVICE="cuda"
    echo "CUDA is available, using GPU"
else
    DEVICE="cpu"
    echo "CUDA not available, using CPU"
fi

# Set ZK flag based on availability
ZK_FLAG=""
if [ "$REAL_ZK" = true ]; then
    ZK_FLAG="--use-real-zk"
    echo "Using real ZK proofs for benchmarks"
fi

# Run benchmarks for all datasets
python scripts/run_benchmarks.py --datasets mnist cifar10 imdb reuters --num-clients 10 --num-rounds 10 --device $DEVICE --output benchmark_results.csv $ZK_FLAG

echo "Benchmarks complete. Results are in benchmark_results.csv"
echo
echo "To visualize results, run:"
echo "  python -c \"import pandas as pd; print(pd.read_csv('benchmark_results.csv').to_markdown(index=False))\"" 