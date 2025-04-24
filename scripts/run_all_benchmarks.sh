#!/bin/bash

# Make the scripts directory
mkdir -p scripts

# Ensure the script is executable
chmod +x scripts/run_benchmarks.py

# Install required dependencies
pip install numpy pandas torch torchvision sklearn

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

# Run benchmarks for all datasets
python scripts/run_benchmarks.py --datasets mnist cifar10 imdb reuters --num-clients 10 --num-rounds 10 --device $DEVICE --output benchmark_results.csv

echo "Benchmarks complete. Results are in benchmark_results.csv" 