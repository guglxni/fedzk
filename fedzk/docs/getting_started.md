# Getting Started with FedZK

This guide will help you get up and running with the FedZK framework for secure federated learning with zero-knowledge proofs.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9 or later
- Node.js 16.0 or later (for circuit compilation)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aaryanguglani/fedzk.git
cd fedzk
```

### 2. Set Up Python Environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Install Circuit Dependencies

```bash
# Install snarkjs and circom
npm install -g snarkjs circom
```

### 4. Set Up ZK Circuits

```bash
python -m fedzk.cli setup
```

## Basic Usage

### 1. Set Up ZK Circuits

First, set up the necessary zero-knowledge circuits and keys:

```bash
python -m fedzk.cli setup
```

This command generates the circuit artifacts required for proof generation and verification.

### 2. Train a Local Model

Create a simple script to train a model and extract gradients:

```python
import torch
from torch import nn
from fedzk.client.trainer import LocalTrainer

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# Create a trainer
trainer = LocalTrainer(model)

# Create synthetic data
inputs = torch.randn(100, 10)
targets = torch.randint(0, 2, (100,))

# Train the model
trainer.train(inputs, targets, epochs=1, batch_size=32)

# Extract and save gradients
gradients = trainer.extract_gradients()
trainer.save_gradients(gradients, "gradients.json")
```

### 3. Generate a Zero-Knowledge Proof

Generate a proof for your model gradients:

```bash
python -m fedzk.cli generate --input gradients.json --output proof.json --secure
```

This command creates a zero-knowledge proof that your gradients satisfy certain constraints (e.g., L2 norm bound) without revealing the actual values.

### 4. Verify the Proof

Verify the generated proof:

```bash
python -m fedzk.cli verify --proof proof.json
```

A successful verification confirms that the gradients satisfy the required constraints.

## Using the Coordinator

### 1. Start the Coordinator Server

```bash
python -m fedzk.coordinator.server --host 127.0.0.1 --port 8000
```

This starts a FastAPI server that handles client registrations and gradient aggregation.

### 2. Register a Client and Submit Proof

```python
import requests
import json

# Register client
response = requests.post(
    "http://127.0.0.1:8000/register",
    json={"client_id": "client1"}
)
print(response.json())

# Submit proof
with open("proof.json", "r") as f:
    proof_data = json.load(f)

response = requests.post(
    "http://127.0.0.1:8000/submit_proof",
    json={
        "client_id": "client1",
        "proof": proof_data["proof"],
        "public_inputs": proof_data["public_inputs"]
    }
)
print(response.json())
```

### 3. Get the Updated Global Model

```python
response = requests.get("http://127.0.0.1:8000/global_model")
global_model = response.json()
print("Updated global model received")
```

## Using the MPC Server

For clients with limited computational resources, the MPC server offers remote proof generation and verification.

### 1. Start the MPC Server

```bash
# Set API keys for authentication
export MPC_API_KEYS="key1,key2,key3"

# Start the server
python -m fedzk.mpc.server --host 127.0.0.1 --port 8001
```

### 2. Generate Proof Remotely

```bash
python -m fedzk.cli generate --input gradients.json --output proof.json --secure --mpc-server http://127.0.0.1:8001 --api-key key1
```

## Batch Processing for Large Models

For models with many parameters, use batch processing:

```bash
python -m fedzk.cli generate --input gradients.json --output proof.json --secure --batch --chunk-size 1000
```

This splits the gradients into chunks of 1000 elements each and generates proofs for each chunk.

## Running Benchmarks

Evaluate the performance of FedZK with different configurations:

```bash
python -m fedzk.cli benchmark run --clients 5 --secure --output benchmark_results.json --csv benchmark_results.csv
```

This simulates 5 clients in parallel and measures various performance metrics.

## Next Steps

- Check the [API Reference](api_reference.md) for detailed information about FedZK's modules and classes
- Review the [Architecture Document](architecture.md) to understand the system design
- Explore advanced features like custom constraints and aggregation strategies

## Troubleshooting

### Common Issues

1. **Circuit Compilation Fails**
   
   Ensure you have the correct versions of snarkjs and circom installed:
   ```bash
   npm list -g snarkjs circom
   ```

2. **Proof Generation Takes Too Long**
   
   For large models, use batch processing with appropriate chunk sizes, or consider using the MPC server for remote proof generation.

3. **API Key Authentication Fails**
   
   Double-check that the environment variable is set correctly and that you're using a valid key:
   ```bash
   echo $MPC_API_KEYS
   ```

### Getting Help

- File an issue on the GitHub repository
- Join the community discussion forum
- Check the FAQ section in the documentation 