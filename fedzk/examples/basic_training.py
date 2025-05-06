#!/usr/bin/env python3
"""
Basic Training Example for FedZK

This example demonstrates how to:
1. Define a simple model
2. Train it locally on sample data
3. Generate a zero-knowledge proof for the gradients
4. Verify the proof locally
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from fedzk.client import LocalTrainer
from fedzk.zk import ZKGenerator, ZKVerifier


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=2):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def main():
    # Create synthetic data
    print("Creating synthetic dataset...")
    num_samples = 100
    input_dim = 10

    # Generate random data
    X = torch.randn(num_samples, input_dim)
    # Generate labels (binary classification)
    y = torch.randint(0, 2, (num_samples,))

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = SimpleModel(input_dim=input_dim)

    # Initialize trainer with loss function and optimizer
    trainer = LocalTrainer(
        model=model,
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.01)
    )

    # Train the model for 5 epochs
    print("Training model for 5 epochs...")
    trainer.train(epochs=5)

    # Extract gradients
    gradients = trainer.get_gradients()
    print(f"Extracted {len(gradients)} gradient tensors")

    # Flatten gradients for ZK proof
    flattened_gradients = []
    for grad in gradients:
        if grad is not None:
            flattened_gradients.extend(grad.flatten().tolist())

    print(f"Total gradient elements: {len(flattened_gradients)}")

    # Generate proof
    print("Generating ZK proof...")
    generator = ZKGenerator(secure=True, max_norm=100.0)
    proof, public_inputs = generator.generate_proof(flattened_gradients)

    print("Proof generated successfully")
    print(f"Public inputs: {public_inputs}")

    # Verify proof
    print("Verifying proof...")
    verifier = ZKVerifier()
    is_valid = verifier.verify_proof(proof, public_inputs)

    print(f"Proof verification result: {'Valid' if is_valid else 'Invalid'}")

    # Save proof and public inputs
    proof_file = "gradient_proof.json"
    print(f"Saving proof to {proof_file}")

    proof_data = {
        "proof": proof,
        "public_inputs": public_inputs
    }

    import json
    with open(proof_file, "w") as f:
        json.dump(proof_data, f, indent=2)

    print("Example completed successfully!")

if __name__ == "__main__":
    main()
