#!/usr/bin/env python3
# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Basic Training Example for FedZK

This example demonstrates how to set up a simple federated learning training process
using the FedZK framework with zero-knowledge proofs.
"""

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Simulating FedZK imports
try:
    from fedzk.client import FedZKClient
    from fedzk.prover import ZKProver
except ImportError:
    print("FedZK not installed. This is just an example file.")

    # Mock classes for example purposes
    class FedZKClient:
        def __init__(self, client_id, coordinator_url):
            self.client_id = client_id
            self.coordinator_url = coordinator_url

        def train(self, model, data, epochs=1):
            print(f"Client {self.client_id} training for {epochs} epochs")
            return model

        def generate_proof(self, model, training_data):
            print(f"Client {self.client_id} generating zero-knowledge proof")
            return {"proof": "mock_proof", "public_inputs": []}

    class ZKProver:
        def prove_training(self, model_updates, training_data):
            return {"proof": "mock_proof", "public_inputs": []}


def create_synthetic_data(num_samples: int = 1000,
                         input_dim: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for demonstration."""
    X = torch.randn(num_samples, input_dim)
    w = torch.randn(input_dim, 1)
    y = X.mm(w) + 0.1 * torch.randn(num_samples, 1)
    return X, y


def create_model() -> nn.Module:
    """Create a simple linear model."""
    return nn.Sequential(
        nn.Linear(10, 1),
    )


def main():
    # Create synthetic data
    X, y = create_synthetic_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = create_model()

    # Initialize FedZK client
    client = FedZKClient(
        client_id="client_1",
        coordinator_url="http://localhost:8000"
    )

    # Perform local training
    print("Starting local training...")
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Just for demonstration - in real code, client.train would handle this
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}")

    # In a real scenario, this would be the actual model training
    updated_model = client.train(model, dataloader, epochs=3)

    # Generate ZK proof of correct training
    proof = client.generate_proof(updated_model, dataset)

    print(f"Training completed with proof: {proof}")
    print("The proof can be verified by the coordinator to ensure honest training.")


if __name__ == "__main__":
    main()
    print("This is a demonstration file. In a real deployment, you would use actual FedZK components.")
