#!/usr/bin/env python3
"""
Federated Learning Example with FedZK

This example demonstrates how to:
1. Start a coordinator server
2. Train multiple client models in parallel
3. Generate ZK proofs for model updates
4. Submit updates to the coordinator
5. Aggregate model updates securely

Note: This is a simulated example running in a single process.
      In a real deployment, clients would run on separate machines.
"""

import asyncio
import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from fedzk.client import LocalTrainer
from fedzk.coordinator import CoordinatorAPI, ModelAggregator


# Define a simple model for federated learning
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

def create_synthetic_data(num_samples=1000, input_dim=10, num_clients=3):
    """Create synthetic data and split among clients"""
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    # Generate labels (binary classification)
    y = torch.randint(0, 2, (num_samples,))

    # Create full dataset
    dataset = TensorDataset(X, y)

    # Split dataset among clients (non-IID)
    client_data = random_split(dataset, [num_samples // num_clients] * num_clients)

    return client_data

def run_coordinator(host="localhost", port=8000):
    """Start the coordinator server in a separate thread"""
    coordinator = CoordinatorAPI()

    # Run the server in a thread
    threading.Thread(
        target=coordinator.start_server,
        args=(host, port),
        daemon=True
    ).start()

    # Wait for server to start
    time.sleep(1)
    print(f"Coordinator server running at http://{host}:{port}")

    return f"http://{host}:{port}"

async def train_client(client_id, dataloader, coordinator_url, secure=True):
    """Train a client model and submit the update to the coordinator"""
    print(f"Client {client_id}: Starting training...")

    # Initialize model
    model = SimpleModel()

    # Initialize trainer
    trainer = LocalTrainer(
        model=model,
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.01)
    )

    # Train model
    print(f"Client {client_id}: Training for 3 epochs...")
    trainer.train(epochs=3)

    # Generate proof
    print(f"Client {client_id}: Generating ZK proof...")
    success = await trainer.generate_and_submit_update(
        coordinator_url=coordinator_url,
        secure=secure
    )

    print(f"Client {client_id}: Update submission {'successful' if success else 'failed'}")
    return success

async def run_federated_learning(num_clients=3, secure=True):
    """Run a federated learning simulation with multiple clients"""
    # Start coordinator
    coordinator_url = run_coordinator()

    # Create synthetic data for clients
    client_datasets = create_synthetic_data(num_clients=num_clients)
    client_dataloaders = [
        DataLoader(ds, batch_size=32, shuffle=True)
        for ds in client_datasets
    ]

    # Run clients in parallel
    client_tasks = [
        train_client(i, dataloader, coordinator_url, secure)
        for i, dataloader in enumerate(client_dataloaders)
    ]

    # Wait for all clients to complete
    results = await asyncio.gather(*client_tasks)

    print("\nFederated learning round completed")
    print(f"Successful updates: {sum(results)}/{len(results)}")

    # Fetch aggregated model from coordinator
    print("Fetching aggregated model from coordinator...")

    # In a real application, you would fetch the model using a GET request
    # Here we simulate by creating an aggregator
    aggregator = ModelAggregator()
    aggregated_model = aggregator.get_aggregated_model()

    if aggregated_model:
        print("Successfully aggregated model updates")
    else:
        print("No aggregated model available")

    print("\nFederated learning simulation completed!")

def main():
    # Run the federated learning simulation
    asyncio.run(run_federated_learning(num_clients=3, secure=True))

if __name__ == "__main__":
    main()
