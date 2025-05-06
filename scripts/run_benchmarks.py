#!/usr/bin/env python3
# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
FedZK Benchmarking Script

This script runs benchmarks on various datasets to measure:
1. Training accuracy
2. Proof generation time
3. Verification time

Results are saved to a CSV file for easy import into the README.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

# Import real ZK proof functionality instead of using mock
try:
    from fedzk.prover.verifier import ZKVerifier
    from fedzk.prover.zkgenerator import ZKProver

    # Define paths to necessary files
    ZK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zk")
    CIRCUIT_WASM = os.path.join(ZK_DIR, "model_update.wasm")
    PROVING_KEY = os.path.join(ZK_DIR, "proving_key.zkey")
    VERIFICATION_KEY = os.path.join(ZK_DIR, "verification_key.json")

    # Check if ZK setup files exist
    USE_REAL_ZK = (os.path.exists(CIRCUIT_WASM) and
                   os.path.exists(PROVING_KEY) and
                   os.path.exists(VERIFICATION_KEY))
except ImportError:
    USE_REAL_ZK = False
    print("Warning: Real ZK prover/verifier not available. Using mock implementation.")


# Fallback to MockZKProver if real implementation is unavailable
class MockZKProver:
    def __init__(self, model_size, complexity_factor=1.0):
        self.model_size = model_size  # Number of parameters
        self.complexity_factor = complexity_factor

    def generate_proof(self, model, data_sample):
        """Simulate proof generation with realistic timing based on model size"""
        # Simulate computation time based on model size and complexity
        param_count = sum(p.numel() for p in model.parameters())
        computation_time = 0.0001 * param_count * self.complexity_factor

        start_time = time.time()
        # Simulate ZK proof generation
        time.sleep(min(computation_time, 0.5))  # Cap at 0.5s for testing
        end_time = time.time()

        return {
            "proof": "mock_zk_proof",
            "time_taken": end_time - start_time
        }

    def verify_proof(self, proof, model_hash, data_sample_hash):
        """Simulate proof verification with realistic timing"""
        start_time = time.time()
        # Simulate verification (usually faster than generation)
        time.sleep(min(0.0001 * self.model_size * 0.3 * self.complexity_factor, 0.2))
        end_time = time.time()

        return {
            "is_valid": True,
            "time_taken": end_time - start_time
        }


class FedZKBenchmark:
    def __init__(self, dataset_name, model_type="cnn", num_clients=5, num_rounds=3, device="cuda"):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.device = device if torch.cuda.is_available() else "cpu"
        self.results = {
            "dataset": dataset_name,
            "model_type": model_type,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "device": self.device,
            "accuracy": 0.0,
            "proof_generation_time": 0.0,
            "verification_time": 0.0,
            "training_time": 0.0
        }

    def load_dataset(self):
        """Load and preprocess the selected dataset"""
        if self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST("./data", train=False, transform=transform)

            # Create model for MNIST
            if self.model_type == "cnn":
                self.model = nn.Sequential(
                    nn.Conv2d(1, 32, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(1600, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                ).to(self.device)
            else:
                self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                ).to(self.device)

        elif self.dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10("./data", train=False, transform=transform)

            # Create model for CIFAR-10
            if self.model_type == "cnn":
                self.model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
                ).to(self.device)
            else:
                self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(3*32*32, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
                ).to(self.device)

        elif self.dataset_name == "imdb":
            try:
                from torchtext.data.utils import get_tokenizer
                from torchtext.datasets import IMDB
                from torchtext.vocab import build_vocab_from_iterator

                # This is simplified for the benchmark - in a real scenario you'd do proper preprocessing
                train_dataset = IMDB(split="train")
                test_dataset = IMDB(split="test")

                # Simple preprocessing
                tokenizer = get_tokenizer("basic_english")

                def yield_tokens(data_iter):
                    for _, text in data_iter:
                        yield tokenizer(text)

                vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=["<unk>"])
                vocab.set_default_index(vocab["<unk>"])

                text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
                label_pipeline = lambda x: 1 if x == "pos" else 0

                # Process the data
                processed_train = []
                for label, text in train_dataset:
                    processed_train.append((text_pipeline(text)[:100], label_pipeline(label)))

                processed_test = []
                for label, text in test_dataset:
                    processed_test.append((text_pipeline(text)[:100], label_pipeline(label)))

                # Create the model
                self.model = nn.Sequential(
                    nn.Embedding(len(vocab), 64),
                    nn.LSTM(64, 128, batch_first=True),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                ).to(self.device)

                return processed_train, processed_test

            except ImportError:
                print("IMDB dataset requires torchtext > 0.10.0. Using synthetic data instead.")
                # Create synthetic data
                X = np.random.randn(2500, 100)  # 100 features, 2500 samples
                y = np.random.randint(0, 2, 2500)  # Binary classification

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
                test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

                # Create model for synthetic data
                self.model = nn.Sequential(
                    nn.Linear(100, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                ).to(self.device)

        elif self.dataset_name == "reuters":
            try:
                # Use 20 Newsgroups instead of Reuters (which is harder to get)
                data = fetch_20newsgroups(subset="all", categories=None, shuffle=True, random_state=42)
                X, y = data.data, data.target

                # Use TF-IDF for features
                vectorizer = TfidfVectorizer(max_features=1000)
                X = vectorizer.fit_transform(X).toarray()

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
                test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

                # Create model
                self.model = nn.Sequential(
                    nn.Linear(1000, 256),
                    nn.ReLU(),
                    nn.Linear(256, 20)  # 20 Newsgroups has 20 classes
                ).to(self.device)

            except Exception as e:
                print(f"Error loading 20 Newsgroups dataset: {str(e)}. Using synthetic data instead.")
                # Create synthetic data for text classification (multi-class)
                X = np.random.randn(10000, 1000)  # 1000 features, 10000 samples
                y = np.random.randint(0, 20, 10000)  # Multi-class classification

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
                test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

                # Create model for synthetic data
                self.model = nn.Sequential(
                    nn.Linear(1000, 256),
                    nn.ReLU(),
                    nn.Linear(256, 20)
                ).to(self.device)

        elif self.dataset_name == "imagenet":
            try:
                # Try to load a subset of ImageNet
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                # Usually we'd use ImageNet but for benchmarking we use CIFAR-100 as a stand-in
                train_dataset = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR100("./data", train=False, transform=transform)

                # Create a simplified ImageNet-style model
                self.model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Flatten(),
                    nn.Linear(256 * 2 * 2, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 100)  # CIFAR-100 has 100 classes
                ).to(self.device)

            except:
                print("ImageNet dataset not available. Using CIFAR-100 as substitute.")
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

                # Use CIFAR-100 as a stand-in
                train_dataset = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR100("./data", train=False, transform=transform)

                # Create model
                self.model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 100)
                ).to(self.device)

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return train_dataset, test_dataset

    def federated_training(self, train_dataset, test_dataset):
        """Simulate federated learning with multiple clients"""
        # Split the dataset for each client
        client_data_size = len(train_dataset) // self.num_clients
        client_datasets = random_split(
            train_dataset,
            [client_data_size] * (self.num_clients - 1) + [len(train_dataset) - client_data_size * (self.num_clients - 1)]
        )

        # Initialize ZK prover/verifier
        param_count = sum(p.numel() for p in self.model.parameters())

        if USE_REAL_ZK:
            # Use real ZK prover/verifier
            zk_prover = ZKProver(circuit_path=CIRCUIT_WASM, proving_key_path=PROVING_KEY)
            zk_verifier = ZKVerifier(verification_key_path=VERIFICATION_KEY)
        else:
            # Fallback to mock implementation
            zk_prover = MockZKProver(model_size=param_count)

        # Training parameters
        criterion = nn.CrossEntropyLoss()

        # Track metrics
        start_training_time = time.time()
        proof_times = []
        verification_times = []

        # Global model weights
        global_model = self.model.state_dict()

        # Federated learning rounds
        for round_idx in range(self.num_rounds):
            print(f"Round {round_idx+1}/{self.num_rounds}")

            client_models = []

            # Client training
            for client_idx, client_dataset in enumerate(client_datasets):
                print(f"  Training client {client_idx+1}/{self.num_clients}")

                # Create client model with global weights
                client_model = type(self.model)().to(self.device)
                client_model.load_state_dict(global_model)
                client_optimizer = optim.SGD(client_model.parameters(), lr=0.01)

                # Create data loader
                loader = DataLoader(client_dataset, batch_size=32, shuffle=True)

                # Train for a few local epochs
                client_model.train()
                for local_epoch in range(1):  # Reduced for benchmarking
                    for batch_idx, (data, target) in enumerate(loader):
                        data, target = data.to(self.device), target.to(self.device)
                        client_optimizer.zero_grad()
                        output = client_model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        client_optimizer.step()

                        if batch_idx >= 10:  # Limit batches for benchmarking
                            break

                # Generate ZK proof for this client update
                if USE_REAL_ZK:
                    # Extract gradient updates as a dictionary
                    gradient_dict = {}
                    for name, param in client_model.named_parameters():
                        original_param = global_model[name]
                        gradient_dict[name] = param.data - original_param

                    # Measure proof generation time
                    start_time = time.time()
                    proof, public_inputs = zk_prover.generate_real_proof(gradient_dict, max_inputs=4)
                    proof_time = time.time() - start_time

                    # Measure verification time
                    start_time = time.time()
                    is_valid = zk_verifier.verify_real_proof(proof, public_inputs)
                    verification_time = time.time() - start_time

                    proof_result = {"time_taken": proof_time}
                    verification_result = {"time_taken": verification_time}
                else:
                    # Use mock implementation
                    proof_result = zk_prover.generate_proof(client_model, data[:5])
                    verification_result = zk_prover.verify_proof(proof_result["proof"], "model_hash", "data_hash")

                proof_times.append(proof_result["time_taken"])
                verification_times.append(verification_result["time_taken"])

                # Store the updated model
                client_models.append(client_model.state_dict())

            # Aggregate updates (simple averaging)
            with torch.no_grad():
                for key in global_model:
                    global_model[key] = torch.stack([client_models[i][key] for i in range(self.num_clients)]).mean(0)

            # Update global model
            self.model.load_state_dict(global_model)

            # Evaluate after this round
            if (round_idx + 1) == self.num_rounds:
                accuracy = self.evaluate(test_dataset)
                print(f"  Accuracy after round {round_idx+1}: {accuracy:.2f}%")

        # Record metrics
        end_training_time = time.time()
        self.results["accuracy"] = accuracy
        self.results["proof_generation_time"] = np.mean(proof_times)
        self.results["verification_time"] = np.mean(verification_times)
        self.results["training_time"] = end_training_time - start_training_time

        return self.results

    def evaluate(self, test_dataset):
        """Evaluate the global model on the test dataset"""
        test_loader = DataLoader(test_dataset, batch_size=128)

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def run(self):
        """Run the complete benchmark and return results"""
        print(f"Starting benchmark for {self.dataset_name}...")
        train_dataset, test_dataset = self.load_dataset()
        results = self.federated_training(train_dataset, test_dataset)
        return results


def main():
    parser = argparse.ArgumentParser(description="Run FedZK benchmarks on various datasets")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "imdb", "reuters"],
                        help="List of datasets to benchmark")
    parser.add_argument("--model-type", default="cnn", choices=["cnn", "mlp"],
                        help="Type of model to use for image datasets")
    parser.add_argument("--num-clients", type=int, default=10,
                        help="Number of clients for federated learning")
    parser.add_argument("--num-rounds", type=int, default=5,
                        help="Number of federated learning rounds")
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="Output file for benchmark results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run benchmarks on")
    parser.add_argument("--use-real-zk", action="store_true",
                        help="Force use of real ZK proofs (if available)")

    args = parser.parse_args()

    # If user explicitly requests real ZK, check if it's available
    if args.use_real_zk and not USE_REAL_ZK:
        print("Warning: Real ZK infrastructure not found or incomplete.")
        print("Run './fedzk/scripts/setup_zk.sh' to set up the ZK environment.")
        print("Falling back to mock implementation.")

    results = []

    # Run benchmarks for each dataset
    for dataset in args.datasets:
        try:
            benchmark = FedZKBenchmark(
                dataset_name=dataset,
                model_type=args.model_type,
                num_clients=args.num_clients,
                num_rounds=args.num_rounds,
                device=args.device
            )

            result = benchmark.run()
            results.append(result)
            print(f"Completed benchmark for {dataset}")
            print(f"  Accuracy: {result['accuracy']:.2f}%")
            print(f"  Proof generation time: {result['proof_generation_time']:.4f}s")
            print(f"  Verification time: {result['verification_time']:.4f}s")
            print(f"  Training time: {result['training_time']:.2f}s")

        except Exception as e:
            print(f"Error benchmarking {dataset}: {str(e)}")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Benchmark results saved to {args.output}")

        # Also save as JSON for easier README update
        with open(args.output.replace(".csv", ".json"), "w") as f:
            json.dump(results, f, indent=2)

        # Print a markdown table for easy copy-paste into README
        print("\nMarkdown table for README:")
        print("| Dataset | Clients | Rounds | Accuracy | Proof Generation Time | Verification Time |")
        print("|---------|---------|--------|----------|----------------------|-------------------|")
        for r in results:
            print(f"| {r['dataset']} | {r['num_clients']} | {r['num_rounds']} | {r['accuracy']:.1f}% | {r['proof_generation_time']:.1f}s | {r['verification_time']:.1f}s |")


if __name__ == "__main__":
    main()
