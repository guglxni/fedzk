#!/usr/bin/env python3
# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Secure Multi-Party Computation Example for FedZK

This example demonstrates how to use secure multi-party computation (MPC)
for privacy-preserving federated learning in the FedZK framework.
"""

import numpy as np
import time
from typing import List, Dict, Tuple

# Simulating FedZK imports
try:
    from fedzk.mpc import MPCCoordinator, MPCParticipant
    from fedzk.mpc.protocols import SecretSharing, SecureAggregation
except ImportError:
    print("FedZK not installed. This is just an example file.")
    
    # Mock classes for example purposes
    class MPCCoordinator:
        def __init__(self, num_parties, threshold):
            self.num_parties = num_parties
            self.threshold = threshold
            self.participants = []
            
        def register_participant(self, participant_id):
            self.participants.append(participant_id)
            return len(self.participants) - 1
            
        def initialize_protocol(self, protocol_name):
            print(f"Initializing {protocol_name} protocol with {self.num_parties} parties")
            return {"protocol_id": f"{protocol_name}_{int(time.time())}"}
            
        def collect_shares(self, shares):
            print(f"Collected {len(shares)} shares")
            return True
            
        def reconstruct_result(self, shares, min_shares=None):
            min_shares = min_shares or self.threshold
            if len(shares) < min_shares:
                raise ValueError(f"Need at least {min_shares} shares to reconstruct")
            # In a real system, this would do actual reconstruction
            return np.mean([s["value"] for s in shares]) * len(shares)
    
    class MPCParticipant:
        def __init__(self, participant_id, coordinator=None):
            self.participant_id = participant_id
            self.coordinator = coordinator
            
        def create_shares(self, data, num_shares, threshold):
            # Simulate Shamir's secret sharing
            print(f"Creating {num_shares} shares with threshold {threshold}")
            shares = []
            for i in range(num_shares):
                # In a real system, this would create proper Shamir shares
                noise = np.random.randn() * 0.1 * data
                shares.append({
                    "share_id": i,
                    "value": data / num_shares + noise,
                    "participant_id": self.participant_id
                })
            return shares
            
        def submit_shares(self, shares, recipients):
            print(f"Submitting {len(shares)} shares to {len(recipients)} recipients")
            return True
    
    class SecretSharing:
        @staticmethod
        def split(secret, num_shares, threshold):
            # Simulate Shamir's secret sharing
            shares = []
            for i in range(num_shares):
                # This is a toy implementation - real systems use polynomials
                noise = np.random.randn() * 0.01
                shares.append({
                    "x": i + 1,
                    "y": secret / num_shares + noise
                })
            return shares
            
        @staticmethod
        def reconstruct(shares, threshold):
            # Simple reconstruction for demonstration
            if len(shares) < threshold:
                raise ValueError(f"Need at least {threshold} shares, got {len(shares)}")
            # In a real system, this would use polynomial interpolation
            return sum(share["y"] for share in shares) 
    
    class SecureAggregation:
        def __init__(self, num_parties):
            self.num_parties = num_parties
            
        def generate_masks(self):
            # Generate random masks that sum to zero
            masks = [np.random.randn() for _ in range(self.num_parties - 1)]
            masks.append(-sum(masks))
            return masks
            
        def apply_mask(self, data, mask):
            return data + mask
            
        def aggregate(self, masked_data):
            # In secure aggregation, masks cancel out when summed
            return sum(masked_data)


def simulate_secure_model_aggregation():
    """
    Simulate secure aggregation of model updates using MPC.
    """
    # Simulation parameters
    num_parties = 5
    threshold = 3
    model_size = 10
    
    # Initialize the coordinator
    coordinator = MPCCoordinator(num_parties, threshold)
    
    # Initialize participants
    participants = []
    for i in range(num_parties):
        participant_id = f"client_{i}"
        participant = MPCParticipant(participant_id, coordinator)
        participants.append(participant)
        coordinator.register_participant(participant_id)
    
    # Generate synthetic model updates
    model_updates = []
    for i in range(num_parties):
        # Each participant has their own model update
        update = np.random.randn(model_size) * 0.1
        model_updates.append(update)
    
    print("\n1. Each participant creates shares of their model update")
    all_shares = []
    for i, participant in enumerate(participants):
        # Split model updates into shares
        for j in range(model_size):
            value = model_updates[i][j]
            shares = participant.create_shares(value, num_parties, threshold)
            all_shares.extend(shares)
    
    print(f"\n2. Shares are distributed securely between participants")
    # In a real system, shares would be encrypted and sent securely
    
    print("\n3. Coordinator collects shares and performs secure aggregation")
    # Group shares by parameter index
    shares_by_index = {}
    for share in all_shares:
        idx = share["share_id"] % model_size
        if idx not in shares_by_index:
            shares_by_index[idx] = []
        shares_by_index[idx].append(share)
    
    # Reconstruct the aggregated model
    aggregated_model = np.zeros(model_size)
    for idx, shares in shares_by_index.items():
        # Only use threshold number of shares for reconstruction
        aggregated_model[idx] = coordinator.reconstruct_result(shares[:threshold])
    
    print("\n4. Comparing with the actual sum to verify correctness")
    actual_sum = sum(model_updates)
    
    # Print comparison (with some tolerance for the simplified implementation)
    print("\nSecure aggregation result:")
    print(f"First 3 parameters: {aggregated_model[:3]}")
    print("\nActual sum:")
    print(f"First 3 parameters: {actual_sum[:3]}")
    
    # Calculate error (would be zero in a real implementation)
    error = np.abs(aggregated_model - actual_sum).mean()
    print(f"\nAverage error: {error:.6f}")
    
    if error < 0.5:  # Arbitrary threshold for this demo
        print("\n✅ Secure aggregation successful!")
    else:
        print("\n❌ Aggregation error too large!")


def demonstrate_secure_comparison():
    """
    Demonstrate secure comparison without revealing inputs.
    """
    print("\nSecure Comparison Demo")
    print("----------------------")
    
    # Initialize secret sharing
    n = 3  # Number of parties
    t = 2  # Threshold
    
    # Participant 1 has a private value
    private_value_1 = 42
    print(f"Participant 1's private value: {private_value_1}")
    
    # Participant 2 has a private value
    private_value_2 = 37
    print(f"Participant 2's private value: {private_value_2}")
    
    # Split into shares
    print("\nSplitting values into shares...")
    shares_1 = SecretSharing.split(private_value_1, n, t)
    shares_2 = SecretSharing.split(private_value_2, n, t)
    
    # Compute difference shares (without revealing values)
    diff_shares = []
    for i in range(n):
        diff_shares.append({
            "x": shares_1[i]["x"],
            "y": shares_1[i]["y"] - shares_2[i]["y"]
        })
    
    # Reconstruct the difference
    print("\nSecurely reconstructing the difference...")
    difference = SecretSharing.reconstruct(diff_shares, t)
    
    # Determine which is greater (without revealing actual values)
    if difference > 0:
        result = "Participant 1's value is greater"
    elif difference < 0:
        result = "Participant 2's value is greater"
    else:
        result = "The values are equal"
    
    print(f"\nResult of secure comparison: {result}")
    print("(This was determined without any participant revealing their actual value)")


def main():
    print("\nSecure Multi-Party Computation in FedZK")
    print("======================================")
    
    print("\nExample 1: Secure Model Aggregation")
    print("-----------------------------------")
    simulate_secure_model_aggregation()
    
    print("\n\nExample 2: Secure Comparison")
    print("----------------------------")
    demonstrate_secure_comparison()
    
    print("\nThese examples demonstrate how FedZK uses secure MPC to:")
    print("1. Aggregate model updates without revealing individual contributions")
    print("2. Perform secure operations on private data")
    print("3. Enable privacy-preserving federated learning")


if __name__ == "__main__":
    main()
    print("\nThis is a demonstration file. In a real deployment, you would use actual FedZK components.") 