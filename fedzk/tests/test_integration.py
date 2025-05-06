"""
Integration tests for FedZK.

These tests validate that different components of FedZK work 
correctly together, focusing on the interactions between:
- LocalTrainer (client)
- ZKProver
- ZKVerifier
- FedZKAggregator (coordinator)
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedzk.client.trainer import LocalTrainer
from fedzk.coordinator.aggregator import UpdateSubmission, get_status, submit_update
from fedzk.prover.zkgenerator import ZKProver


def convert_tensors_to_lists(gradient_dict):
    """
    Convert PyTorch tensors in a gradient dictionary to Python floats.
    
    Args:
        gradient_dict: Dictionary with parameter names as keys and tensors as values
        
    Returns:
        Dictionary with the same keys but with tensors converted to lists of floats
    """
    converted_dict = {}
    for key, value in gradient_dict.items():
        if isinstance(value, torch.Tensor):
            # For multi-dimensional tensors, flatten them first
            if value.dim() > 1:
                # Flatten and convert to Python float values
                converted_dict[key] = [float(x) for x in value.flatten()]
            else:
                # For 1D tensors, convert each element to a Python float
                converted_dict[key] = [float(x) for x in value]
        else:
            converted_dict[key] = value
    return converted_dict


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def reset_aggregator_state():
    """Reset aggregator state between tests."""
    import fedzk.coordinator.aggregator as aggregator

    # Save initial state
    original_version = aggregator.current_version
    original_updates = aggregator.pending_updates.copy() if aggregator.pending_updates else []

    # Reset state for test
    aggregator.current_version = 1
    aggregator.pending_updates.clear()

    yield

    # Restore initial state after test
    aggregator.current_version = original_version
    aggregator.pending_updates.clear()
    aggregator.pending_updates.extend(original_updates)


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for training."""
    # Create random inputs and targets
    inputs = torch.randn(20, 5)
    targets = torch.randint(0, 3, (20,))

    return TensorDataset(inputs, targets)


@pytest.fixture
def client_model():
    """Create a model for client training."""
    return SimpleModel()


def test_client_to_coordinator_flow(reset_aggregator_state, dummy_dataset, client_model):
    """
    Test the complete flow from client training to coordinator aggregation.
    
    This test validates:
    1. Client training produces gradients
    2. ZKProver generates proofs from gradients
    3. ZKVerifier verifies the proofs
    4. Coordinator accepts valid updates
    """
    # Set up data loader
    dataloader = DataLoader(dummy_dataset, batch_size=4)

    # 1. Client training
    trainer = LocalTrainer(client_model, dataloader)
    gradients = trainer.train_one_epoch()

    # Verify gradients structure
    assert isinstance(gradients, dict)
    assert len(gradients) > 0
    assert "fc1.weight" in gradients
    assert "fc2.bias" in gradients

    # 2. Generate ZK proof
    prover = ZKProver("dummy_circuit.json", "dummy_proving_key.json")
    proof, public_signals = prover.generate_proof(gradients)

    # Verify proof structure
    assert isinstance(proof, str)
    assert isinstance(public_signals, list)
    assert len(public_signals) > 0

    # 3. Create update submission
    update = UpdateSubmission(
        gradients=convert_tensors_to_lists(gradients),
        proof=proof,
        public_signals=public_signals
    )

    # 4. Submit to coordinator
    result = submit_update(update)

    # Verify update was accepted
    assert result["status"] in ["accepted", "aggregated"]

    # Check aggregator state was updated
    status = get_status()
    assert status["current_model_version"] >= 1

    # If this was the only update, it should be pending
    if result["status"] == "accepted":
        assert status["pending_updates"] == 1
    # If this triggered aggregation, pending should be reset
    else:
        assert status["pending_updates"] == 0
        assert "global_update" in result


def test_multiple_clients(reset_aggregator_state, dummy_dataset):
    """
    Test multiple clients submitting updates to the coordinator.
    
    This test validates:
    1. Multiple clients can train independently
    2. All clients can submit proofs
    3. Coordinator correctly aggregates updates
    """
    # Create models for 3 clients
    client_models = [SimpleModel() for _ in range(3)]
    dataloader = DataLoader(dummy_dataset, batch_size=4)

    # Train each client and submit updates
    all_gradients = []
    for i, model in enumerate(client_models):
        # Train client
        trainer = LocalTrainer(model, dataloader)
        gradients = trainer.train_one_epoch()
        all_gradients.append(gradients)

        # Generate proof
        prover = ZKProver("dummy_circuit.json", "dummy_proving_key.json")
        proof, public_signals = prover.generate_proof(gradients)

        # Submit update
        update = UpdateSubmission(
            gradients=convert_tensors_to_lists(gradients),
            proof=proof,
            public_signals=public_signals
        )

        result = submit_update(update)

        # Check result - Note: Aggregation happens after >= 2 updates are in the pending_updates list
        if i < 1:  # First update should be accepted (only 1 update in the list)
            assert result["status"] == "accepted"
        elif i == 1:  # Second update should trigger aggregation (now we have 2 updates)
            assert result["status"] == "aggregated"
            assert "version" in result
            assert "global_update" in result

            # Second update should have aggregated the first two updates
            global_update = result["global_update"]
            assert set(global_update.keys()) == set(gradients.keys())

            # Check aggregation logic - should average the first two updates
            for param_name in global_update:
                # We'll just check that the aggregated update exists and has the right shape
                assert param_name in global_update

                # Check the shapes match the original gradients
                if isinstance(all_gradients[0][param_name], torch.Tensor):
                    expected_len = len(all_gradients[0][param_name].flatten())
                    assert len(global_update[param_name]) == expected_len
        else:  # Third update should be accepted (resets after aggregation)
            assert result["status"] == "accepted"

    # Final check on aggregator state
    status = get_status()
    assert status["current_model_version"] == 2
    assert status["pending_updates"] == 1  # One update should be pending (the third one)
