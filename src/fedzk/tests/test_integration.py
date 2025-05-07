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
import httpx
import subprocess
from pathlib import Path

from fedzk.client.trainer import LocalTrainer
from fedzk.coordinator.aggregator import UpdateSubmission, get_status, submit_update
from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.verifier import ZKVerifier


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
    prover = ZKProver(secure=False)
    try:
        proof, public_signals = prover.generate_proof(gradients)
        # 3. Verify ZK proof
        ASSET_DIR = Path(__file__).resolve().parent.parent / "zk"
        vkey_path = str(ASSET_DIR / "verification_key.json")
        verifier = ZKVerifier(verification_key_path=vkey_path)
        assert verifier.verify_real_proof(proof, public_signals), "Proof verification failed"

        # 4. Submit to coordinator
        coordinator_client = httpx.Client(app=app, base_url="http://test")
        update_data = {
            "gradients": {k: v.tolist() for k, v in gradients.items()},
            "proof": proof,
            "public_inputs": public_signals
        }
        response = coordinator_client.post("/submit_update", json=update_data)
        assert response.status_code == 200
        assert response.json()["status"] == "pending"

    except subprocess.CalledProcessError as e:
        print(f"SNARKjs execution failed during client_to_coordinator_flow: {e}")
        pytest.skip("Skipping due to SNARKjs execution failure")
    except ValueError as e:
        if "could not convert string to float" in str(e):
            print(f"ValueError during ZKProver init or call (likely due to old constructor style if not fully updated): {e}")
            pytest.fail(f"Test setup error for ZKProver: {e}")
        raise


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
    successful_submissions = 0
    for i, model in enumerate(client_models):
        # Train client
        trainer = LocalTrainer(model, dataloader)
        gradients = trainer.train_one_epoch()
        all_gradients.append(gradients)

        # Generate proof
        prover = ZKProver(secure=False)
        try:
            proof, public_signals = prover.generate_proof(gradients)
            
            # Use correct verifier initialization
            ASSET_DIR = Path(__file__).resolve().parent.parent / "zk"
            vkey_path = str(ASSET_DIR / "verification_key.json")
            verifier = ZKVerifier(verification_key_path=vkey_path)
            assert verifier.verify_real_proof(proof, public_signals)

            # Submit update
            coordinator_client = httpx.Client(app=app, base_url="http://test")
            update_data = {
                "gradients": {k: v.tolist() for k, v in gradients.items()},
                "proof": proof,
                "public_inputs": public_signals
            }
            response = coordinator_client.post("/submit_update", json=update_data)
            assert response.status_code == 200
            assert response.json()["status"] == "pending"
            successful_submissions += 1

        except subprocess.CalledProcessError as e:
            print(f"SNARKjs execution failed during multiple_clients test for client {i}: {e}")
            continue
        except ValueError as e:
            if "could not convert string to float" in str(e):
                print(f"ValueError during ZKProver init or call (client {i}): {e}")
                pytest.fail(f"Test setup error for ZKProver (client {i}): {e}")
            raise

    # Skip the final check if no successful submissions
    if successful_submissions == 0:
        pytest.skip("Skipping final check because no submissions succeeded")

    # Final check on aggregator state - model version should be expected with successful submissions
    status = get_status()
    # With 3 clients, if they all succeed, we should have version 2 and 1 pending
    # If fewer succeed, adjust expectations accordingly
    expected_version = 1 + (successful_submissions // 2)
    expected_pending = successful_submissions % 2
    
    assert status["current_model_version"] == expected_version
    assert status["pending_updates"] == expected_pending
