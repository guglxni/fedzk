# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Tests for the ZKVerifier class and the aggregation logic in the FedZKAggregator.
Using real ZK infrastructure - NO MOCKING.
Following Ian Sommerville's software engineering best practices.
"""

import os
import pytest
import torch
from pathlib import Path

from fedzk.coordinator.aggregator import (
    UpdateSubmission,
    get_status,
    pending_updates,
    submit_update,
)
from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver


@pytest.fixture
def reset_state():
    """Reset global state between tests."""
    global current_version, pending_updates
    current_version = 1
    pending_updates.clear()
    yield
    # Clean up after each test
    current_version = 1
    pending_updates.clear()


@pytest.fixture
def real_gradients1():
    """Real gradients for first client from actual training."""
    # Create a simple linear model and train it to get real gradients
    model = torch.nn.Linear(4, 2)
    X = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    y = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    
    # Train to get real gradients
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    output = model(X)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    
    # Extract real gradients as tensors
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    
    return gradients


@pytest.fixture
def real_gradients2():
    """Real gradients for second client from actual training."""
    # Create a simple linear model with different initialization
    model = torch.nn.Linear(4, 2)
    X = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]])
    y = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    
    # Train to get real gradients
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    output = model(X)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    
    # Extract real gradients as tensors
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    
    return gradients


@pytest.fixture
def real_proof_and_signals(real_gradients1):
    """Generate real ZK proof and public signals."""
    # Skip if ZK infrastructure not verified
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
    # Generate real ZK proof
    prover = ZKProver(secure=False)
    proof, public_signals = prover.generate_proof(real_gradients1)
    
    return proof, public_signals


def test_verifier_with_real_proof_and_verification_key(real_proof_and_signals):
    """Test that ZKVerifier correctly validates a real proof with a real verification key."""
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
    vkey_path = Path("src/fedzk/zk/verification_key.json")
    
    if not vkey_path.exists():
        vkey_path = Path("setup_artifacts/model_update_verification_key.json")
        
    if not vkey_path.exists():
        pytest.skip("Verification key not found - run setup_zk.sh")
    
    # Get real proof and signals from fixture
    proof, public_signals = real_proof_and_signals
    
    # Create real verifier with actual verification key
    verifier = ZKVerifier(str(vkey_path))
    
    # Use real verification
    is_valid = verifier.verify_proof(proof, public_signals)
    
    assert is_valid is True, "Real ZK proof should be verified as valid"
    
    # Test with invalid inputs
    is_valid_empty = verifier.verify_proof({}, [])
    assert is_valid_empty is False, "Empty proof should be rejected"


def test_status_endpoint(reset_state):
    """Test that the status endpoint returns correct initial state."""
    status = get_status()

    assert status["current_model_version"] == 1
    assert status["pending_updates"] == 0


def test_submit_update_single(reset_state, real_gradients1, real_proof_and_signals):
    """Test submitting a single update with real ZK proof."""
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
    
    # Get real proof and signals
    proof, public_signals = real_proof_and_signals
    
    # Create update submission with real data
    update = UpdateSubmission(
        gradients=real_gradients1,
        proof=proof,
        public_signals=public_signals
    )

    # Setup verifier with real verification key
    vkey_path = Path("src/fedzk/zk/verification_key.json")
    if not vkey_path.exists():
        vkey_path = Path("setup_artifacts/model_update_verification_key.json")
    
    if not vkey_path.exists():
        pytest.skip("Verification key not found - run setup_zk.sh")
    
    # Use real verification in the aggregator
    from fedzk.coordinator import aggregator
    aggregator.verifier = ZKVerifier(str(vkey_path))
    
    # Submit update
    result = submit_update(update)

    assert result["status"] == "accepted"
    assert result["version"] == 1

    # Check aggregator state
    status = get_status()
    assert status["pending_updates"] == 1
    assert status["current_model_version"] == 1


def test_submit_update_with_aggregation(reset_state, real_gradients1, real_gradients2):
    """Test submitting two updates with real ZK proofs, which should trigger aggregation."""
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
    
    # Setup real prover
    prover = ZKProver(secure=False)
    
    # Generate real proofs for both sets of gradients
    proof1, signals1 = prover.generate_proof(real_gradients1)
    proof2, signals2 = prover.generate_proof(real_gradients2)
    
    # Setup verifier with real verification key
    vkey_path = Path("src/fedzk/zk/verification_key.json")
    if not vkey_path.exists():
        vkey_path = Path("setup_artifacts/model_update_verification_key.json")
    
    if not vkey_path.exists():
        pytest.skip("Verification key not found - run setup_zk.sh")
    
    # Use real verification in the aggregator
    from fedzk.coordinator import aggregator
    aggregator.verifier = ZKVerifier(str(vkey_path))
    
    # Submit first update with real proof
    update1 = UpdateSubmission(
        gradients=real_gradients1,
        proof=proof1,
        public_signals=signals1
    )
    result1 = submit_update(update1)
    assert result1["status"] == "accepted"

    # Submit second update with real proof (should trigger aggregation)
    update2 = UpdateSubmission(
        gradients=real_gradients2,
        proof=proof2,
        public_signals=signals2
    )
    result2 = submit_update(update2)

    # Check aggregation result
    assert result2["status"] == "aggregated"
    assert result2["version"] == 2  # Version should be incremented
    assert "global_update" in result2

    # Verify that the aggregation logic is correct (FedAvg)
    global_update = result2["global_update"]
    for param_name, values in global_update.items():
        param_values = values.tolist() if hasattr(values, 'tolist') else values
        param1_values = real_gradients1[param_name].tolist() if hasattr(real_gradients1[param_name], 'tolist') else real_gradients1[param_name]
        param2_values = real_gradients2[param_name].tolist() if hasattr(real_gradients2[param_name], 'tolist') else real_gradients2[param_name]
        
        # Check if shapes match
        if isinstance(param_values, list) and isinstance(param1_values, list) and isinstance(param2_values, list):
            # Handle flat lists
            for i, value in enumerate(param_values):
                expected = (param1_values[i] + param2_values[i]) / 2
                assert abs(value - expected) < 1e-5
        else:
            # Handle tensors - compare element-wise
            assert torch.allclose(
                values, 
                (real_gradients1[param_name] + real_gradients2[param_name]) / 2, 
                rtol=1e-4
            )

    # Check aggregator state
    status = get_status()
    assert status["pending_updates"] == 0  # Should be reset after aggregation
    assert status["current_model_version"] == 2  # Should be incremented
    
    
class TestFedZKAggregatorComprehensive:
    """
    Comprehensive real ZK testing for the FedZK Aggregator.
    Following Ian Sommerville's software engineering best practices.
    """
    
    def test_real_batch_aggregation_with_multiple_models(self, reset_state):
        """Test comprehensive federated learning with multiple real models and ZK proofs."""
        # Skip if ZK infrastructure not available
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        # Create multiple models and train them
        num_clients = 3
        clients = []
        updates = []
        
        # Setup verifier with real verification key
        vkey_path = Path("src/fedzk/zk/verification_key.json")
        if not vkey_path.exists():
            vkey_path = Path("setup_artifacts/model_update_verification_key.json")
        
        if not vkey_path.exists():
            pytest.skip("Verification key not found - run setup_zk.sh")
            
        # Use real verification in the aggregator
        from fedzk.coordinator import aggregator
        aggregator.verifier = ZKVerifier(str(vkey_path))
        
        for i in range(num_clients):
            # Create model and data
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 2)
            )
            X = torch.randn(5, 4)
            y = torch.randn(5, 2)
            
            # Train to get real gradients
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            
            output = model(X)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            
            # Extract real gradients
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            # Generate real ZK proof
            prover = ZKProver(secure=False)
            proof, signals = prover.generate_proof(gradients)
            
            # Create update
            update = UpdateSubmission(
                gradients=gradients,
                proof=proof,
                public_signals=signals
            )
            
            # Store for submission
            clients.append(model)
            updates.append(update)
        
        # Submit updates one by one
        for i, update in enumerate(updates):
            result = submit_update(update)
            
            if i < num_clients - 1:
                assert result["status"] == "accepted"
            else:
                assert result["status"] == "aggregated"
                assert result["version"] == 2
                assert "global_update" in result
        
        # Verify final state
        status = get_status()
        assert status["current_model_version"] == 2
        assert status["pending_updates"] == 0
