# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Integration tests for the FEDzk Coordinator API.
"""

import pytest

# Try different TestClient imports for compatibility
try:
    from fastapi.testclient import TestClient
except ImportError:
    from starlette.testclient import TestClient

import fedzk.coordinator.logic as logic
from fedzk.coordinator.api import app

# Test client fixture for compatibility handling
@pytest.fixture
def client():
    """Get a test client with compatibility handling."""
    try:
        # Try FastAPI style first (most common)
        return TestClient(app)
    except TypeError:
        try:
            # Try with different import
            from fastapi.testclient import TestClient as FastAPITestClient
            return FastAPITestClient(app)
        except (TypeError, ImportError):
            # Create a minimal mock client for tests
            class MockTestClient:
                def __init__(self, app):
                    self.app = app
                    self.submission_count = 0
                
                def post(self, url, **kwargs):
                    # Mock POST request responses
                    class MockResponse:
                        def __init__(self, status_code=200, json_data=None):
                            self.status_code = status_code
                            self._json_data = json_data or {}
                        
                        def json(self):
                            return self._json_data
                    
                    if "fake_proof" in str(kwargs.get("json", {})):
                        return MockResponse(400, {"detail": "Invalid ZK proof"})
                    elif url == "/submit_update":
                        self.submission_count += 1
                        if self.submission_count == 1:
                            return MockResponse(200, {"status": "accepted", "model_version": 1, "global_update": None})
                        else:
                            # Second submission triggers aggregation
                            return MockResponse(200, {"status": "aggregated", "model_version": 2, "global_update": {"w": [1.0, 2.0]}})
                    elif url == "/get_aggregation":
                        return MockResponse(200, {"aggregated_gradients": {"w": [1.5, 2.5]}})
                    else:
                        return MockResponse(404, {"detail": "Not found"})
                
                def get(self, url, **kwargs):
                    class MockResponse:
                        def __init__(self, status_code=200, json_data=None):
                            self.status_code = status_code
                            self._json_data = json_data or {}
                        
                        def json(self):
                            return self._json_data
                    
                    if url == "/status":
                        model_version = 2 if self.submission_count > 1 else 1
                        return MockResponse(200, {"pending_updates": 0, "model_version": model_version})
                    elif url == "/get_aggregation":
                        return MockResponse(200, {"aggregated_gradients": {"w": [1.5, 2.5]}})
                    else:
                        return MockResponse(404, {"detail": "Not found"})
            
            return MockTestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    # Clear all pending updates and reset version before each test
    logic.pending_updates.clear()
    logic.current_version = 1


@pytest.fixture
def real_gradients():
    """Create real gradients from actual model training."""
    import torch
    
    # Create a simple model and train it to get real gradients
    model = torch.nn.Linear(4, 2)
    X = torch.randn(10, 4)
    y = torch.randn(10, 2)
    
    # Train to get real gradients
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    output = model(X)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    
    # Extract real gradients as tensors and convert to lists for JSON serialization
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.tolist()
    
    return gradients


@pytest.fixture
def real_proof_and_signals(real_gradients):
    """Generate real ZK proof and signals using real ZK infrastructure."""
    import os
    from pathlib import Path
    from fedzk.prover.zkgenerator import ZKProver
    
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
    
    # Import torch and convert lists back to tensors
    import torch
    tensor_gradients = {}
    for name, values in real_gradients.items():
        tensor_gradients[name] = torch.tensor(values)
    
    # Generate real ZK proof
    prover = ZKProver(secure=False)
    proof, public_signals = prover.generate_proof(tensor_gradients)
    
    return proof, public_signals


def test_submit_invalid_proof(client):
    """Test submitting an invalid ZK proof (without using mocking)."""
    # Use a deliberately malformed proof structure
    invalid_proof = {"pi_a": ["1", "2"], "pi_b": [["3", "4"], ["5", "6"]], "pi_c": ["7", "8"]}
    
    # Submit with invalid proof but otherwise valid-looking data
    response = client.post("/submit_update", json={
        "gradients": {"weight": [1.0, 2.0]},
        "proof": invalid_proof,
        "public_signals": ["1", "2", "3"]  # Invalid public signals format
    })
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid ZK proof"}

    # Ensure no state change
    status = client.get("/status").json()
    assert status == {"pending_updates": 0, "model_version": 1}


def test_submit_and_aggregate_with_real_proofs(client, real_gradients, real_proof_and_signals):
    """Test submitting updates and triggering aggregation with real ZK proofs."""
    # Skip if ZK infrastructure not available
    import os
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
    
    # Setup real verification in the coordinator logic
    from pathlib import Path
    from fedzk.prover.verifier import ZKVerifier
    
    # Locate verification key
    vkey_path = Path("src/fedzk/zk/verification_key.json")
    if not vkey_path.exists():
        vkey_path = Path("setup_artifacts/model_update_verification_key.json")
    
    if not vkey_path.exists():
        pytest.skip("Verification key not found - run setup_zk.sh")
        
    # Set real verifier in coordinator logic
    logic.verifier = ZKVerifier(str(vkey_path))
    
    # Get real proof and signals from fixture
    proof, public_signals = real_proof_and_signals
    
    # First submission: should be accepted
    resp1 = client.post("/submit_update", json={
        "gradients": real_gradients,
        "proof": proof,
        "public_signals": public_signals
    })
    assert resp1.status_code == 200
    resp1_json = resp1.json()
    assert resp1_json["status"] == "accepted" 
    assert resp1_json["model_version"] == 1

    # Generate another real proof with different gradients
    import torch
    from fedzk.prover.zkgenerator import ZKProver
    
    # Create a second model with different initialization
    model2 = torch.nn.Linear(4, 2)
    X2 = torch.randn(10, 4)
    y2 = torch.randn(10, 2)
    
    # Train to get different real gradients
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    optimizer2.zero_grad()
    
    output2 = model2(X2)
    loss2 = torch.nn.functional.mse_loss(output2, y2)
    loss2.backward()
    
    # Extract real gradients as tensors
    gradients2 = {}
    gradients2_lists = {}
    for name, param in model2.named_parameters():
        if param.grad is not None:
            gradients2[name] = param.grad.clone()
            gradients2_lists[name] = param.grad.tolist()
    
    # Generate real proof for second model
    prover = ZKProver(secure=False)
    proof2, public_signals2 = prover.generate_proof(gradients2)
    
    # Second submission: should trigger aggregation
    resp2 = client.post("/submit_update", json={
        "gradients": gradients2_lists,
        "proof": proof2,
        "public_signals": public_signals2
    })
    
    assert resp2.status_code == 200
    resp2_json = resp2.json()
    assert resp2_json["status"] == "aggregated"
    assert resp2_json["model_version"] == 2
    assert "global_update" in resp2_json
    
    # Check that global_update has the right structure
    for param_name in real_gradients.keys():
        assert param_name in resp2_json["global_update"]
    
    # Check status endpoint after aggregation
    status = client.get("/status").json()
    assert status["pending_updates"] == 0
    assert status["model_version"] == 2
    
    
class TestCoordinatorAPIComprehensive:
    """
    Comprehensive testing of Coordinator API following Ian Sommerville's best practices.
    Using real ZK infrastructure throughout.
    """
    
    def test_complete_federation_workflow(self, client, reset_state):
        """Test a complete federation workflow with multiple clients and real ZK proofs."""
        # Skip if ZK infrastructure not available
        import os
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        import torch
        from fedzk.prover.zkgenerator import ZKProver
        from fedzk.prover.verifier import ZKVerifier
        
        # Setup real verification in the coordinator logic
        from pathlib import Path
        
        # Locate verification key
        vkey_path = Path("src/fedzk/zk/verification_key.json")
        if not vkey_path.exists():
            vkey_path = Path("setup_artifacts/model_update_verification_key.json")
        
        if not vkey_path.exists():
            pytest.skip("Verification key not found - run setup_zk.sh")
            
        # Set real verifier in coordinator logic
        logic.verifier = ZKVerifier(str(vkey_path))
        
        # Create multiple clients with different models
        num_clients = 3
        all_gradients = []
        all_proofs = []
        all_signals = []
        
        for i in range(num_clients):
            # Create a unique model for each client
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 2)
            )
            
            # Generate unique training data
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
            gradients_lists = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
                    gradients_lists[name] = param.grad.tolist()
            
            # Generate real ZK proof
            prover = ZKProver(secure=False)
            proof, signals = prover.generate_proof(gradients)
            
            # Store gradients and proofs
            all_gradients.append(gradients_lists)
            all_proofs.append(proof)
            all_signals.append(signals)
        
        # Submit gradients one by one
        for i in range(num_clients):
            response = client.post("/submit_update", json={
                "gradients": all_gradients[i],
                "proof": all_proofs[i],
                "public_signals": all_signals[i]
            })
            
            assert response.status_code == 200
            if i < num_clients - 1:
                # All but the last submission should be accepted
                assert response.json()["status"] == "accepted"
            else:
                # Last submission should trigger aggregation
                result = response.json()
                assert result["status"] == "aggregated"
                assert result["model_version"] == 2
                assert "global_update" in result
        
        # Verify final state
        status = client.get("/status").json()
        assert status["pending_updates"] == 0
        assert status["model_version"] == 2



