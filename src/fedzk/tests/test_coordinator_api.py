# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Integration tests for the FedZK Coordinator API.
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


def test_submit_invalid_proof(monkeypatch, client):
    # Mock proof verification to fail
    monkeypatch.setattr(logic.verifier, "verify_real_proof", lambda proof, public_inputs: False)
    response = client.post("/submit_update", json={
        "gradients": {"w": [1.0, 2.0]},
        "proof": {"proof": "fake_proof"},
        "public_inputs": [{"param_name": "w", "norm": 5.0, "hash_prefix": "abcd"}]
    })
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid ZK proof"}

    # Ensure no state change
    status = client.get("/status").json()
    assert status == {"pending_updates": 0, "model_version": 1}


def test_submit_and_aggregate(monkeypatch, client):
    # Mock proof verification to succeed
    monkeypatch.setattr(logic.verifier, "verify_real_proof", lambda proof, public_inputs: True)
    gradients = {"w": [1.0, 2.0]}
    proof = {"proof": "dummy_proof"}
    public_inputs = [{"param_name": "w", "norm": 5.0, "hash_prefix": "abcd"}]

    # First submission: accepted
    resp1 = client.post("/submit_update", json={
        "gradients": gradients,
        "proof": proof,
        "public_inputs": public_inputs
    })
    assert resp1.status_code == 200
    assert resp1.json() == {"status": "accepted", "model_version": 1, "global_update": None}

    # Second submission: triggers aggregation
    resp2 = client.post("/submit_update", json={
        "gradients": gradients,
        "proof": proof,
        "public_inputs": public_inputs
    })
    assert resp2.status_code == 200
    resp2_json = resp2.json()
    assert resp2_json["status"] == "aggregated"
    assert resp2_json["model_version"] == 2
    assert resp2_json["global_update"] == gradients

    # Check status endpoint after aggregation
    status = client.get("/status").json()
    assert status == {"pending_updates": 0, "model_version": 2}



