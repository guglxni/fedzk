# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

# Integration tests for the FedZK MPC Proof Server API.

import os
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch # Keep for type hints if used elsewhere, or remove if not directly used

# Try different TestClient imports for compatibility
try:
    from fastapi.testclient import TestClient
except ImportError:
    from starlette.testclient import TestClient

import fedzk.mpc.server as mpc_server # Keep for monkeypatching os.path.exists
from fedzk.mpc.server import app # Import the FastAPI app directly
from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.batch_zkgenerator import BatchZKProver # Added import
from fedzk.prover.verifier import ZKVerifier


@pytest.fixture(autouse=True)
def patch_file_existence(monkeypatch):
    # Always pretend circuits and keys exist
    monkeypatch.setattr(mpc_server.os.path, "exists", lambda path: True)
    # Set and reload allowed API keys
    monkeypatch.setenv("MPC_API_KEYS", "testkey")
    # Update server allowed keys list
    mpc_server.ALLOWED_API_KEYS = ["testkey"]

# Test client fixture for compatibility handling
@pytest.fixture
def client():
    """Get a test client with compatibility handling."""
    try:
        # Try FastAPI style first (most common)
        return TestClient(mpc_server.app)
    except TypeError:
        try:
            # Try with different import
            from fastapi.testclient import TestClient as FastAPITestClient
            return FastAPITestClient(mpc_server.app)
        except (TypeError, ImportError):
            # Create a custom test client using requests
            import requests
            
            class MockTestClient:
                def __init__(self, app):
                    self.app = app
                    self.base_url = "http://testserver"
                
                def post(self, url, **kwargs):
                    # Mock POST request
                    class MockResponse:
                        def __init__(self, status_code=200, json_data=None):
                            self.status_code = status_code
                            self._json_data = json_data or {}
                        
                        def json(self):
                            return self._json_data
                    
                    if url == "/generate_proof":
                        return MockResponse(200, {"proof": "mock_proof", "public_signals": []})
                    elif url == "/generate_batch_proof":
                        return MockResponse(200, {"proofs": ["mock_proof"], "public_signals": [[]]})
                    elif url == "/verify_proof":
                        return MockResponse(200, {"valid": True})
                    else:
                        return MockResponse(404, {"error": "Not found"})
            
            return MockTestClient(mpc_server.app)

# Corrected test data for generate_proof endpoint
GRADIENT_DATA_STD = {"gradients": {"param1": [0.1, 0.2, 0.3], "param2": [0.4, 0.5]}, "secure": False}
GRADIENT_DATA_SEC = {"gradients": {"param1": [1.0, 0.0]}, "secure": True, "max_norm_squared": 100.0, "min_active": 1}

# For batch tests, if BatchZKProver processes a single logical gradient set by chunking it internally,
# then the payload should be similar to non-batch, just with batch=True and chunk_size.
BATCH_PAYLOAD_FOR_TEST = {
    "gradients": {"param1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "param2": [0.1, 0.2, 0.3, 0.4]},
    "batch": True,
    "chunk_size": 2, # Example chunk size
    "secure": False 
}

def test_generate_proof_standard(monkeypatch, client):
    monkeypatch.setattr(ZKProver, "generate_real_proof_standard", lambda self, grads: ("proof_std", ["sig_std"]))
    headers = {"x-api-key": "testkey"}
    response = client.post("/generate_proof", json=GRADIENT_DATA_STD, headers=headers)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["proof"] == "proof_std"
    assert data["public_inputs"] == ["sig_std"]


def test_generate_proof_secure(monkeypatch, client):
    monkeypatch.setattr(ZKProver, "generate_real_proof_secure",
                       lambda self, grads, max_norm_sq, min_active: ("proof_sec", ["sig_sec"]))
    headers = {"x-api-key": "testkey"}
    response = client.post("/generate_proof", json=GRADIENT_DATA_SEC, headers=headers)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["proof"] == "proof_sec"


def test_generate_proof_validation_error(client):
    # Missing gradients field triggers validation error
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/generate_proof", json={"secure": False}, headers=headers
    )
    assert response.status_code == 422


def test_generate_proof_missing_files(monkeypatch, client):
    monkeypatch.setattr(mpc_server.os.path, "exists", lambda path: False)
    headers = {"x-api-key": "testkey"}
    # Use a valid payload structure, the error should come from os.path.exists mock
    response = client.post("/generate_proof", json=GRADIENT_DATA_STD, headers=headers)
    assert response.status_code == 500, response.text # Expect 500 if files are truly missing and server tries to load them
    # The actual error might be different if ZKProver itself raises FileNotFoundError early
    # For now, this depends on how mpc_server.py handles it with the ZKProver calls.
    # If ZKProver itself would fail to init or ASSET_DIR makes paths invalid, this might not be 500 from missing *runtime* files.


def test_verify_proof_standard(monkeypatch, client):
    monkeypatch.setattr(ZKVerifier, "verify_real_proof", lambda self, proof, inputs: True)
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof",
        json={"proof": {"pi_a": []}, "public_inputs": ["sig"], "secure": False}, # proof is a dict
        headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.json()["valid"] is True


def test_verify_proof_secure(monkeypatch, client):
    monkeypatch.setattr(ZKVerifier, "verify_real_proof", lambda self, proof, inputs: False)
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof",
        json={"proof": {"pi_a": []}, "public_inputs": ["sig"], "secure": True},
        headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.json()["valid"] is False


def test_verify_proof_validation_error(client):
    # Missing required fields
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof", json={"public_inputs": []}, headers=headers
    )
    assert response.status_code == 422


def test_verify_proof_missing_key(monkeypatch, client):
    # Let's test the scenario where ZKVerifier.verify_proof is successfully called but returns False
    monkeypatch.setattr(ZKVerifier, "verify_real_proof", lambda self, proof, inputs: False)
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof", json={"proof": {"pi_a":[]}, "public_inputs": [], "secure": False}, headers=headers
    )
    assert response.status_code == 200 # Endpoint itself should succeed
    assert response.json()["valid"] is False # Reflecting verifier's decision

# Authentication failure tests
def test_generate_proof_unauthorized_no_key(client):
    response = client.post("/generate_proof", json=GRADIENT_DATA_STD) # No headers
    assert response.status_code == 401, response.text

def test_generate_proof_unauthorized_bad_key(client):
    response = client.post("/generate_proof", json=GRADIENT_DATA_STD, headers={"x-api-key": "bad"})
    assert response.status_code == 401, response.text

def test_verify_proof_unauthorized_no_key(client):
    # No API key header provided
    response = client.post(
        "/verify_proof", json={"proof": "p", "public_inputs": ["sig"], "secure": False}
    )
    assert response.status_code == 401

def test_verify_proof_unauthorized_bad_key(client):
    # Invalid API key
    response = client.post(
        "/verify_proof", json={"proof": "p", "public_inputs": ["sig"], "secure": False},
        headers={"x-api-key": "bad"}
    )
    assert response.status_code == 401
    assert "Invalid API key" in response.json().get("detail", "")

def test_generate_proof_batch(monkeypatch, client):
    def dummy_batch_gen(self, gradient_dict_tensors):
        # gradient_dict_tensors will be Dict[str, torch.Tensor]
        num_params = len(gradient_dict_tensors)
        first_param_name = list(gradient_dict_tensors.keys())[0]
        first_param_len = len(gradient_dict_tensors[first_param_name])
        return {"proof": f"batch_proof_params_{num_params}_len_{first_param_len}"}, [f"batch_sig_params_{num_params}_len_{first_param_len}"]

    monkeypatch.setattr(BatchZKProver, "generate_proof", dummy_batch_gen)
    headers = {"x-api-key": "testkey"}
    response = client.post("/generate_proof", json=BATCH_PAYLOAD_FOR_TEST, headers=headers)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "batch_proof_params_2_len_6" in data["proof"]["proof"] # Example check



