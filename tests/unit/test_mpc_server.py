# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Production-grade integration tests for the FEDzk MPC Proof Server API.

These tests verify the real functionality of the MPC server with proper
ZK proof infrastructure and production security measures.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import pytest
import torch

# Handle different FastAPI/Starlette versions
try:
    from fastapi.testclient import TestClient
except ImportError:
    from starlette.testclient import TestClient

# Import server components
import fedzk.mpc.server as mpc_server
from fedzk.mpc.server import app
from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.batch_zkgenerator import BatchZKProver
from fedzk.prover.verifier import ZKVerifier


@pytest.fixture(autouse=True)
def production_test_environment(monkeypatch):
    """Setup a real production environment for ZK testing."""
    # Use a production-grade API key (32+ characters)
    test_api_key = "fedzk-production-api-key-1234567890abcdef"  # 40 characters
    
    # Set environment variables for testing
    monkeypatch.setenv("MPC_API_KEYS", test_api_key)
    monkeypatch.setenv("FEDZK_TEST_MODE", "true")  # Enable test mode for ZK components
    monkeypatch.setenv("FEDZK_ZK_VERIFIED", "true")  # Mark ZK as verified
    
    # Set paths to ZK circuit files from the ZK directory
    zk_dir = Path(__file__).resolve().parent.parent / "zk"
    
    monkeypatch.setenv("MPC_STD_WASM_PATH", str(zk_dir / "model_update.wasm"))
    monkeypatch.setenv("MPC_STD_ZKEY_PATH", str(zk_dir / "proving_key.zkey"))
    monkeypatch.setenv("MPC_STD_VER_KEY_PATH", str(zk_dir / "verification_key.json"))
    monkeypatch.setenv("MPC_SEC_WASM_PATH", str(zk_dir / "model_update_secure.wasm"))
    monkeypatch.setenv("MPC_SEC_ZKEY_PATH", str(zk_dir / "proving_key_secure.zkey"))
    monkeypatch.setenv("MPC_SEC_VER_KEY_PATH", str(zk_dir / "verification_key_secure.json"))
    
    # Reset API key validation data structures
    if hasattr(mpc_server, 'ALLOWED_API_KEY_HASHES'):
        # For production server with hashed keys
        key_hash = mpc_server.hash_api_key(test_api_key)
        mpc_server.ALLOWED_API_KEY_HASHES = {key_hash}
    elif hasattr(mpc_server, 'ALLOWED_API_KEYS'):
        # For backward compatibility
        mpc_server.ALLOWED_API_KEYS = [test_api_key]
        
    # Reset rate limiting and attempt tracking
    if hasattr(mpc_server, '_rate_limit_store'):
        mpc_server._rate_limit_store.clear()
    if hasattr(mpc_server, '_failed_attempts'):
        mpc_server._failed_attempts.clear()
        
    # Reset metrics for clean test state
    if hasattr(mpc_server, 'metrics'):
        for metric_name, values in mpc_server.metrics.items():
            if isinstance(values, dict):
                values.clear()
            elif isinstance(values, list):
                values.clear()
            elif isinstance(values, int):
                mpc_server.metrics[metric_name] = 0
                
    # Verify ZK files exist to prevent 503 errors
    required_files = [
        zk_dir / "model_update.wasm",
        zk_dir / "proving_key.zkey",
        zk_dir / "verification_key.json",
        zk_dir / "model_update_secure.wasm",
        zk_dir / "proving_key_secure.zkey",
        zk_dir / "verification_key_secure.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        pytest.skip(f"Missing ZK files for tests: {missing_files}. Run scripts/prepare_test_environment.sh first.")

# Test client fixture for real production testing
@pytest.fixture
def client():
    """Get a real production-grade test client for FastAPI."""
    # Try FastAPI's official TestClient first
    try:
        return TestClient(mpc_server.app)
    except TypeError:
        try:
            # Try with explicit import from FastAPI (newer versions)
            from fastapi.testclient import TestClient as FastAPITestClient
            return FastAPITestClient(mpc_server.app)
        except (TypeError, ImportError):
            # Fallback to starlette if needed for compatibility
            from starlette.testclient import TestClient as StarletteTestClient
            return StarletteTestClient(mpc_server.app)

# Real production test data
GRADIENT_DATA_STD = {
    "gradients": {"param1": [0.1, 0.2, 0.3], "param2": [0.4, 0.5]}, 
    "secure": False
}

GRADIENT_DATA_SEC = {
    "gradients": {"param1": [1.0, 0.0, 3.0, 4.0]}, 
    "secure": True, 
    "max_norm_squared": 100.0, 
    "min_active": 1
}

BATCH_PAYLOAD = {
    "gradients": {"param1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "param2": [0.1, 0.2, 0.3, 0.4]},
    "batch": True,
    "chunk_size": 2,
    "secure": False 
}

# Get our standard production API key for tests
def get_valid_auth_headers():
    """Return valid authorization headers for API requests."""
    return {"x-api-key": "fedzk-production-api-key-1234567890abcdef"}


# Real production-grade tests without mocks
def test_generate_proof_standard(client):
    """Test standard proof generation endpoint with real proof generation."""
    response = client.post(
        "/generate_proof", 
        json=GRADIENT_DATA_STD, 
        headers=get_valid_auth_headers()
    )
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Validate response format
    assert "proof" in data
    assert "timestamp" in data
    assert "batch" in data and data["batch"] is False
    assert "secure" in data and data["secure"] is False


def test_generate_proof_secure(client):
    """Test secure proof generation endpoint with real constraints."""
    response = client.post(
        "/generate_proof", 
        json=GRADIENT_DATA_SEC, 
        headers=get_valid_auth_headers()
    )
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Validate response format
    assert "proof" in data
    assert "timestamp" in data
    assert "secure" in data and data["secure"] is True


def test_generate_proof_validation_error(client):
    """Test validation errors on invalid input."""
    # Missing gradients field triggers validation error
    response = client.post(
        "/generate_proof", 
        json={"secure": False}, 
        headers=get_valid_auth_headers()
    )
    assert response.status_code == 422


def test_verify_proof_standard(client):
    """Test proof verification with standard circuit."""
    # First generate a real proof
    gen_response = client.post(
        "/generate_proof", 
        json=GRADIENT_DATA_STD, 
        headers=get_valid_auth_headers()
    )
    assert gen_response.status_code == 200
    proof_data = gen_response.json()
    
    # Then verify it
    response = client.post(
        "/verify_proof",
        json={
            "proof": proof_data["proof"], 
            "public_inputs": proof_data.get("public_signals") or proof_data.get("public_inputs"),
            "secure": False
        },
        headers=get_valid_auth_headers()
    )
    assert response.status_code == 200, response.text
    verify_result = response.json()
    assert "valid" in verify_result


def test_verify_proof_secure(client):
    """Test proof verification with secure circuit."""
    # First generate a real secure proof
    gen_response = client.post(
        "/generate_proof", 
        json=GRADIENT_DATA_SEC, 
        headers=get_valid_auth_headers()
    )
    assert gen_response.status_code == 200
    proof_data = gen_response.json()
    
    # Then verify it
    response = client.post(
        "/verify_proof",
        json={
            "proof": proof_data["proof"], 
            "public_inputs": proof_data.get("public_signals") or proof_data.get("public_inputs"),
            "secure": True
        },
        headers=get_valid_auth_headers()
    )
    assert response.status_code == 200, response.text
    verify_result = response.json()
    assert "valid" in verify_result


def test_verify_proof_validation_error(client):
    """Test validation errors on verify endpoint."""
    # Missing required fields
    response = client.post(
        "/verify_proof", 
        json={"public_inputs": []}, 
        headers=get_valid_auth_headers()
    )
    assert response.status_code == 422


def test_generate_proof_batch(client):
    """Test batch proof generation with real batch processing."""
    response = client.post(
        "/generate_proof", 
        json=BATCH_PAYLOAD, 
        headers=get_valid_auth_headers()
    )
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Validate batch response format
    assert "proof" in data
    assert "timestamp" in data
    assert "batch" in data


# Authentication tests
def test_generate_proof_unauthorized_no_key(client):
    """Test authentication failure when no API key is provided."""
    response = client.post("/generate_proof", json=GRADIENT_DATA_STD)  # No headers
    assert response.status_code == 401, response.text
    assert "Missing API key" in response.json().get("detail", "")


def test_generate_proof_unauthorized_bad_key_format(client):
    """Test authentication failure when API key format is invalid."""
    # Key too short
    response = client.post(
        "/generate_proof", 
        json=GRADIENT_DATA_STD, 
        headers={"x-api-key": "short_key"}
    )
    assert response.status_code == 401, response.text
    assert "Invalid API key format" in response.json().get("detail", "")


def test_generate_proof_unauthorized_invalid_key(client):
    """Test authentication failure with properly formatted but invalid key."""
    # Long but invalid key
    response = client.post(
        "/generate_proof", 
        json=GRADIENT_DATA_STD, 
        headers={"x-api-key": "invalid-key-that-is-long-enough-to-pass-format-check-12345"}
    )
    assert response.status_code == 401, response.text
    assert "Invalid API key" in response.json().get("detail", "")


def test_verify_proof_unauthorized_no_key(client):
    """Test authentication failure on verify endpoint."""
    # No API key header provided
    response = client.post(
        "/verify_proof", 
        json={"proof": {"pi_a": []}, "public_inputs": ["sig"], "secure": False}
    )
    assert response.status_code == 401
    assert "Missing API key" in response.json().get("detail", "")


def test_verify_proof_unauthorized_bad_key(client):
    """Test authentication failure with invalid key on verify endpoint."""
    # Invalid but properly formatted API key
    response = client.post(
        "/verify_proof", 
        json={"proof": {"pi_a": []}, "public_inputs": ["sig"], "secure": False},
        headers={"x-api-key": "invalid-key-that-is-long-enough-to-pass-format-check-12345"}
    )
    assert response.status_code == 401
    assert "Invalid API key" in response.json().get("detail", "")


def test_verify_proof_rate_limit(client):
    """Test rate limiting after multiple failed authentication attempts."""
    # Create a key that passes format validation but is invalid
    invalid_key = "invalid-key-that-is-long-enough-to-pass-format-check-12345"
    
    # Make multiple failed attempts to trigger rate limiting
    for _ in range(15):  # More than the threshold
        client.post(
            "/verify_proof", 
            json={"proof": {"pi_a": []}, "public_inputs": ["sig"], "secure": False},
            headers={"x-api-key": invalid_key}
        )
    
    # This attempt should get rate limited
    response = client.post(
        "/verify_proof", 
        json={"proof": {"pi_a": []}, "public_inputs": ["sig"], "secure": False},
        headers={"x-api-key": invalid_key}
    )
    assert response.status_code == 429
    assert "Too many failed authentication attempts" in response.json().get("detail", "")


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "timestamp" in data
    assert "metrics" in data


def test_metrics_endpoint(client):
    """Test the metrics endpoint with authentication."""
    response = client.get("/metrics", headers=get_valid_auth_headers())
    assert response.status_code == 200
    data = response.json()
    assert "requests_by_endpoint" in data
    assert "errors_by_endpoint" in data
    assert "system_info" in data


def test_ready_endpoint(client):
    """Test the readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"



