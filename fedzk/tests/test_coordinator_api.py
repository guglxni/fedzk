"""
Integration tests for the FedZK Coordinator API.
"""

import pytest
from fastapi.testclient import TestClient

import fedzk.coordinator.logic as logic
from fedzk.coordinator.api import app


@pytest.fixture(autouse=True)
def reset_state():
    # Clear all pending updates and reset version before each test
    logic.pending_updates.clear()
    logic.current_version = 1


def test_submit_invalid_proof(monkeypatch):
    # Mock proof verification to fail
    monkeypatch.setattr(logic.verifier, "verify_real_proof", lambda proof, public_inputs: False)
    client = TestClient(app)
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


def test_submit_and_aggregate(monkeypatch):
    # Mock proof verification to succeed
    monkeypatch.setattr(logic.verifier, "verify_real_proof", lambda proof, public_inputs: True)
    client = TestClient(app)
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



