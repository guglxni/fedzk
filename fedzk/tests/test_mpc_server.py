# Integration tests for the FedZK MPC Proof Server API.

import pytest
from fastapi.testclient import TestClient

import fedzk.mpc.server as mpc_server
from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver


@pytest.fixture(autouse=True)
def patch_file_existence(monkeypatch):
    # Always pretend circuits and keys exist
    monkeypatch.setattr(mpc_server.os.path, "exists", lambda path: True)
    # Set and reload allowed API keys
    monkeypatch.setenv("MPC_API_KEYS", "testkey")
    # Update server allowed keys list
    mpc_server.ALLOWED_API_KEYS = ["testkey"]

client = TestClient(mpc_server.app)

def test_generate_proof_standard(monkeypatch):
    # Stub standard proof generation
    monkeypatch.setattr(ZKProver, "generate_real_proof", lambda self, grads: ("proof_std", ["sig_std"]))
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/generate_proof",
        json={"gradients": [0.1, 0.2, 0.3], "secure": False},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json() == {"proof": "proof_std", "public_inputs": ["sig_std"]}


def test_generate_proof_secure(monkeypatch):
    # Stub secure proof generation with additional parameters
    monkeypatch.setattr(ZKProver, "generate_real_proof_secure",
                       lambda self, grads, max_norm=100, min_active=3: ("proof_sec", ["sig_sec"]))
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/generate_proof",
        json={"gradients": [1.0, 0.0], "secure": True},
        headers=headers
    )
    assert response.status_code == 200
    result = response.json()
    assert result["proof"] == "proof_sec"
    assert result["public_inputs"] == ["sig_sec"]


def test_generate_proof_validation_error():
    # Missing gradients field triggers validation error
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/generate_proof", json={"secure": False}, headers=headers
    )
    assert response.status_code == 422


def test_generate_proof_missing_files(monkeypatch):
    # Simulate missing circuit/key files
    monkeypatch.setattr(mpc_server.os.path, "exists", lambda path: False)
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/generate_proof", json={"gradients": [0.5], "secure": False}, headers=headers
    )
    assert response.status_code == 500
    assert "Circuit or key not found" in response.json().get("detail", "")


def test_verify_proof_standard(monkeypatch):
    # Stub standard verification
    monkeypatch.setattr(ZKVerifier, "verify_real_proof", lambda self, proof, inputs: True)
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof",
        json={"proof": "p", "public_inputs": ["sig"], "secure": False},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json() == {"valid": True}


def test_verify_proof_secure(monkeypatch):
    # Stub secure verification
    monkeypatch.setattr(ZKVerifier, "verify_real_proof_secure", lambda self, proof, inputs: False)
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof",
        json={"proof": "p", "public_inputs": ["sig"], "secure": True},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json() == {"valid": False}


def test_verify_proof_validation_error():
    # Missing required fields
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof", json={"public_inputs": []}, headers=headers
    )
    assert response.status_code == 422


def test_verify_proof_missing_key(monkeypatch):
    # Simulate missing verification key file
    monkeypatch.setattr(mpc_server.os.path, "exists", lambda path: False)
    headers = {"x-api-key": "testkey"}
    response = client.post(
        "/verify_proof", json={"proof": "p", "public_inputs": [], "secure": False}, headers=headers
    )
    assert response.status_code == 500
    assert "Verification key not found" in response.json().get("detail", "")

# Authentication failure tests
def test_generate_proof_unauthorized_no_key():
    # No API key header provided
    response = client.post(
        "/generate_proof", json={"gradients": [0.1], "secure": False}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}

def test_generate_proof_unauthorized_bad_key():
    # Invalid API key
    response = client.post(
        "/generate_proof", json={"gradients": [0.1], "secure": False},
        headers={"x-api-key": "bad"}
    )
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json().get("detail", "")

def test_verify_proof_unauthorized_no_key():
    # No API key header provided
    response = client.post(
        "/verify_proof", json={"proof": "p", "public_inputs": ["sig"], "secure": False}
    )
    assert response.status_code == 401

def test_verify_proof_unauthorized_bad_key():
    # Invalid API key
    response = client.post(
        "/verify_proof", json={"proof": "p", "public_inputs": ["sig"], "secure": False},
        headers={"x-api-key": "bad"}
    )
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json().get("detail", "")

def test_generate_proof_batch(monkeypatch):
    # Stub proof generation for batch
    def dummy_gen(self, gradient_dict):
        # Account for the float to int conversion (10x scaling)
        total = sum(gradient_dict["gradients"])
        # Divide by 10 to simulate the reverse conversion
        return (f"proof_{int(total//10)}", ["sig"])
    monkeypatch.setattr(ZKProver, "generate_real_proof", dummy_gen)
    headers = {"x-api-key": "testkey"}
    # Three gradient batches for three clients
    batch_payload = {
        "batch": True,
        "gradient_batches": [
            [1.0, 2.0, 3.0],  # sum 6
            [4.0, 5.0],       # sum 9
            [0.5, 0.5, 0.5]   # sum 1.5, int 1
        ]
    }
    response = client.post(
        "/generate_proof",
        json=batch_payload,
        headers=headers
    )
    assert response.status_code == 200
    result = response.json()
    # Expect three proofs
    assert "batch_proofs" in result
    proofs = result["batch_proofs"]
    assert isinstance(proofs, list) and len(proofs) == 3
    # Validate each proof based on summed gradients
    assert proofs[0] == {"proof": "proof_6", "public_inputs": ["sig"]}
    assert proofs[1] == {"proof": "proof_9", "public_inputs": ["sig"]}
    # sum 1.5 truncated to int 1
    assert proofs[2] == {"proof": "proof_1", "public_inputs": ["sig"]}



