"""
Tests for the ZKProver class.
"""

from typing import Dict

import torch

from fedzk.prover.zkgenerator import ZKProver


def create_mock_gradients() -> Dict[str, torch.Tensor]:
    """
    Create mock gradient dictionary similar to what LocalTrainer would return.
    
    Returns:
        Dictionary mapping parameter names to mock gradient tensors
    """
    # Create mock gradients for a simple linear model
    mock_gradients = {
        "fc1.weight": torch.randn(10, 5),  # 10 outputs, 5 inputs
        "fc1.bias": torch.randn(10),
        "fc2.weight": torch.randn(3, 10),  # 3 outputs, 10 inputs
        "fc2.bias": torch.randn(3)
    }

    return mock_gradients


def test_zkprover_init():
    """Test that ZKProver initializes correctly with circuit and proving key paths."""
    # Mock paths
    circuit_path = "/path/to/circuit.json"
    proving_key_path = "/path/to/proving_key.json"

    # Initialize prover
    prover = ZKProver(circuit_path, proving_key_path)

    # Check initialization
    assert prover.circuit_path == circuit_path
    assert prover.proving_key_path == proving_key_path


def test_zkprover_generate_proof():
    """Test that ZKProver.generate_proof returns expected structure."""
    # Initialize prover with dummy paths
    prover = ZKProver("dummy_circuit.json", "dummy_proving_key.json")

    # Create mock gradients
    mock_gradients = create_mock_gradients()

    # Generate proof
    proof, public_signals = prover.generate_proof(mock_gradients)

    # Assertions
    assert isinstance(proof, str), "Proof should be a string"
    assert isinstance(public_signals, list), "Public signals should be a list"
    assert len(public_signals) == len(mock_gradients), "Number of public signals should match number of parameters"

    # Check that proof starts with expected prefix
    assert proof.startswith("dummy_proof_"), "Proof should have the expected prefix"

    # Check public signals content
    for signal in public_signals:
        assert "param_name" in signal, "Each public signal should contain parameter name"
        assert "norm" in signal, "Each public signal should contain norm"
        assert "hash_prefix" in signal, "Each public signal should contain hash prefix"

        # Check that the norm is positive
        assert signal["norm"] > 0, f"Norm for {signal['param_name']} should be > 0"

        # Ensure parameter name exists in original gradient dict
        assert signal["param_name"] in mock_gradients, f"Parameter {signal['param_name']} not found in gradient dict"


def test_zkprover_hash_consistency():
    """Test that ZKProver produces consistent hashes for identical inputs."""
    # Initialize prover
    prover = ZKProver("dummy_circuit.json", "dummy_proving_key.json")

    # Create two identical gradient dictionaries
    tensor = torch.tensor([1.0, 2.0, 3.0])
    grad_dict1 = {"param": tensor}
    grad_dict2 = {"param": tensor.clone()}

    # Generate proofs for both
    proof1, _ = prover.generate_proof(grad_dict1)
    proof2, _ = prover.generate_proof(grad_dict2)

    # Hashes should be identical for identical inputs
    assert proof1 == proof2, "Proof generation should be deterministic for identical inputs"


def test_zkprover_different_inputs():
    """Test that ZKProver produces different proofs for different inputs."""
    # Initialize prover
    prover = ZKProver("dummy_circuit.json", "dummy_proving_key.json")

    # Create two different gradient dictionaries
    grad_dict1 = {"param": torch.tensor([1.0, 2.0, 3.0])}
    grad_dict2 = {"param": torch.tensor([1.0, 2.0, 3.1])}  # Slight change

    # Generate proofs for both
    proof1, _ = prover.generate_proof(grad_dict1)
    proof2, _ = prover.generate_proof(grad_dict2)

    # Proofs should be different for different inputs
    assert proof1 != proof2, "Proofs should differ for different inputs"
