# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Tests for the ZKProver class.
"""

import subprocess
from typing import Dict

import pytest
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
    """Test that ZKProver initializes correctly with default parameters."""
    # Initialize prover with default parameters
    prover = ZKProver(secure=False)

    # Check initialization
    assert prover.secure == False
    assert hasattr(prover, 'wasm_path')
    assert hasattr(prover, 'zkey_path')
    assert prover.max_norm_squared == 100.0
    assert prover.min_active == 1


def test_zkprover_generate_proof():
    """Test that ZKProver.generate_proof returns expected structure."""
    # Initialize prover with secure=False for testing
    prover = ZKProver(secure=False)

    # Create mock gradients
    mock_gradients = create_mock_gradients()

    # For these tests, we expect SNARKjs to fail since it's not installed in CI
    # We'll catch the error and verify the proper exception is raised
    try:
        proof, public_signals = prover.generate_proof(mock_gradients)
        # If it somehow succeeds, verify structure
        assert proof is not None
        assert public_signals is not None
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Expected in CI environment without snarkjs
        pytest.skip(f"SNARKjs not available in test environment: {e}")


def test_zkprover_hash_consistency():
    """Test that ZKProver produces consistent hashes for identical inputs."""
    # Initialize prover
    prover = ZKProver(secure=False)

    # Create two identical gradient dictionaries
    tensor = torch.tensor([1.0, 2.0, 3.0])
    grad_dict1 = {"param": tensor}
    grad_dict2 = {"param": tensor.clone()}

    # Generate proofs for both - catch SNARKjs errors
    try:
        proof1, _ = prover.generate_proof(grad_dict1)
        proof2, _ = prover.generate_proof(grad_dict2)
        # Hashes should be identical for identical inputs
        assert proof1 == proof2, "Proof generation should be deterministic for identical inputs"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"SNARKjs not available in test environment: {e}")


def test_zkprover_different_inputs():
    """Test that ZKProver produces different proofs for different inputs."""
    # Initialize prover
    prover = ZKProver(secure=False)

    # Create two different gradient dictionaries
    grad_dict1 = {"param": torch.tensor([1.0, 2.0, 3.0])}
    grad_dict2 = {"param": torch.tensor([1.0, 2.0, 3.1])}  # Slight change

    # Generate proofs for both - catch SNARKjs errors
    try:
        proof1, _ = prover.generate_proof(grad_dict1)
        proof2, _ = prover.generate_proof(grad_dict2)
        # Proofs should be different for different inputs
        assert proof1 != proof2, "Proof generation should produce different results for different inputs"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"SNARKjs not available in test environment: {e}")
    assert proof1 != proof2, "Proofs should differ for different inputs"
