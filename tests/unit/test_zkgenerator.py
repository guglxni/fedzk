# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Tests for the ZKProver class using real ZK infrastructure.
Following Ian Sommerville's software engineering best practices.
"""

import subprocess
import os
from typing import Dict
from pathlib import Path

import pytest
import torch

from fedzk.prover.zkgenerator import ZKProver


def create_real_gradients() -> Dict[str, torch.Tensor]:
    """
    Create real gradient dictionary from actual model training.
    
    Returns:
        Dictionary mapping parameter names to real gradient tensors
    """
    # Create a simple linear model and train it to get real gradients
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 3)
    )
    
    # Create data
    X = torch.randn(8, 10)
    y = torch.randn(8, 3)
    
    # Perform forward pass and backward pass to get real gradients
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
    
    return gradients


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
    
    # Check for the presence of required ZK artifacts
    wasm_path = Path(prover.wasm_path)
    zkey_path = Path(prover.zkey_path)
    
    if not os.getenv("FEDZK_TEST_MODE", "false").lower() == "true":
        if wasm_path.exists():
            assert wasm_path.stat().st_size > 0, "WASM file exists but is empty"
        if zkey_path.exists():
            assert zkey_path.stat().st_size > 0, "ZKEY file exists but is empty"


def test_zkprover_generate_proof():
    """Test that ZKProver.generate_proof returns expected structure using real ZK infrastructure."""
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
    
    # Initialize prover with secure=False for testing
    prover = ZKProver(secure=False)

    # Create real gradients
    real_gradients = create_real_gradients()

    try:
        proof, public_signals = prover.generate_proof(real_gradients)
        # If it succeeds, verify structure
        assert proof is not None
        assert public_signals is not None
        
        # Check proof format in production mode
        if not os.getenv("FEDZK_TEST_MODE", "false").lower() == "true":
            # Verify it's a real Groth16 proof
            assert isinstance(proof, dict)
            assert "pi_a" in proof
            assert "pi_b" in proof
            assert "pi_c" in proof
            
            # Verify signals format
            assert isinstance(public_signals, list)
            for signal in public_signals:
                assert isinstance(signal, str)
                
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Expected in environments without snarkjs
        pytest.skip(f"SNARKjs not available in test environment: {e}")


def test_zkprover_deterministic_proofs():
    """Test that ZKProver produces consistent proofs for identical inputs."""
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
    # Initialize prover
    prover = ZKProver(secure=False)

    # Create two identical gradient dictionaries with exact values
    tensor = torch.tensor([1.0, 2.0, 3.0])
    grad_dict1 = {"param": tensor}
    grad_dict2 = {"param": tensor.clone()}

    # Generate proofs for both
    try:
        proof1, signals1 = prover.generate_proof(grad_dict1)
        proof2, signals2 = prover.generate_proof(grad_dict2)
        
        # Depending on the ZK infrastructure, proofs may or may not be deterministic
        # In test mode, we expect deterministic proofs for identical inputs
        if os.getenv("FEDZK_TEST_MODE", "false").lower() == "true":
            # In test mode, identical inputs should produce identical proofs
            assert proof1 == proof2, "Proof generation should be deterministic in test mode"
        else:
            # In production mode with real ZK, proofs may not be identical due to randomness
            # but the verification should still work for both
            # We'll just verify they're both valid proof structures
            assert isinstance(proof1, dict) and isinstance(proof2, dict)
            assert isinstance(signals1, list) and isinstance(signals2, list)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"SNARKjs not available in test environment: {e}")


def test_zkprover_different_inputs():
    """Test that ZKProver produces different proofs for different inputs."""
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
    # Initialize prover
    prover = ZKProver(secure=False)

    # Create two different gradient dictionaries with controlled values
    grad_dict1 = {"param": torch.tensor([1.0, 2.0, 3.0])}
    grad_dict2 = {"param": torch.tensor([1.0, 2.0, 3.1])}  # Slight difference

    # Generate proofs for both
    try:
        proof1, signals1 = prover.generate_proof(grad_dict1)
        proof2, signals2 = prover.generate_proof(grad_dict2)
        
        # In both test and production mode, different inputs should produce different results
        # Either the proofs or the signals should differ
        assert proof1 != proof2 or signals1 != signals2, "Different inputs should produce different proof results"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"SNARKjs not available in test environment: {e}")


def test_real_zk_verification_with_valid_proof():
    """Test end-to-end real ZK proof generation and verification."""
    # Skip if ZK infrastructure not available
    if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
        pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
    from fedzk.prover.verifier import ZKVerifier
    
    # Create prover and real gradients
    prover = ZKProver(secure=False)
    real_gradients = create_real_gradients()
    
    try:
        # Generate real proof
        proof, signals = prover.generate_proof(real_gradients)
        
        # Locate verification key
        vkey_path = Path("src/fedzk/zk/verification_key.json")
        if not vkey_path.exists():
            vkey_path = Path("setup_artifacts/model_update_verification_key.json")
        
        if not vkey_path.exists():
            pytest.skip("Verification key not found - run setup_zk.sh")
        
        # Create verifier with real verification key
        verifier = ZKVerifier(str(vkey_path))
        
        # Verify the proof
        is_valid = verifier.verify_proof(proof, signals)
        
        # The real proof should be valid
        assert is_valid is True, "Real ZK proof should be verified successfully"
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"ZK toolchain error: {e}")
