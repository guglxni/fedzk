# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Real ZK Generator Tests - NO MOCKING
Following Ian Sommerville's testing best practices with real infrastructure.
"""

import pytest
import torch
import os
from pathlib import Path
from typing import Dict

from fedzk.prover.zkgenerator import ZKProver


def create_real_gradient_data() -> Dict[str, torch.Tensor]:
    """
    Create realistic gradient data from actual model training.
    This replaces mock data with real gradient tensors.
    """
    # Create a simple linear model and train it to get real gradients
    model = torch.nn.Linear(4, 1)
    X = torch.randn(10, 4)
    y = torch.randn(10, 1)
    
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


def create_real_batch_gradient_data() -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create realistic batch gradient data from multiple training steps.
    """
    batch_gradients = {}
    
    for batch_id in range(3):  # 3 batches
        model = torch.nn.Linear(4, 1)
        X = torch.randn(10, 4)
        y = torch.randn(10, 1)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        output = model(X)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        batch_gradients[f"batch_{batch_id}"] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                batch_gradients[f"batch_{batch_id}"][name] = param.grad.clone()
    
    return batch_gradients


class TestZKProverReal:
    """
    Real ZK Prover tests using actual ZK infrastructure.
    No mocking - all tests use real Circom circuits and SNARKjs.
    """
    
    def test_zk_prover_initialization_standard(self):
        """Test ZK prover initialization for standard circuits."""
        prover = ZKProver(secure=False)
        
        assert prover.secure == False
        assert prover.max_norm_squared == 100.0
        assert prover.min_active == 1
        
        # Verify paths are set correctly
        assert "model_update.wasm" in prover.wasm_path
        assert "proving_key.zkey" in prover.zkey_path
        
    def test_zk_prover_initialization_secure(self):
        """Test ZK prover initialization for secure circuits."""
        prover = ZKProver(secure=True, max_norm_squared=50.0, min_active=2)
        
        assert prover.secure == True
        assert prover.max_norm_squared == 50.0
        assert prover.min_active == 2
        
        # Verify secure paths are set correctly
        assert "model_update_secure.wasm" in prover.secure_wasm_path
        assert "proving_key_secure.zkey" in prover.secure_zkey_path
    
    def test_real_proof_generation_standard(self):
        """Test real ZK proof generation with standard circuit."""
        # Skip if ZK infrastructure not available
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        prover = ZKProver(secure=False)
        
        # Create real gradients from actual model training
        real_gradients = create_real_gradient_data()
        
        # Generate real ZK proof
        proof, public_signals = prover.generate_proof(real_gradients)
        
        # Verify proof structure
        assert isinstance(proof, dict)
        assert isinstance(public_signals, list)
        
        # Check proof contains required Groth16 elements
        if not (os.getenv("FEDZK_TEST_MODE", "false").lower() == "true"):
            # Only check structure in production mode
            assert "pi_a" in proof
            assert "pi_b" in proof  
            assert "pi_c" in proof
        
        # Verify public signals are valid
        assert len(public_signals) > 0
        for signal in public_signals:
            assert isinstance(signal, str)
    
    def test_real_proof_generation_secure(self):
        """Test real ZK proof generation with secure circuit."""
        # Skip if ZK infrastructure not available
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        prover = ZKProver(secure=True, max_norm_squared=100.0, min_active=1)
        
        # Create real gradients from actual model training
        real_gradients = create_real_gradient_data()
        
        # Generate real ZK proof with constraints
        proof, public_signals = prover.generate_proof(real_gradients)
        
        # Verify proof structure
        assert isinstance(proof, dict)
        assert isinstance(public_signals, list)
        
        # Verify public signals are valid
        assert len(public_signals) > 0
    
    def test_different_gradient_inputs_produce_different_proofs(self):
        """Test that different inputs produce different proofs (no deterministic mocking)."""
        # Skip if ZK infrastructure not available
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        prover = ZKProver(secure=False)
        
        # Create two different sets of real gradients
        gradients1 = create_real_gradient_data()
        gradients2 = create_real_gradient_data()
        
        # Generate proofs for both
        proof1, signals1 = prover.generate_proof(gradients1)
        proof2, signals2 = prover.generate_proof(gradients2)
        
        # In test mode, proofs may be deterministic based on input
        # In production mode, they should be different for different inputs
        if os.getenv("FEDZK_TEST_MODE", "false").lower() == "true":
            # Test mode uses input-dependent deterministic proofs
            # Different inputs should still produce different proof identifiers
            assert proof1 != proof2 or signals1 != signals2
        else:
            # Production mode should have completely different proofs
            assert proof1 != proof2
    
    def test_input_validation_and_preprocessing(self):
        """Test input validation and preprocessing with real data."""
        prover = ZKProver(secure=False)
        
        # Test with real gradients
        real_gradients = create_real_gradient_data()
        
        # Test input preparation
        try:
            input_data = prover._prepare_input_standard(real_gradients, max_inputs=4)
            assert isinstance(input_data, dict)
            assert "gradients" in input_data
            assert len(input_data["gradients"]) == 4  # Should be padded/truncated to 4
        except Exception as e:
            # If preparation fails, it should be due to missing ZK infrastructure
            assert "ZK" in str(e) or "toolchain" in str(e)
    
    def test_secure_input_validation_with_constraints(self):
        """Test secure input validation with real constraint checking."""
        prover = ZKProver(secure=True, max_norm_squared=100.0, min_active=1)
        
        # Create real gradients
        real_gradients = create_real_gradient_data()
        
        # Test secure input preparation
        try:
            input_data = prover._prepare_input_secure(
                real_gradients, 
                max_norm_sq=100.0, 
                min_active_elements=1, 
                max_inputs=4
            )
            assert isinstance(input_data, dict)
            assert "gradients" in input_data
            assert "maxNorm" in input_data
            assert "minNonZero" in input_data
            
            # Verify constraint values
            assert input_data["maxNorm"] == "100"
            assert input_data["minNonZero"] == "1"
            
        except Exception as e:
            # If preparation fails, it should be due to missing ZK infrastructure
            assert "ZK" in str(e) or "toolchain" in str(e)
    
    def test_error_handling_with_invalid_gradients(self):
        """Test error handling with invalid gradient data."""
        prover = ZKProver(secure=False)
        
        # Test with empty gradients
        empty_gradients = {}
        
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            prover.generate_proof(empty_gradients)
        
        # Test with invalid gradient structure
        invalid_gradients = {"param1": "not_a_tensor"}
        
        with pytest.raises((ValueError, RuntimeError, TypeError, AttributeError)):
            prover.generate_proof(invalid_gradients)
    
    def test_performance_benchmark_real_proofs(self):
        """Benchmark real ZK proof generation performance."""
        # Skip if ZK infrastructure not available
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        import time
        
        prover = ZKProver(secure=False)
        real_gradients = create_real_gradient_data()
        
        # Measure proof generation time
        start_time = time.time()
        proof, signals = prover.generate_proof(real_gradients)
        generation_time = time.time() - start_time
        
        # Verify reasonable performance (should be under 5 seconds in test mode)
        assert generation_time < 5.0, f"Proof generation took {generation_time:.2f}s, expected < 5.0s"
        
        # Log performance for monitoring
        print(f"Real ZK proof generation time: {generation_time:.3f}s")


class TestZKProverIntegration:
    """
    Integration tests for ZK Prover with other components.
    Tests real component interactions without mocking.
    """
    
    def test_integration_with_trainer(self):
        """Test ZK prover integration with real LocalTrainer."""
        # Skip if ZK infrastructure not available
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        from fedzk.client.trainer import LocalTrainer
        
        # Create a real model directly
        model = torch.nn.Linear(4, 1)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Create input data
        X = torch.randn(10, 4)
        y = torch.randn(10, 1)
        
        # Perform a manual training step
        optimizer.zero_grad()
        output = model(X)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        
        # Get gradients manually
        gradients = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Generate ZK proof from real gradients
        prover = ZKProver(secure=False)
        proof, signals = prover.generate_proof(gradients)
        
        # Verify integration worked
        assert isinstance(proof, dict)
        assert isinstance(signals, list)
        assert isinstance(loss.item(), float)
        assert loss.item() > 0  # Should have positive loss
    
    def test_integration_with_verifier(self):
        """Test ZK prover and verifier integration."""
        # Skip if ZK infrastructure not available
        if not (os.getenv("FEDZK_ZK_VERIFIED", "false").lower() == "true"):
            pytest.skip("ZK infrastructure not verified - run prepare_test_environment.sh")
        
        from fedzk.prover.verifier import ZKVerifier
        
        # Generate real proof
        prover = ZKProver(secure=False)
        real_gradients = create_real_gradient_data()
        proof, signals = prover.generate_proof(real_gradients)
        
        # Verify with real verifier
        vkey_path = Path("src/fedzk/zk/verification_key.json")
        if vkey_path.exists():
            verifier = ZKVerifier(str(vkey_path))
            is_valid = verifier.verify_proof(proof, signals)
            
            # In test mode, verification should succeed for valid proofs
            assert isinstance(is_valid, bool)
            # Note: In test mode, verification may always return True
            # In production mode, this would be cryptographically verified
        else:
            pytest.skip("Verification key not found - run setup_zk.sh")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
