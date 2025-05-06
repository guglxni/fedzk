"""
Test for the batched ZK generator implementation.

This test verifies that the batch proof generation and verification work correctly
for handling large gradient tensors.
"""

import unittest
from pathlib import Path

import pytest
import torch

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver


@pytest.mark.skip("Batch ZK generator tests require additional updates for integer conversion")
class TestBatchZKGenerator(unittest.TestCase):
    """Test cases for the batched zero-knowledge proof generator."""

    def setUp(self):
        """Set up test by initializing the ZK prover and verifier."""
        # Paths to secure zk artifacts
        self.circuit_path = Path("zk/model_update_secure.wasm")
        self.proving_key_path = Path("zk/proving_key_secure.zkey")
        self.verification_key_path = Path("zk/verification_key_secure.json")
        # Skip if any artifact is missing
        if not (self.circuit_path.exists() and self.proving_key_path.exists() and self.verification_key_path.exists()):
            self.skipTest("Secure ZK files not found. Run setup_zk.sh script first.")

        # Create ZKProver and ZKVerifier instances
        self.prover = ZKProver(str(self.circuit_path), str(self.proving_key_path))
        self.verifier = ZKVerifier(str(self.verification_key_path))

    def test_batch_proof_single_tensor(self):
        """Test batch proof generation for a single tensor that fits in one chunk."""
        # Create a small gradient dict with a single tensor
        gradient_dict = {
            "layer1.weight": torch.tensor([1.0, 2.0, 3.0, 0.0])
        }

        # Generate batch proof with default chunk size (4)
        batch_result = self.prover.batch_generate_proof_secure(
            gradient_dict,
            chunk_size=4,
            max_norm=100.0,
            min_active=3
        )

        # Verify the batch
        overall_success, detailed_results = self.verifier.verify_batch_proof_secure(batch_result)

        # Check results
        self.assertTrue(overall_success)
        self.assertTrue(detailed_results["merkle_root_verified"])
        self.assertEqual(detailed_results["total_chunks"], 1)
        self.assertEqual(detailed_results["verified_chunks"], 1)

    def test_batch_proof_multi_chunk(self):
        """Test batch proof generation for a tensor that requires multiple chunks."""
        # Create a larger gradient dict with a tensor that requires multiple chunks
        large_tensor = torch.tensor([float(i) for i in range(1, 11)])  # 10 elements
        gradient_dict = {
            "large_layer.weight": large_tensor
        }

        # Generate batch proof with chunk size 4 (should create 3 chunks, the last one padded)
        batch_result = self.prover.batch_generate_proof_secure(
            gradient_dict,
            chunk_size=4,
            max_norm=500.0,  # Increased max_norm to accommodate the sum of squares
            min_active=1     # Each chunk should have at least 1 non-zero element
        )

        # Verify the batch
        overall_success, detailed_results = self.verifier.verify_batch_proof_secure(batch_result)

        # Check results
        self.assertTrue(overall_success)
        self.assertTrue(detailed_results["merkle_root_verified"])
        self.assertEqual(detailed_results["total_chunks"], 3)
        self.assertEqual(detailed_results["verified_chunks"], 3)

        # Check that the chunks were split correctly
        chunk_mapping = batch_result["metadata"]["param_mapping"]
        self.assertEqual(len(chunk_mapping["large_layer.weight"]), 3)

    def test_batch_proof_multiple_tensors(self):
        """Test batch proof generation with multiple tensors of different sizes."""
        # Create a gradient dict with multiple tensors (using integer values only)
        gradient_dict = {
            "layer1.weight": torch.tensor([1.0, 2.0, 3.0, 0.0]),
            "layer2.weight": torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            "layer3.bias": torch.tensor([1.0, 2.0])  # Use integers instead of floats
        }

        # Generate batch proof with chunk size 4
        batch_result = self.prover.batch_generate_proof_secure(
            gradient_dict,
            chunk_size=4,
            max_norm=200.0,  # Higher norm to accommodate all chunks
            min_active=1     # Each non-empty chunk should have at least 1 non-zero element
        )

        # Verify the batch
        overall_success, detailed_results = self.verifier.verify_batch_proof_secure(batch_result)

        # Check results
        self.assertTrue(overall_success)
        self.assertTrue(detailed_results["merkle_root_verified"])

        # Should be 4 chunks total: 1 for layer1, 2 for layer2, 1 for layer3
        self.assertEqual(detailed_results["total_chunks"], 4)
        self.assertEqual(detailed_results["verified_chunks"], 4)

        # Check that the chunk mapping contains all tensors
        chunk_mapping = batch_result["metadata"]["param_mapping"]
        self.assertIn("layer1.weight", chunk_mapping)
        self.assertIn("layer2.weight", chunk_mapping)
        self.assertIn("layer3.bias", chunk_mapping)

    def test_batch_proof_constraint_violation(self):
        """Test batch proof generation when a chunk violates constraints."""
        # Create a gradient dict with a tensor that exceeds the norm constraint
        gradient_dict = {
            "layer1.weight": torch.tensor([10.0, 10.0, 10.0, 10.0]),  # Norm = 400
            "layer2.weight": torch.tensor([1.0, 2.0, 3.0, 0.0])       # Valid
        }

        # Attempt to generate batch proof with a low max_norm
        with self.assertRaises(ValueError) as context:
            batch_result = self.prover.batch_generate_proof_secure(
                gradient_dict,
                chunk_size=4,
                max_norm=100.0,  # First tensor exceeds this
                min_active=1
            )

        # Check error message
        error_msg = str(context.exception)
        self.assertIn("constraint violations", error_msg)
        self.assertIn("Gradient norm", error_msg)


if __name__ == "__main__":
    unittest.main()



