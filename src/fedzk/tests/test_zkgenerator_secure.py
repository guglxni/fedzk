"""
Test for the secure ZK generator implementation.

This test verifies that the zero-knowledge proof generation works correctly
with the secure circuit with fairness constraints.
"""

import unittest
from pathlib import Path

import torch

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver


class TestSecureZKGenerator(unittest.TestCase):
    """Test cases for the secure zero-knowledge proof generator."""

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

    def test_generate_and_verify_secure_proof_valid(self):
        """Test generating and verifying a valid secure proof."""
        # Create a sample gradient dictionary with valid inputs
        # This should satisfy the constraints (norm <= 10, at least 3 non-zero elements)
        gradient_dict = {
            "layer1.weight": torch.tensor([1.0, 2.0, 3.0, 0.0])
        }

        # Generate secure proof
        proof, public_inputs = self.prover.generate_real_proof_secure(
            gradient_dict,
            max_inputs=4,
            max_norm=100.0,  # Sum of squares must be <= 100
            min_active=3
        )

        # Verify the proof
        result = self.verifier.verify_real_proof_secure(proof, public_inputs["public_inputs"])

        # Assert result is True
        self.assertTrue(result)

        # Check metadata
        self.assertEqual(public_inputs["metadata"]["nonzero_count"], 3)
        self.assertLessEqual(public_inputs["metadata"]["norm"], 100.0)

    def test_generate_secure_proof_invalid_norm(self):
        """Test that generating a proof with norm > max_norm raises an error."""
        # Create a sample gradient dictionary exceeding the norm constraint
        gradient_dict = {
            "layer1.weight": torch.tensor([10.0, 10.0, 10.0, 10.0])  # Norm = 10^2 * 4 = 400
        }

        # Attempt to generate secure proof, should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.prover.generate_real_proof_secure(
                gradient_dict,
                max_inputs=4,
                max_norm=100.0,  # Sum of squares must be <= 100
                min_active=3
            )

        # Check error message contains the expected text
        self.assertIn("Gradient norm", str(context.exception))
        self.assertIn("exceeds maximum", str(context.exception))

    def test_generate_secure_proof_invalid_active(self):
        """Test that generating a proof with too few non-zero elements raises an error."""
        # Create a sample gradient dictionary with not enough non-zero elements
        gradient_dict = {
            "layer1.weight": torch.tensor([1.0, 2.0, 0.0, 0.0])  # Only 2 non-zero values
        }

        # Attempt to generate secure proof, should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.prover.generate_real_proof_secure(
                gradient_dict,
                max_inputs=4,
                max_norm=10.0,
                min_active=3
            )

        # Check error message contains the expected text
        self.assertIn("non-zero gradients", str(context.exception))
        self.assertIn("minimum required", str(context.exception))


if __name__ == "__main__":
    unittest.main()
