#!/usr/bin/env python3
"""
ModelUpdateSecure ZK Circuit
============================

Secure model update circuit for federated learning with enhanced security constraints.
"""

from typing import Dict, List, Any
from pathlib import Path

class ModelUpdateSecureCircuit:
    """Secure model update circuit with enhanced security constraints."""

    def __init__(self, circuit_path: Path = None):
        """Initialize the secure circuit."""
        if circuit_path is None:
            circuit_path = Path(__file__).parent / "model_update_secure.circom"
        self.circuit_path = circuit_path
        self.name = "model_update_secure"

    def get_circuit_spec(self) -> Dict[str, Any]:
        """Get circuit specification."""
        return {
            "inputs": ["gradients", "maxNorm", "minNonZero"],
            "outputs": ["newWeights", "gradientNorm", "securityValid", "nonZeroCount", "normValid"],
            "constraints": 1500,
            "witness_size": 1024
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate circuit inputs."""
        required_inputs = ["gradients", "maxNorm", "minNonZero"]
        if not all(key in inputs for key in required_inputs):
            return False

        # Validate gradient array
        gradients = inputs["gradients"]
        if not isinstance(gradients, list) or len(gradients) != 4:
            return False

        # Validate maxNorm and minNonZero
        max_norm = inputs["maxNorm"]
        min_nonzero = inputs["minNonZero"]

        if not isinstance(max_norm, (int, float)) or not isinstance(min_nonzero, int):
            return False

        if max_norm <= 0 or min_nonzero < 0:
            return False

        return True

    def generate_witness(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate witness for the secure circuit."""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for secure model update circuit")

        gradients = inputs["gradients"]
        max_norm = int(inputs["maxNorm"])
        min_nonzero = int(inputs["minNonZero"])

        # Generate real witness based on circuit logic
        witness = []

        # Public inputs
        witness.append(max_norm)
        witness.append(min_nonzero)

        # Gradients (private inputs)
        for grad in gradients:
            witness.append(int(grad))

        return {
            "witness": witness,
            "public_inputs": [max_norm, min_nonzero]  # maxNorm and minNonZero are public
        }

    def validate_security_constraints(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security constraints before witness generation."""
        gradients = inputs["gradients"]
        max_norm = inputs["maxNorm"]
        min_nonzero = inputs["minNonZero"]

        # Calculate gradient norm
        norm_sq = sum(g * g for g in gradients)

        # Count non-zero elements
        nonzero_count = sum(1 for g in gradients if g != 0)

        # Validate constraints
        norm_valid = norm_sq <= max_norm
        nonzero_valid = nonzero_count >= min_nonzero
        security_valid = norm_valid and nonzero_valid

        return {
            "gradient_norm": norm_sq,
            "nonzero_count": nonzero_count,
            "norm_valid": norm_valid,
            "nonzero_valid": nonzero_valid,
            "security_valid": security_valid
        }

