#!/usr/bin/env python3
"""
ModelUpdateQuantized ZK Circuit
===============================

Circuit for handling quantized floating-point gradients in federated learning.
"""

from typing import Dict, List, Any
from pathlib import Path

class ModelUpdateQuantizedCircuit:
    """Quantized model update circuit with gradient scaling support."""

    def __init__(self, circuit_path: Path = None, scale_factor: int = 1000):
        """Initialize the quantized circuit."""
        if circuit_path is None:
            circuit_path = Path(__file__).parent / "model_update_quantized.circom"
        self.circuit_path = circuit_path
        self.name = "model_update_quantized"
        self.scale_factor = scale_factor

    def get_circuit_spec(self) -> Dict[str, Any]:
        """Get circuit specification."""
        return {
            "inputs": ["quantized_gradients", "scale_factor_input"],
            "outputs": ["original_norm", "quantized_norm", "gradient_count"],
            "constraints": 1200,
            "witness_size": 1536
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate circuit inputs."""
        required_inputs = ["quantized_gradients", "scale_factor_input"]
        if not all(key in inputs for key in required_inputs):
            return False

        # Validate quantized gradients array
        quantized_gradients = inputs["quantized_gradients"]
        if not isinstance(quantized_gradients, list) or len(quantized_gradients) != 4:
            return False

        # Validate scale factor
        scale_factor = inputs["scale_factor_input"]
        if not isinstance(scale_factor, (int, float)):
            return False

        return True

    def quantize_gradients(self, gradients: List[float], scale_factor: int) -> List[int]:
        """Quantize gradients using the specified scale factor."""
        return [int(g * scale_factor) for g in gradients]

    def dequantize_gradients(self, quantized_gradients: List[int], scale_factor: int) -> List[float]:
        """Dequantize gradients back to original scale."""
        return [qg / scale_factor for qg in quantized_gradients]

    def generate_witness(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate witness for the quantized circuit."""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for quantized model update circuit")

        quantized_gradients = inputs["quantized_gradients"]
        scale_factor_input = int(inputs["scale_factor_input"])

        # Validate inputs match circuit expectations (n=4)
        if len(quantized_gradients) != 4:
            raise ValueError("Circuit expects exactly 4 quantized gradients")

        # Generate witness based on circuit logic
        witness = []

        # Public input: scale factor
        witness.append(scale_factor_input)

        # Private inputs: quantized gradients
        for qg in quantized_gradients:
            witness.append(int(qg))

        return {
            "witness": witness,
            "public_inputs": [scale_factor_input]  # scale_factor_input is public
        }

    def validate_quantization(self, original_gradients: List[float],
                            quantized_gradients: List[int],
                            scale_factor: int) -> Dict[str, Any]:
        """Validate the quantization process."""
        # Dequantize and compare
        dequantized = self.dequantize_gradients(quantized_gradients, scale_factor)

        # Calculate quantization error
        errors = [abs(orig - deq) for orig, deq in zip(original_gradients, dequantized)]
        max_error = max(errors)
        avg_error = sum(errors) / len(errors)

        # Calculate norms
        original_norm = sum(g * g for g in original_gradients) ** 0.5
        quantized_norm = sum(qg * qg for qg in quantized_gradients) ** 0.5 / scale_factor

        return {
            "max_quantization_error": max_error,
            "avg_quantization_error": avg_error,
            "original_norm": original_norm,
            "quantized_norm": quantized_norm,
            "quantization_accuracy": 1.0 - (max_error / max(abs(g) for g in original_gradients))
        }

