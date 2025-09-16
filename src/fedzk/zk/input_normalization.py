"""
Input Normalization Module for FEDzk

This module provides functions to quantize floating-point gradients into integers
that can be processed by ZK circuits, while maintaining cryptographic integrity.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch
import math


class GradientQuantizer:
    """
    Handles quantization of floating-point gradients for ZK circuit compatibility.

    This class provides methods to:
    1. Quantize floating-point gradients to integers
    2. Dequantize integers back to floating-point values
    3. Maintain quantization metadata for verification
    4. Ensure quantization preserves gradient relationships
    """

    def __init__(self, scale_factor: int = 1000, max_bits: int = 32):
        """
        Initialize the gradient quantizer.

        Args:
            scale_factor: Multiplier for quantization (higher = more precision)
            max_bits: Maximum bits for quantized values
        """
        self.scale_factor = scale_factor
        self.max_bits = max_bits
        self.max_value = 2**(max_bits - 1) - 1  # Signed integer limit
        self.min_value = -(2**(max_bits - 1))   # Signed integer minimum

    def quantize_gradients(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
        """
        Quantize floating-point gradients to integers for ZK circuits.

        Args:
            gradients: Dictionary of gradient tensors

        Returns:
            Tuple of (quantized_gradients, quantization_metadata)
        """
        quantized_gradients = {}
        quantization_metadata = {
            "scale_factor": self.scale_factor,
            "original_shapes": {},
            "quantization_stats": {},
            "verification_hash": None
        }

        for param_name, grad_tensor in gradients.items():
            # Store original shape
            original_shape = grad_tensor.shape
            quantization_metadata["original_shapes"][param_name] = original_shape

            # Flatten tensor for processing
            flat_gradients = grad_tensor.flatten()

            # Quantize each gradient value
            quantized_values = []
            stats = {
                "original_min": float(flat_gradients.min()),
                "original_max": float(flat_gradients.max()),
                "original_mean": float(flat_gradients.mean()),
                "original_std": float(flat_gradients.std()),
                "quantized_min": None,
                "quantized_max": None,
                "quantization_error": None
            }

            for grad_val in flat_gradients:
                # Quantize: multiply by scale factor and round to nearest integer
                quantized = round(float(grad_val) * self.scale_factor)

                # Clamp to valid range
                quantized = max(self.min_value, min(self.max_value, quantized))

                quantized_values.append(int(quantized))

            # Update statistics
            quantized_array = np.array(quantized_values)
            stats["quantized_min"] = int(quantized_array.min())
            stats["quantized_max"] = int(quantized_array.max())

            # Calculate quantization error
            original_vals = flat_gradients.numpy()
            dequantized_vals = quantized_array.astype(float) / self.scale_factor
            quantization_error = np.mean(np.abs(original_vals - dequantized_vals))
            stats["quantization_error"] = float(quantization_error)

            quantized_gradients[param_name] = quantized_values
            quantization_metadata["quantization_stats"][param_name] = stats

        # Generate verification hash
        import hashlib
        metadata_str = str(sorted(quantization_metadata.items()))
        quantization_metadata["verification_hash"] = hashlib.sha256(
            metadata_str.encode()
        ).hexdigest()

        return quantized_gradients, quantization_metadata

    def dequantize_gradients(self, quantized_gradients: Dict[str, List[int]],
                           quantization_metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Dequantize integer gradients back to floating-point tensors.

        Args:
            quantized_gradients: Quantized gradient values
            quantization_metadata: Metadata from quantization process

        Returns:
            Dictionary of dequantized gradient tensors
        """
        dequantized_gradients = {}

        for param_name, quantized_values in quantized_gradients.items():
            # Convert to numpy array and dequantize
            quantized_array = np.array(quantized_values, dtype=float)
            dequantized_array = quantized_array / quantization_metadata["scale_factor"]

            # Reshape to original dimensions
            original_shape = quantization_metadata["original_shapes"][param_name]
            dequantized_tensor = torch.tensor(dequantized_array).reshape(original_shape)

            dequantized_gradients[param_name] = dequantized_tensor

        return dequantized_gradients

    def validate_quantization_integrity(self, original_gradients: Dict[str, torch.Tensor],
                                      quantized_gradients: Dict[str, List[int]],
                                      quantization_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that quantization preserves gradient integrity.

        Args:
            original_gradients: Original floating-point gradients
            quantized_gradients: Quantized integer gradients
            quantization_metadata: Metadata from quantization

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "integrity_preserved": True,
            "max_quantization_error": 0.0,
            "shape_consistency": True,
            "verification_hash_valid": True,
            "parameter_validation": {}
        }

        # Validate verification hash
        import hashlib
        metadata_copy = quantization_metadata.copy()
        original_hash = metadata_copy.pop("verification_hash", None)
        metadata_str = str(sorted(metadata_copy.items()))
        computed_hash = hashlib.sha256(metadata_str.encode()).hexdigest()

        if original_hash != computed_hash:
            validation_results["verification_hash_valid"] = False
            validation_results["integrity_preserved"] = False

        # Validate each parameter
        for param_name, original_tensor in original_gradients.items():
            param_results = {
                "shape_match": True,
                "quantization_error": 0.0,
                "value_range_valid": True
            }

            if param_name not in quantized_gradients:
                param_results["shape_match"] = False
                validation_results["integrity_preserved"] = False
                continue

            quantized_values = quantized_gradients[param_name]
            original_flat = original_tensor.flatten().numpy()

            # Check shape consistency
            if len(quantized_values) != len(original_flat):
                param_results["shape_match"] = False
                validation_results["integrity_preserved"] = False

            # Check quantization error
            dequantized = np.array(quantized_values, dtype=float) / quantization_metadata["scale_factor"]
            error = np.mean(np.abs(original_flat - dequantized))
            param_results["quantization_error"] = float(error)

            validation_results["max_quantization_error"] = max(
                validation_results["max_quantization_error"],
                param_results["quantization_error"]
            )

            # Check value range
            quantized_array = np.array(quantized_values)
            if quantized_array.min() < self.min_value or quantized_array.max() > self.max_value:
                param_results["value_range_valid"] = False
                validation_results["integrity_preserved"] = False

            validation_results["parameter_validation"][param_name] = param_results

        return validation_results


class AdaptiveQuantizer(GradientQuantizer):
    """
    Adaptive quantizer that automatically determines optimal scale factor
    based on gradient statistics to minimize quantization error.
    """

    def __init__(self, target_precision: float = 1e-6, max_bits: int = 32):
        """
        Initialize adaptive quantizer.

        Args:
            target_precision: Target quantization precision
            max_bits: Maximum bits for quantized values
        """
        super().__init__(scale_factor=1, max_bits=max_bits)
        self.target_precision = target_precision

    def adapt_scale_factor(self, gradients: Dict[str, torch.Tensor]) -> int:
        """
        Determine optimal scale factor based on gradient statistics.

        Args:
            gradients: Gradient tensors to analyze

        Returns:
            Optimal scale factor for quantization
        """
        all_gradients = []
        for grad_tensor in gradients.values():
            all_gradients.extend(grad_tensor.flatten().numpy())

        all_gradients = np.array(all_gradients)

        # Calculate gradient statistics
        grad_min = np.min(np.abs(all_gradients))
        grad_max = np.max(np.abs(all_gradients))
        grad_std = np.std(all_gradients)

        # Use smallest non-zero gradient as reference for precision
        if grad_min > 0:
            # Scale factor to achieve target precision
            scale_factor = int(1.0 / (grad_min * self.target_precision))
        else:
            # Fallback: use standard deviation
            scale_factor = int(1.0 / (grad_std * self.target_precision))

        # Ensure scale factor fits within bit constraints
        max_scale = self.max_value // max(1, int(grad_max))
        scale_factor = min(scale_factor, max_scale)

        # Ensure minimum scale factor of 1
        scale_factor = max(1, scale_factor)

        return scale_factor

    def quantize_gradients_adaptive(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
        """
        Quantize gradients using adaptive scale factor determination.

        Args:
            gradients: Dictionary of gradient tensors

        Returns:
            Tuple of (quantized_gradients, quantization_metadata)
        """
        # Determine optimal scale factor
        self.scale_factor = self.adapt_scale_factor(gradients)

        # Use standard quantization with adaptive scale factor
        return self.quantize_gradients(gradients)


# Utility functions for gradient preprocessing
def normalize_gradients(gradients: Dict[str, torch.Tensor],
                       normalization_type: str = "l2") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Apply normalization to gradients before quantization.

    Args:
        gradients: Original gradients
        normalization_type: Type of normalization ("l2", "l1", "max", "none")

    Returns:
        Tuple of (normalized_gradients, normalization_metadata)
    """
    normalized_gradients = {}
    normalization_metadata = {
        "type": normalization_type,
        "norms": {},
        "original_stats": {}
    }

    for param_name, grad_tensor in gradients.items():
        flat_grad = grad_tensor.flatten()

        # Store original statistics
        normalization_metadata["original_stats"][param_name] = {
            "norm": float(torch.norm(flat_grad)),
            "max": float(torch.max(torch.abs(flat_grad))),
            "mean": float(torch.mean(torch.abs(flat_grad)))
        }

        if normalization_type == "l2":
            # L2 normalization
            norm = torch.norm(flat_grad)
            if norm > 0:
                normalized = flat_grad / norm
                normalization_metadata["norms"][param_name] = float(norm)
            else:
                normalized = flat_grad
                normalization_metadata["norms"][param_name] = 1.0

        elif normalization_type == "l1":
            # L1 normalization
            norm = torch.sum(torch.abs(flat_grad))
            if norm > 0:
                normalized = flat_grad / norm
                normalization_metadata["norms"][param_name] = float(norm)
            else:
                normalized = flat_grad
                normalization_metadata["norms"][param_name] = 1.0

        elif normalization_type == "max":
            # Max normalization
            max_val = torch.max(torch.abs(flat_grad))
            if max_val > 0:
                normalized = flat_grad / max_val
                normalization_metadata["norms"][param_name] = float(max_val)
            else:
                normalized = flat_grad
                normalization_metadata["norms"][param_name] = 1.0

        else:  # "none"
            normalized = flat_grad
            normalization_metadata["norms"][param_name] = 1.0

        # Reshape back to original shape
        normalized_gradients[param_name] = normalized.reshape(grad_tensor.shape)

    return normalized_gradients, normalization_metadata


def create_quantized_proof_input(quantized_gradients: Dict[str, List[int]],
                               quantization_metadata: Dict[str, Any],
                               circuit_type: str = "quantized") -> Dict[str, Any]:
    """
    Create properly formatted input for quantized ZK circuits.

    Args:
        quantized_gradients: Quantized gradient values
        quantization_metadata: Quantization metadata
        circuit_type: Type of circuit to target

    Returns:
        Dictionary with circuit inputs
    """
    circuit_input = {
        "quantized_gradients": [],
        "scale_factor_input": quantization_metadata["scale_factor"]
    }

    # Flatten all quantized gradients into single array for circuit
    all_quantized = []
    for param_gradients in quantized_gradients.values():
        all_quantized.extend(param_gradients)

    circuit_input["quantized_gradients"] = all_quantized

    # Add circuit-specific parameters
    if circuit_type == "secure_quantized":
        circuit_input.update({
            "max_norm_input": 1000000,  # Default max norm
            "min_active_input": 1       # Default min active gradients
        })

    return circuit_input


# Global quantizer instances
default_quantizer = GradientQuantizer(scale_factor=1000)
adaptive_quantizer = AdaptiveQuantizer(target_precision=1e-6)

