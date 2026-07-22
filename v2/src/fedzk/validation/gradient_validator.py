#!/usr/bin/env python3
"""
FEDzk Gradient Data Validation
==============================

Comprehensive validation and sanitization for federated learning gradient data.
Implements defense against adversarial inputs, bounds checking, and data integrity.

Features:
- Gradient tensor validation and sanitization
- Bounds checking for gradient values
- Data type and shape validation
- Defense against adversarial gradient inputs
- Statistical anomaly detection
- Data poisoning prevention
- Production security standards
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import statistics
import time
from scipy import stats
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class GradientValidationError(Exception):
    """Base exception for gradient validation errors."""
    pass

class BoundsViolationError(GradientValidationError):
    """Exception for gradient bounds violations."""
    pass

class ShapeViolationError(GradientValidationError):
    """Exception for gradient shape violations."""
    pass

class TypeViolationError(GradientValidationError):
    """Exception for gradient type violations."""
    pass

class AdversarialInputError(GradientValidationError):
    """Exception for detected adversarial inputs."""
    pass

class DataPoisoningError(GradientValidationError):
    """Exception for detected data poisoning."""
    pass

class ValidationLevel(Enum):
    """Gradient validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

class AdversarialPattern(Enum):
    """Known adversarial patterns for gradient attacks."""
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    GRADIENT_NAN_INF = "gradient_nan_inf"
    GRADIENT_UNIFORM = "gradient_uniform"
    GRADIENT_ZERO = "gradient_zero"
    GRADIENT_OUTLIER = "gradient_outlier"
    GRADIENT_PERIODIC = "gradient_periodic"
    GRADIENT_NOISY = "gradient_noisy"
    GRADIENT_INVERTED = "gradient_inverted"
    GRADIENT_SCALING_ATTACK = "gradient_scaling_attack"

@dataclass
class GradientValidationConfig:
    """Configuration for gradient validation."""
    validation_level: ValidationLevel = ValidationLevel.STRICT
    max_gradient_value: float = 1000.0
    min_gradient_value: float = -1000.0
    max_tensor_size: int = 10000000  # 10M elements
    max_dimensions: int = 4
    allowed_dtypes: List[str] = None
    enable_statistical_analysis: bool = True
    outlier_threshold_sigma: float = 3.0
    enable_adversarial_detection: bool = True
    enable_data_poisoning_detection: bool = True
    enable_shape_consistency: bool = True
    expected_shapes: Optional[Dict[str, Tuple]] = None

    def __post_init__(self):
        if self.allowed_dtypes is None:
            self.allowed_dtypes = ["float32", "float64", "int32", "int64"]

@dataclass
class GradientValidationResult:
    """Result of gradient validation."""
    is_valid: bool
    validation_score: float
    detected_anomalies: List[str]
    statistical_summary: Dict[str, Any]
    adversarial_patterns: List[AdversarialPattern]
    warnings: List[str]
    sanitized_gradients: Optional[Dict[str, Any]]
    validation_metadata: Dict[str, Any]

@dataclass
class GradientStatistics:
    """Statistical summary of gradient data."""
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    skewness: float
    kurtosis: float
    outlier_count: int
    nan_count: int
    inf_count: int
    zero_count: int
    total_elements: int
    shape: Tuple
    dtype: str

class GradientValidator:
    """
    Comprehensive gradient data validator for FEDzk.

    Provides multi-layer validation and sanitization for federated learning
    gradient updates to prevent adversarial attacks and data corruption.
    """

    def __init__(self, config: GradientValidationConfig = None):
        self.config = config or GradientValidationConfig()
        self.validation_history: List[GradientValidationResult] = []
        self.baseline_statistics: Optional[Dict[str, GradientStatistics]] = None
        self.adversarial_patterns_detected: Dict[str, int] = {}

        logger.info("GradientValidator initialized with comprehensive security features")

    def validate_gradients_comprehensive(
        self,
        gradients: Dict[str, Any],
        client_id: str = "unknown",
        expected_shapes: Optional[Dict[str, Tuple]] = None
    ) -> GradientValidationResult:
        """
        Perform comprehensive gradient validation.

        Args:
            gradients: Dictionary of gradient tensors
            client_id: Client identifier for tracking
            expected_shapes: Expected tensor shapes

        Returns:
            GradientValidationResult: Comprehensive validation result
        """
        validation_score = 100.0
        detected_anomalies = []
        statistical_summary = {}
        adversarial_patterns = []
        warnings = []
        sanitized_gradients = None

        try:
            # Phase 1: Basic structural validation
            self._validate_gradient_structure(gradients)

            # Phase 2: Type and shape validation
            shape_score = self._validate_types_and_shapes(gradients, expected_shapes)
            validation_score -= shape_score

            # Phase 3: Bounds checking
            bounds_score = self._validate_bounds(gradients)
            validation_score -= bounds_score

            # Phase 4: Statistical analysis
            if self.config.enable_statistical_analysis:
                stats_score, stats_summary = self._perform_statistical_analysis(gradients)
                validation_score -= stats_score
                statistical_summary = stats_summary

            # Phase 5: Adversarial pattern detection
            if self.config.enable_adversarial_detection:
                adv_patterns = self._detect_adversarial_patterns(gradients)
                adversarial_patterns.extend(adv_patterns)
                validation_score -= len(adv_patterns) * 15

            # Phase 6: Data poisoning detection
            if self.config.enable_data_poisoning_detection:
                poisoning_score = self._detect_data_poisoning(gradients)
                validation_score -= poisoning_score

            # Phase 7: Sanitization (if enabled)
            if self.config.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                sanitized_gradients = self._sanitize_gradients(gradients)

            # Calculate final score
            validation_score = max(0.0, min(100.0, validation_score))

            # Generate warnings based on score and patterns
            if validation_score < 80:
                warnings.append("Low validation score - additional review recommended")
            if validation_score < 60:
                warnings.append("Critical validation score - gradients should be rejected")
            if adversarial_patterns:
                warnings.append(f"Adversarial patterns detected: {[p.value for p in adversarial_patterns]}")

            # Update detection history
            for pattern in adversarial_patterns:
                self.adversarial_patterns_detected[pattern.value] = \
                    self.adversarial_patterns_detected.get(pattern.value, 0) + 1

            result = GradientValidationResult(
                is_valid=validation_score >= 50 and not adversarial_patterns,
                validation_score=validation_score,
                detected_anomalies=detected_anomalies,
                statistical_summary=statistical_summary,
                adversarial_patterns=adversarial_patterns,
                warnings=warnings,
                sanitized_gradients=sanitized_gradients,
                validation_metadata={
                    "client_id": client_id,
                    "validation_level": self.config.validation_level.value,
                    "timestamp": time.time(),
                    "gradient_count": len(gradients),
                    "total_elements": sum(len(g.flatten()) if hasattr(g, 'flatten') else 1 for g in gradients.values())
                }
            )

            # Store result in history
            self.validation_history.append(result)

            logger.info(
                f"Gradient validation completed for client {client_id} - "
                f"Score: {validation_score:.1f}, Patterns: {len(adversarial_patterns)}"
            )

            return result

        except GradientValidationError as e:
            logger.error(f"Gradient validation error for client {client_id}: {e}")
            return GradientValidationResult(
                is_valid=False,
                validation_score=0.0,
                detected_anomalies=[str(e)],
                statistical_summary={},
                adversarial_patterns=[],
                warnings=[f"Validation error: {str(e)}"],
                sanitized_gradients=None,
                validation_metadata={"error": str(e), "client_id": client_id}
            )

    def _validate_gradient_structure(self, gradients: Dict[str, Any]):
        """Validate basic gradient structure."""
        if not isinstance(gradients, dict):
            raise TypeViolationError("Gradients must be a dictionary")

        if not gradients:
            raise ShapeViolationError("Empty gradients dictionary")

        if len(gradients) > 1000:  # Reasonable limit
            raise ShapeViolationError(f"Too many gradient parameters: {len(gradients)}")

    def _validate_types_and_shapes(
        self,
        gradients: Dict[str, Any],
        expected_shapes: Optional[Dict[str, Tuple]]
    ) -> float:
        """Validate gradient types and shapes."""
        penalty_score = 0.0

        for param_name, gradient in gradients.items():
            try:
                # Convert to tensor if needed
                if isinstance(gradient, (list, tuple)):
                    gradient = torch.tensor(gradient)
                elif not isinstance(gradient, (torch.Tensor, np.ndarray)):
                    raise TypeViolationError(f"Unsupported gradient type for {param_name}: {type(gradient)}")

                # Check tensor properties
                if hasattr(gradient, 'dtype'):
                    dtype_str = str(gradient.dtype)
                    if dtype_str not in self.config.allowed_dtypes:
                        penalty_score += 20
                        logger.warning(f"Invalid dtype for {param_name}: {dtype_str}")

                # Check shape
                if hasattr(gradient, 'shape'):
                    shape = gradient.shape

                    # Check dimension limits
                    if len(shape) > self.config.max_dimensions:
                        penalty_score += 15
                        logger.warning(f"Too many dimensions for {param_name}: {len(shape)}")

                    # Check total size
                    total_size = 1
                    for dim in shape:
                        total_size *= dim

                    if total_size > self.config.max_tensor_size:
                        penalty_score += 25
                        logger.warning(f"Tensor too large for {param_name}: {total_size}")

                    # Check expected shape if provided
                    if expected_shapes and param_name in expected_shapes:
                        expected_shape = expected_shapes[param_name]
                        if shape != expected_shape:
                            penalty_score += 10
                            logger.warning(f"Shape mismatch for {param_name}: expected {expected_shape}, got {shape}")

                # Store validated gradient back
                gradients[param_name] = gradient

            except Exception as e:
                logger.error(f"Error validating {param_name}: {e}")
                penalty_score += 30

        return penalty_score

    def _validate_bounds(self, gradients: Dict[str, Any]) -> float:
        """Validate gradient value bounds."""
        penalty_score = 0.0

        for param_name, gradient in gradients.items():
            try:
                # Convert to numpy for easier analysis
                if isinstance(gradient, torch.Tensor):
                    grad_array = gradient.detach().cpu().numpy()
                elif isinstance(gradient, np.ndarray):
                    grad_array = gradient
                else:
                    continue

                # Check for NaN and Inf
                nan_count = np.isnan(grad_array).sum()
                inf_count = np.isinf(grad_array).sum()

                if nan_count > 0:
                    penalty_score += nan_count * 50
                    logger.warning(f"NaN values detected in {param_name}: {nan_count}")

                if inf_count > 0:
                    penalty_score += inf_count * 50
                    logger.warning(f"Inf values detected in {param_name}: {inf_count}")

                # Check bounds
                min_val = np.min(grad_array)
                max_val = np.max(grad_array)

                if min_val < self.config.min_gradient_value:
                    penalty_score += 20
                    logger.warning(f"Gradient too small in {param_name}: {min_val}")

                if max_val > self.config.max_gradient_value:
                    penalty_score += 20
                    logger.warning(f"Gradient too large in {param_name}: {max_val}")

                # Check for all zeros (suspicious)
                zero_count = (grad_array == 0).sum()
                if zero_count == grad_array.size:
                    penalty_score += 30
                    logger.warning(f"All zero gradients in {param_name}")

            except Exception as e:
                logger.error(f"Error in bounds validation for {param_name}: {e}")
                penalty_score += 40

        return penalty_score

    def _perform_statistical_analysis(self, gradients: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Perform statistical analysis of gradients."""
        penalty_score = 0.0
        summary = {}

        for param_name, gradient in gradients.items():
            try:
                # Convert to numpy
                if isinstance(gradient, torch.Tensor):
                    grad_array = gradient.detach().cpu().numpy().flatten()
                elif isinstance(gradient, np.ndarray):
                    grad_array = gradient.flatten()
                else:
                    continue

                # Skip if empty or too small
                if len(grad_array) < 2:
                    continue

                # Calculate statistics
                stats_summary = GradientStatistics(
                    mean=float(np.mean(grad_array)),
                    std=float(np.std(grad_array)),
                    min_val=float(np.min(grad_array)),
                    max_val=float(np.max(grad_array)),
                    median=float(np.median(grad_array)),
                    skewness=float(stats.skew(grad_array)),
                    kurtosis=float(stats.kurtosis(grad_array)),
                    outlier_count=self._count_outliers(grad_array),
                    nan_count=int(np.isnan(grad_array).sum()),
                    inf_count=int(np.isinf(grad_array).sum()),
                    zero_count=int((grad_array == 0).sum()),
                    total_elements=len(grad_array),
                    shape=getattr(gradient, 'shape', (len(grad_array),)),
                    dtype=str(getattr(gradient, 'dtype', type(grad_array[0]).__name__))
                )

                summary[param_name] = asdict(stats_summary)

                # Check for anomalies
                if abs(stats_summary.skewness) > 2.0:
                    penalty_score += 10
                    logger.warning(f"High skewness in {param_name}: {stats_summary.skewness}")

                if stats_summary.kurtosis > 5.0:
                    penalty_score += 10
                    logger.warning(f"High kurtosis in {param_name}: {stats_summary.kurtosis}")

                if stats_summary.outlier_count > len(grad_array) * 0.1:  # >10% outliers
                    penalty_score += 15
                    logger.warning(f"High outlier count in {param_name}: {stats_summary.outlier_count}")

            except Exception as e:
                logger.error(f"Error in statistical analysis for {param_name}: {e}")
                penalty_score += 20

        return penalty_score, summary

    def _count_outliers(self, data: np.ndarray) -> int:
        """Count statistical outliers in data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            threshold = self.config.outlier_threshold_sigma * std

            outliers = np.abs(data - mean) > threshold
            return int(np.sum(outliers))
        except:
            return 0

    def _detect_adversarial_patterns(self, gradients: Dict[str, Any]) -> List[AdversarialPattern]:
        """Detect known adversarial patterns in gradients."""
        detected_patterns = []

        for param_name, gradient in gradients.items():
            try:
                # Convert to numpy
                if isinstance(gradient, torch.Tensor):
                    grad_array = gradient.detach().cpu().numpy().flatten()
                elif isinstance(gradient, np.ndarray):
                    grad_array = gradient.flatten()
                else:
                    continue

                # Skip if too small
                if len(grad_array) < 10:
                    continue

                # Check for gradient explosion
                max_val = np.max(np.abs(grad_array))
                if max_val > self.config.max_gradient_value * 10:
                    detected_patterns.append(AdversarialPattern.GRADIENT_EXPLOSION)

                # Check for gradient vanishing
                non_zero = np.count_nonzero(grad_array)
                if non_zero / len(grad_array) < 0.01:  # <1% non-zero
                    detected_patterns.append(AdversarialPattern.GRADIENT_VANISHING)

                # Check for NaN/Inf
                if np.any(np.isnan(grad_array)) or np.any(np.isinf(grad_array)):
                    detected_patterns.append(AdversarialPattern.GRADIENT_NAN_INF)

                # Check for uniform values
                unique_vals = len(np.unique(grad_array))
                if unique_vals == 1:
                    detected_patterns.append(AdversarialPattern.GRADIENT_UNIFORM)

                # Check for all zeros
                if np.all(grad_array == 0):
                    detected_patterns.append(AdversarialPattern.GRADIENT_ZERO)

                # Check for outliers
                outlier_count = self._count_outliers(grad_array)
                if outlier_count > len(grad_array) * 0.2:  # >20% outliers
                    detected_patterns.append(AdversarialPattern.GRADIENT_OUTLIER)

                # Check for periodic patterns (suspicious)
                if self._detect_periodic_pattern(grad_array):
                    detected_patterns.append(AdversarialPattern.GRADIENT_PERIODIC)

                # Check for excessive noise
                if self._detect_excessive_noise(grad_array):
                    detected_patterns.append(AdversarialPattern.GRADIENT_NOISY)

            except Exception as e:
                logger.error(f"Error detecting adversarial patterns in {param_name}: {e}")

        # Remove duplicates
        return list(set(detected_patterns))

    def _detect_periodic_pattern(self, data: np.ndarray) -> bool:
        """Detect periodic patterns in gradient data."""
        try:
            # Simple autocorrelation check
            if len(data) < 20:
                return False

            # Check for simple periodic patterns
            for period in range(2, min(20, len(data) // 2)):
                correlation = np.corrcoef(data[:-period], data[period:])[0, 1]
                if abs(correlation) > 0.8:  # High correlation suggests periodicity
                    return True

            return False
        except:
            return False

    def _detect_excessive_noise(self, data: np.ndarray) -> bool:
        """Detect excessive noise in gradient data."""
        try:
            if len(data) < 10:
                return False

            # Calculate signal-to-noise ratio
            signal = np.mean(np.abs(data))
            noise = np.std(data)

            if noise > 0:
                snr = signal / noise
                # Low SNR suggests excessive noise
                return snr < 0.1
            else:
                return False
        except:
            return False

    def _detect_data_poisoning(self, gradients: Dict[str, Any]) -> float:
        """Detect potential data poisoning in gradients."""
        penalty_score = 0.0

        try:
            # Compare with baseline statistics if available
            if self.baseline_statistics:
                for param_name, gradient in gradients.items():
                    if param_name in self.baseline_statistics:
                        baseline = self.baseline_statistics[param_name]

                        # Convert current gradient to stats
                        if isinstance(gradient, torch.Tensor):
                            grad_array = gradient.detach().cpu().numpy().flatten()
                        elif isinstance(gradient, np.ndarray):
                            grad_array = gradient.flatten()
                        else:
                            continue

                        current_stats = GradientStatistics(
                            mean=float(np.mean(grad_array)),
                            std=float(np.std(grad_array)),
                            min_val=float(np.min(grad_array)),
                            max_val=float(np.max(grad_array)),
                            median=float(np.median(grad_array)),
                            skewness=float(stats.skew(grad_array)),
                            kurtosis=float(stats.kurtosis(grad_array)),
                            outlier_count=self._count_outliers(grad_array),
                            nan_count=int(np.isnan(grad_array).sum()),
                            inf_count=int(np.isinf(grad_array).sum()),
                            zero_count=int((grad_array == 0).sum()),
                            total_elements=len(grad_array),
                            shape=getattr(gradient, 'shape', (len(grad_array),)),
                            dtype=str(getattr(gradient, 'dtype', type(grad_array[0]).__name__))
                        )

                        # Compare with baseline
                        deviation_score = self._calculate_deviation_score(current_stats, baseline)
                        if deviation_score > 2.0:  # Significant deviation
                            penalty_score += deviation_score * 10
                            logger.warning(f"Statistical deviation detected in {param_name}: {deviation_score}")

        except Exception as e:
            logger.error(f"Error in data poisoning detection: {e}")
            penalty_score += 20

        return penalty_score

    def _calculate_deviation_score(self, current: GradientStatistics, baseline: GradientStatistics) -> float:
        """Calculate deviation score between current and baseline statistics."""
        try:
            # Calculate z-scores for key metrics
            mean_z = abs(current.mean - baseline.mean) / max(baseline.std, 0.001)
            std_z = abs(current.std - baseline.std) / max(baseline.std, 0.001)
            skewness_z = abs(current.skewness - baseline.skewness) / 2.0  # Rough estimate

            # Combine scores
            return (mean_z + std_z + skewness_z) / 3.0
        except:
            return 0.0

    def _sanitize_gradients(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize gradients by clamping values and removing anomalies."""
        sanitized = {}

        for param_name, gradient in gradients.items():
            try:
                # Convert to tensor
                if isinstance(gradient, np.ndarray):
                    tensor = torch.from_numpy(gradient)
                elif isinstance(gradient, torch.Tensor):
                    tensor = gradient.clone()
                else:
                    sanitized[param_name] = gradient
                    continue

                # Clamp values to bounds
                tensor = torch.clamp(tensor,
                                   min=self.config.min_gradient_value,
                                   max=self.config.max_gradient_value)

                # Remove NaN and Inf
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
                tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)

                sanitized[param_name] = tensor

            except Exception as e:
                logger.error(f"Error sanitizing {param_name}: {e}")
                sanitized[param_name] = gradient  # Return original if sanitization fails

        return sanitized

    def establish_baseline(self, gradients: Dict[str, Any]):
        """Establish baseline statistics for anomaly detection."""
        try:
            baseline = {}
            _, stats_summary = self._perform_statistical_analysis(gradients)

            for param_name, stats_dict in stats_summary.items():
                baseline[param_name] = GradientStatistics(**stats_dict)

            self.baseline_statistics = baseline
            logger.info(f"Established baseline statistics for {len(baseline)} parameters")

        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")

    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive validation metrics."""
        metrics = {
            "total_validations": len(self.validation_history),
            "average_score": 0.0,
            "adversarial_patterns_detected": dict(self.adversarial_patterns_detected),
            "validation_level": self.config.validation_level.value,
            "baseline_established": self.baseline_statistics is not None
        }

        if self.validation_history:
            scores = [result.validation_score for result in self.validation_history]
            metrics["average_score"] = sum(scores) / len(scores)
            metrics["min_score"] = min(scores)
            metrics["max_score"] = max(scores)

        # Calculate pattern detection rates
        total_patterns = sum(metrics["adversarial_patterns_detected"].values())
        metrics["total_patterns_detected"] = total_patterns

        return metrics

# Convenience functions for easy usage
def create_gradient_validator(level: ValidationLevel = ValidationLevel.STRICT) -> GradientValidator:
    """
    Create a gradient validator with predefined security levels.

    Args:
        level: Validation strictness level

    Returns:
        GradientValidator: Configured validator
    """
    config = GradientValidationConfig(validation_level=level)

    # Adjust config based on level
    if level == ValidationLevel.BASIC:
        config.enable_adversarial_detection = False
        config.enable_data_poisoning_detection = False
        config.enable_statistical_analysis = False
    elif level == ValidationLevel.PARANOID:
        config.max_gradient_value = 100.0  # Stricter bounds
        config.min_gradient_value = -100.0
        config.outlier_threshold_sigma = 2.0  # More sensitive

    return GradientValidator(config)

def validate_federated_gradients(
    gradients: Dict[str, Any],
    client_id: str = "unknown",
    expected_shapes: Optional[Dict[str, Tuple]] = None
) -> GradientValidationResult:
    """
    Convenience function for federated learning gradient validation.

    Args:
        gradients: Gradient tensors to validate
        client_id: Client identifier
        expected_shapes: Expected tensor shapes

    Returns:
        GradientValidationResult: Validation result
    """
    validator = create_gradient_validator()
    return validator.validate_gradients_comprehensive(gradients, client_id, expected_shapes)
