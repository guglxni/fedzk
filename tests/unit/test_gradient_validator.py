#!/usr/bin/env python3
"""
Tests for Gradient Validator
============================

Comprehensive test suite for FEDzk gradient validation system.
Tests adversarial detection, bounds checking, and data sanitization.
"""

import unittest
import numpy as np
import torch
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from fedzk.validation.gradient_validator import (
    GradientValidator,
    GradientValidationConfig,
    GradientValidationResult,
    GradientValidationError,
    BoundsViolationError,
    ShapeViolationError,
    TypeViolationError,
    AdversarialInputError,
    DataPoisoningError,
    ValidationLevel,
    AdversarialPattern,
    GradientStatistics,
    create_gradient_validator,
    validate_federated_gradients
)

class TestGradientValidator(unittest.TestCase):
    """Test GradientValidator class."""

    def setUp(self):
        """Setup test environment."""
        self.config = GradientValidationConfig(
            validation_level=ValidationLevel.STRICT,
            max_gradient_value=1000.0,
            min_gradient_value=-1000.0,
            enable_statistical_analysis=True,
            enable_adversarial_detection=True,
            outlier_threshold_sigma=3.0
        )

        self.validator = GradientValidator(self.config)

        # Create test gradients
        self.valid_gradients = {
            "layer1.weight": torch.randn(10, 20),
            "layer1.bias": torch.randn(20),
            "layer2.weight": torch.randn(20, 5),
            "layer2.bias": torch.randn(5)
        }

        self.numpy_gradients = {
            "layer1.weight": np.random.randn(10, 20),
            "layer1.bias": np.random.randn(20),
            "layer2.weight": np.random.randn(20, 5),
            "layer2.bias": np.random.randn(5)
        }

    def test_valid_gradients_validation(self):
        """Test validation of valid gradients."""
        result = self.validator.validate_gradients_comprehensive(
            self.valid_gradients, "test_client"
        )

        self.assertIsInstance(result, GradientValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.validation_score, 80)
        self.assertEqual(len(result.detected_anomalies), 0)
        self.assertEqual(len(result.adversarial_patterns), 0)

    def test_numpy_gradients_validation(self):
        """Test validation of NumPy gradients."""
        result = self.validator.validate_gradients_comprehensive(
            self.numpy_gradients, "numpy_client"
        )

        self.assertTrue(result.is_valid)
        self.assertGreater(result.validation_score, 80)

    def test_bounds_violation_detection(self):
        """Test detection of gradient bounds violations."""
        # Create gradients with values outside bounds
        invalid_gradients = {
            "layer1.weight": torch.tensor([[2000.0, -1500.0], [500.0, 800.0]]),
            "layer1.bias": torch.tensor([1200.0, -800.0])
        }

        result = self.validator.validate_gradients_comprehensive(
            invalid_gradients, "bounds_client"
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.validation_score, 100)
        self.assertGreater(len(result.warnings), 0)

    def test_nan_inf_detection(self):
        """Test detection of NaN and Inf values."""
        nan_gradients = {
            "layer1.weight": torch.tensor([[1.0, float('nan')], [3.0, 4.0]]),
            "layer1.bias": torch.tensor([float('inf'), -float('inf')])
        }

        result = self.validator.validate_gradients_comprehensive(
            nan_gradients, "nan_client"
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.validation_score, 50)
        self.assertIn(AdversarialPattern.GRADIENT_NAN_INF, result.adversarial_patterns)

    def test_shape_validation(self):
        """Test gradient shape validation."""
        # Test with expected shapes
        expected_shapes = {
            "layer1.weight": (10, 20),
            "layer1.bias": (20,),
            "layer2.weight": (20, 5),
            "layer2.bias": (5,)
        }

        result = self.validator.validate_gradients_comprehensive(
            self.valid_gradients, "shape_client", expected_shapes
        )

        self.assertTrue(result.is_valid)

        # Test with mismatched shapes
        wrong_shapes = {
            "layer1.weight": torch.randn(5, 10),  # Wrong shape
            "layer1.bias": torch.randn(20)
        }

        result = self.validator.validate_gradients_comprehensive(
            wrong_shapes, "wrong_shape_client", expected_shapes
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.validation_score, 100)

    def test_empty_gradients_validation(self):
        """Test validation of empty gradients."""
        empty_gradients = {}

        with self.assertRaises(ShapeViolationError):
            self.validator.validate_gradients_comprehensive(empty_gradients, "empty_client")

    def test_adversarial_pattern_detection(self):
        """Test detection of adversarial patterns."""

        # Test gradient explosion
        explosion_gradients = {
            "layer1.weight": torch.tensor([[1e10, -1e10], [5e9, 8e9]])
        }

        result = self.validator.validate_gradients_comprehensive(
            explosion_gradients, "explosion_client"
        )

        self.assertIn(AdversarialPattern.GRADIENT_EXPLOSION, result.adversarial_patterns)

        # Test uniform gradients
        uniform_gradients = {
            "layer1.weight": torch.ones(10, 20) * 5.0
        }

        result = self.validator.validate_gradients_comprehensive(
            uniform_gradients, "uniform_client"
        )

        self.assertIn(AdversarialPattern.GRADIENT_UNIFORM, result.adversarial_patterns)

        # Test all zero gradients
        zero_gradients = {
            "layer1.weight": torch.zeros(10, 20)
        }

        result = self.validator.validate_gradients_comprehensive(
            zero_gradients, "zero_client"
        )

        self.assertIn(AdversarialPattern.GRADIENT_ZERO, result.adversarial_patterns)

    def test_statistical_analysis(self):
        """Test statistical analysis of gradients."""
        result = self.validator.validate_gradients_comprehensive(
            self.valid_gradients, "stats_client"
        )

        self.assertIn("layer1.weight", result.statistical_summary)
        self.assertIn("layer2.bias", result.statistical_summary)

        # Check statistical properties
        stats = result.statistical_summary["layer1.weight"]
        self.assertIsInstance(stats["mean"], float)
        self.assertIsInstance(stats["std"], float)
        self.assertIsInstance(stats["min_val"], float)
        self.assertIsInstance(stats["max_val"], float)
        self.assertGreater(stats["total_elements"], 0)

    def test_outlier_detection(self):
        """Test outlier detection in gradients."""
        # Create gradients with outliers
        outlier_gradients = {
            "layer1.weight": torch.tensor([[1.0, 2.0, 3.0, 1000.0], [4.0, 5.0, 6.0, 7.0]])
        }

        result = self.validator.validate_gradients_comprehensive(
            outlier_gradients, "outlier_client"
        )

        # Should detect outliers
        stats = result.statistical_summary["layer1.weight"]
        self.assertGreater(stats["outlier_count"], 0)

    def test_gradient_sanitization(self):
        """Test gradient sanitization functionality."""
        # Create gradients with issues
        unsanitized_gradients = {
            "layer1.weight": torch.tensor([[1500.0, float('nan')], [-1200.0, float('inf')]]),
            "layer1.bias": torch.tensor([800.0, 900.0])
        }

        result = self.validator.validate_gradients_comprehensive(
            unsanitized_gradients, "sanitize_client"
        )

        # Should have sanitized version
        self.assertIsNotNone(result.sanitized_gradients)

        # Check that sanitized gradients are clean
        sanitized = result.sanitized_gradients
        weight_tensor = sanitized["layer1.weight"]

        # Should not have NaN or Inf
        self.assertFalse(torch.isnan(weight_tensor).any())
        self.assertFalse(torch.isinf(weight_tensor).any())

        # Should be clamped to bounds
        self.assertTrue((weight_tensor >= self.config.min_gradient_value).all())
        self.assertTrue((weight_tensor <= self.config.max_gradient_value).all())

    def test_baseline_establishment(self):
        """Test baseline establishment for anomaly detection."""
        # Establish baseline
        self.validator.establish_baseline(self.valid_gradients)

        self.assertIsNotNone(self.validator.baseline_statistics)
        self.assertIn("layer1.weight", self.validator.baseline_statistics)

        # Test with similar gradients (should pass)
        similar_gradients = {
            "layer1.weight": torch.randn(10, 20) * 0.1,  # Similar scale
            "layer1.bias": torch.randn(20) * 0.1
        }

        result = self.validator.validate_gradients_comprehensive(
            similar_gradients, "baseline_client"
        )

        self.assertTrue(result.is_valid)

    def test_validation_levels(self):
        """Test different validation levels."""
        # Test basic level
        basic_validator = create_gradient_validator(ValidationLevel.BASIC)
        result = basic_validator.validate_gradients_comprehensive(
            self.valid_gradients, "basic_client"
        )

        self.assertTrue(result.is_valid)

        # Test paranoid level
        paranoid_config = GradientValidationConfig(
            validation_level=ValidationLevel.PARANOID,
            max_gradient_value=10.0,  # Very strict bounds
            min_gradient_value=-10.0
        )

        paranoid_validator = GradientValidator(paranoid_config)

        # These gradients should fail paranoid validation
        large_gradients = {
            "layer1.weight": torch.tensor([[50.0, -30.0], [25.0, 40.0]])
        }

        result = paranoid_validator.validate_gradients_comprehensive(
            large_gradients, "paranoid_client"
        )

        self.assertFalse(result.is_valid)

    def test_validation_metrics(self):
        """Test validation metrics collection."""
        # Perform several validations
        for i in range(5):
            test_gradients = {
                "layer.weight": torch.randn(5, 5) * (i + 1),
                "layer.bias": torch.randn(5)
            }
            self.validator.validate_gradients_comprehensive(
                test_gradients, f"metrics_client_{i}"
            )

        metrics = self.validator.get_validation_metrics()

        self.assertEqual(metrics["total_validations"], 5)
        self.assertGreater(metrics["average_score"], 0)
        self.assertIn("min_score", metrics)
        self.assertIn("max_score", metrics)

    def test_type_validation(self):
        """Test gradient type validation."""
        # Test with invalid types
        invalid_gradients = {
            "layer1.weight": "invalid_string_type",
            "layer1.bias": {"invalid": "dict_type"}
        }

        with self.assertRaises(TypeViolationError):
            self.validator.validate_gradients_comprehensive(
                invalid_gradients, "type_client"
            )

    def test_large_tensor_handling(self):
        """Test handling of large tensors."""
        # Create a very large tensor (but within limits)
        large_tensor = torch.randn(100, 100)  # 10,000 elements

        large_gradients = {
            "large_layer.weight": large_tensor,
            "large_layer.bias": torch.randn(100)
        }

        result = self.validator.validate_gradients_comprehensive(
            large_gradients, "large_client"
        )

        self.assertTrue(result.is_valid)

        # Test tensor that's too large
        oversized_config = GradientValidationConfig(max_tensor_size=1000)  # Very small limit
        oversized_validator = GradientValidator(oversized_config)

        result = oversized_validator.validate_gradients_comprehensive(
            large_gradients, "oversized_client"
        )

        self.assertFalse(result.is_valid)

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test create_gradient_validator
        validator = create_gradient_validator(ValidationLevel.STRICT)
        self.assertIsInstance(validator, GradientValidator)
        self.assertEqual(validator.config.validation_level, ValidationLevel.STRICT)

        # Test validate_federated_gradients
        result = validate_federated_gradients(self.valid_gradients, "convenience_client")
        self.assertIsInstance(result, GradientValidationResult)
        self.assertTrue(result.is_valid)

class TestGradientValidationConfig(unittest.TestCase):
    """Test GradientValidationConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = GradientValidationConfig()

        self.assertEqual(config.validation_level, ValidationLevel.STRICT)
        self.assertEqual(config.max_gradient_value, 1000.0)
        self.assertEqual(config.min_gradient_value, -1000.0)
        self.assertTrue(config.enable_statistical_analysis)
        self.assertTrue(config.enable_adversarial_detection)

    def test_custom_config(self):
        """Test custom configuration."""
        config = GradientValidationConfig(
            validation_level=ValidationLevel.BASIC,
            max_gradient_value=500.0,
            min_gradient_value=-500.0,
            enable_adversarial_detection=False,
            outlier_threshold_sigma=2.5
        )

        self.assertEqual(config.validation_level, ValidationLevel.BASIC)
        self.assertEqual(config.max_gradient_value, 500.0)
        self.assertEqual(config.min_gradient_value, -500.0)
        self.assertFalse(config.enable_adversarial_detection)
        self.assertEqual(config.outlier_threshold_sigma, 2.5)

class TestAdversarialPattern(unittest.TestCase):
    """Test AdversarialPattern enum."""

    def test_adversarial_patterns(self):
        """Test adversarial pattern definitions."""
        patterns = [
            AdversarialPattern.GRADIENT_EXPLOSION,
            AdversarialPattern.GRADIENT_VANISHING,
            AdversarialPattern.GRADIENT_NAN_INF,
            AdversarialPattern.GRADIENT_UNIFORM,
            AdversarialPattern.GRADIENT_ZERO,
            AdversarialPattern.GRADIENT_OUTLIER,
            AdversarialPattern.GRADIENT_PERIODIC,
            AdversarialPattern.GRADIENT_NOISY,
            AdversarialPattern.GRADIENT_INVERTED,
            AdversarialPattern.GRADIENT_SCALING_ATTACK
        ]

        # Ensure all patterns are defined
        for pattern in patterns:
            self.assertIsInstance(pattern, AdversarialPattern)
            self.assertIsInstance(pattern.value, str)

class TestGradientStatistics(unittest.TestCase):
    """Test GradientStatistics class."""

    def test_statistics_creation(self):
        """Test gradient statistics creation."""
        stats = GradientStatistics(
            mean=0.5,
            std=0.2,
            min_val=-1.0,
            max_val=2.0,
            median=0.4,
            skewness=0.1,
            kurtosis=1.5,
            outlier_count=2,
            nan_count=0,
            inf_count=0,
            zero_count=1,
            total_elements=100,
            shape=(10, 10),
            dtype="float32"
        )

        self.assertEqual(stats.mean, 0.5)
        self.assertEqual(stats.std, 0.2)
        self.assertEqual(stats.total_elements, 100)
        self.assertEqual(stats.shape, (10, 10))
        self.assertEqual(stats.dtype, "float32")

class TestValidationLevel(unittest.TestCase):
    """Test ValidationLevel enum."""

    def test_validation_levels(self):
        """Test validation level definitions."""
        levels = [
            ValidationLevel.BASIC,
            ValidationLevel.STANDARD,
            ValidationLevel.STRICT,
            ValidationLevel.PARANOID
        ]

        for level in levels:
            self.assertIsInstance(level, ValidationLevel)
            self.assertIsInstance(level.value, str)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in gradient validation."""

    def setUp(self):
        self.validator = GradientValidator()

    def test_gradient_validation_error(self):
        """Test GradientValidationError handling."""
        with self.assertRaises(GradientValidationError):
            raise BoundsViolationError("Test bounds violation")

    def test_bounds_violation_error(self):
        """Test BoundsViolationError."""
        with self.assertRaises(BoundsViolationError):
            raise BoundsViolationError("Gradient bounds exceeded")

    def test_shape_violation_error(self):
        """Test ShapeViolationError."""
        with self.assertRaises(ShapeViolationError):
            raise ShapeViolationError("Invalid gradient shape")

    def test_type_violation_error(self):
        """Test TypeViolationError."""
        with self.assertRaises(TypeViolationError):
            raise TypeViolationError("Invalid gradient type")

    def test_adversarial_input_error(self):
        """Test AdversarialInputError."""
        with self.assertRaises(AdversarialInputError):
            raise AdversarialInputError("Adversarial input detected")

    def test_data_poisoning_error(self):
        """Test DataPoisoningError."""
        with self.assertRaises(DataPoisoningError):
            raise DataPoisoningError("Data poisoning detected")

if __name__ == '__main__':
    unittest.main()

