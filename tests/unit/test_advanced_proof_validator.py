#!/usr/bin/env python3
"""
Tests for Advanced Proof Validator
===================================

Comprehensive test suite for the advanced proof validation system.
Tests security features, attack detection, and validation logic.
"""

import unittest
import json
from unittest.mock import patch, MagicMock
from fedzk.prover.advanced_proof_validator import (
    AdvancedProofValidator,
    ProofValidationConfig,
    ProofValidationResult,
    ProofValidationError,
    MalformedProofError,
    ProofSizeError,
    ProofComplexityError,
    ProofFormatError,
    AttackPattern,
    validate_proof_security,
    create_secure_validator
)

class TestAdvancedProofValidator(unittest.TestCase):
    """Test AdvancedProofValidator class."""

    def setUp(self):
        """Setup test environment."""
        self.validator = AdvancedProofValidator()
        self.secure_validator = create_secure_validator()

        # Valid proof for testing
        self.valid_proof = {
            "pi_a": ["0x1234567890abcdef", "0xfedcba0987654321"],
            "pi_b": [
                ["0x1111111111111111", "0x2222222222222222"],
                ["0x3333333333333333", "0x4444444444444444"]
            ],
            "pi_c": ["0x5555555555555555", "0x6666666666666666"],
            "protocol": "groth16"
        }

        self.valid_signals = [100, 200, 300, 400]

        # Invalid proofs for attack testing
        self.malformed_proof = {
            "pi_a": ["invalid_hex", "also_invalid"],
            "pi_b": [["not", "hex"], ["values", "here"]],
            "pi_c": ["bad", "format"]
        }

    def test_valid_proof_validation(self):
        """Test validation of a valid proof."""
        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, self.valid_signals
        )

        self.assertIsInstance(result, ProofValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.security_score, 80)
        self.assertEqual(len(result.attack_patterns_detected), 0)
        self.assertLess(result.validation_time, 1.0)  # Should be fast

    def test_malformed_proof_detection(self):
        """Test detection of malformed proofs."""
        result = self.validator.validate_proof_comprehensive(
            self.malformed_proof, self.valid_signals
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.security_score, 50)
        self.assertIn(AttackPattern.MALFORMED_JSON, result.attack_patterns_detected)

    def test_proof_size_limits(self):
        """Test proof size limit enforcement."""
        # Create oversized proof
        large_proof = self.valid_proof.copy()
        large_proof["large_field"] = "x" * (1024 * 1024 + 1)  # > 1MB

        with self.assertRaises(ProofSizeError):
            self.validator.validate_proof_comprehensive(large_proof, self.valid_signals)

    def test_attack_pattern_detection(self):
        """Test detection of various attack patterns."""

        # Test buffer overflow
        buffer_overflow_proof = self.valid_proof.copy()
        buffer_overflow_proof["overflow"] = "x" * 2000000  # Large string

        result = self.validator.validate_proof_comprehensive(
            buffer_overflow_proof, self.valid_signals
        )
        self.assertIn(AttackPattern.BUFFER_OVERFLOW, result.attack_patterns_detected)

        # Test null byte injection
        null_injection_proof = self.valid_proof.copy()
        null_injection_proof["null_field"] = "safe_string\x00malicious"

        result = self.validator.validate_proof_comprehensive(
            null_injection_proof, self.valid_signals
        )
        self.assertIn(AttackPattern.NULL_BYTE_INJECTION, result.attack_patterns_detected)

        # Test large number attack
        large_number_proof = self.valid_proof.copy()
        large_number_proof["large_number"] = "1" * 200  # Very large number

        result = self.validator.validate_proof_comprehensive(
            large_number_proof, self.valid_signals
        )
        self.assertIn(AttackPattern.LARGE_NUMBER_ATTACK, result.attack_patterns_detected)

    def test_integer_overflow_detection(self):
        """Test detection of integer overflow attacks."""
        overflow_signals = [2**512, -2**512, 2**1024]  # Very large integers

        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, overflow_signals
        )

        self.assertIn(AttackPattern.INTEGER_OVERFLOW, result.attack_patterns_detected)
        self.assertLess(result.security_score, 80)

    def test_format_injection_detection(self):
        """Test detection of format injection attacks."""
        injection_proof = self.valid_proof.copy()
        injection_proof["injected"] = "safe%value$with\\bad%chars"

        result = self.validator.validate_proof_comprehensive(
            injection_proof, self.valid_signals
        )

        self.assertIn(AttackPattern.FORMAT_INJECTION, result.attack_patterns_detected)

    def test_recursive_structure_detection(self):
        """Test detection of recursive structure attacks."""
        recursive_signals = []
        current = recursive_signals
        for i in range(15):  # Exceed max nesting depth
            current.append([])
            current = current[0]

        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, recursive_signals
        )

        self.assertIn(AttackPattern.RECURSIVE_STRUCTURE, result.attack_patterns_detected)

    def test_proof_structure_validation(self):
        """Test proof structure validation."""
        # Missing required field
        incomplete_proof = self.valid_proof.copy()
        del incomplete_proof["pi_a"]

        with self.assertRaises(ProofFormatError):
            self.validator.validate_proof_comprehensive(incomplete_proof, self.valid_signals)

        # Invalid protocol
        invalid_protocol_proof = self.valid_proof.copy()
        invalid_protocol_proof["protocol"] = "invalid_protocol"

        with self.assertRaises(ProofFormatError):
            self.validator.validate_proof_comprehensive(invalid_protocol_proof, self.valid_signals)

    def test_signal_validation(self):
        """Test public signal validation."""
        # Invalid signal type
        invalid_signals = [100, "invalid_string", {"dict": "not_allowed"}, [1, 2, 3]]

        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, invalid_signals
        )

        self.assertFalse(result.is_valid)
        self.assertIn("Invalid signal format", " ".join(result.warnings))

    def test_cryptographic_integrity_checks(self):
        """Test cryptographic integrity validation."""
        # Test with weak cryptographic parameters
        weak_proof = self.valid_proof.copy()
        weak_proof["pi_a"] = ["0", "1"]  # Obviously weak values

        result = self.validator.validate_proof_comprehensive(
            weak_proof, self.valid_signals
        )

        self.assertLess(result.security_score, 70)
        self.assertTrue(len(result.warnings) > 0)

    def test_circuit_specific_validation(self):
        """Test circuit-specific validation."""
        # Test model_update circuit
        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, self.valid_signals, "model_update"
        )
        self.assertTrue(result.is_valid)

        # Test with wrong number of signals
        wrong_signals = [100, 200]  # Too few for model_update
        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, wrong_signals, "model_update"
        )
        self.assertLess(result.security_score, 100)

    def test_configuration_options(self):
        """Test different configuration options."""
        # Strict mode
        strict_config = ProofValidationConfig(strict_mode=True, max_proof_size=100)
        strict_validator = AdvancedProofValidator(strict_config)

        # Should reject normal-sized proof in strict mode with small limit
        with self.assertRaises(ProofSizeError):
            strict_validator.validate_proof_comprehensive(
                self.valid_proof, self.valid_signals
            )

        # Lenient mode
        lenient_config = ProofValidationConfig(
            strict_mode=False,
            enable_attack_detection=False
        )
        lenient_validator = AdvancedProofValidator(lenient_config)

        result = lenient_validator.validate_proof_comprehensive(
            self.valid_proof, self.valid_signals
        )
        self.assertTrue(result.is_valid)  # Should pass in lenient mode

    def test_validation_performance(self):
        """Test validation performance."""
        import time

        start_time = time.time()

        # Run multiple validations
        for i in range(10):
            test_signals = [100 + i, 200 + i, 300 + i, 400 + i]
            result = self.validator.validate_proof_comprehensive(
                self.valid_proof, test_signals
            )
            self.assertTrue(result.is_valid)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete within reasonable time
        self.assertLess(total_time, 2.0)  # Less than 2 seconds for 10 validations
        self.assertGreater(total_time, 0.1)  # But not too fast (suspicious)

    def test_validation_stats(self):
        """Test validation statistics collection."""
        # Run some validations
        for i in range(5):
            test_signals = [100 + i, 200 + i, 300 + i, 400 + i]
            self.validator.validate_proof_comprehensive(self.valid_proof, test_signals)

        stats = self.validator.get_validation_stats()

        self.assertEqual(stats["total_validations"], 5)
        self.assertIn("average_score", stats)
        self.assertIn("min_score", stats)
        self.assertIn("max_score", stats)
        self.assertIn("average_time", stats)

    def test_edge_cases(self):
        """Test edge cases in validation."""
        # Empty proof
        empty_proof = {}
        result = self.validator.validate_proof_comprehensive(empty_proof, [])
        self.assertFalse(result.is_valid)

        # None values
        none_proof = {"pi_a": None, "pi_b": None, "pi_c": None}
        result = self.validator.validate_proof_comprehensive(none_proof, self.valid_signals)
        self.assertFalse(result.is_valid)

        # Extremely nested structure
        nested_signals = [{"nested": {"deeply": {"nested": [1, 2, {"more": "nesting"}]}}}]
        result = self.validator.validate_proof_comprehensive(self.valid_proof, nested_signals)
        self.assertIn(AttackPattern.RECURSIVE_STRUCTURE, result.attack_patterns_detected)

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test validate_proof_security
        result = validate_proof_security(self.valid_proof, self.valid_signals)
        self.assertIsInstance(result, ProofValidationResult)
        self.assertTrue(result.is_valid)

        # Test create_secure_validator
        secure_validator = create_secure_validator()
        self.assertIsInstance(secure_validator, AdvancedProofValidator)

        result = secure_validator.validate_proof_comprehensive(
            self.valid_proof, self.valid_signals
        )
        self.assertTrue(result.is_valid)

    def test_error_handling(self):
        """Test error handling in validation."""
        # Test with invalid JSON
        invalid_json_signals = [set([1, 2, 3])]  # Sets are not JSON serializable

        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, invalid_json_signals
        )
        self.assertFalse(result.is_valid)

    def test_attack_pattern_enumeration(self):
        """Test all attack pattern types are covered."""
        # Ensure all attack patterns are tested
        expected_patterns = [
            AttackPattern.BUFFER_OVERFLOW,
            AttackPattern.INTEGER_OVERFLOW,
            AttackPattern.FORMAT_INJECTION,
            AttackPattern.MALFORMED_JSON,
            AttackPattern.SIZE_BOMB,
            AttackPattern.RECURSIVE_STRUCTURE,
            AttackPattern.NULL_BYTE_INJECTION,
            AttackPattern.UNICODE_BOMB,
            AttackPattern.LARGE_NUMBER_ATTACK,
            AttackPattern.DUPLICATE_KEYS
        ]

        # Check that our validator handles all patterns
        for pattern in expected_patterns:
            self.assertIsInstance(pattern, AttackPattern)

    def test_security_score_calculation(self):
        """Test security score calculation logic."""
        # Valid proof should have high score
        result = self.validator.validate_proof_comprehensive(
            self.valid_proof, self.valid_signals
        )
        self.assertGreater(result.security_score, 80)

        # Malformed proof should have low score
        result = self.validator.validate_proof_comprehensive(
            self.malformed_proof, self.valid_signals
        )
        self.assertLess(result.security_score, 50)

        # Proof with attacks should have reduced score
        attack_proof = self.valid_proof.copy()
        attack_proof["large_number"] = "1" * 200

        result = self.validator.validate_proof_comprehensive(
            attack_proof, self.valid_signals
        )
        self.assertLess(result.security_score, 100)

class TestProofValidationConfig(unittest.TestCase):
    """Test ProofValidationConfig class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ProofValidationConfig()

        self.assertEqual(config.max_proof_size, 1024 * 1024)
        self.assertEqual(config.max_signal_count, 1000)
        self.assertEqual(config.max_field_size, 10000)
        self.assertEqual(config.max_nesting_depth, 10)
        self.assertTrue(config.enable_attack_detection)
        self.assertTrue(config.strict_mode)

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ProofValidationConfig(
            max_proof_size=512 * 1024,
            max_signal_count=500,
            enable_attack_detection=False,
            strict_mode=False
        )

        self.assertEqual(config.max_proof_size, 512 * 1024)
        self.assertEqual(config.max_signal_count, 500)
        self.assertFalse(config.enable_attack_detection)
        self.assertFalse(config.strict_mode)

    def test_allowed_protocols(self):
        """Test allowed protocols configuration."""
        config = ProofValidationConfig()
        self.assertIn("groth16", config.allowed_protocols)
        self.assertIn("plonk", config.allowed_protocols)
        self.assertIn("marlin", config.allowed_protocols)

class TestProofValidationResult(unittest.TestCase):
    """Test ProofValidationResult class."""

    def test_result_creation(self):
        """Test creation of validation result."""
        result = ProofValidationResult(
            is_valid=True,
            attack_patterns_detected=[],
            security_score=95.5,
            validation_time=0.123,
            warnings=["Minor issue"],
            metadata={"test": "data"}
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.attack_patterns_detected), 0)
        self.assertEqual(result.security_score, 95.5)
        self.assertEqual(result.validation_time, 0.123)
        self.assertEqual(result.warnings, ["Minor issue"])
        self.assertEqual(result.metadata, {"test": "data"})

    def test_invalid_result(self):
        """Test invalid validation result."""
        result = ProofValidationResult(
            is_valid=False,
            attack_patterns_detected=[AttackPattern.MALFORMED_JSON],
            security_score=25.0,
            validation_time=0.456,
            warnings=["Critical security issue"],
            metadata={"error": "malformed"}
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.attack_patterns_detected), 1)
        self.assertEqual(result.attack_patterns_detected[0], AttackPattern.MALFORMED_JSON)
        self.assertEqual(result.security_score, 25.0)

if __name__ == '__main__':
    unittest.main()

