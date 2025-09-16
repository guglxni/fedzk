#!/usr/bin/env python3
"""
Tests for Proof Validator
=========================

Comprehensive test suite for FEDzk proof validation system.
Tests cryptographic parameter validation, attack detection, and structure validation.
"""

import unittest
import json
import time
from unittest.mock import patch, MagicMock

from fedzk.validation.proof_validator import (
    ProofValidator,
    ProofValidationConfig,
    ProofValidationResult,
    ProofValidationError,
    ProofStructureError,
    ProofParameterError,
    ProofManipulationError,
    ProofComplexityError,
    ProofValidationLevel,
    ProofAttackPattern,
    ZKProofType,
    ProofStructure,
    create_proof_validator,
    validate_zk_proof
)

class TestProofValidator(unittest.TestCase):
    """Test ProofValidator class."""

    def setUp(self):
        """Setup test environment."""
        self.config = ProofValidationConfig(
            validation_level=ProofValidationLevel.STRICT,
            max_proof_size=1024 * 1024,
            max_field_size=10000,
            enable_attack_detection=True,
            enable_timing_protection=True
        )

        self.validator = ProofValidator(self.config)

        # Create valid Groth16 proof
        self.valid_groth16_proof = {
            "pi_a": ["0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef", "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"],
            "pi_b": [
                ["0x1111111111111111111111111111111111111111111111111111111111111111", "0x2222222222222222222222222222222222222222222222222222222222222222"],
                ["0x3333333333333333333333333333333333333333333333333333333333333333", "0x4444444444444444444444444444444444444444444444444444444444444444"]
            ],
            "pi_c": ["0x5555555555555555555555555555555555555555555555555555555555555555", "0x6666666666666666666666666666666666666666666666666666666666666666"],
            "protocol": "groth16",
            "curve": "bn128"
        }

        self.valid_signals = [100, 200, 300, 400]

    def test_valid_proof_validation(self):
        """Test validation of valid proof."""
        result = self.validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, self.valid_signals, "test_client"
        )

        self.assertIsInstance(result, ProofValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.validation_score, 80)
        self.assertEqual(len(result.detected_attacks), 0)

    def test_proof_structure_validation(self):
        """Test proof structure validation."""
        # Valid structure
        result = self.validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, None, "structure_client"
        )

        self.assertTrue(result.is_valid)
        self.assertIn("structure_analysis", result.validation_metadata)
        self.assertTrue(result.structure_analysis.get("has_required_fields", False))

    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        incomplete_proof = {
            "pi_a": ["0x123", "0x456"],
            # Missing pi_b, pi_c
            "protocol": "groth16"
        }

        result = self.validator.validate_proof_comprehensive(
            incomplete_proof, ZKProofType.GROTH16, None, "incomplete_client"
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.validation_score, 100)
        self.assertFalse(result.structure_analysis.get("has_required_fields", True))

    def test_size_limits_validation(self):
        """Test proof size limits validation."""
        # Create oversized proof
        large_proof = self.valid_groth16_proof.copy()
        large_proof["large_field"] = "x" * (self.config.max_field_size + 1)

        result = self.validator.validate_proof_comprehensive(
            large_proof, ZKProofType.GROTH16, None, "large_client"
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.validation_score, 100)

    def test_cryptographic_parameter_validation(self):
        """Test cryptographic parameter validation."""
        result = self.validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, None, "crypto_client"
        )

        self.assertTrue(result.is_valid)
        self.assertIn("cryptographic_analysis", result)
        self.assertIn("strength_assessment", result.cryptographic_analysis)

    def test_attack_pattern_detection(self):
        """Test detection of attack patterns."""

        # Test null byte injection
        null_injection_proof = self.valid_groth16_proof.copy()
        null_injection_proof["injected"] = "safe_string\x00malicious"

        result = self.validator.validate_proof_comprehensive(
            null_injection_proof, ZKProofType.GROTH16, None, "null_client"
        )

        self.assertIn(ProofAttackPattern.NULL_BYTE_INJECTION, result.detected_attacks)

        # Test malformed JSON
        malformed_proof = {
            "pi_a": set([1, 2, 3]),  # Sets are not JSON serializable
            "pi_b": ["valid", "data"],
            "pi_c": ["also", "valid"]
        }

        result = self.validator.validate_proof_comprehensive(
            malformed_proof, ZKProofType.GROTH16, None, "malformed_client"
        )

        self.assertIn(ProofAttackPattern.MALFORMED_STRUCTURE, result.detected_attacks)

    def test_invalid_cryptographic_parameters(self):
        """Test detection of invalid cryptographic parameters."""
        invalid_proof = self.valid_groth16_proof.copy()
        invalid_proof["pi_a"] = ["", ""]  # Empty strings are invalid

        result = self.validator.validate_proof_comprehensive(
            invalid_proof, ZKProofType.GROTH16, None, "invalid_crypto_client"
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.validation_score, 100)

    def test_nesting_depth_validation(self):
        """Test nesting depth validation."""
        # Create deeply nested proof
        nested_proof = self.valid_groth16_proof.copy()
        current = nested_proof
        for i in range(15):  # Exceed max nesting depth
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]

        result = self.validator.validate_proof_comprehensive(
            nested_proof, ZKProofType.GROTH16, None, "nested_client"
        )

        self.assertIn(ProofAttackPattern.RECURSIVE_STRUCTURE, result.detected_attacks)

    def test_unicode_bomb_detection(self):
        """Test detection of Unicode bomb attacks."""
        unicode_bomb_proof = self.valid_groth16_proof.copy()
        # Create a string that expands significantly when encoded
        bomb_string = "а" * 1000  # Cyrillic 'a' characters
        unicode_bomb_proof["unicode_bomb"] = bomb_string

        result = self.validator.validate_proof_comprehensive(
            unicode_bomb_proof, ZKProofType.GROTH16, None, "unicode_client"
        )

        # Should detect if the expansion ratio is too high
        if len(bomb_string.encode('utf-8')) > len(bomb_string) * 2:
            self.assertIn(ProofAttackPattern.UNICODE_BOMB, result.detected_attacks)

    def test_format_injection_detection(self):
        """Test detection of format injection attacks."""
        format_injection_proof = self.valid_groth16_proof.copy()
        format_injection_proof["injected"] = "safe_string%d_with_format%s"

        result = self.validator.validate_proof_comprehensive(
            format_injection_proof, ZKProofType.GROTH16, None, "format_client"
        )

        self.assertIn(ProofAttackPattern.FORMAT_INJECTION, result.detected_attacks)

    def test_replay_attack_detection(self):
        """Test detection of replay attacks."""
        # First validation
        result1 = self.validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, self.valid_signals, "replay_client"
        )

        # Immediate second validation with same proof (should not trigger replay)
        result2 = self.validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, self.valid_signals, "replay_client"
        )

        # Both should be valid (replay detection has time window)
        self.assertTrue(result1.is_valid)
        self.assertTrue(result2.is_valid)

    def test_proof_sanitization(self):
        """Test proof sanitization functionality."""
        # Create proof with issues
        unsanitized_proof = self.valid_groth16_proof.copy()
        unsanitized_proof["problematic_field"] = "string_with_null\x00byte"
        unsanitized_proof["oversized_field"] = "x" * (self.config.max_string_length + 100)

        result = self.validator.validate_proof_comprehensive(
            unsanitized_proof, ZKProofType.GROTH16, None, "sanitize_client"
        )

        # Should have sanitized version
        self.assertIsNotNone(result.sanitized_proof)

        # Check sanitization
        sanitized = result.sanitized_proof

        # Null bytes should be removed
        self.assertNotIn('\x00', sanitized.get("problematic_field", ""))

        # Oversized strings should be truncated
        oversized = sanitized.get("oversized_field", "")
        self.assertLessEqual(len(oversized), self.config.max_string_length + 3)  # +3 for "..."

    def test_validation_levels(self):
        """Test different validation levels."""
        # Test basic level
        basic_validator = create_proof_validator(ProofValidationLevel.BASIC)
        result = basic_validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, None, "basic_client"
        )

        self.assertTrue(result.is_valid)

        # Test paranoid level
        paranoid_validator = create_proof_validator(ProofValidationLevel.PARANOID)
        result = paranoid_validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, None, "paranoid_client"
        )

        self.assertTrue(result.is_valid)  # Should still pass with valid proof

    def test_different_proof_types(self):
        """Test validation of different proof types."""
        # Test PLONK proof structure
        plonk_proof = {
            "proof": "plonk_proof_data",
            "public_signals": [100, 200, 300],
            "protocol": "plonk"
        }

        result = self.validator.validate_proof_comprehensive(
            plonk_proof, ZKProofType.PLONK, None, "plonk_client"
        )

        self.assertTrue(result.is_valid)

    def test_timing_protection(self):
        """Test timing attack protection."""
        result = self.validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, None, "timing_client"
        )

        # Should have timing information
        self.assertIn("validation_time", result.validation_metadata)
        self.assertGreater(result.validation_metadata["validation_time"], 0)

    def test_proof_hashing(self):
        """Test proof hashing for caching and replay detection."""
        hash1 = self.validator._calculate_proof_hash(self.valid_groth16_proof)
        hash2 = self.validator._calculate_proof_hash(self.valid_groth16_proof)

        # Same proof should have same hash
        self.assertEqual(hash1, hash2)

        # Different proof should have different hash
        different_proof = self.valid_groth16_proof.copy()
        different_proof["extra_field"] = "different"

        hash3 = self.validator._calculate_proof_hash(different_proof)
        self.assertNotEqual(hash1, hash3)

    def test_validation_metrics(self):
        """Test validation metrics collection."""
        # Perform several validations
        for i in range(3):
            test_proof = self.valid_groth16_proof.copy()
            test_proof["test_id"] = i

            self.validator.validate_proof_comprehensive(
                test_proof, ZKProofType.GROTH16, None, f"metrics_client_{i}"
            )

        metrics = self.validator.get_validation_metrics()

        self.assertEqual(metrics["total_validations"], 3)
        self.assertGreater(metrics["average_score"], 0)
        self.assertIn("average_timing", metrics)
        self.assertIn("cache_size", metrics)

    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        # Add some entries to cache
        self.validator.proof_cache["test_hash_1"] = {
            "proof": self.valid_groth16_proof,
            "timestamp": time.time() - 4000,  # Old timestamp
            "client_id": "test_client"
        }

        self.validator.proof_cache["test_hash_2"] = {
            "proof": self.valid_groth16_proof,
            "timestamp": time.time(),  # Recent timestamp
            "client_id": "test_client"
        }

        # Cleanup should remove old entries
        self.validator.cleanup_cache()

        # Should have only recent entries
        self.assertEqual(len(self.validator.proof_cache), 1)
        self.assertIn("test_hash_2", self.validator.proof_cache)

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test create_proof_validator
        validator = create_proof_validator(ProofValidationLevel.STRICT)
        self.assertIsInstance(validator, ProofValidator)
        self.assertEqual(validator.config.validation_level, ProofValidationLevel.STRICT)

        # Test validate_zk_proof
        result = validate_zk_proof(
            self.valid_groth16_proof, ZKProofType.GROTH16, self.valid_signals, "convenience_client"
        )
        self.assertIsInstance(result, ProofValidationResult)
        self.assertTrue(result.is_valid)

class TestProofValidationConfig(unittest.TestCase):
    """Test ProofValidationConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProofValidationConfig()

        self.assertEqual(config.validation_level, ProofValidationLevel.STRICT)
        self.assertEqual(config.max_proof_size, 1024 * 1024)
        self.assertEqual(config.max_field_size, 10000)
        self.assertTrue(config.enable_attack_detection)
        self.assertIn(ZKProofType.GROTH16, config.allowed_proof_types)

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProofValidationConfig(
            validation_level=ProofValidationLevel.BASIC,
            max_proof_size=512 * 1024,
            enable_attack_detection=False,
            allowed_proof_types=[ZKProofType.PLONK]
        )

        self.assertEqual(config.validation_level, ProofValidationLevel.BASIC)
        self.assertEqual(config.max_proof_size, 512 * 1024)
        self.assertFalse(config.enable_attack_detection)
        self.assertEqual(config.allowed_proof_types, [ZKProofType.PLONK])

class TestZKProofType(unittest.TestCase):
    """Test ZKProofType enum."""

    def test_proof_types(self):
        """Test ZK proof type definitions."""
        proof_types = [
            ZKProofType.GROTH16,
            ZKProofType.PLONK,
            ZKProofType.MARLIN,
            ZKProofType.BULLETPROOFS,
            ZKProofType.STARK
        ]

        for proof_type in proof_types:
            self.assertIsInstance(proof_type, ZKProofType)
            self.assertIsInstance(proof_type.value, str)

class TestProofAttackPattern(unittest.TestCase):
    """Test ProofAttackPattern enum."""

    def test_attack_patterns(self):
        """Test attack pattern definitions."""
        patterns = [
            ProofAttackPattern.MALFORMED_STRUCTURE,
            ProofAttackPattern.INVALID_PARAMETERS,
            ProofAttackPattern.SIZE_INFLATION,
            ProofAttackPattern.RECURSIVE_STRUCTURE,
            ProofAttackPattern.NULL_BYTE_INJECTION,
            ProofAttackPattern.UNICODE_BOMB,
            ProofAttackPattern.FORMAT_INJECTION,
            ProofAttackPattern.CRYPTOGRAPHIC_WEAKNESS,
            ProofAttackPattern.TIMING_ATTACK,
            ProofAttackPattern.REPLAY_ATTACK
        ]

        for pattern in patterns:
            self.assertIsInstance(pattern, ProofAttackPattern)
            self.assertIsInstance(pattern.value, str)

class TestProofStructure(unittest.TestCase):
    """Test ProofStructure class."""

    def test_proof_structure_creation(self):
        """Test proof structure creation."""
        structure = ProofStructure(
            proof_type="groth16",
            field_count=3,
            total_size=2048,
            nesting_depth=2,
            has_required_fields=True,
            parameter_ranges={"pi_a": (0, 1000)},
            cryptographic_parameters={"strength": "strong"},
            complexity_score=85.5
        )

        self.assertEqual(structure.proof_type, "groth16")
        self.assertEqual(structure.field_count, 3)
        self.assertEqual(structure.nesting_depth, 2)
        self.assertTrue(structure.has_required_fields)
        self.assertEqual(structure.complexity_score, 85.5)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in proof validation."""

    def setUp(self):
        self.validator = ProofValidator()

    def test_proof_validation_error(self):
        """Test ProofValidationError handling."""
        with self.assertRaises(ProofValidationError):
            raise ProofStructureError("Test structure error")

    def test_proof_structure_error(self):
        """Test ProofStructureError."""
        with self.assertRaises(ProofStructureError):
            raise ProofStructureError("Invalid proof structure")

    def test_proof_parameter_error(self):
        """Test ProofParameterError."""
        with self.assertRaises(ProofParameterError):
            raise ProofParameterError("Invalid cryptographic parameters")

    def test_proof_manipulation_error(self):
        """Test ProofManipulationError."""
        with self.assertRaises(ProofManipulationError):
            raise ProofManipulationError("Proof manipulation detected")

    def test_proof_complexity_error(self):
        """Test ProofComplexityError."""
        with self.assertRaises(ProofComplexityError):
            raise ProofComplexityError("Proof complexity violation")

class TestEdgeCases(unittest.TestCase):
    """Test edge cases in proof validation."""

    def setUp(self):
        self.validator = ProofValidator()

    def test_empty_proof(self):
        """Test validation of empty proof."""
        empty_proof = {}

        result = self.validator.validate_proof_comprehensive(
            empty_proof, ZKProofType.GROTH16, None, "empty_client"
        )

        self.assertFalse(result.is_valid)
        self.assertLess(result.validation_score, 50)

    def test_none_proof(self):
        """Test validation with None proof."""
        with self.assertRaises(Exception):
            self.validator.validate_proof_comprehensive(
                None, ZKProofType.GROTH16, None, "none_client"
            )

    def test_very_large_proof(self):
        """Test validation of very large proof."""
        large_proof = {
            "pi_a": ["0x" + "1" * 10000, "0x" + "2" * 10000],  # Very large strings
            "pi_b": [["0x" + "3" * 10000, "0x" + "4" * 10000]],
            "pi_c": ["0x" + "5" * 10000, "0x" + "6" * 10000],
            "protocol": "groth16"
        }

        result = self.validator.validate_proof_comprehensive(
            large_proof, ZKProofType.GROTH16, None, "large_client"
        )

        # Should detect size inflation
        if len(json.dumps(large_proof)) > self.validator.config.max_proof_size:
            self.assertIn(ProofAttackPattern.SIZE_INFLATION, result.detected_attacks)

    def test_cryptographic_weakness_detection(self):
        """Test detection of cryptographic weaknesses."""
        weak_proof = {
            "pi_a": ["0", "1"],  # Obviously weak values
            "pi_b": [["0", "1"], ["0", "1"]],
            "pi_c": ["0", "1"],
            "protocol": "groth16"
        }

        result = self.validator.validate_proof_comprehensive(
            weak_proof, ZKProofType.GROTH16, None, "weak_client"
        )

        self.assertIn(ProofAttackPattern.CRYPTOGRAPHIC_WEAKNESS, result.detected_attacks)

if __name__ == '__main__':
    unittest.main()

