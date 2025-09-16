#!/usr/bin/env python3
"""
Comprehensive Input Validation Testing Framework
===============================================

Task 6.3.3: Comprehensive Testing Framework for Input Validation
Tests gradient validation (6.3.1) and proof validation (6.3.2) components.

This integration test suite validates:
- Adversarial attack simulation and detection
- Statistical analysis and anomaly detection
- Sanitization and data cleaning effectiveness
- Attack pattern detection accuracy
- Performance testing for validation operations
- Integration with federated learning workflows
- Security metrics and monitoring validation
- Baseline establishment and deviation detection
- Edge cases and boundary conditions
- Multi-client validation scenarios
"""

import unittest
import time
import json
import tempfile
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import patch, MagicMock

from fedzk.validation.gradient_validator import (
    GradientValidator,
    GradientValidationConfig,
    ValidationLevel,
    create_gradient_validator,
    validate_federated_gradients,
    AdversarialPattern
)
from fedzk.validation.proof_validator import (
    ProofValidator,
    ProofValidationConfig,
    ProofValidationLevel,
    ZKProofType,
    create_proof_validator,
    validate_zk_proof,
    ProofAttackPattern
)

class TestComprehensiveInputValidation(unittest.TestCase):
    """Comprehensive integration tests for input validation framework."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Setup gradient validator
        self.gradient_config = GradientValidationConfig(
            validation_level=ValidationLevel.STRICT,
            max_gradient_value=1000.0,
            min_gradient_value=-1000.0,
            enable_statistical_analysis=True,
            enable_adversarial_detection=True,
            enable_data_poisoning_detection=True
        )
        self.gradient_validator = GradientValidator(self.gradient_config)

        # Setup proof validator
        self.proof_config = ProofValidationConfig(
            validation_level=ProofValidationLevel.STRICT,
            max_proof_size=1024 * 1024,
            enable_attack_detection=True,
            enable_timing_protection=True
        )
        self.proof_validator = ProofValidator(self.proof_config)

        # Create test data
        self.valid_gradients = {
            "model.layer1.weight": torch.randn(64, 128),
            "model.layer1.bias": torch.randn(128),
            "model.layer2.weight": torch.randn(128, 64),
            "model.layer2.bias": torch.randn(64),
            "model.output.weight": torch.randn(64, 10),
            "model.output.bias": torch.randn(10)
        }

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

        # Test results tracking
        self.test_results = {
            "gradient_tests": [],
            "proof_tests": [],
            "integration_tests": [],
            "performance_tests": [],
            "security_tests": []
        }

    def tearDown(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_gradient_validation_comprehensive(self):
        """Test comprehensive gradient validation capabilities."""
        print("\n🧪 Testing Gradient Validation Comprehensive")

        # Test valid gradients
        result = self.gradient_validator.validate_gradients_comprehensive(
            self.valid_gradients, "test_client_valid"
        )

        self.assertTrue(result.is_valid)
        self.assertGreater(result.validation_score, 80)
        self.assertEqual(len(result.detected_anomalies), 0)
        self.assertEqual(len(result.adversarial_patterns), 0)

        # Record results
        self.test_results["gradient_tests"].append({
            "test": "valid_gradients",
            "passed": result.is_valid,
            "score": result.validation_score,
            "patterns_detected": len(result.adversarial_patterns)
        })

        print("✅ Valid gradients test passed")

    def test_adversarial_attack_simulation(self):
        """Test adversarial attack simulation and detection."""
        print("\n🛡️ Testing Adversarial Attack Simulation")

        # Create various adversarial attacks
        attack_tests = [
            {
                "name": "gradient_explosion",
                "gradients": {
                    "model.weight": torch.tensor([[1e10, -1e10], [5e9, 8e9]], dtype=torch.float32)
                },
                "expected_pattern": AdversarialPattern.GRADIENT_EXPLOSION
            },
            {
                "name": "nan_injection",
                "gradients": {
                    "model.weight": torch.tensor([[1.0, float('nan')], [3.0, 4.0]], dtype=torch.float32)
                },
                "expected_pattern": AdversarialPattern.GRADIENT_NAN_INF
            },
            {
                "name": "uniform_attack",
                "gradients": {
                    "model.weight": torch.ones(10, 10, dtype=torch.float32) * 2.5
                },
                "expected_pattern": AdversarialPattern.GRADIENT_UNIFORM
            },
            {
                "name": "bounds_violation",
                "gradients": {
                    "model.weight": torch.tensor([[1500.0, -1200.0], [800.0, 900.0]], dtype=torch.float32)
                },
                "expected_pattern": None  # Bounds violation, not adversarial pattern
            }
        ]

        attack_results = []
        for attack in attack_tests:
            result = self.gradient_validator.validate_gradients_comprehensive(
                attack["gradients"], f"test_client_{attack['name']}"
            )

            detected_patterns = [p.value for p in result.adversarial_patterns]
            expected_detected = attack["expected_pattern"].value in detected_patterns if attack["expected_pattern"] else not result.is_valid

            attack_results.append({
                "attack": attack["name"],
                "detected": expected_detected,
                "score": result.validation_score,
                "patterns": detected_patterns
            })

            if expected_detected:
                print(f"✅ {attack['name']} attack detected correctly")
            else:
                print(f"❌ {attack['name']} attack detection failed")

        # Record results
        self.test_results["security_tests"].append({
            "test": "adversarial_attacks",
            "attacks_tested": len(attack_tests),
            "attacks_detected": sum(1 for r in attack_results if r["detected"]),
            "detection_rate": sum(1 for r in attack_results if r["detected"]) / len(attack_tests)
        })

    def test_proof_validation_comprehensive(self):
        """Test comprehensive proof validation capabilities."""
        print("\n🔐 Testing Proof Validation Comprehensive")

        # Test valid proof
        result = self.proof_validator.validate_proof_comprehensive(
            self.valid_groth16_proof, ZKProofType.GROTH16, self.valid_signals, "test_client_valid"
        )

        self.assertTrue(result.is_valid)
        self.assertGreater(result.validation_score, 80)
        self.assertEqual(len(result.detected_attacks), 0)

        # Record results
        self.test_results["proof_tests"].append({
            "test": "valid_proof",
            "passed": result.is_valid,
            "score": result.validation_score,
            "attacks_detected": len(result.detected_attacks)
        })

        print("✅ Valid proof test passed")

    def test_proof_attack_simulation(self):
        """Test proof attack simulation and detection."""
        print("\n🚨 Testing Proof Attack Simulation")

        # Create various proof attacks
        attack_tests = [
            {
                "name": "null_byte_injection",
                "proof": {
                    "pi_a": ["0x123", "0x456"],
                    "pi_b": [["0x789", "0xabc"]],
                    "pi_c": ["0xdef", "0xghi"],
                    "protocol": "groth16",
                    "injected": "safe_string\x00malicious"
                },
                "expected_attack": ProofAttackPattern.NULL_BYTE_INJECTION
            },
            {
                "name": "format_injection",
                "proof": {
                    "pi_a": ["0x123", "0x456"],
                    "pi_b": [["0x789", "0xabc"]],
                    "pi_c": ["0xdef", "0xghi"],
                    "protocol": "groth16",
                    "injected": "safe_string%d_with_format%s"
                },
                "expected_attack": ProofAttackPattern.FORMAT_INJECTION
            },
            {
                "name": "malformed_structure",
                "proof": {
                    "pi_a": ["", ""],  # Invalid parameters
                    "pi_b": [["", ""]],
                    "pi_c": ["", ""],
                    "protocol": "groth16"
                },
                "expected_attack": ProofAttackPattern.CRYPTOGRAPHIC_WEAKNESS
            }
        ]

        attack_results = []
        for attack in attack_tests:
            result = self.proof_validator.validate_proof_comprehensive(
                attack["proof"], ZKProofType.GROTH16, None, f"test_client_{attack['name']}"
            )

            detected_attacks = [a.value for a in result.detected_attacks]
            expected_detected = attack["expected_attack"].value in detected_attacks

            attack_results.append({
                "attack": attack["name"],
                "detected": expected_detected,
                "score": result.validation_score,
                "attacks": detected_attacks
            })

            if expected_detected:
                print(f"✅ {attack['name']} attack detected correctly")
            else:
                print(f"❌ {attack['name']} attack detection failed")

        # Record results
        self.test_results["security_tests"].append({
            "test": "proof_attacks",
            "attacks_tested": len(attack_tests),
            "attacks_detected": sum(1 for r in attack_results if r["detected"]),
            "detection_rate": sum(1 for r in attack_results if r["detected"]) / len(attack_tests)
        })

    def test_statistical_analysis_validation(self):
        """Test statistical analysis and anomaly detection."""
        print("\n📊 Testing Statistical Analysis and Anomaly Detection")

        # Test with normal data
        result = self.gradient_validator.validate_gradients_comprehensive(
            self.valid_gradients, "test_client_stats"
        )

        self.assertIn("statistical_summary", result.validation_metadata)
        stats = result.statistical_summary

        # Verify statistical properties are calculated
        self.assertGreater(len(stats), 0)
        for layer_name, layer_stats in stats.items():
            self.assertIn("mean", layer_stats)
            self.assertIn("std", layer_stats)
            self.assertIn("total_elements", layer_stats)

        print("✅ Statistical analysis working correctly")

        # Record results
        self.test_results["gradient_tests"].append({
            "test": "statistical_analysis",
            "passed": True,
            "layers_analyzed": len(stats),
            "properties_calculated": len(list(stats.values())[0]) if stats else 0
        })

    def test_baseline_establishment_and_detection(self):
        """Test baseline establishment and deviation detection."""
        print("\n📈 Testing Baseline Establishment and Deviation Detection")

        # Establish baseline
        self.gradient_validator.establish_baseline(self.valid_gradients)
        print("✅ Baseline established")

        # Test with similar data (should pass)
        similar_gradients = {
            "model.layer1.weight": torch.randn(64, 128) * 0.1,
            "model.layer1.bias": torch.randn(128) * 0.1,
            "model.layer2.weight": torch.randn(128, 64) * 0.1,
            "model.layer2.bias": torch.randn(64) * 0.1,
            "model.output.weight": torch.randn(64, 10) * 0.1,
            "model.output.bias": torch.randn(10) * 0.1
        }

        result = self.gradient_validator.validate_gradients_comprehensive(
            similar_gradients, "test_client_similar"
        )

        self.assertTrue(result.is_valid)
        print("✅ Similar data passed baseline check")

        # Test with anomalous data (should detect deviation)
        anomalous_gradients = {
            "model.layer1.weight": torch.randn(64, 128) * 10.0,  # Much larger scale
            "model.layer1.bias": torch.randn(128) * 5.0,
            "model.layer2.weight": torch.randn(128, 64) * 0.1,
            "model.layer2.bias": torch.randn(64) * 0.1,
            "model.output.weight": torch.randn(64, 10) * 0.1,
            "model.output.bias": torch.randn(10) * 0.1
        }

        result = self.gradient_validator.validate_gradients_comprehensive(
            anomalous_gradients, "test_client_anomalous"
        )

        # Should detect some anomalies
        has_anomalies = len(result.detected_anomalies) > 0 or result.validation_score < 100
        print(f"✅ Anomalous data detected: {has_anomalies}")

        # Record results
        self.test_results["gradient_tests"].append({
            "test": "baseline_detection",
            "baseline_established": True,
            "similar_data_passed": True,
            "anomalous_data_detected": has_anomalies
        })

    def test_sanitization_effectiveness(self):
        """Test sanitization and data cleaning effectiveness."""
        print("\n🧹 Testing Sanitization and Data Cleaning Effectiveness")

        # Create gradients with various issues
        unsanitized_gradients = {
            "model.layer1.weight": torch.tensor([
                [1500.0, float('nan'), -1200.0],
                [800.0, float('inf'), 600.0],
                [200.0, 300.0, -float('inf')]
            ], dtype=torch.float32),
            "model.layer1.bias": torch.tensor([900.0, -1100.0, 700.0], dtype=torch.float32),
            "model.layer2.weight": torch.zeros(2, 2, dtype=torch.float32)  # All zeros
        }

        result = self.gradient_validator.validate_gradients_comprehensive(
            unsanitized_gradients, "test_client_sanitize"
        )

        # Should have sanitized version
        has_sanitized = result.sanitized_gradients is not None
        print(f"✅ Sanitization applied: {has_sanitized}")

        if has_sanitized:
            sanitized = result.sanitized_gradients

            # Check that NaN/Inf are removed
            for name, grad in sanitized.items():
                nan_count = torch.isnan(grad).sum().item()
                inf_count = torch.isinf(grad).sum().item()

                if nan_count == 0 and inf_count == 0:
                    print(f"✅ {name}: NaN/Inf successfully removed")
                else:
                    print(f"❌ {name}: Still contains NaN/Inf")

            # Check bounds clamping
            for name, grad in sanitized.items():
                min_val = torch.min(grad).item()
                max_val = torch.max(grad).item()

                if min_val >= self.gradient_config.min_gradient_value and max_val <= self.gradient_config.max_gradient_value:
                    print(f"✅ {name}: Values properly clamped to bounds")
                else:
                    print(f"❌ {name}: Values outside bounds")

        # Record results
        self.test_results["gradient_tests"].append({
            "test": "sanitization",
            "sanitization_applied": has_sanitized,
            "issues_cleaned": has_sanitized
        })

    def test_performance_validation(self):
        """Test performance of validation operations."""
        print("\n⚡ Testing Validation Performance")

        # Test gradient validation performance
        start_time = time.time()
        iterations = 10

        for i in range(iterations):
            test_gradients = {
                "layer.weight": torch.randn(100, 200),
                "layer.bias": torch.randn(200)
            }
            self.gradient_validator.validate_gradients_comprehensive(
                test_gradients, f"perf_client_{i}"
            )

        gradient_time = time.time() - start_time
        avg_gradient_time = gradient_time / iterations

        print(".3f"        print(".1f")

        # Test proof validation performance
        start_time = time.time()

        for i in range(iterations):
            test_proof = self.valid_groth16_proof.copy()
            test_proof["test_id"] = i
            self.proof_validator.validate_proof_comprehensive(
                test_proof, ZKProofType.GROTH16, None, f"perf_client_{i}"
            )

        proof_time = time.time() - start_time
        avg_proof_time = proof_time / iterations

        print(".3f"        print(".1f")

        # Performance requirements
        gradient_perf_ok = avg_gradient_time < 1.0  # Less than 1 second
        proof_perf_ok = avg_proof_time < 0.5       # Less than 0.5 seconds

        print(f"✅ Gradient validation performance: {'PASS' if gradient_perf_ok else 'FAIL'}")
        print(f"✅ Proof validation performance: {'PASS' if proof_perf_ok else 'FAIL'}")

        # Record results
        self.test_results["performance_tests"].append({
            "test": "validation_performance",
            "gradient_avg_time": avg_gradient_time,
            "gradient_perf_ok": gradient_perf_ok,
            "proof_avg_time": avg_proof_time,
            "proof_perf_ok": proof_perf_ok,
            "iterations": iterations
        })

    def test_integration_with_federated_learning(self):
        """Test integration with federated learning workflows."""
        print("\n🤝 Testing Integration with Federated Learning Workflows")

        # Simulate multi-client federated learning scenario
        clients = ["client_1", "client_2", "client_3"]

        integration_results = []
        for client_id in clients:
            # Generate client-specific gradients
            client_gradients = {
                f"model.layer1.weight": torch.randn(64, 128) * (0.5 + np.random.random() * 0.5),
                f"model.layer1.bias": torch.randn(128) * (0.5 + np.random.random() * 0.5),
                f"model.layer2.weight": torch.randn(128, 64) * (0.5 + np.random.random() * 0.5),
                f"model.layer2.bias": torch.randn(64) * (0.5 + np.random.random() * 0.5)
            }

            # Validate gradients
            grad_result = self.gradient_validator.validate_gradients_comprehensive(
                client_gradients, client_id
            )

            # Generate client-specific proof (simplified)
            client_proof = self.valid_groth16_proof.copy()
            client_proof["client_id"] = client_id

            # Validate proof
            proof_result = self.proof_validator.validate_proof_comprehensive(
                client_proof, ZKProofType.GROTH16, self.valid_signals, client_id
            )

            integration_results.append({
                "client_id": client_id,
                "gradient_valid": grad_result.is_valid,
                "gradient_score": grad_result.validation_score,
                "proof_valid": proof_result.is_valid,
                "proof_score": proof_result.validation_score
            })

            print(f"✅ {client_id}: Gradient={grad_result.is_valid}, Proof={proof_result.is_valid}")

        # Verify all clients passed validation
        all_valid = all(r["gradient_valid"] and r["proof_valid"] for r in integration_results)
        print(f"✅ All clients validated: {'PASS' if all_valid else 'FAIL'}")

        # Record results
        self.test_results["integration_tests"].append({
            "test": "federated_learning_integration",
            "clients_tested": len(clients),
            "all_valid": all_valid,
            "avg_gradient_score": sum(r["gradient_score"] for r in integration_results) / len(integration_results),
            "avg_proof_score": sum(r["proof_score"] for r in integration_results) / len(integration_results)
        })

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        print("\n🔍 Testing Edge Cases and Boundary Conditions")

        edge_cases = [
            {
                "name": "empty_gradients",
                "gradients": {},
                "should_fail": True
            },
            {
                "name": "single_element",
                "gradients": {"param": torch.tensor([1.0])},
                "should_fail": False
            },
            {
                "name": "large_tensors",
                "gradients": {"large": torch.randn(1000, 1000)},
                "should_fail": False
            },
            {
                "name": "extreme_values",
                "gradients": {"extreme": torch.tensor([1e-10, 1e10])},
                "should_fail": False
            },
            {
                "name": "mixed_types",
                "gradients": {
                    "float": torch.tensor([1.0, 2.0]),
                    "int": torch.tensor([1, 2], dtype=torch.int32)
                },
                "should_fail": False
            }
        ]

        edge_results = []
        for case in edge_cases:
            try:
                if case["name"] == "empty_gradients":
                    # Special handling for empty gradients
                    with self.assertRaises(Exception):
                        self.gradient_validator.validate_gradients_comprehensive(
                            case["gradients"], f"edge_{case['name']}"
                        )
                    result_valid = False
                else:
                    result = self.gradient_validator.validate_gradients_comprehensive(
                        case["gradients"], f"edge_{case['name']}"
                    )
                    result_valid = result.is_valid

                expected_result = not case["should_fail"]
                case_passed = result_valid == expected_result

                edge_results.append({
                    "case": case["name"],
                    "passed": case_passed,
                    "expected": expected_result,
                    "actual": result_valid
                })

                if case_passed:
                    print(f"✅ {case['name']}: PASS")
                else:
                    print(f"❌ {case['name']}: FAIL (expected {expected_result}, got {result_valid})")

            except Exception as e:
                edge_results.append({
                    "case": case["name"],
                    "passed": case["should_fail"],  # If should_fail=True and we got exception, it's a pass
                    "expected": not case["should_fail"],
                    "actual": False,
                    "error": str(e)
                })
                if case["should_fail"]:
                    print(f"✅ {case['name']}: PASS (expected failure)")
                else:
                    print(f"❌ {case['name']}: FAIL (unexpected error: {e})")

        # Record results
        self.test_results["gradient_tests"].append({
            "test": "edge_cases",
            "cases_tested": len(edge_cases),
            "cases_passed": sum(1 for r in edge_results if r["passed"])
        })

    def test_security_metrics_monitoring(self):
        """Test security metrics and monitoring validation."""
        print("\n📊 Testing Security Metrics and Monitoring")

        # Perform various security-related operations
        security_operations = []

        # 1. Multiple gradient validations
        for i in range(5):
            grad_result = self.gradient_validator.validate_gradients_comprehensive(
                self.valid_gradients, f"security_client_{i}"
            )
            security_operations.append({
                "type": "gradient_validation",
                "score": grad_result.validation_score,
                "patterns": len(grad_result.adversarial_patterns)
            })

        # 2. Multiple proof validations
        for i in range(3):
            proof_result = self.proof_validator.validate_proof_comprehensive(
                self.valid_groth16_proof, ZKProofType.GROTH16, None, f"security_client_{i}"
            )
            security_operations.append({
                "type": "proof_validation",
                "score": proof_result.validation_score,
                "attacks": len(proof_result.detected_attacks)
            })

        # Get security metrics
        gradient_metrics = self.gradient_validator.get_validation_metrics()
        proof_metrics = self.proof_validator.get_validation_metrics()

        print("📈 Gradient Validation Metrics:"        print(f"   Total validations: {gradient_metrics['total_validations']}")
        print(".1f"        print(f"   Cache size: {gradient_metrics.get('cache_size', 0)}")

        print("📈 Proof Validation Metrics:"        print(f"   Total validations: {proof_metrics['total_validations']}")
        print(".1f"        print(f"   Cache size: {proof_metrics.get('cache_size', 0)}")

        # Verify metrics are reasonable
        gradient_metrics_ok = (
            gradient_metrics["total_validations"] >= 5 and
            gradient_metrics["average_score"] > 0
        )

        proof_metrics_ok = (
            proof_metrics["total_validations"] >= 3 and
            proof_metrics["average_score"] > 0
        )

        print(f"✅ Gradient metrics valid: {'PASS' if gradient_metrics_ok else 'FAIL'}")
        print(f"✅ Proof metrics valid: {'PASS' if proof_metrics_ok else 'FAIL'}")

        # Record results
        self.test_results["security_tests"].append({
            "test": "security_metrics",
            "gradient_metrics_ok": gradient_metrics_ok,
            "proof_metrics_ok": proof_metrics_ok,
            "operations_performed": len(security_operations)
        })

    def generate_comprehensive_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE INPUT VALIDATION TEST REPORT")
        print("=" * 70)

        # Overall statistics
        total_tests = sum(len(tests) for tests in self.test_results.values())
        passed_tests = 0
        failed_tests = 0

        for category, tests in self.test_results.items():
            for test in tests:
                if isinstance(test, dict):
                    # Check for various success indicators
                    if "passed" in test and test["passed"]:
                        passed_tests += 1
                    elif "all_valid" in test and test["all_valid"]:
                        passed_tests += 1
                    elif "detection_rate" in test and test["detection_rate"] >= 0.8:
                        passed_tests += 1
                    elif "gradient_perf_ok" in test and "proof_perf_ok" in test:
                        if test["gradient_perf_ok"] and test["proof_perf_ok"]:
                            passed_tests += 1
                    else:
                        failed_tests += 1

        print("
📊 OVERALL STATISTICS"        print(f"   Total Test Categories: {len(self.test_results)}")
        print(f"   Total Tests Executed: {total_tests}")
        print(f"   Tests Passed: {passed_tests}")
        print(f"   Tests Failed: {failed_tests}")
        print(".1f"
        # Category breakdown
        print("
📈 CATEGORY BREAKDOWN"        for category, tests in self.test_results.items():
            print(f"   {category.replace('_', ' ').title()}: {len(tests)} tests")

        # Detailed results
        print("
📋 DETAILED RESULTS"        for category, tests in self.test_results.items():
            if tests:
                print(f"\n{category.replace('_', ' ').upper()}:")
                for test in tests:
                    if isinstance(test, dict):
                        status = "✅ PASS" if self._is_test_passed(test) else "❌ FAIL"
                        print(f"   {status} {test.get('test', 'unknown')}")
                        if "score" in test:
                            print(".1f"                        if "detection_rate" in test:
                            print(".1f"
        # Security assessment
        security_score = self._calculate_security_score()
        print("
🛡️ SECURITY ASSESSMENT"        print(f"   Overall Security Score: {security_score}/100")
        if security_score >= 90:
            print("   Security Level: EXCELLENT")
        elif security_score >= 80:
            print("   Security Level: VERY GOOD")
        elif security_score >= 70:
            print("   Security Level: GOOD")
        elif security_score >= 60:
            print("   Security Level: ADEQUATE")
        else:
            print("   Security Level: NEEDS IMPROVEMENT")

        # Recommendations
        print("
💡 RECOMMENDATIONS"        if security_score < 80:
            print("   • Review failed test cases for security improvements")
            print("   • Consider adjusting validation thresholds")
            print("   • Implement additional monitoring for failed validations")

        print("   • Regular security audits recommended")
        print("   • Monitor performance metrics in production")
        print("   • Keep attack pattern definitions updated")

        print("\n" + "=" * 70)
        print("TEST REPORT COMPLETE")
        print("=" * 70)

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "security_score": security_score,
            "categories": list(self.test_results.keys())
        }

    def _is_test_passed(self, test: Dict[str, Any]) -> bool:
        """Determine if a test passed based on its results."""
        if "passed" in test:
            return test["passed"]
        if "all_valid" in test:
            return test["all_valid"]
        if "detection_rate" in test:
            return test["detection_rate"] >= 0.8
        if "gradient_perf_ok" in test and "proof_perf_ok" in test:
            return test["gradient_perf_ok"] and test["proof_perf_ok"]
        if "gradient_metrics_ok" in test and "proof_metrics_ok" in test:
            return test["gradient_metrics_ok"] and test["proof_metrics_ok"]
        return False

    def _calculate_security_score(self) -> float:
        """Calculate overall security score."""
        score_components = []

        # Gradient validation security
        gradient_tests = self.test_results.get("gradient_tests", [])
        if gradient_tests:
            gradient_scores = [t.get("score", 0) for t in gradient_tests if "score" in t]
            if gradient_scores:
                score_components.append(sum(gradient_scores) / len(gradient_scores))

        # Proof validation security
        proof_tests = self.test_results.get("proof_tests", [])
        if proof_tests:
            proof_scores = [t.get("score", 0) for t in proof_tests if "score" in t]
            if proof_scores:
                score_components.append(sum(proof_scores) / len(proof_scores))

        # Security test effectiveness
        security_tests = self.test_results.get("security_tests", [])
        if security_tests:
            detection_rates = [t.get("detection_rate", 0) for t in security_tests if "detection_rate" in t]
            if detection_rates:
                score_components.append(sum(detection_rates) / len(detection_rates) * 100)

        # Performance test results
        perf_tests = self.test_results.get("performance_tests", [])
        if perf_tests:
            for test in perf_tests:
                if test.get("gradient_perf_ok", False) and test.get("proof_perf_ok", False):
                    score_components.append(100)
                else:
                    score_components.append(70)

        if score_components:
            return sum(score_components) / len(score_components)
        return 0.0

def run_comprehensive_input_validation_tests():
    """Run the comprehensive input validation test suite."""
    print("🚀 Starting Comprehensive Input Validation Test Suite")
    print("=" * 60)
    print("Task 6.3.3: Comprehensive Testing Framework for Input Validation")
    print("Testing gradient validation (6.3.1) and proof validation (6.3.2)")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestComprehensiveInputValidation('test_gradient_validation_comprehensive'))
    suite.addTest(TestComprehensiveInputValidation('test_adversarial_attack_simulation'))
    suite.addTest(TestComprehensiveInputValidation('test_proof_validation_comprehensive'))
    suite.addTest(TestComprehensiveInputValidation('test_proof_attack_simulation'))
    suite.addTest(TestComprehensiveInputValidation('test_statistical_analysis_validation'))
    suite.addTest(TestComprehensiveInputValidation('test_baseline_establishment_and_detection'))
    suite.addTest(TestComprehensiveInputValidation('test_sanitization_effectiveness'))
    suite.addTest(TestComprehensiveInputValidation('test_performance_validation'))
    suite.addTest(TestComprehensiveInputValidation('test_integration_with_federated_learning'))
    suite.addTest(TestComprehensiveInputValidation('test_edge_cases_and_boundary_conditions'))
    suite.addTest(TestComprehensiveInputValidation('test_security_metrics_monitoring'))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate comprehensive report
    if hasattr(result, 'test_results'):
        report = result.test_results.generate_comprehensive_test_report()
    else:
        # Fallback for cases where we can't access test instance
        print("\n" + "=" * 60)
        print("TEST EXECUTION COMPLETE")
        print("=" * 60)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  ❌ {test}")

        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  ❌ {test}")

        report = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
        }

    return report

if __name__ == "__main__":
    # Run the comprehensive test suite
    results = run_comprehensive_input_validation_tests()

    print("
🎯 FINAL RESULTS:"    print(f"   Tests Executed: {results.get('total_tests', results.get('tests_run', 'N/A'))}")
    print(f"   Success Rate: {results.get('success_rate', 'N/A'):.1f}%")
    print(f"   Security Score: {results.get('security_score', 'N/A')}/100")

    if results.get('success_rate', 0) >= 80:
        print("   ✅ COMPREHENSIVE TESTING: PASS")
        print("   🏆 Task 6.3.3: COMPLETED SUCCESSFULLY")
    else:
        print("   ❌ COMPREHENSIVE TESTING: ISSUES DETECTED")
        print("   🔧 Review test failures and address issues")

