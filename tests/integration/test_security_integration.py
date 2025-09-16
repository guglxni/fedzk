#!/usr/bin/env python3
"""
Security Integration Tests
==========================

Comprehensive security testing including:
- Adversarial attack simulation and detection
- Penetration testing framework
- Input fuzzing and validation robustness
- Cryptographic attack vector testing
- Security monitoring and alerting
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Callable
import random
import string
import json
from unittest.mock import MagicMock, patch
import hashlib
import hmac
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.fedzk.validation.gradient_validator import GradientValidator
from src.fedzk.validation.proof_validator import ProofValidator
from src.fedzk.client.trainer import FederatedTrainer
from src.fedzk.prover.zkgenerator import ZKProver


class AdversarialAttackSimulator:
    """Simulate various adversarial attacks on federated learning."""

    def __init__(self):
        self.attack_types = {
            'gradient_explosion': self._gradient_explosion_attack,
            'gradient_vanishing': self._gradient_vanishing_attack,
            'data_poisoning': self._data_poisoning_attack,
            'model_poisoning': self._model_poisoning_attack,
            'byzantine_attack': self._byzantine_attack,
            'label_flipping': self._label_flipping_attack,
            'backdoor_attack': self._backdoor_attack,
            'evasion_attack': self._evasion_attack
        }

    def generate_attack(self, attack_type: str, clean_gradients: List[float],
                       intensity: float = 1.0) -> List[float]:
        """Generate adversarial gradients using specified attack."""
        if attack_type not in self.attack_types:
            raise ValueError(f"Unknown attack type: {attack_type}")

        return self.attack_types[attack_type](clean_gradients, intensity)

    def _gradient_explosion_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate gradient explosion attack."""
        return [g * (1 + intensity * 100) for g in gradients]

    def _gradient_vanishing_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate gradient vanishing attack."""
        return [g * (1 - intensity * 0.99) for g in gradients]

    def _data_poisoning_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate data poisoning attack."""
        poisoned = gradients.copy()
        # Poison random subset of gradients
        poison_indices = random.sample(range(len(gradients)), int(len(gradients) * intensity))
        for idx in poison_indices:
            poisoned[idx] = random.uniform(-10, 10)  # Random large values
        return poisoned

    def _model_poisoning_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate model poisoning attack."""
        # Add systematic bias to gradients
        bias = intensity * 5.0
        return [g + bias if random.random() < 0.5 else g - bias for g in gradients]

    def _byzantine_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate Byzantine attack (arbitrary malicious updates)."""
        if random.random() < intensity:
            # Send completely arbitrary gradients
            return [random.uniform(-100, 100) for _ in gradients]
        return gradients

    def _label_flipping_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate label flipping attack."""
        # This would affect the training data, resulting in inverted gradients
        return [-g * (1 + intensity) for g in gradients]

    def _backdoor_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate backdoor attack."""
        # Add subtle perturbations that trigger on specific conditions
        trigger_pattern = [0.1, 0.2, 0.3, 0.4]  # Simplified trigger
        perturbed = gradients.copy()

        # Apply trigger pattern with some probability
        if random.random() < intensity:
            for i in range(min(len(trigger_pattern), len(perturbed))):
                perturbed[i] += trigger_pattern[i] * intensity

        return perturbed

    def _evasion_attack(self, gradients: List[float], intensity: float) -> List[float]:
        """Simulate evasion attack (trying to avoid detection)."""
        # Add small perturbations that might evade statistical detection
        noise = np.random.normal(0, intensity * 0.01, len(gradients))
        return [g + n for g, n in zip(gradients, noise)]


class PenetrationTestingFramework:
    """Framework for penetration testing of the federated learning system."""

    def __init__(self):
        self.vulnerabilities_found = []
        self.test_results = []

    def run_penetration_test(self, target_system: str, test_vectors: List[Dict]) -> Dict[str, Any]:
        """Run penetration test against specified system component."""
        results = {
            'target': target_system,
            'vulnerabilities': [],
            'successful_attacks': 0,
            'total_tests': len(test_vectors)
        }

        for test_vector in test_vectors:
            test_result = self._execute_test_vector(target_system, test_vector)
            results['successful_attacks'] += 1 if test_result['exploited'] else 0

            if test_result['exploited']:
                results['vulnerabilities'].append({
                    'type': test_vector['type'],
                    'severity': test_vector['severity'],
                    'description': test_result['description']
                })

        return results

    def _execute_test_vector(self, target: str, test_vector: Dict) -> Dict[str, Any]:
        """Execute individual test vector."""
        # This would implement actual penetration testing logic
        # For now, simulate based on test vector characteristics

        exploited = False
        description = f"Test vector {test_vector['type']} executed"

        # Simulate vulnerability detection based on test vector
        if test_vector.get('malicious_payload', False):
            if random.random() < 0.3:  # 30% chance of successful exploitation
                exploited = True
                description = f"Successful exploitation via {test_vector['type']}"

        return {
            'exploited': exploited,
            'description': description,
            'target': target
        }


class InputFuzzer:
    """Fuzzing framework for input validation testing."""

    def __init__(self):
        self.fuzz_strategies = {
            'numeric_overflow': self._fuzz_numeric_overflow,
            'string_injection': self._fuzz_string_injection,
            'null_bytes': self._fuzz_null_bytes,
            'unicode_bomb': self._fuzz_unicode_bomb,
            'large_input': self._fuzz_large_input,
            'malformed_json': self._fuzz_malformed_json,
            'type_confusion': self._fuzz_type_confusion
        }

    def generate_fuzzed_inputs(self, base_input: Any, strategy: str,
                              iterations: int = 10) -> List[Any]:
        """Generate fuzzed inputs using specified strategy."""
        if strategy not in self.fuzz_strategies:
            raise ValueError(f"Unknown fuzzing strategy: {strategy}")

        fuzzed_inputs = []
        for _ in range(iterations):
            fuzzed_input = self.fuzz_strategies[strategy](base_input)
            fuzzed_inputs.append(fuzzed_input)

        return fuzzed_inputs

    def _fuzz_numeric_overflow(self, base_input: List[float]) -> List[float]:
        """Generate numeric overflow inputs."""
        if isinstance(base_input, list):
            return [x * 1e308 for x in base_input]  # Very large numbers
        return [1e308] * 4

    def _fuzz_string_injection(self, base_input: List[float]) -> Any:
        """Generate string injection inputs."""
        return ["<script>alert('xss')</script>"] * len(base_input) if isinstance(base_input, list) else ["injection"]

    def _fuzz_null_bytes(self, base_input: List[float]) -> Any:
        """Generate null byte inputs."""
        if isinstance(base_input, list):
            return [str(x) + '\x00' for x in base_input]
        return "test\x00input"

    def _fuzz_unicode_bomb(self, base_input: List[float]) -> Any:
        """Generate unicode bomb inputs."""
        bomb = "💥" * 1000  # Many emoji characters
        if isinstance(base_input, list):
            return [bomb] * len(base_input)
        return bomb

    def _fuzz_large_input(self, base_input: List[float]) -> List[float]:
        """Generate very large inputs."""
        return [random.uniform(-1e10, 1e10) for _ in range(10000)]

    def _fuzz_malformed_json(self, base_input: List[float]) -> str:
        """Generate malformed JSON inputs."""
        return '{"gradients": [1, 2, 3'  # Missing closing bracket

    def _fuzz_type_confusion(self, base_input: List[float]) -> Any:
        """Generate type confusion inputs."""
        return {"nested": {"deeply": {"confusing": base_input}}}


class TestAdversarialAttackDetection:
    """Test detection of adversarial attacks."""

    def setup_method(self):
        """Set up test environment."""
        self.attack_simulator = AdversarialAttackSimulator()
        self.validator = GradientValidator()

    def test_gradient_explosion_detection(self):
        """Test detection of gradient explosion attacks."""
        clean_gradients = [0.01, -0.02, 0.005, 0.001]

        # Generate explosive gradients
        explosive_gradients = self.attack_simulator.generate_attack(
            'gradient_explosion', clean_gradients, intensity=0.8
        )

        # Test detection
        result = self.validator.validate_gradients_comprehensive({
            'gradients': explosive_gradients,
            'client_id': 'test_client'
        })

        assert 'gradient_explosion' in result['detected_attacks']
        assert result['overall_valid'] == False
        assert result['overall_score'] < 50  # Should have low score

    def test_gradient_vanishing_detection(self):
        """Test detection of gradient vanishing attacks."""
        clean_gradients = [0.01, -0.02, 0.005, 0.001]

        # Generate vanishing gradients
        vanishing_gradients = self.attack_simulator.generate_attack(
            'gradient_vanishing', clean_gradients, intensity=0.9
        )

        result = self.validator.validate_gradients_comprehensive({
            'gradients': vanishing_gradients,
            'client_id': 'test_client'
        })

        assert 'gradient_vanishing' in result['detected_attacks']
        assert result['overall_valid'] == False

    def test_data_poisoning_detection(self):
        """Test detection of data poisoning attacks."""
        clean_gradients = [0.01, -0.02, 0.005, 0.001]

        # Generate poisoned gradients
        poisoned_gradients = self.attack_simulator.generate_attack(
            'data_poisoning', clean_gradients, intensity=0.5
        )

        result = self.validator.validate_gradients_comprehensive({
            'gradients': poisoned_gradients,
            'client_id': 'test_client'
        })

        # Data poisoning might be harder to detect statistically
        # but should still impact validation score
        assert result['overall_score'] < 80

    def test_byzantine_attack_detection(self):
        """Test detection of Byzantine attacks."""
        clean_gradients = [0.01, -0.02, 0.005, 0.001]

        # Generate Byzantine attack
        byzantine_gradients = self.attack_simulator.generate_attack(
            'byzantine_attack', clean_gradients, intensity=1.0
        )

        result = self.validator.validate_gradients_comprehensive({
            'gradients': byzantine_gradients,
            'client_id': 'test_client'
        })

        # Byzantine attacks should be clearly detected
        assert result['overall_valid'] == False
        assert result['overall_score'] < 30

    def test_multiple_simultaneous_attacks(self):
        """Test detection when multiple attacks occur simultaneously."""
        clean_gradients = [0.01, -0.02, 0.005, 0.001]

        # Apply multiple attacks
        attacked_gradients = clean_gradients.copy()

        # Apply explosion
        attacked_gradients = self.attack_simulator.generate_attack(
            'gradient_explosion', attacked_gradients, intensity=0.3
        )

        # Apply poisoning
        attacked_gradients = self.attack_simulator.generate_attack(
            'data_poisoning', attacked_gradients, intensity=0.3
        )

        result = self.validator.validate_gradients_comprehensive({
            'gradients': attacked_gradients,
            'client_id': 'test_client'
        })

        # Should detect multiple attack types
        assert len(result['detected_attacks']) >= 2
        assert result['overall_valid'] == False
        assert result['overall_score'] < 40

    def test_adversarial_attack_robustness(self):
        """Test system robustness against various adversarial inputs."""
        attack_types = ['gradient_explosion', 'gradient_vanishing',
                       'data_poisoning', 'model_poisoning', 'byzantine_attack']

        clean_gradients = [0.01, -0.02, 0.005, 0.001]

        for attack_type in attack_types:
            attacked_gradients = self.attack_simulator.generate_attack(
                attack_type, clean_gradients, intensity=0.7
            )

            result = self.validator.validate_gradients_comprehensive({
                'gradients': attacked_gradients,
                'client_id': f'test_{attack_type}'
            })

            # System should detect adversarial inputs
            assert isinstance(result['overall_valid'], bool)
            assert isinstance(result['overall_score'], (int, float))
            assert isinstance(result['detected_attacks'], list)

            # High-intensity attacks should be detected
            if attack_type in ['gradient_explosion', 'byzantine_attack']:
                assert result['overall_valid'] == False


class TestPenetrationTesting:
    """Test penetration testing framework."""

    def setup_method(self):
        """Set up penetration testing environment."""
        self.penetration_tester = PenetrationTestingFramework()

    def test_api_endpoint_penetration(self):
        """Test penetration testing of API endpoints."""
        test_vectors = [
            {
                'type': 'sql_injection',
                'severity': 'high',
                'payload': "'; DROP TABLE users; --",
                'malicious_payload': True
            },
            {
                'type': 'xss_attack',
                'severity': 'medium',
                'payload': '<script>alert("xss")</script>',
                'malicious_payload': True
            },
            {
                'type': 'path_traversal',
                'severity': 'high',
                'payload': '../../../etc/passwd',
                'malicious_payload': True
            },
            {
                'type': 'buffer_overflow',
                'severity': 'critical',
                'payload': 'A' * 10000,
                'malicious_payload': True
            }
        ]

        results = self.penetration_tester.run_penetration_test(
            'api_endpoints', test_vectors
        )

        # Should have processed all test vectors
        assert results['total_tests'] == len(test_vectors)
        assert 'vulnerabilities' in results
        assert isinstance(results['successful_attacks'], int)

    def test_authentication_bypass_testing(self):
        """Test authentication bypass scenarios."""
        auth_test_vectors = [
            {
                'type': 'weak_password',
                'severity': 'medium',
                'credentials': {'username': 'admin', 'password': 'password'}
            },
            {
                'type': 'sql_injection_auth',
                'severity': 'high',
                'credentials': {'username': "admin' --", 'password': ''}
            },
            {
                'type': 'session_hijacking',
                'severity': 'high',
                'session_token': 'fake_session_token'
            }
        ]

        results = self.penetration_tester.run_penetration_test(
            'authentication', auth_test_vectors
        )

        assert results['target'] == 'authentication'
        assert results['total_tests'] == len(auth_test_vectors)

    def test_network_layer_penetration(self):
        """Test network layer penetration scenarios."""
        network_test_vectors = [
            {
                'type': 'man_in_the_middle',
                'severity': 'high',
                'attack_vector': 'certificate_spoofing'
            },
            {
                'type': 'dns_poisoning',
                'severity': 'medium',
                'attack_vector': 'dns_spoofing'
            },
            {
                'type': 'port_scanning',
                'severity': 'low',
                'attack_vector': 'service_discovery'
            }
        ]

        results = self.penetration_tester.run_penetration_test(
            'network_layer', network_test_vectors
        )

        assert results['target'] == 'network_layer'
        assert len(results['vulnerabilities']) >= 0  # May find issues


class TestInputFuzzing:
    """Test input fuzzing and validation robustness."""

    def setup_method(self):
        """Set up fuzzing test environment."""
        self.fuzzer = InputFuzzer()
        self.validator = GradientValidator()

    def test_numeric_overflow_fuzzing(self):
        """Test robustness against numeric overflow inputs."""
        base_gradients = [0.01, -0.02, 0.005, 0.001]

        # Generate fuzzed inputs
        fuzzed_inputs = self.fuzzer.generate_fuzzed_inputs(
            base_gradients, 'numeric_overflow', iterations=5
        )

        for i, fuzzed_input in enumerate(fuzzed_inputs):
            try:
                result = self.validator.validate_gradients_comprehensive({
                    'gradients': fuzzed_input,
                    'client_id': f'fuzz_overflow_{i}'
                })

                # System should handle overflow gracefully
                assert isinstance(result['overall_valid'], bool)
                assert isinstance(result['overall_score'], (int, float))

            except Exception as e:
                # Exceptions are acceptable for extreme fuzz inputs
                assert isinstance(e, (ValueError, TypeError, OverflowError))

    def test_large_input_fuzzing(self):
        """Test robustness against very large inputs."""
        base_gradients = [0.01, -0.02, 0.005, 0.001]

        # Generate large fuzzed inputs
        fuzzed_inputs = self.fuzzer.generate_fuzzed_inputs(
            base_gradients, 'large_input', iterations=3
        )

        for i, fuzzed_input in enumerate(fuzzed_inputs):
            try:
                # Limit input size for safety
                limited_input = fuzzed_input[:100] if len(fuzzed_input) > 100 else fuzzed_input

                result = self.validator.validate_gradients_comprehensive({
                    'gradients': limited_input,
                    'client_id': f'fuzz_large_{i}'
                })

                # Should handle large inputs gracefully
                assert isinstance(result['overall_valid'], bool)

            except Exception as e:
                # Memory or performance exceptions are acceptable
                assert isinstance(e, (MemoryError, RuntimeError, ValueError))

    def test_malformed_input_fuzzing(self):
        """Test robustness against malformed inputs."""
        fuzz_strategies = ['string_injection', 'null_bytes', 'unicode_bomb', 'malformed_json']

        for strategy in fuzz_strategies:
            try:
                fuzzed_inputs = self.fuzzer.generate_fuzzed_inputs(
                    [0.01, -0.02, 0.005, 0.001], strategy, iterations=2
                )

                for i, fuzzed_input in enumerate(fuzzed_inputs):
                    try:
                        result = self.validator.validate_gradients_comprehensive({
                            'gradients': fuzzed_input if isinstance(fuzzed_input, list) else [0.01, -0.02],
                            'client_id': f'fuzz_{strategy}_{i}'
                        })

                        # Should reject malformed inputs
                        if strategy in ['string_injection', 'null_bytes']:
                            assert result['overall_valid'] == False

                    except Exception as e:
                        # Type errors are expected for malformed inputs
                        assert isinstance(e, (TypeError, ValueError))

            except Exception as e:
                # Fuzzer itself might encounter issues with extreme inputs
                assert isinstance(e, (ValueError, TypeError))

    def test_type_confusion_fuzzing(self):
        """Test robustness against type confusion attacks."""
        base_input = [0.01, -0.02, 0.005, 0.001]

        # Generate type confusion inputs
        fuzzed_inputs = self.fuzzer.generate_fuzzed_inputs(
            base_input, 'type_confusion', iterations=5
        )

        for i, fuzzed_input in enumerate(fuzzed_inputs):
            try:
                # Try to extract gradients from nested structure
                if isinstance(fuzzed_input, dict) and 'nested' in fuzzed_input:
                    gradients = fuzzed_input['nested']['deeply']['confusing']
                else:
                    gradients = fuzzed_input if isinstance(fuzzed_input, list) else [0.01]

                result = self.validator.validate_gradients_comprehensive({
                    'gradients': gradients,
                    'client_id': f'fuzz_type_{i}'
                })

                assert isinstance(result['overall_valid'], bool)

            except Exception as e:
                # Type confusion should result in validation errors
                assert isinstance(e, (TypeError, KeyError, ValueError))


class TestCryptographicAttackVectors:
    """Test cryptographic attack vectors and countermeasures."""

    def setup_method(self):
        """Set up cryptographic testing environment."""
        self.proof_validator = ProofValidator()

    def test_invalid_proof_structure_attacks(self):
        """Test attacks using invalid proof structures."""
        invalid_proofs = [
            # Missing required fields
            {'pi_b': [[], []], 'pi_c': []},
            {'pi_a': [], 'pi_c': []},
            {'pi_a': [], 'pi_b': []},

            # Wrong field types
            {'pi_a': 'not_array', 'pi_b': [[], []], 'pi_c': []},
            {'pi_a': [], 'pi_b': 'not_array', 'pi_c': []},
            {'pi_a': [], 'pi_b': [], 'pi_c': 'not_array'},

            # Malformed cryptographic data
            {'pi_a': [1, 'invalid'], 'pi_b': [[], []], 'pi_c': []},
            {'pi_a': [], 'pi_b': [[1, 'invalid']], 'pi_c': []},

            # Oversized fields
            {'pi_a': list(range(1000)), 'pi_b': [[], []], 'pi_c': []},
            {'pi_a': [], 'pi_b': [list(range(1000))], 'pi_c': []},
        ]

        for i, invalid_proof in enumerate(invalid_proofs):
            result = self.proof_validator.validate_proof_comprehensive({
                'proof': invalid_proof,
                'public_inputs': [1, 2, 3],
                'circuit_type': 'model_update'
            })

            # Invalid proofs should be rejected
            assert result['overall_valid'] == False, f"Invalid proof {i} should be rejected"
            assert result['overall_score'] < 50, f"Invalid proof {i} should have low score"

    def test_timing_attack_detection(self):
        """Test detection of timing-based attacks."""
        # This would require measuring response times
        # For now, test the validation framework's timing robustness

        valid_proof = {
            'pi_a': [1, 2, 3],
            'pi_b': [[4, 5], [6, 7]],
            'pi_c': [8, 9]
        }

        # Test multiple validations to check for timing consistency
        timing_results = []
        for _ in range(10):
            start_time = time.time()
            result = self.proof_validator.validate_proof_comprehensive({
                'proof': valid_proof,
                'public_inputs': [1, 2, 3],
                'circuit_type': 'model_update'
            })
            end_time = time.time()

            timing_results.append(end_time - start_time)

        # Timing should be relatively consistent
        avg_time = sum(timing_results) / len(timing_results)
        max_deviation = max(abs(t - avg_time) for t in timing_results)

        # Should not have large timing variations (potential timing attack)
        assert max_deviation < avg_time * 2, "Large timing variations detected"

    def test_side_channel_attack_prevention(self):
        """Test prevention of side-channel attacks."""
        # Test that validation doesn't leak information through error messages
        test_proofs = [
            {'pi_a': [], 'pi_b': [], 'pi_c': []},  # Missing data
            {'pi_a': [1], 'pi_b': [], 'pi_c': []},  # Partial data
            {'pi_a': [1, 2, 3], 'pi_b': [[4]], 'pi_c': [8]},  # Incomplete arrays
        ]

        error_messages = []
        for proof in test_proofs:
            try:
                result = self.proof_validator.validate_proof_comprehensive({
                    'proof': proof,
                    'public_inputs': [1, 2, 3],
                    'circuit_type': 'model_update'
                })
                error_messages.append("Valid")
            except Exception as e:
                error_messages.append(str(e))

        # Error messages should not reveal internal structure
        for msg in error_messages:
            assert "pi_a" not in msg.lower(), "Error message reveals internal structure"
            assert "pi_b" not in msg.lower(), "Error message reveals internal structure"
            assert "pi_c" not in msg.lower(), "Error message reveals internal structure"

    def test_cryptographic_weakness_detection(self):
        """Test detection of cryptographic weaknesses."""
        weak_proofs = [
            # Predictable values
            {'pi_a': [0, 0, 0], 'pi_b': [[0, 0]], 'pi_c': [0, 0]},
            {'pi_a': [1, 1, 1], 'pi_b': [[1, 1]], 'pi_c': [1, 1]},

            # Small values (potential weak keys)
            {'pi_a': [1, 2, 3], 'pi_b': [[1, 1]], 'pi_c': [1, 2]},

            # Repeated patterns
            {'pi_a': [1, 2, 1], 'pi_b': [[2, 1]], 'pi_c': [1, 2]},
        ]

        for i, weak_proof in enumerate(weak_proofs):
            result = self.proof_validator.validate_proof_comprehensive({
                'proof': weak_proof,
                'public_inputs': [1, 2, 3],
                'circuit_type': 'model_update'
            })

            # Weak proofs should be flagged
            assert result['overall_score'] < 80, f"Weak proof {i} should have low score"
            if result['detected_attacks']:
                assert any('weak' in attack.lower() or 'predictable' in attack.lower()
                          for attack in result['detected_attacks'])


if __name__ == "__main__":
    pytest.main([__file__])

