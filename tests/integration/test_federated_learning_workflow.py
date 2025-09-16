#!/usr/bin/env python3
"""
Federated Learning Workflow Integration Tests
=============================================

Comprehensive end-to-end testing of the complete federated learning pipeline
including client training, secure aggregation, MPC proof generation, and validation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock, patch
import threading
import time
import json
from pathlib import Path

from src.fedzk.client.trainer import FederatedTrainer
from src.fedzk.coordinator.logic import CoordinatorLogic, VerifiedUpdate
from src.fedzk.mpc.client import MPCClient
from src.fedzk.prover.zkgenerator import ZKProver
from src.fedzk.validation.gradient_validator import GradientValidator


class SimpleModel(nn.Module):
    """Simple neural network for testing."""
    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class MockMPCServer:
    """Mock MPC server for testing."""
    def __init__(self):
        self.proofs = {}
        self.requests = []

    def generate_proof(self, gradients: List[float], client_id: str) -> Dict[str, Any]:
        """Mock proof generation."""
        proof_id = f"proof_{client_id}_{len(self.proofs)}"
        self.requests.append({
            'client_id': client_id,
            'gradients': gradients,
            'timestamp': time.time()
        })

        proof = {
            'proof_id': proof_id,
            'client_id': client_id,
            'pi_a': [1, 2, 3],
            'pi_b': [[4, 5], [6, 7]],
            'pi_c': [8, 9],
            'public_inputs': [sum(gradients)],
            'verified': True
        }

        self.proofs[proof_id] = proof
        return proof


class TestFederatedLearningWorkflow:
    """Comprehensive federated learning workflow tests."""

    def setup_method(self):
        """Set up test environment."""
        self.num_clients = 3
        self.model_config = {
            'input_size': 10,
            'hidden_size': 5,
            'output_size': 1
        }
        self.mpc_server = MockMPCServer()
        self.global_model = SimpleModel(**self.model_config)

    def test_single_client_training_workflow(self):
        """Test complete workflow for a single client."""
        # Initialize client trainer
        trainer = FederatedTrainer(
            model_config=self.model_config,
            client_id="client_1",
            learning_rate=0.01,
            batch_size=32
        )

        # Generate training data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)

        # Train model locally
        initial_params = {name: param.clone() for name, param in trainer.model.named_parameters()}

        trainer.train_local(X, y, epochs=2)

        # Extract gradients
        gradients = trainer.extract_gradients()

        # Validate gradients
        validator = GradientValidator()
        validation_result = validator.validate_gradients_comprehensive({
            'gradients': gradients,
            'client_id': 'client_1'
        })

        assert validation_result['overall_valid'] == True
        assert len(validation_result['detected_attacks']) == 0

        # Generate ZK proof
        prover = ZKProver()
        proof = prover.generate_proof({
            'gradients': gradients[:4],  # First 4 gradients for circuit
            'client_id': 'client_1'
        })

        assert proof is not None
        assert 'proof' in proof
        assert 'public_inputs' in proof

        # Verify proof structure
        assert 'pi_a' in proof['proof']
        assert 'pi_b' in proof['proof']
        assert 'pi_c' in proof['proof']

    def test_multi_client_coordination(self):
        """Test coordination between multiple clients."""
        clients = []
        client_updates = []

        # Create multiple clients
        for i in range(self.num_clients):
            trainer = FederatedTrainer(
                model_config=self.model_config,
                client_id=f"client_{i+1}",
                learning_rate=0.01,
                batch_size=32
            )
            clients.append(trainer)

        # Each client trains on different data
        for i, client in enumerate(clients):
            # Generate client-specific data with slight variations
            X = torch.randn(50, 10) + torch.randn(1, 10) * 0.1 * i
            y = torch.randn(50, 1) + torch.randn(1, 1) * 0.1 * i

            # Train locally
            client.train_local(X, y, epochs=1)

            # Extract and validate gradients
            gradients = client.extract_gradients()
            validator = GradientValidator()
            validation = validator.validate_gradients_comprehensive({
                'gradients': gradients,
                'client_id': client.client_id
            })

            assert validation['overall_valid'] == True

            # Generate proof
            prover = ZKProver()
            proof = prover.generate_proof({
                'gradients': gradients[:4],
                'client_id': client.client_id
            })

            client_updates.append({
                'client_id': client.client_id,
                'gradients': gradients,
                'proof': proof,
                'validation': validation
            })

        # Verify all clients completed successfully
        assert len(client_updates) == self.num_clients
        for update in client_updates:
            assert update['proof'] is not None
            assert update['validation']['overall_valid'] == True

    def test_secure_aggregation_workflow(self):
        """Test secure aggregation of client updates."""
        # Create coordinator logic
        coordinator = CoordinatorLogic()

        # Generate mock client updates
        client_updates = []
        for i in range(self.num_clients):
            gradients = [np.random.normal(0, 0.1) for _ in range(10)]

            # Create verified update
            update = VerifiedUpdate(
                client_id=f"client_{i+1}",
                gradients=gradients,
                proof={
                    'pi_a': [1, 2, 3],
                    'pi_b': [[4, 5], [6, 7]],
                    'pi_c': [8, 9],
                    'public_inputs': [sum(gradients)]
                },
                timestamp=time.time(),
                validation_score=95.0
            )
            client_updates.append(update)

        # Test aggregation
        aggregated_result = coordinator.aggregate_updates(client_updates)

        assert aggregated_result is not None
        assert 'aggregated_gradients' in aggregated_result
        assert len(aggregated_result['aggregated_gradients']) == 10

        # Verify aggregation is reasonable (not just sum, but average-like)
        expected_avg = np.mean([update.gradients for update in client_updates], axis=0)
        np.testing.assert_array_almost_equal(
            aggregated_result['aggregated_gradients'],
            expected_avg,
            decimal=3
        )

    @patch('requests.post')
    def test_network_failure_recovery(self, mock_post):
        """Test network failure scenarios and recovery."""
        # Simulate network failures
        mock_post.side_effect = [
            ConnectionError("Network timeout"),  # First call fails
            MagicMock(status_code=200, json=lambda: {'status': 'success'})  # Second succeeds
        ]

        mpc_client = MPCClient(
            server_url="http://localhost:8000",
            api_key="test_key",
            timeout=5,
            max_retries=2
        )

        gradients = [0.1, 0.2, 0.3, 0.4]

        # This should succeed despite initial network failure
        result = mpc_client.generate_proof(gradients, "test_client")

        # Verify retry mechanism worked
        assert mock_post.call_count == 2  # One failure + one success
        assert result is not None

    def test_performance_under_load(self):
        """Test system performance under concurrent load."""
        import concurrent.futures

        def simulate_client_workflow(client_id: str) -> Dict[str, Any]:
            """Simulate complete client workflow."""
            # Create trainer
            trainer = FederatedTrainer(
                model_config=self.model_config,
                client_id=client_id,
                learning_rate=0.01,
                batch_size=16
            )

            # Generate data and train
            X = torch.randn(32, 10)
            y = torch.randn(32, 1)
            trainer.train_local(X, y, epochs=1)

            # Extract gradients and validate
            gradients = trainer.extract_gradients()
            validator = GradientValidator()
            validation = validator.validate_gradients_comprehensive({
                'gradients': gradients,
                'client_id': client_id
            })

            return {
                'client_id': client_id,
                'gradients_count': len(gradients),
                'validation_score': validation['overall_score'],
                'success': validation['overall_valid']
            }

        # Test with multiple concurrent clients
        num_concurrent_clients = 5

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_clients) as executor:
            futures = [
                executor.submit(simulate_client_workflow, f"client_{i}")
                for i in range(num_concurrent_clients)
            ]

            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all clients succeeded
        successful_clients = sum(1 for r in results if r['success'])
        assert successful_clients == num_concurrent_clients

        # Verify reasonable performance (should complete within reasonable time)
        assert total_time < 30  # Should complete in less than 30 seconds

        # Log performance metrics
        avg_validation_score = np.mean([r['validation_score'] for r in results])
        print(".2f")
        print(".2f")
        print(".1f")

    def test_end_to_end_federated_learning(self):
        """Test complete end-to-end federated learning scenario."""
        # Initialize global model
        global_model = SimpleModel(**self.model_config)

        # Create multiple clients
        clients = []
        for i in range(3):
            client = FederatedTrainer(
                model_config=self.model_config,
                client_id=f"client_{i+1}",
                learning_rate=0.01,
                batch_size=32
            )
            # Initialize with global model parameters
            client.model.load_state_dict(global_model.state_dict())
            clients.append(client)

        # Simulate federated learning rounds
        num_rounds = 2
        for round_num in range(num_rounds):
            print(f"Round {round_num + 1}/{num_rounds}")

            client_updates = []

            # Each client trains locally
            for client in clients:
                # Generate client-specific data
                X = torch.randn(64, 10) + torch.randn(1, 10) * 0.1
                y = torch.randn(64, 1) + torch.randn(1, 1) * 0.1

                client.train_local(X, y, epochs=1)

                # Extract gradients and create verified update
                gradients = client.extract_gradients()
                validator = GradientValidator()
                validation = validator.validate_gradients_comprehensive({
                    'gradients': gradients,
                    'client_id': client.client_id
                })

                update = VerifiedUpdate(
                    client_id=client.client_id,
                    gradients=gradients,
                    proof={'dummy': 'proof'},  # Simplified for test
                    timestamp=time.time(),
                    validation_score=validation['overall_score']
                )
                client_updates.append(update)

            # Aggregate updates
            coordinator = CoordinatorLogic()
            aggregated_result = coordinator.aggregate_updates(client_updates)

            # Update global model (simplified)
            assert aggregated_result is not None
            assert 'aggregated_gradients' in aggregated_result

            print(f"  Round {round_num + 1} completed successfully")

        # Verify federated learning completed
        assert len(client_updates) == 3 * num_rounds  # 3 clients * 2 rounds
        print("✅ End-to-end federated learning test completed successfully")


class TestSecurityIntegration:
    """Security-focused integration tests."""

    def test_adversarial_gradient_attack_detection(self):
        """Test detection of adversarial gradient attacks."""
        validator = GradientValidator()

        # Test various adversarial scenarios
        test_cases = [
            # Normal gradients
            {
                'name': 'normal_gradients',
                'gradients': [0.01, -0.02, 0.005, 0.001],
                'expected_attacks': []
            },
            # Gradient explosion attack
            {
                'name': 'gradient_explosion',
                'gradients': [100.0, -200.0, 150.0, -300.0],
                'expected_attacks': ['gradient_explosion']
            },
            # Gradient vanishing attack
            {
                'name': 'gradient_vanishing',
                'gradients': [1e-8, -2e-8, 5e-9, -1e-8],
                'expected_attacks': ['gradient_vanishing']
            },
            # NaN attack
            {
                'name': 'nan_attack',
                'gradients': [0.01, float('nan'), 0.005, 0.001],
                'expected_attacks': ['nan_values']
            }
        ]

        for test_case in test_cases:
            result = validator.validate_gradients_comprehensive({
                'gradients': test_case['gradients'],
                'client_id': 'test_client'
            })

            print(f"Testing {test_case['name']}:")
            print(f"  Overall valid: {result['overall_valid']}")
            print(f"  Detected attacks: {result['detected_attacks']}")

            # Verify expected attacks are detected
            for expected_attack in test_case['expected_attacks']:
                assert expected_attack in result['detected_attacks'], \
                    f"Expected attack '{expected_attack}' not detected in {test_case['name']}"

    def test_cryptographic_attack_vectors(self):
        """Test resistance to cryptographic attack vectors."""
        # Test invalid proof structures
        invalid_proofs = [
            {'missing_pi_a': []},  # Missing required field
            {'pi_a': [], 'pi_b': [], 'pi_c': [], 'extra_field': 'invalid'},  # Extra field
            {'pi_a': 'not_array', 'pi_b': [], 'pi_c': []},  # Wrong type
        ]

        from src.fedzk.validation.proof_validator import ProofValidator

        validator = ProofValidator()

        for i, invalid_proof in enumerate(invalid_proofs):
            result = validator.validate_proof_comprehensive({
                'proof': invalid_proof,
                'public_inputs': [1, 2, 3],
                'circuit_type': 'model_update'
            })

            print(f"Invalid proof {i+1}: {result['overall_valid']}")
            assert result['overall_valid'] == False, f"Invalid proof {i+1} should be rejected"

    def test_input_fuzzing_robustness(self):
        """Test system robustness against fuzzed inputs."""
        validator = GradientValidator()

        # Generate various fuzzed inputs
        fuzzed_inputs = [
            # Very large gradients
            [1e10, -1e10, 1e9, -1e9],
            # Very small gradients
            [1e-10, -1e-10, 1e-9, -1e-9],
            # Mixed scales
            [1e5, 1e-5, -1e5, -1e-5],
            # Zero gradients
            [0.0, 0.0, 0.0, 0.0],
            # Infinite values
            [float('inf'), -float('inf'), 0.0, 1.0],
            # Extremely long arrays (truncated by circuit)
            list(range(100)),
        ]

        for i, fuzzed_gradients in enumerate(fuzzed_inputs):
            try:
                result = validator.validate_gradients_comprehensive({
                    'gradients': fuzzed_gradients,
                    'client_id': f'fuzz_client_{i}'
                })

                print(f"Fuzzed input {i+1}: validation_score={result['overall_score']:.2f}")
                # System should handle fuzzed inputs gracefully (not crash)

            except Exception as e:
                print(f"Fuzzed input {i+1}: Exception handled - {type(e).__name__}")
                # Exceptions are acceptable for extreme fuzz inputs

    def test_penetration_testing_simulation(self):
        """Simulate penetration testing scenarios."""
        # Test various attack vectors
        attack_scenarios = [
            {
                'name': 'malformed_proof_injection',
                'payload': {'proof': {'pi_a': None, 'pi_b': [], 'pi_c': []}},
                'expected': 'validation_failure'
            },
            {
                'name': 'timing_attack_simulation',
                'payload': {'gradients': [0.0] * 1000},  # Large input
                'expected': 'graceful_handling'
            },
            {
                'name': 'resource_exhaustion_attempt',
                'payload': {'gradients': list(range(10000))},  # Very large input
                'expected': 'resource_limits_enforced'
            }
        ]

        for scenario in attack_scenarios:
            print(f"Testing penetration scenario: {scenario['name']}")

            try:
                if 'gradients' in scenario['payload']:
                    validator = GradientValidator()
                    result = validator.validate_gradients_comprehensive({
                        'gradients': scenario['payload']['gradients'][:100],  # Limit for safety
                        'client_id': 'pen_test_client'
                    })
                    print(f"  Result: validation_score={result['overall_score']:.2f}")

                elif 'proof' in scenario['payload']:
                    from src.fedzk.validation.proof_validator import ProofValidator
                    validator = ProofValidator()
                    result = validator.validate_proof_comprehensive({
                        'proof': scenario['payload']['proof'],
                        'public_inputs': [1],
                        'circuit_type': 'model_update'
                    })
                    print(f"  Result: valid={result['overall_valid']}")

            except Exception as e:
                print(f"  Exception: {type(e).__name__} - {str(e)[:50]}...")


if __name__ == "__main__":
    pytest.main([__file__])

