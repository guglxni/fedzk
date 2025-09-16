#!/usr/bin/env python3
"""
Integration Tests for Task 9 Optimization Components
==================================================

End-to-end integration testing for all Task 9 optimization features.
"""

import unittest
import asyncio
import time
import json
import threading
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mock dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    import sys
    sys.modules['psutil'] = MagicMock()
    PSUTIL_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    import sys
    sys.modules['msgpack'] = MagicMock()
    MSGPACK_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    import sys
    sys.modules['numpy'] = MagicMock()
    NUMPY_AVAILABLE = False

# Import optimization components
try:
    from fedzk.zk.circuits.model_update_optimized import OptimizedModelUpdateCircuit
    from fedzk.zk.proof_optimizer import OptimizedZKProofGenerator, ProofGenerationConfig
    from fedzk.core.resource_optimizer import OptimizedResourceManager, ResourceConfig
    from fedzk.core.scalability_manager import ScalabilityManager, ScalabilityConfig
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"Optimization components not available: {e}")


class TestCircuitProofIntegration(unittest.TestCase):
    """Integration tests for circuit and proof generation optimization"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        self.circuit = OptimizedModelUpdateCircuit()
        self.proof_generator = OptimizedZKProofGenerator()

    def test_circuit_to_proof_pipeline(self):
        """Test complete pipeline from circuit optimization to proof generation"""
        # 1. Prepare test inputs
        test_inputs = {
            "gradients": [0.1, -0.2, 0.3, 0.05, 0.15],
            "weights": [1.0, 0.8, 1.2, 0.9, 0.95],
            "learningRate": 0.01,
            "maxNormBound": 10000,
            "minNonZeroBound": 1
        }

        # 2. Generate optimized witness
        witness = self.circuit.generate_witness(test_inputs)
        self.assertIn("witness", witness)
        self.assertIn("metadata", witness)

        # 3. Generate proof using optimized generator
        proof_request = {
            "circuit": "model_update_optimized",
            "inputs": test_inputs
        }

        proof_result = asyncio.run(self.proof_generator.generate_optimized_proof(
            proof_request["circuit"],
            proof_request["inputs"],
            {"memory_optimization": True, "parallel": True}
        ))

        self.assertTrue(proof_result.get("success", False))
        self.assertIn("proof", proof_result)
        self.assertIn("metadata", proof_result)

    def test_batch_circuit_proof_integration(self):
        """Test batch processing integration"""
        # Create batch of test inputs
        batch_inputs = []
        for i in range(5):
            inputs = {
                "gradients": [0.1 * (i+1), -0.2 * (i+1), 0.3 * (i+1), 0.05 * (i+1)],
                "weights": [1.0, 0.8, 1.2, 0.9],
                "learningRate": 0.01
            }
            batch_inputs.append(inputs)

        # Generate batch witnesses
        start_time = time.time()
        batch_witnesses = self.circuit.generate_batch_witness(batch_inputs)
        witness_time = time.time() - start_time

        self.assertEqual(batch_witnesses["batch_size"], 5)
        self.assertLess(witness_time, 2.0)  # Should be fast

        # Generate batch proofs
        proof_requests = [
            {"circuit": "model_update_optimized", "inputs": inputs}
            for inputs in batch_inputs
        ]

        start_time = time.time()
        proof_results = asyncio.run(self.proof_generator.generate_proofs_async(proof_requests))
        proof_time = time.time() - start_time

        successful_proofs = sum(1 for r in proof_results if r.get("success", False))
        self.assertEqual(successful_proofs, 5)
        self.assertLess(proof_time, 5.0)  # Should complete within reasonable time

    def test_optimization_feature_integration(self):
        """Test integration of multiple optimization features"""
        test_inputs = {
            "gradients": [0.1, -0.2, 0.3, 0.05] * 25,  # Larger dataset
            "weights": [1.0, 0.8, 1.2, 0.9] * 25,
            "learningRate": 0.01
        }

        # Test with all optimizations enabled
        optimization_flags = {
            "memory_optimization": True,
            "parallel": True,
            "gpu_acceleration": False,  # Mock for testing
            "proof_compression": True
        }

        proof_request = {
            "circuit": "model_update_optimized",
            "inputs": test_inputs
        }

        result = asyncio.run(self.proof_generator.generate_optimized_proof(
            proof_request["circuit"],
            proof_request["inputs"],
            optimization_flags
        ))

        self.assertTrue(result.get("success", False))

        # Verify optimization metadata
        metadata = result.get("metadata", {})
        if metadata:
            self.assertIn("parallel_execution", metadata)
            self.assertIn("memory_optimized", metadata)


class TestResourceScalabilityIntegration(unittest.TestCase):
    """Integration tests for resource optimization and scalability"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        self.resource_optimizer = OptimizedResourceManager()
        self.scalability_manager = ScalabilityManager()

        # Register test workers
        for i in range(3):
            self.scalability_manager.register_worker_node(
                f"integration-worker-{i}",
                "localhost",
                10000 + i
            )

    def test_resource_scalability_pipeline(self):
        """Test complete resource and scalability pipeline"""
        # 1. Prepare large test data
        large_data = {
            "gradients": [0.1] * 500,
            "weights": [1.0] * 500,
            "metadata": {
                "model_version": "v3.0",
                "batch_size": 500,
                "optimization_level": "maximum"
            }
        }

        # 2. Optimize data transmission
        optimized_data = self.resource_optimizer.optimize_request(large_data)
        self.assertIn("data", optimized_data)

        # 3. Process through scalability layer
        result = asyncio.run(self.scalability_manager.process_request(large_data))
        self.assertTrue(result.get("success", False))

        # 4. Verify resource efficiency
        resource_stats = self.resource_optimizer.get_resource_stats()
        scalability_stats = self.scalability_manager.get_scalability_stats()

        self.assertIn("memory_pools", resource_stats)
        self.assertIn("horizontal_scaling", scalability_stats)

    def test_concurrent_resource_scalability(self):
        """Test concurrent resource usage with scalability"""
        # Create multiple concurrent requests
        concurrent_requests = []
        for i in range(10):
            request = {
                "gradients": [0.1] * (20 + i),
                "weights": [1.0] * (20 + i),
                "request_id": f"concurrent-{i}"
            }
            concurrent_requests.append(request)

        # Process requests concurrently
        start_time = time.time()

        async def process_concurrent():
            tasks = [self.scalability_manager.process_request(req)
                    for req in concurrent_requests]
            return await asyncio.gather(*tasks)

        results = asyncio.run(process_concurrent())
        total_time = time.time() - start_time

        # Verify all requests processed successfully
        successful_requests = sum(1 for r in results if r and r.get("success"))
        self.assertEqual(successful_requests, len(concurrent_requests))

        # Verify reasonable performance
        self.assertLess(total_time, 10.0)  # Should complete within 10 seconds

        # Check resource usage
        resource_stats = self.resource_optimizer.get_resource_stats()
        self.assertIn("connection_pools", resource_stats)

    def test_load_balancing_resource_integration(self):
        """Test load balancing with resource optimization"""
        # Create requests with different sizes to test load balancing
        test_requests = [
            {"gradients": [0.1] * 10, "weights": [1.0] * 10, "size": "small"},
            {"gradients": [0.1] * 50, "weights": [1.0] * 50, "size": "medium"},
            {"gradients": [0.1] * 100, "weights": [1.0] * 100, "size": "large"},
            {"gradients": [0.1] * 10, "weights": [1.0] * 10, "size": "small"}
        ]

        # Process requests and track worker assignments
        worker_assignments = {}

        for request in test_requests:
            result = asyncio.run(self.scalability_manager.process_request(request))
            if result and result.get("success"):
                worker_id = result.get("worker_id")
                if worker_id:
                    worker_assignments[worker_id] = worker_assignments.get(worker_id, 0) + 1

        # Verify load distribution
        self.assertGreater(len(worker_assignments), 0)

        # Check load balancing stats
        lb_stats = self.scalability_manager.load_balancer.get_load_stats()
        self.assertIn("load_distribution", lb_stats)


class TestEndToEndOptimization(unittest.TestCase):
    """End-to-end tests for complete optimization pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        # Initialize complete optimization pipeline
        self.circuit = OptimizedModelUpdateCircuit()
        self.proof_generator = OptimizedZKProofGenerator()
        self.resource_optimizer = OptimizedResourceManager()
        self.scalability_manager = ScalabilityManager()

        # Set up test workers
        for i in range(2):
            self.scalability_manager.register_worker_node(
                f"e2e-worker-{i}",
                "localhost",
                11000 + i
            )

    def test_complete_federated_learning_pipeline(self):
        """Test complete federated learning pipeline with all optimizations"""
        # Simulate federated learning scenario
        participants = 3
        model_updates = []

        # Generate model updates from multiple participants
        for participant_id in range(participants):
            # Participant data
            gradients = [0.1 * (participant_id + 1) + 0.05 * i for i in range(8)]
            weights = [1.0 - 0.1 * participant_id + 0.02 * i for i in range(8)]

            participant_data = {
                "participant_id": participant_id,
                "gradients": gradients,
                "weights": weights,
                "learning_rate": 0.01,
                "model_version": "v2.1"
            }

            model_updates.append(participant_data)

        # Process each participant's update
        processed_updates = []
        proof_results = []

        for update in model_updates:
            # 1. Optimize data transmission
            optimized_data = self.resource_optimizer.optimize_request(update)
            processed_updates.append(optimized_data)

            # 2. Generate optimized witness
            witness = self.circuit.generate_witness(update)

            # 3. Generate optimized proof
            proof_request = {
                "circuit": "model_update_optimized",
                "inputs": update
            }

            proof_result = asyncio.run(self.proof_generator.generate_optimized_proof(
                proof_request["circuit"],
                proof_request["inputs"]
            ))
            proof_results.append(proof_result)

        # Verify all components worked
        self.assertEqual(len(processed_updates), participants)
        self.assertEqual(len(proof_results), participants)

        for proof_result in proof_results:
            self.assertTrue(proof_result.get("success", False))

        # Aggregate results (simulate coordinator)
        aggregated_result = {
            "total_participants": participants,
            "successful_proofs": sum(1 for p in proof_results if p.get("success")),
            "total_processing_time": sum(p.get("execution_time", 0) for p in proof_results),
            "optimization_applied": True
        }

        self.assertEqual(aggregated_result["successful_proofs"], participants)
        self.assertGreater(aggregated_result["total_processing_time"], 0)

    def test_performance_optimization_verification(self):
        """Test and verify performance optimizations"""
        # Test data for performance verification
        performance_test_data = {
            "gradients": [0.1] * 100,
            "weights": [1.0] * 100,
            "learning_rate": 0.01,
            "test_mode": "performance_verification"
        }

        # Measure baseline performance
        start_time = time.time()

        # 1. Circuit optimization
        witness = self.circuit.generate_witness(performance_test_data)

        # 2. Resource optimization
        optimized_data = self.resource_optimizer.optimize_request(performance_test_data)

        # 3. Scalability processing
        scalability_result = asyncio.run(
            self.scalability_manager.process_request(performance_test_data)
        )

        # 4. Proof generation
        proof_result = asyncio.run(self.proof_generator.generate_optimized_proof(
            "model_update_optimized",
            performance_test_data
        ))

        total_time = time.time() - start_time

        # Verify all optimizations worked
        self.assertIn("witness", witness)
        self.assertIn("data", optimized_data)
        self.assertTrue(scalability_result.get("success", False))
        self.assertTrue(proof_result.get("success", False))

        # Verify performance is reasonable
        self.assertLess(total_time, 5.0)  # Complete pipeline within 5 seconds

        # Generate performance report
        performance_report = {
            "total_time": total_time,
            "components_tested": 4,
            "optimizations_applied": [
                "circuit_optimization",
                "resource_optimization",
                "scalability_processing",
                "proof_generation"
            ],
            "success_rate": 1.0 if all([
                "witness" in witness,
                "data" in optimized_data,
                scalability_result.get("success", False),
                proof_result.get("success", False)
            ]) else 0.0
        }

        self.assertEqual(performance_report["success_rate"], 1.0)

    def test_optimization_failure_handling(self):
        """Test optimization pipeline failure handling"""
        # Test with invalid data to verify error handling
        invalid_data = {
            "gradients": [],  # Empty gradients
            "weights": [1.0, 2.0],
            "invalid_field": "test"
        }

        # Test circuit validation
        is_valid = self.circuit.validate_inputs(invalid_data)
        self.assertFalse(is_valid)

        # Test with minimal valid data
        valid_data = {
            "gradients": [0.1, 0.2],
            "weights": [1.0, 1.1],
            "learning_rate": 0.01
        }

        # Verify pipeline works with valid data
        witness = self.circuit.generate_witness(valid_data)
        optimized_data = self.resource_optimizer.optimize_request(valid_data)
        scalability_result = asyncio.run(
            self.scalability_manager.process_request(valid_data)
        )

        # All should succeed with valid data
        self.assertIn("witness", witness)
        self.assertIn("data", optimized_data)
        self.assertTrue(scalability_result.get("success", False))


class TestOptimizationMonitoring(unittest.TestCase):
    """Tests for optimization monitoring and metrics"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        self.proof_generator = OptimizedZKProofGenerator()
        self.resource_optimizer = OptimizedResourceManager()
        self.scalability_manager = ScalabilityManager()

    def test_optimization_metrics_collection(self):
        """Test collection of optimization metrics"""
        # Generate optimization activity
        test_data = {"gradients": [0.1, 0.2, 0.3], "weights": [1.0, 1.1, 1.2]}

        # Perform optimization operations
        asyncio.run(self.proof_generator.generate_optimized_proof(
            "model_update_optimized", test_data
        ))
        self.resource_optimizer.optimize_request(test_data)
        asyncio.run(self.scalability_manager.process_request(test_data))

        # Collect metrics from all components
        proof_stats = self.proof_generator.get_optimization_stats()
        resource_stats = self.resource_optimizer.get_resource_stats()
        scalability_stats = self.scalability_manager.get_scalability_stats()

        # Verify metrics structure
        self.assertIn("cache_stats", proof_stats)
        self.assertIn("memory_pools", resource_stats)
        self.assertIn("horizontal_scaling", scalability_stats)

        # Verify metrics contain expected data
        cache_stats = proof_stats["cache_stats"]
        self.assertIn("cache_size", cache_stats)
        self.assertIn("hit_ratio", cache_stats)

    def test_resource_usage_tracking(self):
        """Test tracking of resource usage during optimization"""
        # Perform resource-intensive operations
        large_data = {
            "gradients": [0.1] * 200,
            "weights": [1.0] * 200,
            "metadata": {"size": "large", "optimization_test": True}
        }

        # Track resource usage before
        initial_memory = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0

        # Perform optimizations
        optimized = self.resource_optimizer.optimize_request(large_data)
        result = asyncio.run(self.scalability_manager.process_request(large_data))

        # Track resource usage after
        final_memory = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0

        # Verify operation completed
        self.assertIn("data", optimized)
        self.assertTrue(result.get("success", False))

        # Resource monitoring should work (even with mocks)
        monitoring_stats = self.resource_optimizer.get_resource_stats()
        self.assertIn("resource_monitoring", monitoring_stats)


if __name__ == '__main__':
    unittest.main()
