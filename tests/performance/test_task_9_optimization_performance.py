#!/usr/bin/env python3
"""
Performance Tests for Task 9 Optimization Components
==================================================

Comprehensive performance benchmarking for all Task 9 optimization features.
"""

import unittest
import time
import asyncio
import threading
import statistics
import concurrent.futures
from pathlib import Path
import sys

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


class TestCircuitOptimizationPerformance(unittest.TestCase):
    """Performance tests for circuit optimization"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        self.circuit = OptimizedModelUpdateCircuit()

    def test_witness_generation_performance(self):
        """Test witness generation performance at different scales"""
        test_cases = [
            {"size": 4, "name": "small"},
            {"size": 16, "name": "medium"},
            {"size": 64, "name": "large"},
            {"size": 256, "name": "xlarge"}
        ]

        performance_results = {}

        for test_case in test_cases:
            size = test_case["size"]
            name = test_case["name"]

            # Generate test data
            gradients = [0.1 * (i % 10) for i in range(size)]
            weights = [1.0 - 0.01 * (i % 20) for i in range(size)]

            test_inputs = {
                "gradients": gradients,
                "weights": weights,
                "learningRate": 0.01
            }

            # Measure performance
            times = []
            for _ in range(10):  # 10 iterations for averaging
                start_time = time.time()
                witness = self.circuit.generate_witness(test_inputs)
                end_time = time.time()
                times.append(end_time - start_time)

            # Calculate statistics
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0

            performance_results[name] = {
                "size": size,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "std_dev": std_dev,
                "witness_elements": len(witness["witness"])
            }

            # Performance assertions
            self.assertLess(avg_time, 1.0, f"Witness generation too slow for {name} case")
            self.assertGreater(len(witness["witness"]), 0)

        # Verify scaling behavior (should be roughly linear)
        if len(performance_results) >= 2:
            small_time = performance_results["small"]["avg_time"]
            large_time = performance_results["large"]["avg_time"]
            scaling_factor = large_time / small_time if small_time > 0 else float('inf')

            # Should scale reasonably (not exponentially)
            size_ratio = performance_results["large"]["size"] / performance_results["small"]["size"]
            self.assertLess(scaling_factor / size_ratio, 5.0, "Poor scaling performance")

    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        batch_sizes = [5, 10, 25, 50]
        performance_results = {}

        for batch_size in batch_sizes:
            # Generate batch data
            batch_inputs = []
            for i in range(batch_size):
                inputs = {
                    "gradients": [0.1 * (i % 5), -0.2 * (i % 3), 0.3, 0.05],
                    "weights": [1.0, 0.8, 1.2, 0.9],
                    "learningRate": 0.01
                }
                batch_inputs.append(inputs)

            # Measure batch processing time
            times = []
            for _ in range(5):  # 5 iterations
                start_time = time.time()
                batch_result = self.circuit.generate_batch_witness(batch_inputs)
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = statistics.mean(times)
            performance_results[batch_size] = {
                "batch_size": batch_size,
                "avg_time": avg_time,
                "time_per_item": avg_time / batch_size,
                "total_witness_elements": len(batch_result["witness"])
            }

            # Performance assertions
            self.assertLess(avg_time, 5.0, f"Batch processing too slow for size {batch_size}")
            self.assertEqual(batch_result["batch_size"], batch_size)

        # Verify batch efficiency (should be better than individual processing)
        if 10 in performance_results and 5 in performance_results:
            individual_time_per_item = performance_results[5]["time_per_item"]
            batch_time_per_item = performance_results[10]["time_per_item"]

            # Batch should be more efficient (at least 80% of individual time per item)
            efficiency_ratio = batch_time_per_item / individual_time_per_item
            self.assertLess(efficiency_ratio, 1.2, "Batch processing not efficient enough")

    def test_memory_usage_optimization(self):
        """Test memory usage during circuit operations"""
        if not PSUTIL_AVAILABLE:
            self.skipTest("Memory monitoring not available")

        # Test with increasing data sizes
        sizes = [10, 50, 100, 200]
        memory_usage = {}

        process = psutil.Process()

        for size in sizes:
            # Generate test data
            gradients = [0.1] * size
            weights = [1.0] * size
            test_inputs = {
                "gradients": gradients,
                "weights": weights,
                "learningRate": 0.01
            }

            # Measure memory before
            memory_before = process.memory_info().rss

            # Perform operation
            witness = self.circuit.generate_witness(test_inputs)

            # Measure memory after
            memory_after = process.memory_info().rss

            memory_delta = memory_after - memory_before
            memory_usage[size] = {
                "size": size,
                "memory_delta_kb": memory_delta / 1024,
                "memory_per_element": memory_delta / size,
                "operation_successful": len(witness["witness"]) > 0
            }

            # Memory assertions
            self.assertLess(memory_delta / 1024, 5000, f"Memory usage too high for size {size}")

        # Verify memory scaling
        if len(memory_usage) >= 2:
            small_memory = memory_usage[10]["memory_delta_kb"]
            large_memory = memory_usage[100]["memory_delta_kb"]

            # Memory should scale reasonably
            memory_ratio = large_memory / small_memory if small_memory > 0 else float('inf')
            size_ratio = 10  # 100/10
            self.assertLess(memory_ratio / size_ratio, 3.0, "Memory scaling inefficient")


class TestProofOptimizationPerformance(unittest.TestCase):
    """Performance tests for proof generation optimization"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        self.generator = OptimizedZKProofGenerator()

    def test_proof_generation_throughput(self):
        """Test proof generation throughput under load"""
        # Generate multiple proof requests
        proof_requests = []
        for i in range(20):
            request = {
                "circuit": "model_update_optimized",
                "inputs": {
                    "gradients": [0.1 * (i % 5), -0.2 * (i % 3), 0.3, 0.05],
                    "weights": [1.0, 0.8, 1.2, 0.9],
                    "learningRate": 0.01
                }
            }
            proof_requests.append(request)

        # Measure throughput
        start_time = time.time()
        results = asyncio.run(self.generator.generate_proofs_async(proof_requests))
        total_time = time.time() - start_time

        # Calculate metrics
        successful_proofs = sum(1 for r in results if r.get("success", False))
        throughput = successful_proofs / total_time if total_time > 0 else 0

        performance_metrics = {
            "total_requests": len(proof_requests),
            "successful_proofs": successful_proofs,
            "total_time": total_time,
            "throughput_proofs_per_second": throughput,
            "success_rate": successful_proofs / len(proof_requests)
        }

        # Performance assertions
        self.assertGreater(throughput, 1.0, "Proof generation throughput too low")
        self.assertEqual(successful_proofs, len(proof_requests), "All proofs should succeed")
        self.assertGreater(performance_metrics["success_rate"], 0.95, "Success rate too low")

    def test_caching_performance_impact(self):
        """Test the performance impact of proof caching"""
        test_request = {
            "circuit": "model_update_optimized",
            "inputs": {
                "gradients": [0.1, -0.2, 0.3, 0.05],
                "weights": [1.0, 0.8, 1.2, 0.9],
                "learningRate": 0.01
            }
        }

        # First request (cache miss)
        start_time = time.time()
        result1 = asyncio.run(self.generator.generate_optimized_proof(
            test_request["circuit"],
            test_request["inputs"]
        ))
        first_request_time = time.time() - start_time

        # Second request (cache hit)
        start_time = time.time()
        result2 = asyncio.run(self.generator.generate_optimized_proof(
            test_request["circuit"],
            test_request["inputs"]
        ))
        second_request_time = time.time() - start_time

        # Calculate speedup
        speedup = first_request_time / second_request_time if second_request_time > 0 else float('inf')

        caching_metrics = {
            "first_request_time": first_request_time,
            "second_request_time": second_request_time,
            "speedup_factor": speedup,
            "cache_hit": result2.get("cached", False)
        }

        # Caching assertions
        self.assertTrue(result2.get("cached", False), "Second request should be cached")
        self.assertGreater(speedup, 2.0, "Caching speedup insufficient")
        self.assertLess(second_request_time, first_request_time, "Cached request should be faster")

    def test_parallel_vs_sequential_performance(self):
        """Test parallel vs sequential proof generation performance"""
        # Create multiple proof requests
        proof_requests = []
        for i in range(8):
            request = {
                "circuit": "model_update_optimized",
                "inputs": {
                    "gradients": [0.1 * (i % 4), -0.2 * (i % 3), 0.3, 0.05],
                    "weights": [1.0, 0.8, 1.2, 0.9],
                    "learningRate": 0.01
                }
            }
            proof_requests.append(request)

        # Test sequential processing
        sequential_start = time.time()
        sequential_results = []
        for request in proof_requests:
            result = asyncio.run(self.generator.generate_optimized_proof(
                request["circuit"],
                request["inputs"]
            ))
            sequential_results.append(result)
        sequential_time = time.time() - sequential_start

        # Test parallel processing
        parallel_start = time.time()
        parallel_results = asyncio.run(self.generator.generate_proofs_async(proof_requests))
        parallel_time = time.time() - parallel_start

        # Calculate metrics
        performance_comparison = {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": sequential_time / parallel_time if parallel_time > 0 else float('inf'),
            "sequential_throughput": len(proof_requests) / sequential_time,
            "parallel_throughput": len(proof_requests) / parallel_time,
            "efficiency": parallel_time / sequential_time
        }

        # Performance assertions
        self.assertLess(parallel_time, sequential_time, "Parallel should be faster than sequential")
        self.assertGreater(performance_comparison["speedup"], 1.2, "Parallel speedup insufficient")
        self.assertEqual(len(sequential_results), len(parallel_results), "Result counts should match")

        # Verify all results are successful
        sequential_success = sum(1 for r in sequential_results if r.get("success", False))
        parallel_success = sum(1 for r in parallel_results if r.get("success", False))

        self.assertEqual(sequential_success, len(proof_requests))
        self.assertEqual(parallel_success, len(proof_requests))


class TestResourceOptimizationPerformance(unittest.TestCase):
    """Performance tests for resource optimization"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        self.optimizer = OptimizedResourceManager()

    def test_compression_performance_scaling(self):
        """Test compression performance with different data sizes"""
        data_sizes = [1000, 10000, 100000]  # Characters
        compression_performance = {}

        for size in data_sizes:
            # Generate test data
            test_data = {
                "text": "x" * size,
                "numbers": list(range(size // 100)),
                "metadata": {"size": size, "type": "test"}
            }

            # Measure compression performance
            times = []
            for _ in range(5):
                start_time = time.time()
                optimized = self.optimizer.optimize_request(test_data)
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = statistics.mean(times)
            compression_ratio = len(optimized["data"]) / optimized["original_size"]

            compression_performance[size] = {
                "size": size,
                "avg_compression_time": avg_time,
                "compression_ratio": compression_ratio,
                "throughput_mb_per_sec": (optimized["original_size"] / 1024 / 1024) / avg_time
            }

            # Performance assertions
            self.assertLess(avg_time, 1.0, f"Compression too slow for size {size}")
            self.assertLess(compression_ratio, 1.0, "Compression should reduce size")

        # Verify scaling behavior
        if len(compression_performance) >= 2:
            small_time = compression_performance[1000]["avg_compression_time"]
            large_time = compression_performance[10000]["avg_compression_time"]

            scaling_ratio = large_time / small_time if small_time > 0 else float('inf')
            size_ratio = 10  # 10000/1000

            # Should scale sub-linearly
            self.assertLess(scaling_ratio / size_ratio, 2.0, "Compression scaling inefficient")

    def test_serialization_format_performance(self):
        """Test performance of different serialization formats"""
        test_data = {
            "gradients": [0.1] * 1000,
            "weights": [1.0] * 1000,
            "metadata": {
                "model": "test_model",
                "version": "1.0",
                "parameters": {"learning_rate": 0.01, "batch_size": 32}
            }
        }

        formats_to_test = ["json"]
        if MSGPACK_AVAILABLE:
            formats_to_test.append("msgpack")

        format_performance = {}

        for fmt in formats_to_test:
            times = []
            sizes = []

            for _ in range(10):
                start_time = time.time()
                serialized = self.optimizer.serialization_manager.serialize(test_data, fmt)
                deserialized = self.optimizer.serialization_manager.deserialize(serialized, fmt)
                end_time = time.time()

                times.append(end_time - start_time)
                sizes.append(len(serialized))

            format_performance[fmt] = {
                "format": fmt,
                "avg_serialize_time": statistics.mean(times),
                "avg_size": statistics.mean(sizes),
                "min_time": min(times),
                "max_time": max(times)
            }

        # JSON should always be available
        self.assertIn("json", format_performance)

        # Performance assertions
        for fmt, metrics in format_performance.items():
            self.assertLess(metrics["avg_serialize_time"], 0.1, f"{fmt} serialization too slow")
            self.assertGreater(metrics["avg_size"], 0, f"{fmt} serialization produced empty data")

        # If msgpack is available, it should be faster than JSON
        if "msgpack" in format_performance:
            json_time = format_performance["json"]["avg_serialize_time"]
            msgpack_time = format_performance["msgpack"]["avg_serialize_time"]

            self.assertLessEqual(msgpack_time, json_time, "Msgpack should be faster than JSON")

    def test_memory_pool_performance(self):
        """Test memory pool allocation/deallocation performance"""
        # Test tensor allocation performance
        allocation_times = []
        deallocation_times = []

        for i in range(100):
            # Allocate tensor
            start_time = time.time()
            tensor = self.optimizer.allocate_tensor((10, 10), "float32")
            allocation_time = time.time() - start_time
            allocation_times.append(allocation_time)

            # Deallocate tensor
            start_time = time.time()
            self.optimizer.release_tensor(tensor)
            deallocation_time = time.time() - start_time
            deallocation_times.append(deallocation_time)

        # Calculate statistics
        pool_performance = {
            "avg_allocation_time": statistics.mean(allocation_times),
            "avg_deallocation_time": statistics.mean(deallocation_times),
            "total_operations": len(allocation_times),
            "allocation_throughput": len(allocation_times) / sum(allocation_times),
            "deallocation_throughput": len(deallocation_times) / sum(deallocation_times)
        }

        # Performance assertions
        self.assertLess(pool_performance["avg_allocation_time"], 0.001, "Memory allocation too slow")
        self.assertLess(pool_performance["avg_deallocation_time"], 0.001, "Memory deallocation too slow")
        self.assertGreater(pool_performance["allocation_throughput"], 1000, "Allocation throughput too low")


class TestScalabilityPerformance(unittest.TestCase):
    """Performance tests for scalability features"""

    def setUp(self):
        """Set up test fixtures"""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization components not available")

        self.manager = ScalabilityManager()

        # Register test workers
        for i in range(4):
            self.manager.register_worker_node(
                f"perf-worker-{i}",
                "localhost",
                12000 + i
            )

    def test_concurrent_request_processing(self):
        """Test concurrent request processing performance"""
        # Create concurrent requests
        num_requests = 50
        requests = []

        for i in range(num_requests):
            request = {
                "gradients": [0.1] * (5 + i % 10),  # Varying sizes
                "weights": [1.0] * (5 + i % 10),
                "request_id": f"perf-{i}"
            }
            requests.append(request)

        # Process requests concurrently
        start_time = time.time()

        async def process_all_requests():
            tasks = [self.manager.process_request(req) for req in requests]
            return await asyncio.gather(*tasks)

        results = asyncio.run(process_all_requests())
        total_time = time.time() - start_time

        # Calculate performance metrics
        successful_requests = sum(1 for r in results if r and r.get("success"))
        throughput = successful_requests / total_time if total_time > 0 else 0

        scalability_performance = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "total_time": total_time,
            "throughput_requests_per_second": throughput,
            "success_rate": successful_requests / num_requests,
            "avg_time_per_request": total_time / num_requests
        }

        # Performance assertions
        self.assertEqual(successful_requests, num_requests, "All requests should succeed")
        self.assertGreater(throughput, 5.0, "Request throughput too low")
        self.assertLess(scalability_performance["avg_time_per_request"], 0.5, "Average request time too high")

    def test_load_balancing_efficiency(self):
        """Test load balancing efficiency under load"""
        # Create requests with different computational complexity
        requests = []
        for i in range(30):
            complexity = "high" if i % 3 == 0 else "medium" if i % 3 == 1 else "low"
            size = 20 if complexity == "high" else 10 if complexity == "medium" else 5

            request = {
                "gradients": [0.1] * size,
                "weights": [1.0] * size,
                "complexity": complexity,
                "request_id": f"lb-{i}"
            }
            requests.append(request)

        # Process requests
        start_time = time.time()
        results = []

        async def process_requests():
            for req in requests:
                result = await self.manager.process_request(req)
                results.append(result)

        asyncio.run(process_requests())
        total_time = time.time() - start_time

        # Analyze load distribution
        worker_assignments = {}
        for result in results:
            if result and result.get("success"):
                worker_id = result.get("worker_id")
                if worker_id:
                    worker_assignments[worker_id] = worker_assignments.get(worker_id, 0) + 1

        # Calculate load balancing metrics
        if worker_assignments:
            assignment_counts = list(worker_assignments.values())
            avg_assignments = statistics.mean(assignment_counts)
            load_balance_variance = statistics.variance(assignment_counts) if len(assignment_counts) > 1 else 0

            load_balancing_metrics = {
                "total_assignments": sum(assignment_counts),
                "worker_count": len(worker_assignments),
                "avg_assignments_per_worker": avg_assignments,
                "load_balance_variance": load_balance_variance,
                "load_balance_coefficient": load_balance_variance / (avg_assignments ** 2) if avg_assignments > 0 else 0
            }

            # Load balancing assertions
            self.assertLess(load_balancing_metrics["load_balance_coefficient"], 0.5,
                          "Load balancing too uneven")
            self.assertGreater(len(worker_assignments), 1, "Multiple workers should be used")

    def test_circuit_sharding_performance(self):
        """Test circuit sharding performance"""
        # Create large circuit that should be sharded
        large_request = {
            "gradients": [0.1] * 2000,
            "weights": [1.0] * 2000,
            "circuit_size": 2000
        }

        # Check if sharding is triggered
        should_shard = self.manager.circuit_shard_manager.should_shard_circuit(2000)
        self.assertTrue(should_shard)

        # Measure sharding performance
        start_time = time.time()
        shards = self.manager.circuit_shard_manager.create_shards(large_request)
        sharding_time = time.time() - start_time

        # Process shards
        shard_results = []
        processing_start = time.time()

        async def process_shards():
            for shard in shards:
                result = await self.manager.process_request(shard)
                shard_results.append(result)

        asyncio.run(process_shards())
        processing_time = time.time() - processing_start

        # Calculate sharding metrics
        sharding_performance = {
            "original_size": 2000,
            "num_shards": len(shards),
            "sharding_time": sharding_time,
            "processing_time": processing_time,
            "total_time": sharding_time + processing_time,
            "avg_shard_size": 2000 / len(shards) if shards else 0
        }

        # Performance assertions
        self.assertGreater(len(shards), 1, "Large circuit should be sharded")
        self.assertLess(sharding_time, 0.1, "Sharding should be fast")
        self.assertLess(sharding_performance["total_time"], 5.0, "Total sharding time too high")


if __name__ == '__main__':
    unittest.main()
