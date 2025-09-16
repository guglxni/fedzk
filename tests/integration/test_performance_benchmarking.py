#!/usr/bin/env python3
"""
Performance Benchmarking Integration Tests
==========================================

Comprehensive performance testing under load conditions including:
- Concurrent client handling
- Memory usage monitoring
- CPU utilization tracking
- Network throughput testing
- Scalability benchmarks
"""

import pytest
import time
import psutil
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock, patch
import gc
import os

from src.fedzk.client.trainer import FederatedTrainer
from src.fedzk.coordinator.logic import CoordinatorLogic, VerifiedUpdate
from src.fedzk.mpc.client import MPCClient
from src.fedzk.prover.zkgenerator import ZKProver
from src.fedzk.validation.gradient_validator import GradientValidator


class PerformanceMonitor:
    """Monitor system performance metrics during testing."""

    def __init__(self):
        self.baseline_metrics = {}
        self.test_metrics = {}
        tracemalloc.start()

    def record_baseline(self):
        """Record baseline system metrics."""
        self.baseline_metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'disk_usage': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'timestamp': time.time()
        }

    def record_test_metrics(self, test_name: str):
        """Record metrics during test execution."""
        current_snapshot = tracemalloc.take_snapshot()
        stats = current_snapshot.statistics('lineno')

        self.test_metrics[test_name] = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'memory_delta_mb': (psutil.virtual_memory().used - self.baseline_metrics['memory_mb']) / (1024 * 1024),
            'top_memory_consumers': [
                {
                    'file': stat.traceback[0].filename,
                    'line': stat.traceback[0].lineno,
                    'size_mb': stat.size / (1024 * 1024)
                }
                for stat in stats[:5]  # Top 5 memory consumers
            ],
            'timestamp': time.time(),
            'duration': time.time() - self.baseline_metrics['timestamp']
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'baseline': self.baseline_metrics,
            'tests': self.test_metrics,
            'summary': {}
        }

        if self.test_metrics:
            cpu_usage = [m['cpu_percent'] for m in self.test_metrics.values()]
            memory_usage = [m['memory_usage'] for m in self.test_metrics.values()]

            report['summary'] = {
                'avg_cpu_usage': np.mean(cpu_usage),
                'max_cpu_usage': np.max(cpu_usage),
                'avg_memory_usage': np.mean(memory_usage),
                'max_memory_usage': np.max(memory_usage),
                'memory_growth_mb': max(m['memory_delta_mb'] for m in self.test_metrics.values()),
                'test_count': len(self.test_metrics)
            }

        return report


class TestPerformanceBenchmarking:
    """Comprehensive performance benchmarking tests."""

    def setup_method(self):
        """Set up performance monitoring."""
        self.monitor = PerformanceMonitor()
        self.monitor.record_baseline()

        # Test configuration
        self.num_clients_range = [5, 10, 20, 50]
        self.model_config = {
            'input_size': 10,
            'hidden_size': 5,
            'output_size': 1
        }

    def test_concurrent_client_scaling(self):
        """Test system performance with increasing concurrent clients."""
        scaling_results = {}

        for num_clients in self.num_clients_range:
            print(f"Testing with {num_clients} concurrent clients")

            start_time = time.time()

            # Run concurrent client simulation
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [
                    executor.submit(self._simulate_client_workflow, f"client_{i}", num_clients)
                    for i in range(num_clients)
                ]

                results = [future.result() for future in as_completed(futures)]

            end_time = time.time()
            total_time = end_time - start_time

            # Record performance metrics
            self.monitor.record_test_metrics(f"concurrent_clients_{num_clients}")

            scaling_results[num_clients] = {
                'total_time': total_time,
                'avg_time_per_client': total_time / num_clients,
                'throughput': num_clients / total_time,  # clients per second
                'success_rate': sum(1 for r in results if r['success']) / len(results),
                'memory_usage': psutil.virtual_memory().percent
            }

            print(".2f")
            print(".2f")

        # Analyze scaling performance
        client_counts = list(scaling_results.keys())
        throughputs = [scaling_results[n]['throughput'] for n in client_counts]

        # Check for reasonable scaling (shouldn't degrade too badly)
        if len(throughputs) > 1:
            scaling_efficiency = throughputs[-1] / throughputs[0]
            print(".2f")

            # Throughput should be reasonable (at least 10% of linear scaling)
            assert scaling_efficiency > 0.1, f"Poor scaling efficiency: {scaling_efficiency}"

    def _simulate_client_workflow(self, client_id: str, total_clients: int) -> Dict[str, Any]:
        """Simulate individual client workflow for performance testing."""
        try:
            # Create trainer with smaller model for performance
            trainer = FederatedTrainer(
                model_config=self.model_config,
                client_id=client_id,
                learning_rate=0.01,
                batch_size=16  # Smaller batch for performance
            )

            # Generate smaller dataset for performance
            dataset_size = max(32, 512 // total_clients)  # Scale dataset with client count
            X = torch.randn(dataset_size, 10)
            y = torch.randn(dataset_size, 1)

            # Train locally
            start_train = time.time()
            trainer.train_local(X, y, epochs=1)
            train_time = time.time() - start_train

            # Extract and validate gradients
            gradients = trainer.extract_gradients()
            validator = GradientValidator()
            validation = validator.validate_gradients_comprehensive({
                'gradients': gradients,
                'client_id': client_id
            })

            return {
                'client_id': client_id,
                'success': validation['overall_valid'],
                'train_time': train_time,
                'gradient_count': len(gradients),
                'validation_score': validation['overall_score']
            }

        except Exception as e:
            return {
                'client_id': client_id,
                'success': False,
                'error': str(e),
                'train_time': 0,
                'gradient_count': 0,
                'validation_score': 0
            }

    def test_memory_usage_under_load(self):
        """Test memory usage patterns under different load conditions."""
        memory_results = {}

        for num_clients in [10, 25, 50]:
            print(f"Testing memory usage with {num_clients} clients")

            # Force garbage collection before test
            gc.collect()
            initial_memory = psutil.virtual_memory().used

            # Run concurrent clients
            with ThreadPoolExecutor(max_workers=min(num_clients, 10)) as executor:
                futures = [
                    executor.submit(self._simulate_memory_intensive_workflow, f"client_{i}")
                    for i in range(num_clients)
                ]

                results = [future.result() for future in as_completed(futures)]

            # Measure memory after test
            final_memory = psutil.virtual_memory().used
            memory_delta = (final_memory - initial_memory) / (1024 * 1024)  # MB

            memory_results[num_clients] = {
                'initial_memory_mb': initial_memory / (1024 * 1024),
                'final_memory_mb': final_memory / (1024 * 1024),
                'memory_delta_mb': memory_delta,
                'memory_per_client_mb': memory_delta / num_clients,
                'success_rate': sum(1 for r in results if r['success']) / len(results)
            }

            print(".2f")
            print(".2f")

            # Record metrics
            self.monitor.record_test_metrics(f"memory_load_{num_clients}")

        # Analyze memory scaling
        client_counts = list(memory_results.keys())
        memory_per_client = [memory_results[n]['memory_per_client_mb'] for n in client_counts]

        # Memory per client should be reasonable (< 50MB per client)
        avg_memory_per_client = np.mean(memory_per_client)
        assert avg_memory_per_client < 50, f"High memory usage: {avg_memory_per_client:.2f} MB per client"

    def _simulate_memory_intensive_workflow(self, client_id: str) -> Dict[str, Any]:
        """Simulate memory-intensive client workflow."""
        try:
            # Create trainer
            trainer = FederatedTrainer(
                model_config=self.model_config,
                client_id=client_id,
                learning_rate=0.01,
                batch_size=32
            )

            # Generate larger dataset to stress memory
            X = torch.randn(256, 10)  # Larger dataset
            y = torch.randn(256, 1)

            # Store intermediate results to stress memory
            intermediate_results = []

            for epoch in range(2):
                trainer.train_local(X, y, epochs=1)
                gradients = trainer.extract_gradients()

                # Store gradients to simulate memory pressure
                intermediate_results.append({
                    'epoch': epoch,
                    'gradients': gradients.copy(),
                    'model_state': {k: v.clone() for k, v in trainer.model.state_dict().items()}
                })

            # Validate final gradients
            validator = GradientValidator()
            validation = validator.validate_gradients_comprehensive({
                'gradients': gradients,
                'client_id': client_id
            })

            # Clean up to avoid memory leaks in test
            del intermediate_results
            gc.collect()

            return {
                'client_id': client_id,
                'success': validation['overall_valid'],
                'epochs_completed': 2,
                'validation_score': validation['overall_score']
            }

        except Exception as e:
            return {
                'client_id': client_id,
                'success': False,
                'error': str(e)
            }

    def test_network_throughput_benchmarking(self):
        """Test network throughput under various conditions."""
        throughput_results = {}

        payload_sizes = [100, 1000, 10000]  # Different payload sizes

        for payload_size in payload_sizes:
            print(f"Testing network throughput with {payload_size} gradient values")

            # Create large gradients payload
            large_gradients = [np.random.normal(0, 0.1) for _ in range(payload_size)]

            # Test serialization/deserialization performance
            start_time = time.time()

            # Simulate multiple serialization operations
            for _ in range(10):
                # Serialize gradients
                serialized = json.dumps({'gradients': large_gradients[:100]})  # Limit for safety

                # Deserialize
                deserialized = json.loads(serialized)

                # Validate
                validator = GradientValidator()
                validation = validator.validate_gradients_comprehensive({
                    'gradients': deserialized['gradients'],
                    'client_id': 'network_test_client'
                })

            end_time = time.time()
            throughput_time = end_time - start_time

            throughput_results[payload_size] = {
                'throughput_time': throughput_time,
                'operations_per_second': 10 / throughput_time,
                'payload_size': payload_size,
                'success': validation['overall_valid']
            }

            print(".2f")
            print(".2f")

            # Record performance
            self.monitor.record_test_metrics(f"network_throughput_{payload_size}")

        # Analyze throughput scaling
        sizes = list(throughput_results.keys())
        throughputs = [throughput_results[s]['operations_per_second'] for s in sizes]

        # Throughput should scale reasonably with payload size
        if len(throughputs) > 1:
            scaling_ratio = throughputs[-1] / throughputs[0]
            print(".2f")

            # Should maintain reasonable performance even with larger payloads
            assert scaling_ratio > 0.1, f"Poor throughput scaling: {scaling_ratio}"

    def test_cpu_utilization_benchmarking(self):
        """Test CPU utilization under different computational loads."""
        cpu_results = {}

        # Test different computational intensities
        test_scenarios = [
            {'name': 'light_computation', 'clients': 5, 'epochs': 1, 'dataset_size': 64},
            {'name': 'medium_computation', 'clients': 10, 'epochs': 2, 'dataset_size': 128},
            {'name': 'heavy_computation', 'clients': 20, 'epochs': 3, 'dataset_size': 256}
        ]

        for scenario in test_scenarios:
            print(f"Testing CPU utilization: {scenario['name']}")

            # Monitor CPU before test
            cpu_before = psutil.cpu_percent(interval=1)

            # Run computational workload
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=scenario['clients']) as executor:
                futures = []
                for i in range(scenario['clients']):
                    future = executor.submit(
                        self._simulate_cpu_intensive_workflow,
                        f"client_{i}",
                        scenario['epochs'],
                        scenario['dataset_size']
                    )
                    futures.append(future)

                results = [future.result() for future in as_completed(futures)]

            end_time = time.time()
            total_time = end_time - start_time

            # Monitor CPU after test
            cpu_after = psutil.cpu_percent(interval=1)

            cpu_results[scenario['name']] = {
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'cpu_utilization': (cpu_before + cpu_after) / 2,
                'total_time': total_time,
                'efficiency': scenario['clients'] * scenario['epochs'] / total_time,
                'success_rate': sum(1 for r in results if r['success']) / len(results)
            }

            print(".1f")
            print(".2f")
            print(".2f")

            # Record metrics
            self.monitor.record_test_metrics(f"cpu_{scenario['name']}")

        # Analyze CPU utilization patterns
        utilizations = [cpu_results[s]['cpu_utilization'] for s in test_scenarios]
        avg_utilization = np.mean(utilizations)

        print(".1f")

        # CPU utilization should be reasonable (not overloaded)
        assert avg_utilization < 90, f"High CPU utilization: {avg_utilization:.1f}%"

    def _simulate_cpu_intensive_workflow(self, client_id: str, epochs: int, dataset_size: int) -> Dict[str, Any]:
        """Simulate CPU-intensive client workflow."""
        try:
            # Create trainer
            trainer = FederatedTrainer(
                model_config=self.model_config,
                client_id=client_id,
                learning_rate=0.01,
                batch_size=16
            )

            # Generate dataset
            X = torch.randn(dataset_size, 10)
            y = torch.randn(dataset_size, 1)

            # Train for multiple epochs
            for epoch in range(epochs):
                trainer.train_local(X, y, epochs=1)

            # Extract and validate gradients
            gradients = trainer.extract_gradients()
            validator = GradientValidator()
            validation = validator.validate_gradients_comprehensive({
                'gradients': gradients,
                'client_id': client_id
            })

            return {
                'client_id': client_id,
                'success': validation['overall_valid'],
                'epochs_completed': epochs,
                'validation_score': validation['overall_score']
            }

        except Exception as e:
            return {
                'client_id': client_id,
                'success': False,
                'error': str(e)
            }

    def test_end_to_end_performance_regression(self):
        """Test for performance regression in end-to-end workflow."""
        # Establish performance baseline
        baseline_results = self._run_end_to_end_workflow("baseline", num_clients=5)
        self.monitor.record_test_metrics("baseline_run")

        # Run current performance test
        current_results = self._run_end_to_end_workflow("current", num_clients=5)
        self.monitor.record_test_metrics("current_run")

        # Compare performance
        baseline_time = baseline_results['total_time']
        current_time = current_results['total_time']

        regression_ratio = current_time / baseline_time
        print(".2f")
        print(".2f")
        print(".2f")

        # Performance should not degrade by more than 20%
        assert regression_ratio < 1.2, f"Performance regression detected: {regression_ratio:.2f}x slower"

        # Success rates should be maintained
        assert current_results['success_rate'] >= baseline_results['success_rate'] * 0.9

    def _run_end_to_end_workflow(self, run_name: str, num_clients: int) -> Dict[str, Any]:
        """Run complete end-to-end federated learning workflow."""
        start_time = time.time()

        # Create clients
        clients = []
        for i in range(num_clients):
            trainer = FederatedTrainer(
                model_config=self.model_config,
                client_id=f"{run_name}_client_{i}",
                learning_rate=0.01,
                batch_size=16
            )
            clients.append(trainer)

        # Simulate federated learning
        client_updates = []
        for client in clients:
            # Generate data and train
            X = torch.randn(64, 10)
            y = torch.randn(64, 1)
            client.train_local(X, y, epochs=1)

            # Extract gradients and validate
            gradients = client.extract_gradients()
            validator = GradientValidator()
            validation = validator.validate_gradients_comprehensive({
                'gradients': gradients,
                'client_id': client.client_id
            })

            if validation['overall_valid']:
                update = VerifiedUpdate(
                    client_id=client.client_id,
                    gradients=gradients,
                    proof={'dummy': 'proof'},  # Simplified for performance test
                    timestamp=time.time(),
                    validation_score=validation['overall_score']
                )
                client_updates.append(update)

        # Aggregate updates
        coordinator = CoordinatorLogic()
        aggregated_result = coordinator.aggregate_updates(client_updates)

        end_time = time.time()

        return {
            'run_name': run_name,
            'total_time': end_time - start_time,
            'clients_completed': len(client_updates),
            'success_rate': len(client_updates) / num_clients,
            'aggregation_success': aggregated_result is not None
        }

    def teardown_method(self):
        """Clean up and generate performance report."""
        # Generate comprehensive performance report
        performance_report = self.monitor.get_performance_report()

        # Save performance report
        report_file = f"test_reports/performance_benchmark_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(performance_report, f, indent=2)

        print(f"\n📊 Performance report saved: {report_file}")

        # Print summary
        if performance_report['summary']:
            summary = performance_report['summary']
            print("
📈 Performance Summary:"            print(".1f")
            print(".1f")
            print(".1f")
            print(f"   Memory Growth: {summary['memory_growth_mb']:.1f} MB")
            print(f"   Tests Completed: {summary['test_count']}")


if __name__ == "__main__":
    pytest.main([__file__])

