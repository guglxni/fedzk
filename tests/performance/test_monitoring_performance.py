"""
Performance Tests for Monitoring and Logging Systems
=====================================================

Tests performance characteristics of:
- Metrics collection under load
- Log processing throughput
- Memory usage patterns
- Concurrent access performance
- Scalability benchmarks
"""

import unittest
import time
import threading
import concurrent.futures
import psutil
import os
import statistics
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from fedzk.monitoring.metrics import FEDzkMetricsCollector
    from fedzk.logging.structured_logger import FEDzkLogger
    from fedzk.logging.log_aggregation import LogAggregator
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    print(f"Monitoring components not available: {e}")


class TestMetricsPerformance(unittest.TestCase):
    """Performance tests for metrics collection"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Monitoring components not available")

        self.collector = FEDzkMetricsCollector("perf-test")

    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'collector'):
            del self.collector

    def test_high_throughput_metrics_collection(self):
        """Test metrics collection performance under high throughput"""
        # Measure baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss

        # Test high-volume metrics recording
        start_time = time.time()
        num_operations = 10000

        for i in range(num_operations):
            self.collector.record_request("GET", f"/api/test/{i % 100}", 200, 0.001)

        end_time = time.time()
        duration = end_time - start_time

        # Calculate performance metrics
        operations_per_second = num_operations / duration
        memory_used = process.memory_info().rss - baseline_memory
        memory_per_operation = memory_used / num_operations

        print("\n📊 High Throughput Metrics Performance:")
        print(f"   • Operations: {num_operations:,}")
        print(".2f")
        print(".0f")
        print(".1f")
        print(".0f")
        # Performance assertions
        self.assertGreater(operations_per_second, 1000,
                          "Metrics collection should handle > 1000 ops/sec")
        self.assertLess(memory_per_operation, 1000,
                       "Memory per operation should be < 1KB")

    def test_concurrent_metrics_access(self):
        """Test metrics collection under concurrent access"""
        num_threads = 10
        operations_per_thread = 1000
        total_operations = num_threads * operations_per_thread

        def worker_thread(thread_id):
            """Worker thread for concurrent metrics recording"""
            for i in range(operations_per_thread):
                self.collector.record_request(
                    "GET",
                    f"/api/thread/{thread_id}/op/{i}",
                    200,
                    0.001
                )

        # Measure execution time
        start_time = time.time()

        # Start concurrent threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.time()
        duration = end_time - start_time

        # Calculate performance metrics
        operations_per_second = total_operations / duration

        print("\n🔄 Concurrent Access Performance:")
        print(f"   • Threads: {num_threads}")
        print(f"   • Total Operations: {total_operations:,}")
        print(".2f")
        print(".0f")
        # Verify all operations were recorded
        metrics_output = self.collector.get_metrics_output()
        self.assertIn(f"fedzk_requests_total {total_operations}", metrics_output)

        # Performance assertions
        self.assertGreater(operations_per_second, 5000,
                          "Concurrent access should handle > 5000 ops/sec")
        self.assertLess(duration, 5.0,
                       "Concurrent operations should complete within 5 seconds")

    def test_metrics_export_performance(self):
        """Test performance of metrics export operations"""
        # Populate collector with test data
        for i in range(5000):
            self.collector.record_request("GET", f"/test/{i}", 200, 0.001)

        # Measure export performance
        num_exports = 100
        export_times = []

        for _ in range(num_exports):
            start_time = time.time()
            output = self.collector.get_metrics_output()
            end_time = time.time()
            export_times.append(end_time - start_time)

        # Calculate statistics
        avg_export_time = statistics.mean(export_times)
        p95_export_time = statistics.quantiles(export_times, n=20)[18] if len(export_times) >= 20 else max(export_times)
        min_export_time = min(export_times)
        max_export_time = max(export_times)

        print("\n📤 Metrics Export Performance:")
        print(f"   • Exports tested: {num_exports}")
        print(".6f")
        print(".6f")
        print(".6f")
        print(".6f")
        print(f"   • Output size: {len(output):,} bytes")

        # Performance assertions
        self.assertLess(avg_export_time, 0.01,
                       "Average export time should be < 10ms")
        self.assertLess(p95_export_time, 0.05,
                       "95th percentile export time should be < 50ms")

    def test_memory_usage_patterns(self):
        """Test memory usage patterns during metrics collection"""
        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss

        memory_samples = []

        # Gradually increase metrics volume and measure memory
        for batch_size in [1000, 2000, 5000, 10000]:
            # Record batch of metrics
            for i in range(batch_size):
                self.collector.record_request("GET", f"/memory/test/{i}", 200, 0.001)

            # Measure memory after batch
            current_memory = process.memory_info().rss
            memory_used = current_memory - baseline_memory
            memory_samples.append(memory_used)

            print(f"   After {batch_size} metrics: {memory_used / 1024 / 1024:.1f} MB")

        # Calculate memory growth rate
        if len(memory_samples) >= 2:
            memory_growth_rate = (memory_samples[-1] - memory_samples[0]) / memory_samples[0]
            print(".2%")

            # Memory growth should be reasonable (< 500% for 10x increase)
            self.assertLess(memory_growth_rate, 5.0,
                           "Memory growth should be reasonable with increased load")


class TestLoggingPerformance(unittest.TestCase):
    """Performance tests for logging system"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Logging components not available")

        self.logger = FEDzkLogger("perf-test")

        # Add file handler for testing
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "perf_test.log")
        self.logger.add_file_handler(self.log_file)

    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_high_frequency_logging(self):
        """Test logging performance under high frequency"""
        num_log_entries = 50000

        start_time = time.time()

        for i in range(num_log_entries):
            self.logger.log_structured("info", f"Performance test entry {i}", {
                "entry_id": i,
                "batch": i // 1000,
                "data": f"test_data_{i % 100}"
            })

        end_time = time.time()
        duration = end_time - start_time

        logs_per_second = num_log_entries / duration

        # Check log file was created and has content
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn("Performance test entry", content)

        print("
📝 High Frequency Logging Performance:"        print(f"   • Log entries: {num_log_entries:,}")
        print(".2f")
        print(".0f")
        print(".2f")
        # Performance assertions
        self.assertGreater(logs_per_second, 5000,
                          "Logging should handle > 5000 entries/sec")

    def test_structured_logging_overhead(self):
        """Test overhead of structured logging vs simple logging"""
        num_iterations = 10000

        # Test simple logging
        simple_logger = FEDzkLogger("simple-test")
        start_time = time.time()
        for i in range(num_iterations):
            simple_logger.logger.info(f"Simple log {i}")
        simple_duration = time.time() - start_time

        # Test structured logging
        start_time = time.time()
        for i in range(num_iterations):
            self.logger.log_structured("info", f"Structured log {i}", {
                "entry_id": i,
                "metadata": {"key": f"value_{i % 10}"}
            })
        structured_duration = time.time() - start_time

        # Calculate overhead
        overhead_ratio = structured_duration / simple_duration

        print("
🔧 Structured Logging Overhead:")
        print(".3f")
        print(".3f")
        print(".2f")
        print(".2f")

        # Overhead should be reasonable (< 3x)
        self.assertLess(overhead_ratio, 3.0,
                       "Structured logging overhead should be < 3x simple logging")

    def test_concurrent_logging_performance(self):
        """Test logging performance under concurrent access"""
        num_threads = 8
        logs_per_thread = 5000
        total_logs = num_threads * logs_per_thread

        def logging_worker(thread_id):
            """Worker thread for concurrent logging"""
            for i in range(logs_per_thread):
                self.logger.log_structured("info", f"Thread {thread_id} log {i}", {
                    "thread_id": thread_id,
                    "log_sequence": i,
                    "thread_data": f"thread_{thread_id}_data"
                })

        # Measure concurrent logging performance
        start_time = time.time()

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=logging_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        end_time = time.time()
        duration = end_time - start_time

        logs_per_second = total_logs / duration

        print("
🎯 Concurrent Logging Performance:"        print(f"   • Threads: {num_threads}")
        print(f"   • Total logs: {total_logs:,}")
        print(".2f")
        print(".0f")
        print(".2f"))

        # Verify log file contains expected entries
        with open(self.log_file, 'r') as f:
            content = f.read()
            # Should contain logs from all threads
            for thread_id in range(num_threads):
                self.assertIn(f"Thread {thread_id}", content)

        # Performance assertions
        self.assertGreater(logs_per_second, 10000,
                          "Concurrent logging should handle > 10000 logs/sec")

    def test_log_file_rotation_performance(self):
        """Test performance impact of log file rotation"""
        # Configure small rotation size for testing
        small_log_file = os.path.join(self.temp_dir, "rotation_test.log")
        rotation_logger = FEDzkLogger("rotation-test")
        rotation_logger.add_file_handler(small_log_file, max_bytes=1024, backup_count=3)

        # Generate enough logs to trigger rotation
        num_logs_for_rotation = 200

        start_time = time.time()
        for i in range(num_logs_for_rotation):
            rotation_logger.log_structured("info", f"Rotation test log {i}", {
                "log_number": i,
                "data": "x" * 50  # Pad to trigger rotation faster
            })

        end_time = time.time()
        rotation_duration = end_time - start_time

        logs_per_second_with_rotation = num_logs_for_rotation / rotation_duration

        # Check that rotation files were created
        rotation_files = [f for f in os.listdir(self.temp_dir) if "rotation_test.log" in f]
        self.assertGreater(len(rotation_files), 1,
                          "Log rotation should create multiple files")

        print("\n🔄 Log Rotation Performance:")
        print(f"   • Logs generated: {num_logs_for_rotation}")
        print(".2f")
        print(".0f")
        print(f"   • Rotation files created: {len(rotation_files)}")

        # Performance should still be reasonable with rotation
        self.assertGreater(logs_per_second_with_rotation, 1000,
                          "Log rotation should maintain > 1000 logs/sec")


class TestAggregationPerformance(unittest.TestCase):
    """Performance tests for log aggregation system"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Aggregation components not available")

        self.aggregator = LogAggregator(retention_hours=1)

    def test_log_aggregation_throughput(self):
        """Test log aggregation throughput"""
        num_logs = 100000

        # Generate test logs
        test_logs = []
        base_time = time.time()

        for i in range(num_logs):
            test_logs.append({
                "timestamp_epoch": base_time + i * 0.001,  # Spread over time
                "level": ["INFO", "WARNING", "ERROR"][i % 3],
                "message": f"Test log message {i}",
                "service": f"service-{i % 10}",
                "performance_metric": i % 100 == 0,  # Every 100th log
                "operation": f"operation-{i % 20}" if i % 100 == 0 else None,
                "duration_seconds": (i % 100) * 0.01 if i % 100 == 0 else None
            })

        # Measure aggregation performance
        start_time = time.time()

        for log_entry in test_logs:
            self.aggregator.aggregate_log(log_entry)

        end_time = time.time()
        duration = end_time - start_time

        logs_per_second = num_logs / duration

        print("
📊 Log Aggregation Throughput:"        print(f"   • Logs processed: {num_logs:,}")
        print(".2f")
        print(".0f"))

        # Performance assertions
        self.assertGreater(logs_per_second, 50000,
                          "Log aggregation should handle > 50,000 logs/sec")

        # Verify aggregation results
        stats = self.aggregator.get_statistics()

        self.assertEqual(stats['total_logs'], num_logs)
        self.assertEqual(stats['logs_by_level']['INFO'], num_logs // 3)
        self.assertEqual(stats['logs_by_level']['WARNING'], num_logs // 3)
        self.assertEqual(stats['logs_by_level']['ERROR'], num_logs // 3)

    def test_performance_insights_generation(self):
        """Test performance of insights generation"""
        # Add performance metrics
        num_perf_logs = 5000

        for i in range(num_perf_logs):
            self.aggregator.aggregate_log({
                "timestamp_epoch": time.time(),
                "level": "INFO",
                "message": f"Performance log {i}",
                "performance_metric": True,
                "operation": f"op-{i % 50}",
                "duration_seconds": (i % 100) * 0.01
            })

        # Measure insights generation performance
        start_time = time.time()
        insights = self.aggregator._get_performance_insights(time.time() - 3600)
        end_time = time.time()

        insights_duration = end_time - start_time

        print("\n🔍 Performance Insights Generation:")
        print(f"   • Performance logs: {num_perf_logs}")
        print(".6f")
        print(f"   • Operations analyzed: {len(insights)}")

        # Performance assertions
        self.assertLess(insights_duration, 1.0,
                       "Insights generation should complete within 1 second")
        self.assertGreater(len(insights), 0,
                          "Should generate insights for operations")

    def test_memory_usage_with_large_dataset(self):
        """Test memory usage with large log datasets"""
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss

        # Add large number of logs
        num_logs = 50000

        for i in range(num_logs):
            self.aggregator.aggregate_log({
                "timestamp_epoch": time.time(),
                "level": "INFO",
                "message": f"Memory test log {i}" + "x" * 100,  # Large message
                "service": f"service-{i % 20}",
                "extra_data": {"field1": f"value{i}", "field2": i * 2.5}
            })

        current_memory = process.memory_info().rss
        memory_used = current_memory - baseline_memory
        memory_per_log = memory_used / num_logs

        print("\n💾 Memory Usage with Large Dataset:")
        print(f"   • Logs processed: {num_logs:,}")
        print(".1f")
        print(".0f")

        # Memory usage should be reasonable
        self.assertLess(memory_per_log, 2000,
                       "Memory per log should be < 2KB")

        # Verify functionality is maintained
        stats = self.aggregator.get_statistics()
        self.assertEqual(stats['total_logs'], num_logs)


class TestSystemResourceMonitoring(unittest.TestCase):
    """Tests for monitoring system resource usage"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Resource monitoring not available")

        self.collector = FEDzkMetricsCollector("resource-test")
        self.logger = FEDzkLogger("resource-test")
        self.aggregator = LogAggregator()

    def test_cpu_monitoring_overhead(self):
        """Test CPU overhead of monitoring system"""
        process = psutil.Process(os.getpid())

        # Measure baseline CPU
        baseline_cpu = process.cpu_percent(interval=0.1)

        # Perform monitoring operations
        start_time = time.time()

        for i in range(10000):
            self.collector.record_request("GET", f"/cpu/test/{i}", 200, 0.001)
            self.logger.log_structured("debug", f"CPU test {i}", {"iteration": i})

            if i % 1000 == 0:
                self.aggregator.aggregate_log({
                    "timestamp_epoch": time.time(),
                    "level": "INFO",
                    "message": f"CPU monitoring log {i}",
                    "service": "cpu-test"
                })

        end_time = time.time()
        duration = end_time - start_time

        # Measure CPU usage during operations
        cpu_during_operations = process.cpu_percent(interval=0.1)

        operations_per_second = 10000 / duration

        print("\n⚡ CPU Monitoring Overhead:")
        print(".2f")
        print(".1f")
        print(".0f")
        print(".1f")

        # CPU usage should be reasonable (< 50%)
        self.assertLess(cpu_during_operations, 50.0,
                       "CPU usage during monitoring should be < 50%")

    def test_end_to_end_performance(self):
        """Test end-to-end performance of complete monitoring pipeline"""
        # Simulate complete monitoring workflow
        num_iterations = 5000

        start_time = time.time()

        for i in range(num_iterations):
            # Record metrics
            self.collector.record_request("POST", f"/api/workflow/{i}", 200, 0.005)

            # Log structured data
            self.logger.log_structured("info", f"Workflow step {i}", {
                "step_id": i,
                "workflow_id": f"wf-{i % 100}",
                "data_size": (i % 50) * 1000
            })

            # Aggregate logs
            if i % 100 == 0:
                self.aggregator.aggregate_log({
                    "timestamp_epoch": time.time(),
                    "level": "INFO",
                    "message": f"Workflow milestone {i}",
                    "service": "workflow-engine",
                    "performance_metric": True,
                    "operation": "workflow_step",
                    "duration_seconds": 0.005
                })

        end_time = time.time()
        duration = end_time - start_time

        # Calculate comprehensive performance metrics
        total_operations = num_iterations * 3  # metrics + logging + aggregation
        operations_per_second = total_operations / duration

        print("
🚀 End-to-End Performance:"        print(f"   • Workflow iterations: {num_iterations:,}")
        print(f"   • Total operations: {total_operations:,}")
        print(".2f")
        print(".0f"))

        # Verify all components worked
        metrics_output = self.collector.get_metrics_output()
        self.assertIn(f"fedzk_requests_total {num_iterations}", metrics_output)

        stats = self.aggregator.get_statistics()
        self.assertGreaterEqual(stats['total_logs'], num_iterations // 100)

        # Performance assertions
        self.assertGreater(operations_per_second, 10000,
                          "End-to-end performance should handle > 10,000 ops/sec")


if __name__ == '__main__':
    unittest.main()
