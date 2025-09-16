"""
Comprehensive Testing Suite for Monitoring and Observability
============================================================

Tests for tasks 8.3.1 (Metrics Collection) and 8.3.2 (Logging Infrastructure)
including unit tests, integration tests, security tests, and performance tests.
"""

import unittest
import json
import time
import threading
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock prometheus_client if not available
try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    import sys
    sys.modules['prometheus_client'] = MagicMock()
    PROMETHEUS_AVAILABLE = False

try:
    from fedzk.monitoring.metrics import (
        FEDzkMetricsCollector, ZKProofMetrics
    )
    from fedzk.logging.structured_logger import (
        FEDzkLogger, get_logger, initialize_logging,
        FEDzkJSONFormatter, FEDzkSecurityFormatter
    )
    from fedzk.logging.log_aggregation import (
        LogAggregator, get_log_pipeline
    )
    from fedzk.logging.security_compliance import (
        SecurityEventLogger, AuditLogger,
        ComplianceStandard, SecurityEventType
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    print(f"Monitoring components not available: {e}")

# Final check - monitoring is only available if we have both components AND prometheus
if not PROMETHEUS_AVAILABLE:
    MONITORING_AVAILABLE = False
    print("Prometheus client not available - running with limited functionality")


class TestMetricsCollectionSystem(unittest.TestCase):
    """Unit tests for metrics collection system (Task 8.3.1)"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Monitoring components not available")

        self.collector = FEDzkMetricsCollector("test-service")

    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'collector'):
            # Registry cleanup would happen here
            pass

    def test_basic_metrics_collection(self):
        """Test basic metrics collection functionality"""
        # Test HTTP request metrics
        self.collector.record_request("GET", "/health", 200, 0.05)
        self.collector.record_request("POST", "/api/proof", 201, 0.25)
        self.collector.record_request("GET", "/metrics", 200, 0.02)

        # Test authentication metrics
        self.collector.record_auth_attempt("jwt", True)
        self.collector.record_auth_attempt("api_key", True)
        self.collector.record_auth_attempt("invalid", False)

        # Verify metrics output
        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total", output)
        self.assertIn("fedzk_auth_attempts_total", output)
        self.assertIn("method=\"GET\"", output)
        self.assertIn("status=\"200\"", output)

    def test_zk_proof_metrics(self):
        """Test ZK proof-specific metrics collection"""
        # Create test proof metrics
        proof_metrics = ZKProofMetrics(
            proof_generation_time=1.25,
            proof_size_bytes=2048,
            verification_time=0.35,
            circuit_complexity=1000,
            success=True,
            proof_type="federated_aggregation"
        )

        # Record metrics
        self.collector.record_proof_generation(proof_metrics)

        # Verify ZK metrics
        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_zk_proof_generation_total", output)
        self.assertIn("fedzk_zk_proof_generation_duration_seconds", output)
        self.assertIn("fedzk_zk_proof_size_bytes", output)
        self.assertIn("circuit_type=\"federated_aggregation\"", output)

    def test_federated_learning_metrics(self):
        """Test federated learning metrics collection"""
        # Record FL operations
        self.collector.record_fl_round("completed")
        self.collector.record_fl_round("completed")
        self.collector.record_fl_round("failed")

        # Record MPC operations
        self.collector.record_mpc_operation("secure_aggregation", True, 0.45)
        self.collector.record_mpc_operation("key_generation", True, 0.12)
        self.collector.record_mpc_operation("signature_verification", False, 0.08)

        # Verify FL metrics
        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_fl_rounds_total", output)
        self.assertIn("fedzk_mpc_operations_total", output)
        self.assertIn("fedzk_mpc_computation_duration_seconds", output)

    def test_circuit_complexity_tracking(self):
        """Test circuit complexity metrics"""
        self.collector.record_circuit_complexity("test_circuit", 2048, 1024)
        self.collector.record_circuit_complexity("ml_circuit", 4096, 2048)

        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_zk_circuit_complexity", output)

    def test_security_metrics(self):
        """Test security-related metrics"""
        # Record security events
        self.collector.record_security_event("login_attempt", "info")
        self.collector.record_security_event("suspicious_activity", "warning")
        self.collector.record_security_event("brute_force_attack", "error")

        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_security_events_total", output)

    def test_metrics_output_format(self):
        """Test Prometheus metrics output format"""
        self.collector.record_request("GET", "/test", 200, 0.1)

        output = self.collector.get_metrics_output()

        # Check Prometheus format compliance
        lines = output.split('\n')
        metric_lines = [line for line in lines if line.strip() and not line.startswith('#')]

        for line in metric_lines:
            # Should have metric name, labels, and value
            parts = line.split()
            self.assertGreaterEqual(len(parts), 2)
            self.assertTrue(parts[0].replace('_', '').replace('{', '').replace('}', '').isalnum())

    def test_metrics_thread_safety(self):
        """Test metrics collection is thread-safe"""
        results = []

        def record_metrics(thread_id):
            for i in range(100):
                self.collector.record_request("GET", f"/test/{i}", 200, 0.01)
            results.append(f"thread_{thread_id}_done")

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=record_metrics, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 5)
        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total 500", output)


class TestStructuredLoggingSystem(unittest.TestCase):
    """Unit tests for structured logging system (Task 8.3.2)"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Logging components not available")

    def test_json_formatter(self):
        """Test JSON formatter functionality"""
        formatter = FEDzkJSONFormatter("test-service")

        # Create a mock log record
        record = Mock()
        record.levelname = "INFO"
        record.getMessage.return_value = "Test message"
        record.name = "test.logger"
        record.created = time.time()
        record.__dict__.update({
            'hostname': 'test-host',
            'pid': 12345,
            'structured_data': {'custom_field': 'custom_value'}
        })

        # Format the record
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Verify structure
        self.assertEqual(parsed['level'], 'INFO')
        self.assertEqual(parsed['service'], 'test-service')
        self.assertEqual(parsed['message'], 'Test message')
        self.assertEqual(parsed['logger'], 'test.logger')
        self.assertIn('timestamp', parsed)
        self.assertEqual(parsed['custom_field'], 'custom_value')

    def test_security_formatter(self):
        """Test security formatter with compliance checking"""
        formatter = FEDzkSecurityFormatter("test-service")

        # Create a mock log record
        record = Mock()
        record.levelname = "WARNING"
        record.getMessage.return_value = "User authentication failed"
        record.name = "security.auth"
        record.created = time.time()

        # Format the record
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Verify security fields
        self.assertIn('compliance', parsed)
        self.assertIn('security_level', parsed)
        self.assertIn('data_classification', parsed)
        self.assertIn('audit_trail', parsed)

        # Verify compliance checking
        self.assertIn('gdpr', parsed['compliance'])
        self.assertIn('pci_dss', parsed['compliance'])

    def test_logger_initialization(self):
        """Test logger initialization and configuration"""
        logger = FEDzkLogger("test-service", "DEBUG")

        # Verify logger configuration
        self.assertEqual(logger.logger.level, 10)  # DEBUG level
        self.assertEqual(logger.service_name, "test-service")

        # Test basic logging
        logger.logger.info("Test message", extra={"structured_data": {"test": True}})

    def test_request_context(self):
        """Test request context functionality"""
        logger = FEDzkLogger("test-service")

        # Set request context
        logger.set_request_context(
            request_id="req-123",
            user_id="user-456",
            correlation_id="corr-789"
        )

        # Log with context
        logger.log_structured("info", "Test with context", {"action": "test"})

        # Clear context
        logger.clear_request_context()

        # Verify context is cleared
        self.assertNotIn('request_id', logger._request_context.__dict__)


class TestLogAggregationSystem(unittest.TestCase):
    """Integration tests for log aggregation system"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Log aggregation not available")

        self.aggregator = LogAggregator(retention_hours=1)

    def test_log_aggregation(self):
        """Test log aggregation functionality"""
        # Add sample logs
        logs = [
            {"level": "INFO", "message": "Service started", "service": "coordinator"},
            {"level": "WARNING", "message": "High memory usage", "service": "mpc"},
            {"level": "ERROR", "message": "Connection failed", "service": "zk"},
            {"level": "INFO", "message": "Request processed", "service": "coordinator"},
        ]

        for log in logs:
            log['timestamp_epoch'] = time.time()
            self.aggregator.aggregate_log(log)

        # Get statistics
        stats = self.aggregator.get_statistics()

        # Verify aggregation
        self.assertEqual(stats['total_logs'], 4)
        self.assertEqual(stats['logs_by_level']['INFO'], 2)
        self.assertEqual(stats['logs_by_level']['WARNING'], 1)
        self.assertEqual(stats['logs_by_level']['ERROR'], 1)
        self.assertEqual(stats['logs_by_service']['coordinator'], 2)

    def test_error_pattern_detection(self):
        """Test error pattern detection"""
        error_logs = [
            {"level": "ERROR", "message": "Connection timeout to database", "timestamp_epoch": time.time()},
            {"level": "ERROR", "message": "Failed to connect to Redis", "timestamp_epoch": time.time()},
            {"level": "ERROR", "message": "Database connection refused", "timestamp_epoch": time.time()},
            {"level": "ERROR", "message": "Redis server unavailable", "timestamp_epoch": time.time()},
        ]

        for log in error_logs:
            self.aggregator.aggregate_log(log)

        patterns = self.aggregator.error_patterns

        # Should detect connection errors
        self.assertIn('connection_error', patterns)
        self.assertGreater(patterns['connection_error'], 0)

    def test_performance_insights(self):
        """Test performance insights generation"""
        # Add performance-related logs
        perf_logs = [
            {"performance_metric": True, "operation": "zk_proof", "duration_seconds": 1.2, "timestamp_epoch": time.time()},
            {"performance_metric": True, "operation": "zk_proof", "duration_seconds": 0.8, "timestamp_epoch": time.time()},
            {"performance_metric": True, "operation": "database_query", "duration_seconds": 0.15, "timestamp_epoch": time.time()},
        ]

        for log in perf_logs:
            self.aggregator.aggregate_log(log)

        insights = self.aggregator._get_performance_insights(time.time() - 3600)

        # Verify performance insights
        self.assertIn('zk_proof', insights)
        self.assertIn('database_query', insights)

    def test_anomaly_detection(self):
        """Test anomaly detection in logs"""
        # Add normal logs
        for i in range(10):
            self.aggregator.aggregate_log({
                "level": "INFO",
                "message": "Normal operation",
                "timestamp_epoch": time.time()
            })

        # Add some errors (but not anomalous)
        for i in range(2):
            self.aggregator.aggregate_log({
                "level": "ERROR",
                "message": "Minor error",
                "timestamp_epoch": time.time()
            })

        # Check for anomalies
        anomalies = self.aggregator._detect_anomalies(time.time() - 3600)

        # Should not detect anomalies with this pattern
        error_anomalies = [a for a in anomalies if a['type'] == 'high_error_rate']
        self.assertEqual(len(error_anomalies), 0)


class TestSecurityComplianceLogging(unittest.TestCase):
    """Security and compliance testing for logging system"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Security logging not available")

        self.security_logger = SecurityEventLogger("test-security")
        self.audit_logger = AuditLogger("test-audit")

    def test_security_event_logging(self):
        """Test security event logging"""
        # Log authentication events
        event1 = self.security_logger.log_security_event(
            SecurityEventType.AUTHENTICATION_SUCCESS,
            user_id="user-123",
            source_ip="192.168.1.100",
            success=True
        )

        event2 = self.security_logger.log_security_event(
            SecurityEventType.AUTHENTICATION_FAILURE,
            user_id="user-456",
            source_ip="10.0.0.50",
            success=False
        )

        # Verify events are logged
        self.assertIsInstance(event1, str)
        self.assertIsInstance(event2, str)
        self.assertTrue(event1.startswith("sec_"))
        self.assertTrue(event2.startswith("sec_"))

    def test_compliance_validation(self):
        """Test compliance validation"""
        from fedzk.logging.security_compliance import ComplianceChecker

        checker = ComplianceChecker()

        # Test GDPR compliance
        gdpr_compliant = checker.check_compliance(
            "User john@example.com logged in",
            [ComplianceStandard.GDPR]
        )
        self.assertFalse(gdpr_compliant['gdpr'])  # Contains email

        gdpr_compliant = checker.check_compliance(
            "User logged in successfully",
            [ComplianceStandard.GDPR]
        )
        self.assertTrue(gdpr_compliant['gdpr'])  # No PII

    def test_audit_trail(self):
        """Test audit trail functionality"""
        audit_id = self.audit_logger.log_audit_entry(
            action="USER_LOGIN",
            resource="sessions",
            user_id="user-123",
            success=True,
            before_state={"status": "logged_out"},
            after_state={"status": "logged_in"}
        )

        # Verify audit ID
        self.assertIsInstance(audit_id, str)
        self.assertTrue(audit_id.startswith("audit_"))

        # Verify audit integrity
        integrity = self.audit_logger.verify_audit_integrity(audit_id)
        self.assertTrue(integrity)

    def test_data_classification(self):
        """Test data classification functionality"""
        from fedzk.logging.security_compliance import DataClassification

        checker = ComplianceChecker()

        # Test different classification levels
        public_data = checker.classify_data("Service started successfully")
        confidential_data = checker.classify_data("User email: test@example.com")
        restricted_data = checker.classify_data("Database password: secret123")

        self.assertEqual(public_data, DataClassification.PUBLIC)
        self.assertEqual(confidential_data, DataClassification.CONFIDENTIAL)
        self.assertEqual(restricted_data, DataClassification.RESTRICTED)


class TestEndToEndMonitoring(unittest.TestCase):
    """End-to-end testing for monitoring and observability"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("End-to-end testing not available")

        self.collector = FEDzkMetricsCollector("e2e-test")
        self.logger = FEDzkLogger("e2e-test")
        self.aggregator = LogAggregator()

    def test_complete_workflow(self):
        """Test complete monitoring workflow"""
        # Simulate a complete workflow
        request_id = "req-e2e-123"

        # 1. Set request context
        self.logger.set_request_context(request_id=request_id, user_id="user-123")

        # 2. Record initial metrics
        start_time = time.time()
        self.collector.record_request("POST", "/api/train", 200, 0.05)

        # 3. Log structured events
        self.logger.log_structured("info", "Starting federated learning round", {
            "round": 1,
            "participants": 3
        })

        # 4. Record FL operations
        self.collector.record_fl_round("completed")
        self.collector.record_mpc_operation("secure_aggregation", True, 0.45)

        # 5. Record ZK proof generation
        proof_metrics = ZKProofMetrics(
            proof_generation_time=1.2,
            proof_size_bytes=1536,
            verification_time=0.3,
            circuit_complexity=512,
            success=True,
            proof_type="federated_aggregation"
        )
        self.collector.record_proof_generation(proof_metrics)

        # 6. Log completion
        self.logger.log_structured("info", "Federated learning round completed", {
            "round": 1,
            "duration": time.time() - start_time,
            "status": "success"
        })

        # 7. Verify metrics collection
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total", metrics_output)
        self.assertIn("fedzk_fl_rounds_total", metrics_output)
        self.assertIn("fedzk_zk_proof_generation_total", metrics_output)

        # 8. Clear context
        self.logger.clear_request_context()

    def test_error_handling_workflow(self):
        """Test error handling in monitoring workflow"""
        # Simulate error scenario
        self.logger.set_request_context(request_id="req-error-456", user_id="user-456")

        # Record failed request
        self.collector.record_request("POST", "/api/train", 500, 0.1)

        # Log error
        self.logger.log_structured("error", "Federated learning round failed", {
            "round": 2,
            "error": "MPC computation failed",
            "participants": 3
        })

        # Record failed operations
        self.collector.record_fl_round("failed")
        self.collector.record_mpc_operation("secure_aggregation", False, 0.05)

        # Verify error metrics
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total", metrics_output)
        self.assertIn("fedzk_fl_rounds_total", metrics_output)

        self.logger.clear_request_context()


class TestPerformanceMonitoring(unittest.TestCase):
    """Performance testing for monitoring components"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Performance testing not available")

        self.collector = FEDzkMetricsCollector("perf-test")

    def test_high_throughput_metrics(self):
        """Test metrics collection under high throughput"""
        import concurrent.futures

        def record_batch_metrics(batch_id):
            for i in range(100):
                self.collector.record_request("GET", f"/api/test/{i}", 200, 0.01)

        # Run concurrent metric recording
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(record_batch_metrics, i) for i in range(4)]
            concurrent.futures.wait(futures)

        # Verify all metrics were recorded
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total 400", metrics_output)

    def test_memory_efficiency(self):
        """Test memory efficiency of monitoring components"""
        import psutil
        import os

        initial_memory = psutil.Process(os.getpid()).memory_info().rss

        # Record many metrics
        for i in range(10000):
            self.collector.record_request("GET", f"/api/test/{i}", 200, 0.01)

        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for 10k metrics)
        self.assertLess(memory_increase, 50 * 1024 * 1024)

    def test_metrics_export_performance(self):
        """Test performance of metrics export"""
        # Record metrics
        for i in range(1000):
            self.collector.record_request("GET", f"/test/{i}", 200, 0.01)

        # Measure export time
        import time
        start_time = time.time()
        output = self.collector.get_metrics_output()
        export_time = time.time() - start_time

        # Export should be fast (< 0.1 seconds)
        self.assertLess(export_time, 0.1)
        self.assertGreater(len(output), 0)


class TestComplianceMonitoring(unittest.TestCase):
    """Compliance testing for monitoring standards"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Compliance testing not available")

        from fedzk.logging.security_compliance import ComplianceChecker
        self.checker = ComplianceChecker()

    def test_gdpr_compliance(self):
        """Test GDPR compliance validation"""
        # Test cases that should fail GDPR
        gdpr_violations = [
            "User email: john@example.com logged in",
            "Phone: +1-555-123-4567 registered",
            "SSN: 123-45-6789 updated",
            "IP: 192.168.1.100, User: John Doe",
        ]

        for test_case in gdpr_violations:
            compliance = self.checker.check_compliance(test_case, [ComplianceStandard.GDPR])
            self.assertFalse(compliance['gdpr'],
                           f"Should detect GDPR violation in: {test_case}")

        # Test cases that should pass GDPR
        gdpr_compliant = [
            "User logged in successfully",
            "Request processed in 0.05 seconds",
            "Service started on port 8000",
            "Federated learning round completed",
        ]

        for test_case in gdpr_compliant:
            compliance = self.checker.check_compliance(test_case, [ComplianceStandard.GDPR])
            self.assertTrue(compliance['gdpr'],
                          f"Should be GDPR compliant: {test_case}")

    def test_pci_dss_compliance(self):
        """Test PCI DSS compliance validation"""
        # Test cases that should fail PCI DSS
        pci_violations = [
            "Card: 4111-1111-1111-1111 processed",
            "CVV: 123 verified",
            "Expiry: 12/25 validated",
        ]

        for test_case in pci_violations:
            compliance = self.checker.check_compliance(test_case, [ComplianceStandard.PCI_DSS])
            self.assertFalse(compliance['pci_dss'],
                           f"Should detect PCI DSS violation in: {test_case}")

    def test_data_classification(self):
        """Test automatic data classification"""
        from fedzk.logging.security_compliance import DataClassification

        test_cases = [
            ("Service started successfully", DataClassification.PUBLIC),
            ("User session created", DataClassification.INTERNAL),
            ("Email: test@example.com", DataClassification.CONFIDENTIAL),
            ("Password: secret123", DataClassification.RESTRICTED),
            ("Credit card: 4111111111111111", DataClassification.RESTRICTED),
        ]

        for data, expected_class in test_cases:
            classification = self.checker.classify_data(data)
            self.assertEqual(classification, expected_class,
                           f"Misclassified '{data}' as {classification}, expected {expected_class}")


if __name__ == '__main__':
    # Configure test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestMetricsCollectionSystem,
        TestStructuredLoggingSystem,
        TestLogAggregationSystem,
        TestSecurityComplianceLogging,
        TestEndToEndMonitoring,
        TestPerformanceMonitoring,
        TestComplianceMonitoring,
    ]

    for test_class in test_classes:
        if MONITORING_AVAILABLE:
            suite.addTests(loader.loadTestsFromTestClass(test_class))
        else:
            print(f"Skipping {test_class.__name__} - monitoring not available")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("MONITORING COMPREHENSIVE TESTING SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ ALL MONITORING TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")

        if result.failures:
            print(f"\nFailures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")

    print("="*70)
