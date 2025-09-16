"""
Integration Tests for Monitoring and Logging Infrastructure
==========================================================

Tests the complete integration between metrics collection (8.3.1)
and logging infrastructure (8.3.2) components.
"""

import unittest
import json
import time
import tempfile
import os
import threading
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mock prometheus_client if not available
try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    import sys
    sys.modules['prometheus_client'] = MagicMock()
    PROMETHEUS_AVAILABLE = False

try:
    from fedzk.monitoring.metrics import FEDzkMetricsCollector
    from fedzk.logging.structured_logger import FEDzkLogger, get_logger
    from fedzk.logging.log_aggregation import LogAggregator, ElasticsearchLogShipper
    from fedzk.logging.security_compliance import (
        SecurityEventLogger, AuditLogger,
        SecurityEventType, ComplianceStandard
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    print(f"Monitoring components not available: {e}")


class TestMetricsLoggingIntegration(unittest.TestCase):
    """Integration tests for metrics and logging systems"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Monitoring components not available")

        self.collector = FEDzkMetricsCollector("integration-test")
        self.logger = FEDzkLogger("integration-test")
        self.aggregator = LogAggregator(retention_hours=1)

    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'logger'):
            self.logger.clear_request_context()

    def test_metrics_and_logging_workflow(self):
        """Test complete workflow with metrics and logging integration"""
        # Set up request context
        request_id = "req-integration-123"
        self.logger.set_request_context(request_id=request_id, user_id="user-integration")

        # 1. Record metrics
        start_time = time.time()
        self.collector.record_request("POST", "/api/federated/train", 200, 0.05)

        # 2. Log structured event
        self.logger.log_structured("info", "Federated learning training started", {
            "model_type": "neural_network",
            "dataset_size": 10000,
            "participants": 5
        })

        # 3. Record FL metrics
        self.collector.record_fl_round("in_progress")
        self.collector.record_mpc_operation("key_exchange", True, 0.12)

        # 4. Log progress
        self.logger.log_performance_metric("fl_initialization", 0.12, True, {
            "participants_ready": 5,
            "security_keys_exchanged": True
        })

        # 5. Simulate ZK proof generation
        from fedzk.monitoring.metrics import ZKProofMetrics
        proof_metrics = ZKProofMetrics(
            proof_generation_time=1.45,
            proof_size_bytes=2048,
            verification_time=0.25,
            circuit_complexity=1024,
            success=True,
            proof_type="model_aggregation"
        )
        self.collector.record_proof_generation(proof_metrics)

        # 6. Log completion
        total_duration = time.time() - start_time
        self.logger.log_structured("info", "Federated learning round completed", {
            "round": 1,
            "duration_seconds": total_duration,
            "proof_generated": True,
            "participants_completed": 5
        })

        # 7. Record completion metrics
        self.collector.record_fl_round("completed")
        self.collector.record_request("GET", "/api/status", 200, 0.02)

        # Verify metrics
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total 2", metrics_output)
        self.assertIn("fedzk_fl_rounds_total", metrics_output)
        self.assertIn("fedzk_zk_proof_generation_total", metrics_output)

    def test_error_scenario_integration(self):
        """Test error scenario with integrated monitoring"""
        # Set up context
        self.logger.set_request_context(request_id="req-error-456", user_id="user-error")

        # 1. Record failed request
        self.collector.record_request("POST", "/api/federated/train", 500, 0.08)

        # 2. Log error
        self.logger.log_structured("error", "MPC computation failed", {
            "error_code": "MPC_KEY_EXCHANGE_FAILED",
            "participant_id": "participant-3",
            "retry_count": 2
        })

        # 3. Record failed operations
        self.collector.record_mpc_operation("secure_aggregation", False, 0.05)
        self.collector.record_fl_round("failed")

        # 4. Log security event
        self.logger.log_security_event("computation_failure", "medium", {
            "failure_type": "MPC_error",
            "affected_participants": ["participant-3"]
        })

        # 5. Record security metrics
        self.collector.record_security_event("computation_error", "warning")

        # Verify error tracking
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total", metrics_output)
        self.assertIn("fedzk_security_events_total", metrics_output)

    def test_log_aggregation_with_metrics(self):
        """Test log aggregation working with metrics collection"""
        # Create sample log entries that would be generated by the system
        log_entries = [
            {
                "timestamp_epoch": time.time(),
                "level": "INFO",
                "message": "ZK proof generation started",
                "service": "zk-service",
                "request_id": "req-123",
                "performance_metric": True,
                "operation": "proof_generation",
                "duration_seconds": 0.5
            },
            {
                "timestamp_epoch": time.time(),
                "level": "INFO",
                "message": "MPC computation completed",
                "service": "mpc-service",
                "request_id": "req-123",
                "performance_metric": True,
                "operation": "mpc_computation",
                "duration_seconds": 0.8
            },
            {
                "timestamp_epoch": time.time(),
                "level": "WARNING",
                "message": "High memory usage detected",
                "service": "coordinator",
                "request_id": "req-123",
                "memory_percent": 85.5
            }
        ]

        # Aggregate logs
        for log_entry in log_entries:
            self.aggregator.aggregate_log(log_entry)

        # Get statistics
        stats = self.aggregator.get_statistics()

        # Verify aggregation
        self.assertEqual(stats['total_logs'], 3)
        self.assertEqual(stats['logs_by_level']['INFO'], 2)
        self.assertEqual(stats['logs_by_level']['WARNING'], 1)
        self.assertEqual(stats['logs_by_service']['zk-service'], 1)
        self.assertEqual(stats['logs_by_service']['mpc-service'], 1)
        self.assertEqual(stats['logs_by_service']['coordinator'], 1)

        # Check performance insights
        insights = self.aggregator._get_performance_insights(time.time() - 3600)
        self.assertIn('proof_generation', insights)
        self.assertIn('mpc_computation', insights)

    def test_security_event_integration(self):
        """Test security event integration across components"""
        # Initialize security components
        security_logger = SecurityEventLogger("security-test")
        audit_logger = AuditLogger("audit-test")

        # 1. Log authentication event
        auth_event_id = security_logger.log_security_event(
            SecurityEventType.AUTHENTICATION_SUCCESS,
            user_id="user-security",
            source_ip="192.168.1.100",
            success=True,
            details={"method": "jwt", "session_duration": 3600}
        )

        # 2. Log corresponding audit event
        audit_event_id = audit_logger.log_audit_entry(
            action="USER_AUTHENTICATION",
            resource="sessions",
            user_id="user-security",
            success=True,
            before_state={"authenticated": False},
            after_state={"authenticated": True, "session_id": "sess-123"}
        )

        # 3. Record metrics for security events
        self.collector.record_auth_attempt("jwt", True)
        self.collector.record_security_event("authentication_success", "info")

        # Verify integration
        self.assertTrue(auth_event_id.startswith("sec_"))
        self.assertTrue(audit_event_id.startswith("audit_"))

        # Verify metrics
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_auth_attempts_total", metrics_output)
        self.assertIn("fedzk_security_events_total", metrics_output)

        # Verify audit integrity
        audit_integrity = audit_logger.verify_audit_integrity(audit_event_id)
        self.assertTrue(audit_integrity)

    def test_multi_threaded_integration(self):
        """Test integration under multi-threaded load"""
        results = []

        def worker_thread(thread_id):
            """Worker thread for concurrent testing"""
            # Create thread-specific logger
            thread_logger = FEDzkLogger(f"thread-{thread_id}")
            thread_collector = FEDzkMetricsCollector(f"thread-{thread_id}")

            # Set thread context
            thread_logger.set_request_context(
                request_id=f"req-thread-{thread_id}",
                user_id=f"user-{thread_id}"
            )

            # Perform operations
            for i in range(50):
                # Record metrics
                thread_collector.record_request("GET", f"/api/test/{i}", 200, 0.01)

                # Log events
                thread_logger.log_structured("info", f"Thread operation {i}", {
                    "thread_id": thread_id,
                    "operation": i
                })

            thread_logger.clear_request_context()
            results.append(f"thread-{thread_id}-completed")

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Verify results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.endswith("-completed"))

    @patch('fedzk.logging.log_aggregation.Elasticsearch')
    def test_elasticsearch_integration(self, mock_es):
        """Test Elasticsearch integration for log shipping"""
        # Mock Elasticsearch client
        mock_client = Mock()
        mock_es.return_value = mock_client

        # Create log shipper
        shipper = ElasticsearchLogShipper(
            hosts=["localhost:9200"],
            index_prefix="test-logs"
        )

        # Ship test logs
        test_logs = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "level": "INFO",
                "message": "Test log entry 1",
                "service": "test-service"
            },
            {
                "timestamp": "2025-01-15T10:00:01Z",
                "level": "ERROR",
                "message": "Test error entry",
                "service": "test-service"
            }
        ]

        for log_entry in test_logs:
            shipper.ship_log(log_entry)

        # Force flush
        shipper._flush_buffer()

        # Verify Elasticsearch was called
        mock_client.bulk.assert_called()


class TestEndToEndMonitoring(unittest.TestCase):
    """End-to-end tests for complete monitoring system"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("End-to-end testing not available")

        self.collector = FEDzkMetricsCollector("e2e-test")
        self.logger = FEDzkLogger("e2e-test")
        self.aggregator = LogAggregator()
        self.security_logger = SecurityEventLogger("e2e-security")
        self.audit_logger = AuditLogger("e2e-audit")

    def test_complete_federated_learning_workflow(self):
        """Test complete federated learning workflow with full monitoring"""
        workflow_start = time.time()

        # Phase 1: Initialization
        self.logger.set_request_context(
            request_id="workflow-e2e-001",
            user_id="coordinator-user",
            correlation_id="workflow-001"
        )

        # Record initialization metrics
        self.collector.record_request("POST", "/api/workflow/init", 200, 0.1)

        # Log initialization
        self.logger.log_structured("info", "Federated learning workflow initialized", {
            "workflow_id": "workflow-001",
            "total_participants": 3,
            "model_type": "neural_network",
            "security_level": "zk_proofs"
        })

        # Security audit for workflow start
        audit_id = self.audit_logger.log_audit_entry(
            action="WORKFLOW_START",
            resource="federated_learning",
            user_id="coordinator-user",
            success=True,
            compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.SOX]
        )

        # Phase 2: Participant Registration
        for participant_id in ["participant-1", "participant-2", "participant-3"]:
            # Record participant registration
            self.collector.record_request("POST", "/api/participants/register", 201, 0.05)

            # Log participant registration
            self.logger.log_structured("info", f"Participant {participant_id} registered", {
                "participant_id": participant_id,
                "registration_time": time.time(),
                "status": "active"
            })

            # Security event for participant registration
            self.security_logger.log_security_event(
                SecurityEventType.DATA_ACCESS,
                user_id=f"user-{participant_id}",
                resource="participant_registry",
                action="REGISTER",
                success=True,
                details={"participant_id": participant_id}
            )

        # Phase 3: MPC Key Exchange
        self.logger.log_structured("info", "MPC key exchange started", {
            "phase": "key_exchange",
            "participants": 3
        })

        # Record MPC operations
        for i in range(3):
            self.collector.record_mpc_operation("key_exchange", True, 0.15)

        self.logger.log_performance_metric("mpc_key_exchange", 0.45, True, {
            "keys_exchanged": 3,
            "security_verified": True
        })

        # Phase 4: Training Rounds
        for round_num in range(1, 4):
            round_start = time.time()

            # Record round start
            self.collector.record_fl_round("in_progress")

            # Log round start
            self.logger.log_structured("info", f"Training round {round_num} started", {
                "round": round_num,
                "participants": 3,
                "batch_size": 100
            })

            # Simulate participant training
            for participant_id in ["participant-1", "participant-2", "participant-3"]:
                # Record participant training
                self.collector.record_request("POST", "/api/train/batch", 200, 0.08)

                # Log participant training completion
                self.logger.log_structured("info", f"Participant {participant_id} training completed", {
                    "participant_id": participant_id,
                    "round": round_num,
                    "samples_processed": 100,
                    "loss": 0.25 - (round_num * 0.05)  # Decreasing loss
                })

            # Generate ZK proof for round
            from fedzk.monitoring.metrics import ZKProofMetrics
            proof_metrics = ZKProofMetrics(
                proof_generation_time=1.2 + (round_num * 0.1),
                proof_size_bytes=1536,
                verification_time=0.25,
                circuit_complexity=512,
                success=True,
                proof_type=f"round_{round_num}_aggregation"
            )
            self.collector.record_proof_generation(proof_metrics)

            # Log round completion
            round_duration = time.time() - round_start
            self.logger.log_structured("info", f"Training round {round_num} completed", {
                "round": round_num,
                "duration_seconds": round_duration,
                "zk_proof_generated": True,
                "proof_size_bytes": proof_metrics.proof_size_bytes
            })

            # Record round completion
            self.collector.record_fl_round("completed")

        # Phase 5: Final Aggregation and Verification
        self.logger.log_structured("info", "Final model aggregation started", {
            "phase": "final_aggregation",
            "total_rounds": 3
        })

        # Final proof generation
        final_proof = ZKProofMetrics(
            proof_generation_time=2.1,
            proof_size_bytes=2048,
            verification_time=0.4,
            circuit_complexity=1024,
            success=True,
            proof_type="final_aggregation"
        )
        self.collector.record_proof_generation(final_proof)

        # Phase 6: Workflow Completion
        workflow_duration = time.time() - workflow_start

        # Log completion
        self.logger.log_structured("info", "Federated learning workflow completed", {
            "workflow_id": "workflow-001",
            "total_duration_seconds": workflow_duration,
            "rounds_completed": 3,
            "participants": 3,
            "zk_proofs_generated": 4,  # 3 rounds + 1 final
            "status": "success"
        })

        # Record final metrics
        self.collector.record_request("GET", "/api/workflow/status", 200, 0.02)

        # Security audit for workflow completion
        final_audit_id = self.audit_logger.log_audit_entry(
            action="WORKFLOW_COMPLETE",
            resource="federated_learning",
            user_id="coordinator-user",
            success=True,
            compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.SOX]
        )

        # Final security event
        self.security_logger.log_security_event(
            SecurityEventType.DATA_ACCESS,
            user_id="coordinator-user",
            resource="final_model",
            action="EXPORT",
            success=True,
            details={"workflow_id": "workflow-001", "model_size": 2048}
        )

        # Clear context
        self.logger.clear_request_context()

        # Verify complete workflow
        metrics_output = self.collector.get_metrics_output()

        # Check request metrics
        self.assertIn("fedzk_requests_total", metrics_output)

        # Check FL metrics
        self.assertIn("fedzk_fl_rounds_total", metrics_output)

        # Check ZK proof metrics
        self.assertIn("fedzk_zk_proof_generation_total", metrics_output)

        # Check MPC metrics
        self.assertIn("fedzk_mpc_operations_total", metrics_output)

        # Verify audit integrity
        self.assertTrue(self.audit_logger.verify_audit_integrity(audit_id))
        self.assertTrue(self.audit_logger.verify_audit_integrity(final_audit_id))

        print(f"✅ Complete FL workflow executed in {workflow_duration:.2f} seconds")
        print(f"   • 3 training rounds completed")
        print(f"   • 4 ZK proofs generated")
        print(f"   • 3 participants processed")
        print(f"   • Full security audit trail maintained")


class TestFailoverMonitoring(unittest.TestCase):
    """Failover testing for monitoring systems"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Failover testing not available")

        self.collector = FEDzkMetricsCollector("failover-test")
        self.logger = FEDzkLogger("failover-test")

    @patch('fedzk.logging.log_aggregation.Elasticsearch')
    def test_elasticsearch_failover(self, mock_es):
        """Test Elasticsearch failover scenarios"""
        # Mock connection failure
        mock_es.side_effect = Exception("Connection failed")

        from fedzk.logging.log_aggregation import ElasticsearchLogShipper

        # Create shipper (should handle connection failure gracefully)
        shipper = ElasticsearchLogShipper(hosts=["localhost:9200"])

        # Try to ship logs
        test_log = {"message": "Test log", "level": "INFO"}
        shipper.ship_log(test_log)  # Should not raise exception

        # Verify buffer contains the log
        self.assertEqual(len(shipper.buffer), 1)
        self.assertEqual(shipper.buffer[0], test_log)

    def test_logger_failover(self):
        """Test logger failover when handlers fail"""
        # Create logger with file handler that might fail
        with tempfile.TemporaryDirectory() as temp_dir:
            # Try to create logger with invalid file path
            invalid_path = "/invalid/path/log.txt"

            # This should not crash the application
            logger = FEDzkLogger("failover-test")

            # Try to add invalid file handler
            try:
                logger.add_file_handler(invalid_path)
            except Exception:
                pass  # Expected to fail

            # Logger should still work for console output
            logger.logger.info("Test message after handler failure")

    def test_metrics_collection_failover(self):
        """Test metrics collection under resource pressure"""
        # Fill up memory with metrics to test limits
        for i in range(10000):
            self.collector.record_request("GET", f"/test/{i}", 200, 0.001)

        # Should still be able to add more metrics
        self.collector.record_request("POST", "/final", 200, 0.01)

        # Verify metrics are still being collected
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total", metrics_output)


class TestMonitoringAlerts(unittest.TestCase):
    """Alerting tests for monitoring systems"""

    def setUp(self):
        """Set up test fixtures"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Alerting tests not available")

        self.collector = FEDzkMetricsCollector("alert-test")
        self.logger = FEDzkLogger("alert-test")

    def test_error_rate_alerts(self):
        """Test error rate alerting"""
        # Generate normal traffic
        for i in range(90):
            self.collector.record_request("GET", "/api/normal", 200, 0.01)

        # Generate errors
        for i in range(15):  # 15% error rate
            self.collector.record_request("GET", "/api/error", 500, 0.01)

        # Log errors
        for i in range(15):
            self.logger.log_structured("error", f"Test error {i}", {
                "error_code": "TEST_ERROR",
                "component": "test_service"
            })

        # In a real system, this would trigger alerts
        # Here we just verify the metrics are captured correctly
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total 105", metrics_output)

    def test_performance_degradation_alerts(self):
        """Test performance degradation alerting"""
        # Simulate normal performance
        for i in range(10):
            self.collector.record_request("GET", "/api/fast", 200, 0.01)

        # Simulate degraded performance
        for i in range(10):
            self.collector.record_request("GET", "/api/slow", 200, 0.5)  # 10x slower

        # Record slow performance metrics
        for i in range(10):
            self.logger.log_performance_metric("slow_operation", 0.5, True, {
                "operation": "database_query",
                "degraded": True
            })

        # Verify performance metrics are captured
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total 20", metrics_output)

    def test_security_incident_alerts(self):
        """Test security incident alerting"""
        from fedzk.logging.security_compliance import SecurityEventType

        # Initialize security logger
        security_logger = SecurityEventLogger("security-alert-test")

        # Generate multiple failed authentication attempts
        for i in range(6):  # More than threshold
            security_logger.log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                user_id="suspicious-user",
                source_ip="192.168.1.100",
                success=False,
                details={"attempt": i + 1}
            )

            # Record metrics
            self.collector.record_auth_attempt("password", False)
            self.collector.record_security_event("auth_failure", "medium")

        # In a real system, this would trigger brute force alerts
        # Here we verify the events are logged
        metrics_output = self.collector.get_metrics_output()
        self.assertIn("fedzk_auth_attempts_total 6", metrics_output)
        self.assertIn("fedzk_security_events_total 6", metrics_output)


if __name__ == '__main__':
    unittest.main()
