"""
Tests for FEDzk Monitoring and Metrics Collection
"""

import unittest
import time
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fedzk.monitoring.metrics import (
        FEDzkMetricsCollector,
        ZKProofMetrics,
        DistributedTracingManager
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@unittest.skipUnless(MONITORING_AVAILABLE, "Monitoring dependencies not available")
class TestFEDzkMetrics(unittest.TestCase):
    """Test FEDzk metrics collection"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = FEDzkMetricsCollector("test-service")

    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'collector'):
            # Clear registry
            pass

    def test_basic_metrics_recording(self):
        """Test basic metrics recording"""
        # Record HTTP request
        self.collector.record_request("GET", "/health", 200, 0.05)

        # Record authentication
        self.collector.record_auth_attempt("jwt", True)

        # Record security event
        self.collector.record_security_event("login", "info")

        # Verify metrics output contains our data
        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_requests_total", output)
        self.assertIn("fedzk_auth_attempts_total", output)
        self.assertIn("fedzk_security_events_total", output)

    def test_zk_proof_metrics(self):
        """Test ZK proof metrics recording"""
        proof_metrics = ZKProofMetrics(
            proof_generation_time=1.25,
            proof_size_bytes=2048,
            verification_time=0.35,
            circuit_complexity=1000,
            success=True,
            proof_type="test_circuit"
        )

        self.collector.record_proof_generation(proof_metrics)

        # Verify proof metrics
        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_zk_proof_generation_total", output)
        self.assertIn("fedzk_zk_proof_generation_duration_seconds", output)
        self.assertIn("fedzk_zk_proof_size_bytes", output)

    def test_fl_metrics(self):
        """Test federated learning metrics"""
        # Record FL round
        self.collector.record_fl_round("completed")

        # Record MPC operation
        self.collector.record_mpc_operation("aggregation", True, 0.45)

        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_fl_rounds_total", output)
        self.assertIn("fedzk_mpc_operations_total", output)
        self.assertIn("fedzk_mpc_computation_duration_seconds", output)

    def test_circuit_complexity(self):
        """Test circuit complexity recording"""
        self.collector.record_circuit_complexity("test_circuit", 2048, 1024)

        output = self.collector.get_metrics_output()
        self.assertIn("fedzk_zk_circuit_complexity", output)

    def test_metrics_output_format(self):
        """Test metrics output format"""
        output = self.collector.get_metrics_output()

        # Should be valid Prometheus format
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

        # Should contain HELP and TYPE lines
        lines = output.split('\n')
        help_lines = [line for line in lines if line.startswith('# HELP')]
        type_lines = [line for line in lines if line.startswith('# TYPE')]

        self.assertTrue(len(help_lines) > 0)
        self.assertTrue(len(type_lines) > 0)


@unittest.skipUnless(MONITORING_AVAILABLE, "Monitoring dependencies not available")
class TestDistributedTracing(unittest.TestCase):
    """Test distributed tracing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.tracer = DistributedTracingManager("test-service")

    def test_tracer_initialization(self):
        """Test tracer initialization"""
        self.assertIsNotNone(self.tracer)

    @patch('opentelemetry.trace.get_tracer')
    def test_span_creation(self, mock_get_tracer):
        """Test span creation"""
        mock_span = Mock()
        mock_tracer = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        mock_get_tracer.return_value = mock_tracer

        with self.tracer.create_span("test_operation") as span:
            self.assertIsNotNone(span)
            self.tracer.add_span_attribute(span, "test_key", "test_value")
            self.tracer.record_span_event(span, "test_event")


class TestMetricsIntegration(unittest.TestCase):
    """Test metrics integration scenarios"""

    @unittest.skipUnless(MONITORING_AVAILABLE, "Monitoring dependencies not available")
    def test_full_workflow(self):
        """Test complete metrics workflow"""
        collector = FEDzkMetricsCollector("integration-test")

        # Simulate a complete workflow
        # 1. Authentication
        collector.record_auth_attempt("jwt", True)

        # 2. ZK proof generation
        proof_metrics = ZKProofMetrics(
            proof_generation_time=0.8,
            proof_size_bytes=1024,
            verification_time=0.2,
            circuit_complexity=512,
            success=True,
            proof_type="integration_test"
        )
        collector.record_proof_generation(proof_metrics)

        # 3. FL operations
        collector.record_fl_round("completed")
        collector.record_mpc_operation("secure_aggregation", True, 0.3)

        # 4. Security monitoring
        collector.record_security_event("workflow_complete", "info")

        # Verify all metrics are captured
        output = collector.get_metrics_output()

        expected_metrics = [
            "fedzk_auth_attempts_total",
            "fedzk_zk_proof_generation_total",
            "fedzk_fl_rounds_total",
            "fedzk_mpc_operations_total",
            "fedzk_security_events_total"
        ]

        for metric in expected_metrics:
            self.assertIn(metric, output)


if __name__ == '__main__':
    unittest.main()
