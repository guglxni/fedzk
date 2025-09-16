#!/usr/bin/env python3
"""
FEDzk Monitoring Standalone Demo
===============================

Demonstrates the monitoring system without requiring full FEDzk dependencies.
This demo shows how to integrate Prometheus metrics, distributed tracing,
and performance dashboards into any Python application.
"""

import sys
import time
import logging
import json
from typing import Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock classes for demonstration when dependencies are missing
class MockPrometheusCollector:
    """Mock Prometheus collector for demo purposes"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics = {
            "requests_total": 0,
            "request_duration_sum": 0,
            "request_duration_count": 0,
            "zk_proofs_generated": 0,
            "zk_proofs_failed": 0,
            "auth_attempts": 0,
            "security_events": 0
        }
        logger.info(f"Mock Prometheus collector initialized for {service_name}")

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.metrics["requests_total"] += 1
        self.metrics["request_duration_sum"] += duration
        self.metrics["request_duration_count"] += 1
        logger.info(f"Recorded request: {method} {endpoint} -> {status} ({duration:.3f}s)")

    def record_zk_proof(self, proof_type: str, success: bool, generation_time: float):
        """Record ZK proof metrics"""
        if success:
            self.metrics["zk_proofs_generated"] += 1
        else:
            self.metrics["zk_proofs_failed"] += 1
        logger.info(f"Recorded ZK proof: {proof_type} ({'success' if success else 'failed'}) in {generation_time:.3f}s")

    def record_auth_attempt(self, method: str, success: bool):
        """Record authentication metrics"""
        self.metrics["auth_attempts"] += 1
        logger.info(f"Recorded auth attempt: {method} ({'success' if success else 'failed'})")

    def record_security_event(self, event_type: str, severity: str):
        """Record security event"""
        self.metrics["security_events"] += 1
        logger.info(f"Recorded security event: {event_type} ({severity})")

    def get_metrics_output(self) -> str:
        """Get Prometheus metrics output"""
        output = f"# FEDzk Metrics for {self.service_name}\n"

        # Request metrics
        output += f"fedzk_requests_total {self.metrics['requests_total']}\n"

        if self.metrics["request_duration_count"] > 0:
            avg_duration = self.metrics["request_duration_sum"] / self.metrics["request_duration_count"]
            output += f"fedzk_request_duration_average_seconds {avg_duration:.6f}\n"

        # ZK metrics
        output += f"fedzk_zk_proofs_generated_total {self.metrics['zk_proofs_generated']}\n"
        output += f"fedzk_zk_proofs_failed_total {self.metrics['zk_proofs_failed']}\n"

        # Security metrics
        output += f"fedzk_auth_attempts_total {self.metrics['auth_attempts']}\n"
        output += f"fedzk_security_events_total {self.metrics['security_events']}\n"

        return output

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        return {
            "service_name": self.service_name,
            "timestamp": time.time(),
            "metrics": self.metrics.copy()
        }


class MockTracingManager:
    """Mock distributed tracing manager"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.enabled = True
        self.spans = []
        logger.info(f"Mock tracing manager initialized for {service_name}")

    def create_span(self, name: str):
        """Create a mock span context manager"""
        from contextlib import contextmanager

        @contextmanager
        def span_context():
            span = {
                "name": name,
                "start_time": time.time(),
                "attributes": {},
                "events": []
            }
            self.spans.append(span)
            logger.info(f"Created span: {name}")
            try:
                yield span
            finally:
                span["end_time"] = time.time()
                duration = span["end_time"] - span["start_time"]
                logger.info(f"Completed span: {name} ({duration:.3f}s)")

        return span_context()

    def add_span_attribute(self, span, key: str, value: Any):
        """Add attribute to span"""
        if span and "attributes" in span:
            span["attributes"][key] = value
            logger.info(f"Added attribute to span {span['name']}: {key}={value}")

    def record_span_event(self, span, name: str, attributes: Dict[str, Any] = None):
        """Record event in span"""
        if span and "events" in span:
            event = {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {}
            }
            span["events"].append(event)
            logger.info(f"Recorded event in span {span['name']}: {name}")


def demo_basic_monitoring():
    """Demonstrate basic monitoring capabilities"""
    print("📊 DEMO: Basic Monitoring Setup")
    print("=" * 50)

    # Initialize monitoring
    collector = MockPrometheusCollector("fedzk-demo")
    tracer = MockTracingManager("fedzk-demo")

    print("✅ Monitoring components initialized")
    print("   • Prometheus metrics collector")
    print("   • Distributed tracing manager")
    print()

    return collector, tracer


def demo_http_metrics(collector):
    """Demonstrate HTTP request monitoring"""
    print("🌐 DEMO: HTTP Request Monitoring")
    print("=" * 50)

    # Simulate various HTTP requests
    requests = [
        ("GET", "/health", 200, 0.023),
        ("POST", "/api/proof/generate", 201, 0.847),
        ("GET", "/api/proof/status", 200, 0.034),
        ("POST", "/api/auth/login", 200, 0.156),
        ("GET", "/api/metrics", 200, 0.012),
        ("POST", "/api/model/update", 400, 0.234),  # Error case
    ]

    print("Recording HTTP requests...")
    for method, endpoint, status, duration in requests:
        collector.record_request(method, endpoint, status, duration)
        status_icon = "✅" if status < 400 else "❌"
        print(".3f")

    print()
    print("📈 HTTP Metrics Summary:")
    metrics = collector.get_metrics_dict()
    print(f"   • Total requests: {metrics['metrics']['requests_total']}")
    if metrics['metrics']['request_duration_count'] > 0:
        avg_duration = metrics['metrics']['request_duration_sum'] / metrics['metrics']['request_duration_count']
        print(".3f")
    print()


def demo_zk_monitoring(collector, tracer):
    """Demonstrate ZK proof monitoring"""
    print("🔐 DEMO: ZK Proof Monitoring")
    print("=" * 50)

    # Simulate ZK proof generation with tracing
    proof_types = [
        ("federated_aggregation", True, 1.245),
        ("model_update", True, 0.987),
        ("secure_multiplication", False, 0.654),  # Failed proof
        ("privacy_preservation", True, 2.134),
    ]

    print("Recording ZK proof generations with distributed tracing...")

    for proof_type, success, generation_time in proof_types:
        # Create trace span
        with tracer.create_span(f"zk_proof_{proof_type}") as span:
            tracer.add_span_attribute(span, "proof_type", proof_type)
            tracer.add_span_attribute(span, "expected_success", success)

            # Simulate proof generation steps
            tracer.record_span_event(span, "circuit_compilation_started")
            time.sleep(0.01)  # Simulate work

            tracer.record_span_event(span, "witness_generation_completed")
            time.sleep(0.01)  # Simulate work

            if success:
                tracer.record_span_event(span, "proof_generation_completed")
                tracer.add_span_attribute(span, "generation_time", generation_time)
            else:
                tracer.record_span_event(span, "proof_generation_failed")
                tracer.add_span_attribute(span, "failure_reason", "circuit_constraint_violation")

        # Record metrics
        collector.record_zk_proof(proof_type, success, generation_time)

        status_icon = "✅" if success else "❌"
        print(".3f")

    print()
    print("📊 ZK Metrics Summary:")
    metrics = collector.get_metrics_dict()
    print(f"   • Successful proofs: {metrics['metrics']['zk_proofs_generated']}")
    print(f"   • Failed proofs: {metrics['metrics']['zk_proofs_failed']}")
    success_rate = metrics['metrics']['zk_proofs_generated'] / (metrics['metrics']['zk_proofs_generated'] + metrics['metrics']['zk_proofs_failed']) * 100
    print(".1f")
    print()


def demo_security_monitoring(collector):
    """Demonstrate security monitoring"""
    print("🔒 DEMO: Security Monitoring")
    print("=" * 50)

    # Simulate authentication attempts
    auth_attempts = [
        ("jwt", True),
        ("api_key", True),
        ("basic_auth", False),
        ("oauth", True),
        ("invalid_token", False),
        ("expired_session", False),
    ]

    print("Recording authentication attempts...")
    for method, success in auth_attempts:
        collector.record_auth_attempt(method, success)
        status_icon = "✅" if success else "❌"
        print(f"   {status_icon} {method}: {'Successful' if success else 'Failed'}")

    print()

    # Simulate security events
    security_events = [
        ("login_attempt", "info"),
        ("password_change", "info"),
        ("suspicious_activity", "warning"),
        ("brute_force_attempt", "error"),
        ("unauthorized_access", "error"),
        ("ssl_certificate_expiry", "warning"),
    ]

    print("Recording security events...")
    for event_type, severity in security_events:
        collector.record_security_event(event_type, severity)
        severity_icon = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨"
        }.get(severity, "❓")
        print(f"   {severity_icon} {event_type}: {severity.upper()}")

    print()
    print("📊 Security Metrics Summary:")
    metrics = collector.get_metrics_dict()
    print(f"   • Authentication attempts: {metrics['metrics']['auth_attempts']}")
    print(f"   • Security events: {metrics['metrics']['security_events']}")
    print()


def demo_prometheus_output(collector):
    """Demonstrate Prometheus metrics output"""
    print("📊 DEMO: Prometheus Metrics Output")
    print("=" * 50)

    print("Prometheus-formatted metrics output:")
    print("-" * 40)

    output = collector.get_metrics_output()
    lines = output.split('\n')

    for i, line in enumerate(lines, 1):
        if line.strip():
            print("2d")

    print()
    print("💡 This output can be scraped by Prometheus and visualized in Grafana")
    print()


def demo_dashboard_config():
    """Show dashboard configuration examples"""
    print("📈 DEMO: Dashboard Configuration")
    print("=" * 50)

    print("Sample Grafana dashboard panels for FEDzk monitoring:")
    print()

    dashboard_panels = [
        {
            "title": "ZK Proof Generation Rate",
            "query": "rate(fedzk_zk_proofs_generated_total[5m])",
            "description": "Rate of successful ZK proof generations"
        },
        {
            "title": "Request Latency (95th percentile)",
            "query": "histogram_quantile(0.95, rate(fedzk_request_duration_seconds_bucket[5m]))",
            "description": "95th percentile request latency"
        },
        {
            "title": "Authentication Success Rate",
            "query": "rate(fedzk_auth_attempts_total{success='true'}[5m]) / rate(fedzk_auth_attempts_total[5m]) * 100",
            "description": "Percentage of successful authentication attempts"
        },
        {
            "title": "Active Security Events",
            "query": "rate(fedzk_security_events_total[5m])",
            "description": "Rate of security events by severity"
        }
    ]

    for i, panel in enumerate(dashboard_panels, 1):
        print(f"{i}. {panel['title']}")
        print(f"   Query: {panel['query']}")
        print(f"   Purpose: {panel['description']}")
        print()


def main():
    """Run the complete monitoring demo"""
    print("🚀 FEDzk Monitoring & Observability Standalone Demo")
    print("=" * 70)
    print("This demo showcases the comprehensive monitoring system capabilities")
    print("including Prometheus metrics, distributed tracing, and dashboard examples.")
    print("No external dependencies required - uses mock implementations for demonstration.")
    print()

    try:
        # Initialize monitoring
        collector, tracer = demo_basic_monitoring()

        # Demonstrate different monitoring aspects
        demo_http_metrics(collector)
        demo_zk_monitoring(collector, tracer)
        demo_security_monitoring(collector)
        demo_prometheus_output(collector)
        demo_dashboard_config()

        print("=" * 70)
        print("🎉 MONITORING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("📋 IMPLEMENTATION SUMMARY:")
        print("✅ Prometheus metrics integration")
        print("✅ Custom ZK proof generation metrics")
        print("✅ Distributed tracing capabilities")
        print("✅ Security monitoring and alerting")
        print("✅ Performance dashboard configurations")
        print("✅ Helm chart monitoring integration")
        print()
        print("🔧 PRODUCTION INTEGRATION:")
        print("• Install prometheus-client and opentelemetry packages")
        print("• Configure monitoring in Helm values")
        print("• Deploy with ServiceMonitor for automatic scraping")
        print("• Import dashboard JSON files into Grafana")
        print("• Set up Jaeger for distributed tracing")
        print()
        print("📊 AVAILABLE DASHBOARDS:")
        print("• FEDzk Overview Dashboard")
        print("• FEDzk ZK Proof Monitoring")
        print("• FEDzk Federated Learning Monitoring")
        print("• FEDzk Security Monitoring")
        print()
        print("🚀 READY FOR PRODUCTION MONITORING!")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
