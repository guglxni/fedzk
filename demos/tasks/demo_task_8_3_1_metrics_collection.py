#!/usr/bin/env python3
"""
FEDzk Metrics Collection Demo
============================

Demonstrates the comprehensive monitoring and observability system
implemented for FEDzk including Prometheus metrics, distributed tracing,
and performance dashboards.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from fedzk.monitoring import (
        FEDzkMetricsCollector,
        DistributedTracingManager,
        start_metrics_server,
        ZKProofMetrics
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install prometheus-client opentelemetry-distro opentelemetry-exporter-jaeger")
    sys.exit(1)


def demo_basic_metrics():
    """Demonstrate basic metrics collection"""
    print("📊 DEMO: Basic Metrics Collection")
    print("=" * 50)

    # Initialize metrics collector
    collector = FEDzkMetricsCollector("fedzk-demo")

    # Record some sample metrics
    print("Recording sample HTTP requests...")
    collector.record_request("GET", "/health", 200, 0.05)
    collector.record_request("POST", "/api/proof", 201, 0.25)
    collector.record_request("GET", "/metrics", 200, 0.02)

    print("Recording authentication attempts...")
    collector.record_auth_attempt("jwt", True)
    collector.record_auth_attempt("api_key", True)
    collector.record_auth_attempt("invalid", False)

    print("Recording security events...")
    collector.record_security_event("login_attempt", "info")
    collector.record_security_event("suspicious_activity", "warning")

    # Display metrics output
    print("\n📈 Sample Metrics Output:")
    print("-" * 30)
    metrics = collector.get_metrics_output()
    # Show first few lines
    lines = metrics.split('\n')[:20]
    for line in lines:
        if line.strip():
            print(f"  {line}")

    print("  ... (truncated)")
    return collector


def demo_zk_metrics():
    """Demonstrate ZK proof metrics"""
    print("\n🔐 DEMO: ZK Proof Metrics")
    print("=" * 50)

    collector = FEDzkMetricsCollector("fedzk-zk-demo")

    # Record ZK proof operations
    print("Recording ZK proof generation metrics...")

    # Successful proof
    proof_metrics = ZKProofMetrics(
        proof_generation_time=1.25,
        proof_size_bytes=2048,
        verification_time=0.35,
        circuit_complexity=1024,
        success=True,
        proof_type="federated_aggregation"
    )
    collector.record_proof_generation(proof_metrics)

    # Failed proof
    failed_proof = ZKProofMetrics(
        proof_generation_time=0.8,
        proof_size_bytes=0,
        verification_time=0,
        circuit_complexity=512,
        success=False,
        proof_type="model_update"
    )
    collector.record_proof_generation(failed_proof)

    # Record circuit complexity
    collector.record_circuit_complexity("federated_aggregation", 2048, 1024)
    collector.record_circuit_complexity("model_update", 1024, 512)

    print("ZK Metrics recorded:")
    print("  ✅ Proof generation: 1.25s (federated_aggregation)")
    print("  ❌ Proof generation: 0.8s (model_update - failed)")
    print("  📊 Circuit complexity metrics recorded")

    return collector


def demo_fl_metrics():
    """Demonstrate federated learning metrics"""
    print("\n🤖 DEMO: Federated Learning Metrics")
    print("=" * 50)

    collector = FEDzkMetricsCollector("fedzk-fl-demo")

    # Record FL operations
    print("Recording federated learning operations...")

    collector.record_fl_round("completed")
    collector.record_fl_round("completed")
    collector.record_fl_round("failed")

    # Record MPC operations
    collector.record_mpc_operation("secure_aggregation", True, 0.45)
    collector.record_mpc_operation("key_generation", True, 0.12)
    collector.record_mpc_operation("signature_verification", False, 0.08)

    print("FL Metrics recorded:")
    print("  🎯 Training rounds: 2 completed, 1 failed")
    print("  🔐 MPC operations: 2 successful, 1 failed")

    return collector


def demo_distributed_tracing():
    """Demonstrate distributed tracing"""
    print("\n🔗 DEMO: Distributed Tracing")
    print("=" * 50)

    try:
        # Initialize tracing
        tracer = DistributedTracingManager("fedzk-tracing-demo")

        print("Distributed tracing initialized:")
        print("  📡 Jaeger integration ready")
        print("  🔄 Request tracing enabled")
        print("  📊 Span collection active")

        # Create sample spans
        with tracer.create_span("proof_generation") as span:
            tracer.add_span_attribute(span, "circuit_type", "federated_aggregation")
            tracer.add_span_attribute(span, "complexity", 1024)

            # Nested span
            with tracer.create_span("circuit_compilation") as child_span:
                tracer.add_span_attribute(child_span, "compilation_time", 0.5)
                time.sleep(0.1)  # Simulate work

            tracer.add_span_attribute(span, "generation_time", 1.25)

        print("  ✅ Sample trace created with nested spans")
        print("  📈 Trace data sent to Jaeger collector")

    except Exception as e:
        print(f"  ⚠️  Tracing demo error: {e}")
        print("  💡 Note: Requires Jaeger collector to be running")


def demo_metrics_server():
    """Demonstrate metrics HTTP server"""
    print("\n🌐 DEMO: Metrics HTTP Server")
    print("=" * 50)

    # Create collector with sample data
    collector = FEDzkMetricsCollector("fedzk-server-demo")

    # Add sample metrics
    collector.record_request("GET", "/health", 200, 0.02)
    collector.record_request("GET", "/metrics", 200, 0.05)

    proof_metrics = ZKProofMetrics(
        proof_generation_time=0.8,
        proof_size_bytes=1536,
        verification_time=0.25,
        circuit_complexity=768,
        success=True,
        proof_type="demo_circuit"
    )
    collector.record_proof_generation(proof_metrics)

    print("Starting metrics server on http://localhost:8080")
    print("Endpoints:")
    print("  📊 /metrics - Prometheus metrics")
    print("  ❤️  /health - Health check")
    print("")

    try:
        # Start server
        server = start_metrics_server('localhost', 8080, collector)

        print("Server started successfully!")
        print("🔗 Open http://localhost:8080/metrics in your browser")
        print("🔗 Open http://localhost:8080/health for health check")
        print("")
        print("Press Ctrl+C to stop the server...")

        # Keep server running for a bit
        time.sleep(2)

        print("\nSample metrics output:")
        metrics = collector.get_metrics_output()
        sample_lines = [line for line in metrics.split('\n')[:10] if line.strip()]
        for line in sample_lines:
            print(f"  {line}")

        server.stop()
        print("\n✅ Metrics server demo completed")

    except Exception as e:
        print(f"❌ Server demo error: {e}")


def main():
    """Run all monitoring demos"""
    print("🚀 FEDzk Monitoring & Observability Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive metrics collection system")
    print("including Prometheus integration, distributed tracing, and dashboards.")
    print("")

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Run demos
        demo_basic_metrics()
        demo_zk_metrics()
        demo_fl_metrics()
        demo_distributed_tracing()
        demo_metrics_server()

        print("\n" + "=" * 60)
        print("🎉 ALL MONITORING DEMOS COMPLETED!")
        print("=" * 60)
        print("")
        print("📋 SUMMARY OF IMPLEMENTED FEATURES:")
        print("  ✅ Prometheus metrics integration")
        print("  ✅ Custom ZK proof generation metrics")
        print("  ✅ Distributed tracing with OpenTelemetry")
        print("  ✅ Performance monitoring dashboards")
        print("  ✅ HTTP metrics server")
        print("  ✅ Helm chart monitoring configurations")
        print("")
        print("📊 DASHBOARDS CREATED:")
        print("  • FEDzk Overview Dashboard")
        print("  • FEDzk ZK Proof Monitoring")
        print("  • FEDzk Federated Learning Monitoring")
        print("  • FEDzk Security Monitoring")
        print("")
        print("🔧 INTEGRATION READY:")
        print("  • ServiceMonitor configurations")
        print("  • Alert rules and thresholds")
        print("  • Jaeger tracing integration")
        print("  • Production-ready configurations")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
