"""
FEDzk Metrics Collection System
===============================

Comprehensive metrics collection for FEDzk components including:
- Prometheus integration
- Custom ZK proof generation metrics
- Performance monitoring
- Distributed tracing
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import threading
import psutil
import os

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None
    Counter = None
    Histogram = None
    Gauge = None
    Summary = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Install with: pip install opentelemetry-distro opentelemetry-exporter-jaeger")

# Optional import for configuration
try:
    from fedzk.config import EnvironmentConfigManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Logger not yet defined, use print
    print("WARNING: FEDzk configuration not available, using defaults")

logger = logging.getLogger(__name__)


class MetricsType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class ZKProofMetrics:
    """Metrics specific to ZK proof operations"""
    proof_generation_time: float
    proof_size_bytes: int
    verification_time: float
    circuit_complexity: int
    success: bool
    proof_type: str


# Functional metrics implementations for when Prometheus is not available
class FunctionalCounter:
    """Functional counter implementation without external dependencies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0):
        """Increment the counter"""
        with self._lock:
            self._value += amount

    def add(self, amount: float):
        """Add to the counter"""
        self.inc(amount)

    def _child_samples(self):
        """Return sample data for compatibility"""
        return [(self.name, {}, self._value)]


class FunctionalHistogram:
    """Functional histogram implementation without external dependencies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._samples = []
        self._lock = threading.Lock()

    def observe(self, value: float):
        """Observe a value"""
        with self._lock:
            self._samples.append(value)

    def _child_samples(self):
        """Return sample data for compatibility"""
        return [(f"{self.name}_count", {}, len(self._samples)),
                (f"{self.name}_sum", {}, sum(self._samples) if self._samples else 0)]


class FunctionalGauge:
    """Functional gauge implementation without external dependencies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float):
        """Set the gauge value"""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0):
        """Increment the gauge"""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0):
        """Decrement the gauge"""
        with self._lock:
            self._value -= amount

    def _child_samples(self):
        """Return sample data for compatibility"""
        return [(self.name, {}, self._value)]


class FEDzkMetricsCollector:
    """
    Comprehensive metrics collector for FEDzk system
    """

    def __init__(self, service_name: str = "fedzk", registry: Optional[Any] = None):
        # Allow functional metrics even without Prometheus
        self.service_name = service_name
        if PROMETHEUS_AVAILABLE and CollectorRegistry is not None:
            self.registry = registry or CollectorRegistry()
        else:
            self.registry = None

        # Initialize metrics
        self._setup_base_metrics()
        self._setup_zk_metrics()
        self._setup_api_metrics()
        self._setup_security_metrics()

        logger.info(f"Initialized FEDzk metrics collector for service: {service_name}")

    def _setup_base_metrics(self):
        """Setup basic system metrics"""
        if not PROMETHEUS_AVAILABLE or Counter is None:
            # Functional metrics implementation without external dependencies
            self.request_total = FunctionalCounter('fedzk_requests_total', 'Total number of requests')
            self.request_duration = FunctionalHistogram('fedzk_request_duration_seconds', 'Request duration in seconds')
            self.memory_usage = FunctionalGauge('fedzk_memory_usage_bytes', 'Memory usage in bytes')
            self.cpu_usage = FunctionalGauge('fedzk_cpu_usage_percent', 'CPU usage percentage')
            self.active_connections = FunctionalGauge('fedzk_active_connections', 'Number of active connections')
            return

        # Request metrics
        self.request_total = Counter(
            'fedzk_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'fedzk_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )

        # System metrics
        self.memory_usage = Gauge(
            'fedzk_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'fedzk_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        # Active connections
        self.active_connections = Gauge(
            'fedzk_active_connections',
            'Number of active connections',
            ['component'],
            registry=self.registry
        )

    def _setup_zk_metrics(self):
        """Setup ZK-specific metrics"""
        if not PROMETHEUS_AVAILABLE or Counter is None:
            # Functional metrics implementation
            self.proof_generation_total = FunctionalCounter('fedzk_zk_proof_generation_total', 'Total number of ZK proof generations')
            self.proof_generation_duration = FunctionalHistogram('fedzk_zk_proof_generation_duration_seconds', 'ZK proof generation duration')
            self.proof_verification_total = FunctionalCounter('fedzk_zk_proof_verification_total', 'Total number of ZK proof verifications')
            self.proof_verification_duration = FunctionalHistogram('fedzk_zk_proof_verification_duration_seconds', 'ZK proof verification duration')
            self.proof_size_bytes = FunctionalHistogram('fedzk_zk_proof_size_bytes', 'ZK proof size in bytes')
            self.circuit_complexity = FunctionalGauge('fedzk_zk_circuit_complexity', 'Circuit complexity metrics')
            self.trusted_setup_duration = FunctionalHistogram('fedzk_zk_trusted_setup_duration_seconds', 'Trusted setup duration')
            return

        # Proof generation metrics
        self.proof_generation_total = Counter(
            'fedzk_zk_proof_generation_total',
            'Total number of ZK proof generations',
            ['circuit_type', 'success'],
            registry=self.registry
        )

        self.proof_generation_duration = Histogram(
            'fedzk_zk_proof_generation_duration_seconds',
            'ZK proof generation duration',
            ['circuit_type'],
            registry=self.registry
        )

        self.proof_verification_total = Counter(
            'fedzk_zk_proof_verification_total',
            'Total number of ZK proof verifications',
            ['circuit_type', 'success'],
            registry=self.registry
        )

        self.proof_verification_duration = Histogram(
            'fedzk_zk_proof_verification_duration_seconds',
            'ZK proof verification duration',
            ['circuit_type'],
            registry=self.registry
        )

        # Proof size metrics
        self.proof_size_bytes = Histogram(
            'fedzk_zk_proof_size_bytes',
            'ZK proof size in bytes',
            ['circuit_type'],
            registry=self.registry
        )

        # Circuit complexity
        self.circuit_complexity = Gauge(
            'fedzk_zk_circuit_complexity',
            'Circuit complexity metrics',
            ['circuit_type', 'metric'],
            registry=self.registry
        )

        # Trusted setup metrics
        self.trusted_setup_duration = Histogram(
            'fedzk_zk_trusted_setup_duration_seconds',
            'Trusted setup duration',
            ['phase'],
            registry=self.registry
        )

    def _setup_api_metrics(self):
        """Setup API-specific metrics"""
        if not PROMETHEUS_AVAILABLE or Counter is None:
            # Functional metrics implementation
            self.fl_rounds_total = FunctionalCounter('fedzk_fl_rounds_total', 'Total federated learning rounds')
            self.fl_participants = FunctionalGauge('fedzk_fl_participants_active', 'Number of active FL participants')
            self.fl_model_updates = FunctionalCounter('fedzk_fl_model_updates_total', 'Total model updates received')
            self.mpc_operations_total = FunctionalCounter('fedzk_mpc_operations_total', 'Total MPC operations')
            self.mpc_computation_duration = FunctionalHistogram('fedzk_mpc_computation_duration_seconds', 'MPC computation duration')
            return

        # Federated learning metrics
        self.fl_rounds_total = Counter(
            'fedzk_fl_rounds_total',
            'Total federated learning rounds',
            ['status'],
            registry=self.registry
        )

        self.fl_participants = Gauge(
            'fedzk_fl_participants_active',
            'Number of active FL participants',
            registry=self.registry
        )

        self.fl_model_updates = Counter(
            'fedzk_fl_model_updates_total',
            'Total model updates received',
            ['participant_id'],
            registry=self.registry
        )

        # MPC metrics
        self.mpc_operations_total = Counter(
            'fedzk_mpc_operations_total',
            'Total MPC operations',
            ['operation_type', 'success'],
            registry=self.registry
        )

        self.mpc_computation_duration = Histogram(
            'fedzk_mpc_computation_duration_seconds',
            'MPC computation duration',
            ['operation_type'],
            registry=self.registry
        )

    def _setup_security_metrics(self):
        """Setup security-related metrics"""
        if not PROMETHEUS_AVAILABLE or Counter is None:
            # Functional metrics implementation
            self.auth_attempts_total = FunctionalCounter('fedzk_auth_attempts_total', 'Total authentication attempts')
            self.tls_handshakes_total = FunctionalCounter('fedzk_tls_handshakes_total', 'Total TLS handshakes')
            self.encryption_operations = FunctionalCounter('fedzk_encryption_operations_total', 'Total encryption/decryption operations')
            self.security_events_total = FunctionalCounter('fedzk_security_events_total', 'Total security events')
            return

        # Authentication metrics
        self.auth_attempts_total = Counter(
            'fedzk_auth_attempts_total',
            'Total authentication attempts',
            ['method', 'success'],
            registry=self.registry
        )

        # TLS metrics
        self.tls_handshakes_total = Counter(
            'fedzk_tls_handshakes_total',
            'Total TLS handshakes',
            ['version', 'success'],
            registry=self.registry
        )

        # Encryption metrics
        self.encryption_operations = Counter(
            'fedzk_encryption_operations_total',
            'Total encryption/decryption operations',
            ['operation', 'algorithm'],
            registry=self.registry
        )

        # Security events
        self.security_events_total = Counter(
            'fedzk_security_events_total',
            'Total security events',
            ['event_type', 'severity'],
            registry=self.registry
        )

    # ZK Proof Metrics Methods
    def record_proof_generation(self, metrics: ZKProofMetrics):
        """Record ZK proof generation metrics"""
        if PROMETHEUS_AVAILABLE and hasattr(self.proof_generation_total, 'labels'):
            # Prometheus implementation
            labels = {
                'circuit_type': metrics.proof_type,
                'success': str(metrics.success).lower()
            }

            self.proof_generation_total.labels(**labels).inc()
            self.proof_generation_duration.labels(
                circuit_type=metrics.proof_type
            ).observe(metrics.proof_generation_time)

            if metrics.success:
                self.proof_verification_duration.labels(
                    circuit_type=metrics.proof_type
                ).observe(metrics.verification_time)

                self.proof_size_bytes.labels(
                    circuit_type=metrics.proof_type
                ).observe(metrics.proof_size_bytes)
        else:
            # Functional implementation
            self.proof_generation_total.inc()
            self.proof_generation_duration.observe(metrics.proof_generation_time)

            if metrics.success:
                self.proof_verification_duration.observe(metrics.verification_time)
                self.proof_size_bytes.observe(metrics.proof_size_bytes)

    def record_circuit_complexity(self, circuit_type: str, constraints: int, variables: int):
        """Record circuit complexity metrics"""
        if PROMETHEUS_AVAILABLE and hasattr(self.circuit_complexity, 'labels'):
            # Prometheus implementation
            self.circuit_complexity.labels(
                circuit_type=circuit_type, metric='constraints'
            ).set(constraints)

            self.circuit_complexity.labels(
                circuit_type=circuit_type, metric='variables'
            ).set(variables)
        else:
            # Functional implementation - store as gauge value
            self.circuit_complexity.set(constraints + variables)

    # API Metrics Methods
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        if PROMETHEUS_AVAILABLE and hasattr(self.request_total, 'labels'):
            # Prometheus implementation
            self.request_total.labels(
                method=method,
                endpoint=endpoint,
                status=str(status)
            ).inc()

            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        else:
            # Functional implementation
            self.request_total.inc()
            self.request_duration.observe(duration)

    def record_fl_round(self, status: str = "completed"):
        """Record federated learning round"""
        if PROMETHEUS_AVAILABLE and hasattr(self.fl_rounds_total, 'labels'):
            # Prometheus implementation
            self.fl_rounds_total.labels(status=status).inc()
        else:
            # Functional implementation
            self.fl_rounds_total.inc()

    def record_mpc_operation(self, operation_type: str, success: bool, duration: float):
        """Record MPC operation metrics"""
        if PROMETHEUS_AVAILABLE and hasattr(self.mpc_operations_total, 'labels'):
            # Prometheus implementation
            self.mpc_operations_total.labels(
                operation_type=operation_type,
                success=str(success).lower()
            ).inc()

            if success:
                self.mpc_computation_duration.labels(
                    operation_type=operation_type
                ).observe(duration)
        else:
            # Functional implementation
            self.mpc_operations_total.inc()
            if success:
                self.mpc_computation_duration.observe(duration)

    # Security Metrics Methods
    def record_auth_attempt(self, method: str, success: bool):
        """Record authentication attempt"""
        if PROMETHEUS_AVAILABLE and hasattr(self.auth_attempts_total, 'labels'):
            # Prometheus implementation
            self.auth_attempts_total.labels(
                method=method,
                success=str(success).lower()
            ).inc()
        else:
            # Functional implementation
            self.auth_attempts_total.inc()

    def record_security_event(self, event_type: str, severity: str):
        """Record security event"""
        if PROMETHEUS_AVAILABLE and hasattr(self.security_events_total, 'labels'):
            # Prometheus implementation
            self.security_events_total.labels(
                event_type=event_type,
                severity=severity
            ).inc()
        else:
            # Functional implementation
            self.security_events_total.inc()

    # System Metrics Methods
    def update_system_metrics(self):
        """Update system resource metrics"""
        import psutil

        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.labels(type='total').set(memory.total)
            self.memory_usage.labels(type='used').set(memory.used)
            self.memory_usage.labels(type='available').set(memory.available)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)

        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def get_metrics_output(self) -> str:
        """Get Prometheus metrics output"""
        if not PROMETHEUS_AVAILABLE or generate_latest is None:
            # Functional metrics output
            output = "# FEDzk Functional Metrics\n"

            # Add base metrics
            if hasattr(self, 'request_total'):
                for sample in self.request_total._child_samples():
                    output += f"{sample[0]} {sample[2]}\n"

            if hasattr(self, 'memory_usage'):
                # Set actual memory usage
                try:
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    self.memory_usage.set(memory_mb)
                except:
                    pass
                for sample in self.memory_usage._child_samples():
                    output += f"{sample[0]} {sample[2]}\n"

            if hasattr(self, 'cpu_usage'):
                # Set actual CPU usage
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    self.cpu_usage.set(cpu_percent)
                except:
                    pass
                for sample in self.cpu_usage._child_samples():
                    output += f"{sample[0]} {sample[2]}\n"

            return output
        return generate_latest(self.registry).decode('utf-8')

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary for custom exporters"""
        return {
            'service_name': self.service_name,
            'timestamp': time.time(),
            'registry': str(self.registry)
        }


class FEDzkMetricsMiddleware:
    """
    Middleware for automatic metrics collection in web applications
    """

    def __init__(self, collector: FEDzkMetricsCollector):
        self.collector = collector

    def __call__(self, environ, start_response):
        """WSGI middleware for request metrics"""
        start_time = time.time()

        def custom_start_response(status, headers, *args):
            # Record request metrics
            duration = time.time() - start_time
            method = environ.get('REQUEST_METHOD', 'UNKNOWN')
            path = environ.get('PATH_INFO', '/')

            # Extract status code
            status_code = int(status.split()[0]) if status else 200

            self.collector.record_request(method, path, status_code, duration)

            return start_response(status, headers, *args)

        return custom_start_response


class ZKProofMetricsTracker:
    """
    Context manager and decorator for tracking ZK proof operations
    """

    def __init__(self, collector: FEDzkMetricsCollector, proof_type: str):
        self.collector = collector
        self.proof_type = proof_type
        self.start_time = None
        self.metrics = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and self.metrics:
            generation_time = time.time() - self.start_time
            self.metrics.proof_generation_time = generation_time
            self.collector.record_proof_generation(self.metrics)

    def record_proof_metrics(self, proof_size: int, verification_time: float,
                           circuit_complexity: int, success: bool):
        """Record proof-specific metrics"""
        self.metrics = ZKProofMetrics(
            proof_generation_time=0,  # Will be set in __exit__
            proof_size_bytes=proof_size,
            verification_time=verification_time,
            circuit_complexity=circuit_complexity,
            success=success,
            proof_type=self.proof_type
        )


def track_zk_proof_operation(proof_type: str):
    """
    Decorator to track ZK proof operations
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get collector from kwargs or create new one
            collector = kwargs.get('metrics_collector')
            if not collector:
                # Try to get from first arg if it's a class instance
                if args and hasattr(args[0], 'metrics_collector'):
                    collector = args[0].metrics_collector

            if collector:
                with ZKProofMetricsTracker(collector, proof_type) as tracker:
                    try:
                        result = func(*args, **kwargs)

                        # Extract metrics from result if available
                        if hasattr(result, 'proof_size'):
                            tracker.record_proof_metrics(
                                proof_size=result.proof_size,
                                verification_time=getattr(result, 'verification_time', 0),
                                circuit_complexity=getattr(result, 'circuit_complexity', 0),
                                success=True
                            )
                        else:
                            # Generic success tracking
                            tracker.record_proof_metrics(0, 0, 0, True)

                        return result
                    except Exception as e:
                        tracker.record_proof_metrics(0, 0, 0, False)
                        raise
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator


class DistributedTracingManager:
    """
    Distributed tracing manager using OpenTelemetry
    """

    def __init__(self, service_name: str, jaeger_host: str = "localhost",
                 jaeger_port: int = 6831):
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            self.enabled = False
            return

        self.enabled = True
        self.service_name = service_name

        # Set up tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()

        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )

        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Instrument requests
        RequestsInstrumentor().instrument()

        self.tracer = trace.get_tracer(__name__)
        logger.info(f"Initialized distributed tracing for {service_name}")

    def create_span(self, name: str, kind: str = "internal"):
        """Create a new span"""
        if not self.enabled:
            return None

        return self.tracer.start_as_current_span(name)

    def add_span_attribute(self, span, key: str, value: Any):
        """Add attribute to span"""
        if span and self.enabled:
            span.set_attribute(key, value)

    def record_span_event(self, span, name: str, attributes: Dict[str, Any] = None):
        """Record event in span"""
        if span and self.enabled:
            span.add_event(name, attributes or {})


# Global instances
_metrics_collector = None
_tracing_manager = None


def get_metrics_collector() -> FEDzkMetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = FEDzkMetricsCollector()
    return _metrics_collector


def get_tracing_manager() -> DistributedTracingManager:
    """Get or create global tracing manager"""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = DistributedTracingManager("fedzk")
    return _tracing_manager


def initialize_monitoring(service_name: str = "fedzk",
                          enable_tracing: bool = True) -> tuple:
    """
    Initialize monitoring system
    Returns: (metrics_collector, tracing_manager)
    """
    global _metrics_collector, _tracing_manager

    # Initialize metrics
    if PROMETHEUS_AVAILABLE:
        _metrics_collector = FEDzkMetricsCollector(service_name)
        logger.info("Metrics collection initialized")
    else:
        logger.warning("Metrics collection disabled - prometheus-client not available")
        _metrics_collector = None

    # Initialize tracing
    if enable_tracing and OPENTELEMETRY_AVAILABLE:
        _tracing_manager = DistributedTracingManager(service_name)
        logger.info("Distributed tracing initialized")
    else:
        logger.warning("Distributed tracing disabled")
        _tracing_manager = None

    return _metrics_collector, _tracing_manager


# Health check endpoint for metrics
def health_check():
    """Health check for metrics system"""
    return {
        "status": "healthy",
        "metrics_enabled": PROMETHEUS_AVAILABLE,
        "tracing_enabled": OPENTELEMETRY_AVAILABLE,
        "timestamp": time.time()
    }
