"""
FEDzk Monitoring and Observability
=================================

Comprehensive monitoring solution for FEDzk including:
- Prometheus metrics collection
- Distributed tracing
- Performance dashboards
- Alerting rules
"""

from .metrics import (
    FEDzkMetricsCollector,
    FEDzkMetricsMiddleware,
    ZKProofMetricsTracker,
    DistributedTracingManager,
    track_zk_proof_operation,
    get_metrics_collector,
    get_tracing_manager,
    initialize_monitoring,
    ZKProofMetrics
)

from .metrics_server import (
    MetricsHTTPServer,
    start_metrics_server,
    setup_metrics_endpoint
)

__all__ = [
    # Core metrics
    'FEDzkMetricsCollector',
    'FEDzkMetricsMiddleware',
    'ZKProofMetricsTracker',
    'DistributedTracingManager',
    'ZKProofMetrics',

    # Tracing and monitoring
    'track_zk_proof_operation',
    'get_metrics_collector',
    'get_tracing_manager',
    'initialize_monitoring',

    # HTTP server
    'MetricsHTTPServer',
    'start_metrics_server',
    'setup_metrics_endpoint'
]

__version__ = "1.0.0"
