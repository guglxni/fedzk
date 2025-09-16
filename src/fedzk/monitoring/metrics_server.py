"""
FEDzk Metrics HTTP Server
========================

Simple HTTP server for exposing Prometheus metrics
"""

import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import time

from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving Prometheus metrics"""

    def __init__(self, collector, *args, **kwargs):
        self.collector = collector
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/metrics':
            self._serve_metrics()
        elif self.path == '/health':
            self._serve_health()
        else:
            self._serve_404()

    def _serve_metrics(self):
        """Serve Prometheus metrics"""
        try:
            metrics_output = self.collector.get_metrics_output()
            self.send_response(200)
            self.send_header('Content-Type', self.collector.CONTENT_TYPE_LATEST)
            self.send_header('Content-Length', len(metrics_output))
            self.end_headers()
            self.wfile.write(metrics_output.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Error generating metrics")

    def _serve_health(self):
        """Serve health check"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "fedzk-metrics"
        }
        import json
        self.wfile.write(json.dumps(health_data).encode('utf-8'))

    def _serve_404(self):
        """Serve 404 for unknown endpoints"""
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"Metrics server: {format % args}")


class MetricsHTTPServer:
    """
    HTTP server for exposing Prometheus metrics
    """

    def __init__(self, collector, host: str = '0.0.0.0', port: int = 8000):
        self.collector = collector
        self.host = host
        self.port = port
        self.server = None
        self.thread = None

    def _create_handler(self):
        """Create handler class with collector"""
        collector = self.collector

        class HandlerWithCollector(MetricsHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(collector, *args, **kwargs)

        return HandlerWithCollector

    def start(self):
        """Start the metrics server in a background thread"""
        if self.server:
            logger.warning("Metrics server already running")
            return

        try:
            handler_class = self._create_handler()
            self.server = HTTPServer((self.host, self.port), handler_class)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

            logger.info(f"Metrics server started on {self.host}:{self.port}")
            logger.info(f"Metrics endpoint: http://{self.host}:{self.port}/metrics")
            logger.info(f"Health endpoint: http://{self.host}:{self.port}/health")

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    def stop(self):
        """Stop the metrics server"""
        if self.server:
            logger.info("Stopping metrics server...")
            self.server.shutdown()
            self.server.server_close()
            self.server = None

            if self.thread:
                self.thread.join(timeout=5)
                self.thread = None

            logger.info("Metrics server stopped")

    def is_running(self) -> bool:
        """Check if the server is running"""
        return self.server is not None and self.thread is not None and self.thread.is_alive()


def start_metrics_server(host: str = '0.0.0.0', port: int = 8000,
                         collector=None) -> MetricsHTTPServer:
    """
    Start a metrics HTTP server
    """
    if collector is None:
        collector = get_metrics_collector()

    server = MetricsHTTPServer(collector, host, port)
    server.start()
    return server


# Integration functions for easy use in applications
def setup_metrics_endpoint(host: str = '0.0.0.0', port: int = 8000,
                          service_name: str = "fedzk"):
    """
    Setup and start metrics endpoint for a service
    """
    from .metrics import FEDzkMetricsCollector

    collector = FEDzkMetricsCollector(service_name)
    server = start_metrics_server(host, port, collector)

    return collector, server


if __name__ == "__main__":
    # Example usage
    import sys
    import time

    host = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    print(f"Starting metrics server on {host}:{port}")

    collector, server = setup_metrics_endpoint(host, port)

    # Add some example metrics
    collector.record_request("GET", "/test", 200, 0.1)
    collector.record_proof_generation({
        'proof_generation_time': 1.5,
        'proof_size_bytes': 1024,
        'verification_time': 0.5,
        'circuit_complexity': 1000,
        'success': True,
        'proof_type': 'test_circuit'
    })

    print("Metrics server running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping metrics server...")
        server.stop()
        print("Metrics server stopped.")
