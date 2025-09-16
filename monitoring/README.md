# FEDzk Monitoring and Observability

## Overview

FEDzk includes a comprehensive monitoring and observability system that provides deep insights into system performance, security, and operational health. The monitoring stack includes Prometheus metrics collection, distributed tracing, and performance dashboards.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FEDzk Pods    │    │   Prometheus    │    │     Grafana     │
│                 │    │                 │    │                 │
│ • Coordinator   │────▶  ServiceMonitor │────▶   Dashboards    │
│ • MPC Server    │    │  • /metrics     │    │  • Overview     │
│ • ZK Compiler   │    │  • Scraping     │    │  • ZK Proofs    │
│                 │    │                 │    │  • FL Monitor   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │     Jaeger      │
                    │   Tracing       │
                    └─────────────────┘
```

## Components

### 1. Prometheus Metrics Collection

#### Core Metrics
- **HTTP Requests**: `fedzk_requests_total` - Request count by method, endpoint, status
- **Request Duration**: `fedzk_request_duration_seconds` - Request latency histograms
- **Active Connections**: `fedzk_active_connections` - Current connection counts
- **System Resources**: CPU, memory usage metrics

#### ZK Proof Metrics
- **Proof Generation**: `fedzk_zk_proof_generation_total` - Proof generation counters
- **Generation Duration**: `fedzk_zk_proof_generation_duration_seconds` - Proof generation time
- **Proof Size**: `fedzk_zk_proof_size_bytes` - Generated proof sizes
- **Verification Metrics**: Success rates and verification times
- **Circuit Complexity**: Constraint and variable counts

#### Federated Learning Metrics
- **Training Rounds**: `fedzk_fl_rounds_total` - FL round completion status
- **Active Participants**: `fedzk_fl_participants_active` - Current participant count
- **MPC Operations**: `fedzk_mpc_operations_total` - Secure computation operations

#### Security Metrics
- **Authentication**: `fedzk_auth_attempts_total` - Auth success/failure rates
- **TLS Handshakes**: `fedzk_tls_handshakes_total` - TLS connection metrics
- **Security Events**: `fedzk_security_events_total` - Security incident tracking

### 2. Distributed Tracing

#### Features
- **OpenTelemetry Integration**: Full tracing support with Jaeger
- **Request Tracing**: End-to-end request tracking
- **Span Attributes**: Rich metadata for debugging
- **Performance Analysis**: Latency breakdown and bottleneck identification

#### Configuration
```yaml
monitoring:
  tracing:
    enabled: true
    jaeger:
      host: "jaeger-collector"
      port: 6831
```

### 3. Performance Dashboards

#### Available Dashboards

##### 1. FEDzk Overview Dashboard
- System resource utilization (CPU, memory)
- Request rates and latency
- Active connections
- Error rates and success rates

##### 2. FEDzk ZK Proof Monitoring
- Proof generation rates and success rates
- Proof size distributions
- Circuit complexity metrics
- Verification performance

##### 3. FEDzk Federated Learning Monitoring
- Training round completion rates
- Active participant tracking
- MPC operation performance
- Model update frequencies

##### 4. FEDzk Security Monitoring
- Authentication success rates
- TLS handshake performance
- Security event tracking
- Encryption operation metrics

## Installation and Setup

### 1. Prometheus Setup

```bash
# Add Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus
```

### 2. Grafana Setup

```bash
# Add Helm repository
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Grafana
helm install grafana grafana/grafana
```

### 3. Jaeger Setup (Optional)

```bash
# Add Helm repository
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

# Install Jaeger
helm install jaeger jaegertracing/jaeger
```

### 4. Deploy FEDzk with Monitoring

```bash
# Deploy FEDzk with monitoring enabled
helm install fedzk ./helm/fedzk \
  --set monitoring.enabled=true \
  --set monitoring.prometheus.enabled=true \
  --set monitoring.grafana.enabled=true
```

## Metrics Endpoints

### Application Metrics
- **Coordinator**: `http://fedzk-coordinator:8000/metrics`
- **MPC Server**: `http://fedzk-mpc:8001/metrics`
- **ZK Compiler**: `http://fedzk-zk:3000/metrics`

### Health Checks
- **Coordinator**: `http://fedzk-coordinator:8000/health`
- **MPC Server**: `http://fedzk-mpc:8001/health`
- **ZK Compiler**: `http://fedzk-zk:3000/health`

## Alerting Rules

### Pre-configured Alerts

#### System Alerts
- **High CPU Usage**: CPU > 90% for 5 minutes
- **High Memory Usage**: Memory > 90% for 5 minutes

#### Application Alerts
- **ZK Proof Failures**: Proof generation failures detected
- **Low FL Participants**: Fewer than 1 active participant

#### Security Alerts
- **Authentication Failures**: High auth failure rate
- **Security Events**: Suspicious activity detected

## Usage Examples

### Python Integration

```python
from fedzk.monitoring import FEDzkMetricsCollector, track_zk_proof_operation

# Initialize collector
collector = FEDzkMetricsCollector("my-service")

# Record HTTP request
collector.record_request("POST", "/api/proof", 200, 0.25)

# Track ZK proof operation
@track_zk_proof_operation("federated_aggregation")
def generate_proof(circuit_data):
    # Your proof generation logic
    return proof_result

# Record custom metrics
collector.record_security_event("login_attempt", "info")
```

### Metrics Server

```python
from fedzk.monitoring.metrics_server import setup_metrics_endpoint

# Start metrics server
collector, server = setup_metrics_endpoint(
    host="0.0.0.0",
    port=8000,
    service_name="my-service"
)

# Server runs in background, serving /metrics and /health
```

## Configuration

### Helm Values

```yaml
monitoring:
  enabled: true

  prometheus:
    enabled: true
    scrapeInterval: "30s"
    evaluationInterval: "30s"

    serviceMonitor:
      enabled: true
      interval: "30s"
      scrapeTimeout: "10s"

  tracing:
    enabled: true
    jaeger:
      host: "jaeger-collector"
      port: 6831

  grafana:
    enabled: true
    dashboards:
      enabled: true
```

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
1. Check ServiceMonitor configuration
2. Verify metrics endpoint accessibility
3. Check Prometheus target status

#### Tracing Not Working
1. Ensure Jaeger is running and accessible
2. Check Jaeger host/port configuration
3. Verify OpenTelemetry packages are installed

#### Dashboard Not Loading
1. Check Grafana datasource configuration
2. Verify dashboard JSON syntax
3. Check Grafana permissions

### Debug Commands

```bash
# Check metrics endpoint
curl http://fedzk-coordinator:8000/metrics

# Check Prometheus targets
kubectl get servicemonitors

# Check Grafana dashboards
kubectl get configmaps -l grafana_dashboard=1

# View Jaeger traces
kubectl port-forward svc/jaeger-query 16686:16686
```

## Performance Considerations

### Resource Requirements
- **Prometheus**: 2-4 CPU cores, 4-8GB RAM
- **Grafana**: 1-2 CPU cores, 1-2GB RAM
- **Jaeger**: 2-4 CPU cores, 2-4GB RAM

### Scaling Recommendations
- Use persistent volumes for Prometheus and Grafana
- Configure appropriate retention policies
- Use external databases for large deployments

## Security

### Best Practices
- Use HTTPS for metrics endpoints in production
- Configure authentication for Grafana
- Use network policies to restrict monitoring traffic
- Regularly rotate monitoring credentials

### Compliance
- Metrics data retention policies
- Audit logging for monitoring access
- Data encryption at rest and in transit

## Support

For issues with the monitoring system:
1. Check the troubleshooting section
2. Review Prometheus/Grafana logs
3. Verify network connectivity between components
4. Check resource utilization of monitoring pods
