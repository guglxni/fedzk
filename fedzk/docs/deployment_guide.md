# Deployment Guide for FedZK

This guide provides detailed instructions for deploying FedZK in production environments.

## Deployment Architecture

FedZK consists of three main components:

1. **Clients**: Running on participant devices/servers
2. **Coordinator**: Central server for aggregating updates
3. **MPC Server**: Optional server for remote proof operations

### Recommended Architecture

```
┌───────────┐     ┌───────────────┐     ┌───────────┐
│  Client   │────▶│  Coordinator  │◀────│  Client   │
│           │     │               │     │           │
└─────┬─────┘     └───────┬───────┘     └─────┬─────┘
      │                   │                   │
      │                   ▼                   │
      │           ┌───────────────┐          │
      └──────────▶│   MPC Server  │◀─────────┘
                  │               │
                  └───────────────┘
```

## System Requirements

### Coordinator Server

- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 20+ GB
- OS: Linux (Ubuntu 20.04+ recommended)
- Network: Public IP with ports 8000/TCP open

### MPC Server

- CPU: 8+ cores
- RAM: 16+ GB
- GPU: Optional, improves proof generation performance
- Disk: 50+ GB
- OS: Linux (Ubuntu 20.04+ recommended)
- Network: Public IP with ports 8001/TCP open

### Clients

- CPU: 2+ cores
- RAM: 4+ GB
- Disk: 10+ GB for data and models
- Network: Internet connection to reach coordinator and MPC server

## Deployment Options

### Docker Deployment

FedZK provides Docker images for easy deployment:

```bash
# Pull images
docker pull aaryanguglani/fedzk-coordinator:latest
docker pull aaryanguglani/fedzk-mpc:latest

# Run coordinator
docker run -d -p 8000:8000 --name fedzk-coordinator \
  -v /path/to/config:/app/config \
  aaryanguglani/fedzk-coordinator:latest

# Run MPC server
docker run -d -p 8001:8001 --name fedzk-mpc \
  -e MPC_API_KEYS=key1,key2,key3 \
  -v /path/to/zk:/app/zk \
  aaryanguglani/fedzk-mpc:latest
```

### Docker Compose

Use the provided `docker-compose.yml`:

```yaml
version: '3'

services:
  coordinator:
    image: aaryanguglani/fedzk-coordinator:latest
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped

  mpc:
    image: aaryanguglani/fedzk-mpc:latest
    ports:
      - "8001:8001"
    volumes:
      - ./zk:/app/zk
    environment:
      - MPC_API_KEYS=key1,key2,key3
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

Start with:

```bash
docker-compose up -d
```

### Kubernetes Deployment

Deploy using Kubernetes manifests:

```bash
kubectl apply -f kubernetes/coordinator-deployment.yaml
kubectl apply -f kubernetes/mpc-deployment.yaml
kubectl apply -f kubernetes/coordinator-service.yaml
kubectl apply -f kubernetes/mpc-service.yaml
```

## Configuration

### Environment Variables

#### Coordinator

| Variable | Description | Default |
|----------|-------------|---------|
| `COORDINATOR_HOST` | Host to bind the server | `0.0.0.0` |
| `COORDINATOR_PORT` | Port to listen on | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MIN_CLIENTS` | Minimum clients for aggregation | `3` |
| `SECURE_MODE` | Enable secure verification | `True` |

#### MPC Server

| Variable | Description | Default |
|----------|-------------|---------|
| `MPC_HOST` | Host to bind the server | `0.0.0.0` |
| `MPC_PORT` | Port to listen on | `8001` |
| `MPC_API_KEYS` | Comma-separated API keys | Required |
| `LOG_LEVEL` | Logging level | `INFO` |
| `BATCH_ENABLED` | Enable batch processing | `True` |

### Configuration Files

Both servers also support YAML configuration files:

```yaml
# coordinator-config.yaml
host: 0.0.0.0
port: 8000
log_level: INFO
min_clients: 3
secure_mode: true

# mpc-config.yaml
host: 0.0.0.0
port: 8001
log_level: INFO
batch_enabled: true
```

## Security Considerations

### API Key Management

For MPC server, use strong, random API keys and rotate them regularly:

```bash
# Generate a secure API key
openssl rand -hex 32
```

### TLS Configuration

Both servers should be deployed behind HTTPS. Use a reverse proxy like Nginx:

```nginx
server {
    listen 443 ssl;
    server_name coordinator.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Network Security

- Use firewalls to restrict access to coordinator and MPC servers
- Consider using VPN for connections between components
- Use API key authentication for all MPC server endpoints

## Scaling Considerations

### Coordinator Scaling

For large-scale deployments:

1. Use load balancing with multiple coordinator instances
2. Implement a shared database for state management
3. Use Redis for caching and coordination

### MPC Server Scaling

For high-throughput deployments:

1. Deploy multiple MPC servers behind a load balancer
2. Distribute API keys across instances
3. Consider GPU acceleration for proof generation

## Monitoring and Logging

### Logging Configuration

Configure logging to a centralized service:

```python
import logging
from logging.handlers import SysLogHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        SysLogHandler(address=('logs.example.com', 514))
    ]
)
```

### Prometheus Metrics

Both servers expose Prometheus metrics on `/metrics` endpoints:

- Request counters
- Latency histograms
- Success/failure rates
- Resource usage

### Health Checks

Health check endpoints are available at:

- Coordinator: `/health`
- MPC Server: `/health`

## Backup and Recovery

### Database Backups

If using a database for state management:

```bash
# PostgreSQL example
pg_dump -U username -d database -f backup.sql
```

### ZK Artifacts Backup

Back up the ZK circuit artifacts:

```bash
rsync -av /path/to/zk/ /path/to/backup/
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**: Check network configuration and firewall rules
2. **Authentication Failures**: Verify API keys are correctly configured
3. **Resource Exhaustion**: Increase memory and CPU allocations
4. **Circuit Compatibility**: Ensure all components use compatible circuit versions

### Logs Analysis

Extract errors from logs:

```bash
grep ERROR /var/log/fedzk.log
```

## Upgrading

When upgrading to new versions:

1. Back up all configuration and data
2. Update Docker images or binaries
3. Run migrations if necessary
4. Verify system functionality after upgrade

For zero-downtime upgrades, use blue-green deployment or rolling updates in Kubernetes. 