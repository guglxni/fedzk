# FEDzk Helm Chart

![Version: 1.0.0](https://img.shields.io/badge/Version-1.0.0-informational?style=flat-square)
![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square)
![AppVersion: 1.0.0](https://img.shields.io/badge/AppVersion-1.0.0-informational?style=flat-square)

A Helm chart for deploying FEDzk - Zero-Knowledge Federated Learning Framework on Kubernetes.

## Overview

FEDzk is a production-grade framework for zero-knowledge federated learning that enables privacy-preserving machine learning across distributed data sources using advanced cryptographic techniques.

This Helm chart provides a complete Kubernetes deployment solution with:
- **Horizontal scaling** with HPA (Horizontal Pod Autoscaler)
- **High availability** with Pod Disruption Budgets
- **Rolling updates** for zero-downtime deployments
- **Security hardening** with Network Policies and RBAC
- **Monitoring integration** with Prometheus and Grafana
- **Resource management** with quotas and limits
- **Ingress configuration** for external access

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Coordinator   │    │     MPC Server  │    │  ZK Toolchain   │
│                 │    │                 │    │                 │
│ • Client Mgmt   │◄──►│ • Proof Gen     │◄──►│ • Circom        │
│ • Aggregation   │    │ • Validation    │    │ • SNARKjs       │
│ • Orchestration │    │ • MPC Protocol  │    │ • Compilation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┴────────────────────────┘
                          ┌─────────────────┐
                          │   PostgreSQL    │
                          │     Redis       │
                          │   Monitoring    │
                          └─────────────────┘
```

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PV provisioner support for persistent storage (if using embedded databases)

## Quick Start

### Add Helm Repository
```bash
# Add the FEDzk Helm repository (when available)
helm repo add fedzk https://charts.fedzk.io
helm repo update
```

### Install Chart
```bash
# Install with default values
helm install fedzk ./helm/fedzk

# Install with custom values
helm install fedzk ./helm/fedzk -f my-values.yaml

# Install in specific namespace
helm install fedzk ./helm/fedzk -n fedzk-namespace --create-namespace
```

### Access the Application
```bash
# Get service information
kubectl get svc -l app.kubernetes.io/name=fedzk

# Port forward for local access
kubectl port-forward svc/fedzk-coordinator 8000:8000

# Access via ingress (if enabled)
curl https://fedzk.yourdomain.com/health
```

## Configuration

### Global Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Global Docker image registry | `""` |
| `global.imagePullSecrets` | Global Docker registry secret names | `[]` |
| `global.storageClass` | Global storage class for PVCs | `""` |

### FEDzk Coordinator

| Parameter | Description | Default |
|-----------|-------------|---------|
| `coordinator.enabled` | Enable coordinator deployment | `true` |
| `coordinator.replicaCount` | Number of coordinator replicas | `3` |
| `coordinator.image.repository` | Coordinator image repository | `fedzk` |
| `coordinator.image.tag` | Coordinator image tag | `latest` |
| `coordinator.service.port` | Coordinator service port | `8000` |
| `coordinator.resources.limits.cpu` | CPU limit | `1000m` |
| `coordinator.resources.limits.memory` | Memory limit | `1Gi` |
| `coordinator.resources.requests.cpu` | CPU request | `500m` |
| `coordinator.resources.requests.memory` | Memory request | `512Mi` |
| `coordinator.autoscaling.enabled` | Enable HPA | `true` |
| `coordinator.autoscaling.minReplicas` | Minimum replicas | `3` |
| `coordinator.autoscaling.maxReplicas` | Maximum replicas | `10` |
| `coordinator.pdb.enabled` | Enable Pod Disruption Budget | `true` |

### MPC Server

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mpc.enabled` | Enable MPC server deployment | `true` |
| `mpc.replicaCount` | Number of MPC server replicas | `2` |
| `mpc.image.repository` | MPC server image repository | `fedzk` |
| `mpc.service.port` | MPC server service port | `8001` |
| `mpc.autoscaling.enabled` | Enable HPA | `true` |
| `mpc.autoscaling.minReplicas` | Minimum replicas | `2` |
| `mpc.autoscaling.maxReplicas` | Maximum replicas | `8` |

### ZK Toolchain

| Parameter | Description | Default |
|-----------|-------------|---------|
| `zk.enabled` | Enable ZK toolchain deployment | `true` |
| `zk.replicaCount` | Number of ZK toolchain replicas | `1` |
| `zk.image.repository` | ZK toolchain image repository | `fedzk/zk-toolchain` |
| `zk.service.port` | ZK toolchain service port | `3000` |

### Database Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `postgresql.enabled` | Enable embedded PostgreSQL | `true` |
| `externalDatabase.host` | External database host | `""` |
| `externalDatabase.port` | External database port | `5432` |
| `externalDatabase.database` | External database name | `""` |
| `externalDatabase.username` | External database username | `""` |
| `externalDatabase.password` | External database password | `""` |

### Ingress Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class name | `"nginx"` |
| `ingress.hosts[0].host` | Ingress host | `fedzk.example.com` |
| `ingress.tls[0].secretName` | TLS secret name | `fedzk-tls` |
| `ingress.tls[0].hosts` | TLS hosts | `[fedzk.example.com]` |

### Monitoring

| Parameter | Description | Default |
|-----------|-------------|---------|
| `monitoring.enabled` | Enable monitoring | `true` |
| `monitoring.prometheus.enabled` | Enable Prometheus integration | `true` |
| `monitoring.grafana.enabled` | Enable Grafana dashboards | `true` |

## Scaling Configuration

### Horizontal Pod Autoscaling

The chart includes HPA configurations for automatic scaling based on CPU and memory utilization:

```yaml
coordinator:
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
```

### Manual Scaling

Scale individual components:

```bash
# Scale coordinator
kubectl scale deployment fedzk-coordinator --replicas=5

# Scale MPC servers
kubectl scale deployment fedzk-mpc --replicas=4
```

## High Availability

### Pod Disruption Budgets

The chart includes PDBs to ensure minimum availability during cluster maintenance:

```yaml
coordinator:
  pdb:
    enabled: true
    minAvailable: 2
```

### Rolling Updates

Configure rolling update strategy for zero-downtime deployments:

```yaml
rollingUpdate:
  enabled: true
  maxUnavailable: "25%"
  maxSurge: "25%"
  timeout: 600
```

## Security

### Network Policies

Enable network segmentation:

```yaml
security:
  networkPolicy:
    enabled: true
```

### RBAC

Configure Role-Based Access Control:

```yaml
rbac:
  create: true
  rules:
    - apiGroups: [""]
      resources: ["pods", "services"]
      verbs: ["get", "list", "watch"]
```

### Security Context

Configure pod security contexts:

```yaml
coordinator:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
      - ALL
```

## Monitoring

### Prometheus Integration

Enable metrics collection:

```yaml
monitoring:
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: 30s
```

### Grafana Dashboards

Access pre-configured dashboards:

```bash
# Port forward Grafana
kubectl port-forward svc/grafana 3000:3000

# Access at http://localhost:3000
# Default credentials: admin/admin
```

## Backup and Recovery

### Database Backup

Configure automated backups:

```yaml
backup:
  enabled: false
  schedule: "0 2 * * *"
  retention: "30d"
  storage:
    size: "100Gi"
    className: "standard"
```

### Disaster Recovery

The deployment includes:
- Persistent volume claims for data persistence
- Pod disruption budgets for high availability
- Rolling update strategies for safe deployments
- Health checks for automatic recovery

## Troubleshooting

### Common Issues

1. **Pods not starting**
   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **Service not accessible**
   ```bash
   kubectl get svc
   kubectl describe svc <service-name>
   ```

3. **Ingress not working**
   ```bash
   kubectl get ingress
   kubectl describe ingress <ingress-name>
   ```

### Health Checks

Check component health:

```bash
# Coordinator health
curl http://fedzk-coordinator:8000/health

# MPC server health
curl http://fedzk-mpc:8001/health

# ZK toolchain health
kubectl exec -it <zk-pod> -- circom --version
```

## Upgrade Guide

### Minor Version Upgrades

```bash
# Update Helm repository
helm repo update

# Upgrade release
helm upgrade fedzk ./helm/fedzk
```

### Major Version Upgrades

1. Review release notes
2. Backup data
3. Update values.yaml if needed
4. Perform upgrade with downtime window

```bash
# Upgrade with backup
helm upgrade fedzk ./helm/fedzk --backup
```

## Uninstall

```bash
# Uninstall release
helm uninstall fedzk

# Clean up PVCs (if desired)
kubectl delete pvc -l app.kubernetes.io/name=fedzk
```

## Development

### Local Development

Use the development Docker Compose setup:

```bash
docker-compose up -d
```

### Chart Development

Test chart changes:

```bash
# Lint chart
helm lint ./helm/fedzk

# Template chart
helm template fedzk ./helm/fedzk

# Dry-run install
helm install fedzk ./helm/fedzk --dry-run
```

## Support

- **Documentation**: https://docs.fedzk.io
- **Issues**: https://github.com/fedzk/fedzk/issues
- **Discussions**: https://github.com/fedzk/fedzk/discussions

## License

This chart is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

