# 🚀 FEDZK Deployment Guide

This guide covers deploying FEDZK in various environments including cloud platforms, on-premises, and containerized deployments.

## Table of Contents

1. [Quick Deployment](#quick-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [AWS Deployment](#aws-deployment)
5. [Azure Deployment](#azure-deployment)
6. [GCP Deployment](#gcp-deployment)
7. [On-Premises Deployment](#on-premises-deployment)
8. [High Availability Setup](#high-availability-setup)
9. [Monitoring and Observability](#monitoring-and-observability)

## Quick Deployment

### Using Docker Compose (Recommended for Development)

```yaml
# docker-compose.yml
version: '3.8'
services:
  fedzk-coordinator:
    image: fedzk/coordinator:latest
    ports:
      - "8000:8000"
    environment:
      - FEDZK_ENV=production
      - FEDZK_DATABASE_URL=postgresql://user:pass@db:5432/fedzk
    depends_on:
      - db
      - redis

  fedzk-worker:
    image: fedzk/worker:latest
    environment:
      - FEDZK_COORDINATOR_URL=http://coordinator:8000
      - FEDZK_WORKER_ID=worker-1
    depends_on:
      - coordinator

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=fedzk
      - POSTGRES_USER=fedzk
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

```bash
# Deploy with Docker Compose
docker-compose up -d

# Check deployment status
docker-compose ps
```

## Docker Deployment

### Building FEDZK Docker Images

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash fedzk
RUN chown -R fedzk:fedzk /app
USER fedzk

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import fedzk; print('FEDZK is healthy')"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "fedzk.server"]
```

```bash
# Build the image
docker build -t fedzk:latest .

# Run the container
docker run -d \
  --name fedzk-server \
  -p 8000:8000 \
  -e FEDZK_ENV=production \
  -e FEDZK_DATABASE_URL=postgresql://user:pass@localhost:5432/fedzk \
  fedzk:latest
```

### Multi-Stage Docker Build

```dockerfile
# Multi-stage Dockerfile for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    git

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash fedzk

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=fedzk:fedzk . .

# Switch to non-root user
USER fedzk

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "fedzk.server"]
```

## Kubernetes Deployment

### Complete Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fedzk-coordinator
  labels:
    app: fedzk
    component: coordinator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fedzk
      component: coordinator
  template:
    metadata:
      labels:
        app: fedzk
        component: coordinator
    spec:
      containers:
      - name: fedzk-coordinator
        image: fedzk/coordinator:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: FEDZK_ENV
          value: "production"
        - name: FEDZK_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: fedzk-secrets
              key: database-url
        - name: FEDZK_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: fedzk-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: fedzk-data
          mountPath: /app/data
      volumes:
      - name: fedzk-data
        persistentVolumeClaim:
          claimName: fedzk-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: fedzk-coordinator
  labels:
    app: fedzk
spec:
  selector:
    app: fedzk
    component: coordinator
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fedzk-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fedzk-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.fedzk.example.com
    secretName: fedzk-tls
  rules:
  - host: api.fedzk.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fedzk-coordinator
            port:
              number: 8000
```

### Helm Chart Deployment

```yaml
# values.yaml
replicaCount: 3

image:
  repository: fedzk/coordinator
  tag: "v1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: api.fedzk.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: fedzk-tls
      hosts:
        - api.fedzk.example.com

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi

database:
  host: postgresql
  port: 5432
  database: fedzk
  username: fedzk
  password: ""  # Set via secret

redis:
  host: redis
  port: 6379
  password: ""  # Set via secret

zk:
  enabled: true
  circuitPath: /app/circuits
```

```bash
# Install with Helm
helm repo add fedzk https://charts.fedzk.io
helm install fedzk fedzk/fedzk -f values.yaml

# Upgrade deployment
helm upgrade fedzk fedzk/fedzk -f values.yaml

# Check status
kubectl get pods -l app=fedzk
```

## AWS Deployment

### Using Amazon ECS

```yaml
# task-definition.json
{
  "family": "fedzk-task",
  "taskRoleArn": "arn:aws:iam::123456789012:role/fedzk-task-role",
  "executionRoleArn": "arn:aws:iam::123456789012:role/fedzk-execution-role",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "fedzk-coordinator",
      "image": "fedzk/coordinator:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FEDZK_ENV",
          "value": "production"
        },
        {
          "name": "FEDZK_DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/fedzk"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fedzk",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### Using AWS Fargate

```bash
# Create cluster
aws ecs create-cluster --cluster-name fedzk-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster fedzk-cluster \
  --service-name fedzk-service \
  --task-definition fedzk-task \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345]}"
```

### Using Amazon EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name fedzk-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed

# Install FEDZK with Helm
helm repo add fedzk https://charts.fedzk.io
helm install fedzk fedzk/fedzk \
  --set database.host=rds-endpoint \
  --set redis.host=elasticache-endpoint
```

## Azure Deployment

### Using Azure Container Instances

```bash
# Create resource group
az group create --name fedzk-rg --location eastus

# Create container instance
az container create \
  --resource-group fedzk-rg \
  --name fedzk-container \
  --image fedzk/coordinator:latest \
  --ports 8000 \
  --environment-variables FEDZK_ENV=production FEDZK_DATABASE_URL=postgresql://user:pass@server:5432/fedzk \
  --cpu 1 \
  --memory 2 \
  --dns-name-label fedzk-api \
  --ports 8000
```

### Using Azure Kubernetes Service (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group fedzk-rg \
  --name fedzk-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group fedzk-rg --name fedzk-cluster

# Deploy FEDZK
helm install fedzk fedzk/fedzk \
  --set database.host=postgres-server \
  --set redis.host=redis-cache
```

## GCP Deployment

### Using Google Kubernetes Engine (GKE)

```bash
# Create GKE cluster
gcloud container clusters create fedzk-cluster \
  --num-nodes=3 \
  --zone=us-central1-a \
  --machine-type=e2-medium

# Get credentials
gcloud container clusters get-credentials fedzk-cluster --zone=us-central1-a

# Deploy FEDZK
helm install fedzk fedzk/fedzk \
  --set database.host=cloudsql-instance \
  --set redis.host=redis-instance
```

### Using Cloud Run

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: fedzk-coordinator
spec:
  template:
    spec:
      containers:
      - image: gcr.io/project-id/fedzk/coordinator:latest
        ports:
        - containerPort: 8080
        env:
        - name: FEDZK_ENV
          value: "production"
        - name: FEDZK_DATABASE_URL
          value: "postgresql://user:pass@instance:5432/fedzk"
        resources:
          limits:
            cpu: "1000m"
            memory: "1Gi"
```

## On-Premises Deployment

### Bare Metal Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip postgresql redis-server nginx

# Create FEDZK user
sudo useradd -m -s /bin/bash fedzk
sudo usermod -aG sudo fedzk

# Install FEDZK
sudo -u fedzk pip3 install fedzk

# Configure database
sudo -u postgres createdb fedzk
sudo -u postgres createuser fedzk
sudo -u postgres psql -c "ALTER USER fedzk PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE fedzk TO fedzk;"

# Configure Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Configure nginx
sudo tee /etc/nginx/sites-available/fedzk << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/fedzk /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Configure FEDZK
sudo -u fedzk tee /home/fedzk/.fedzk/config.yaml << EOF
environment: production
database:
  url: postgresql://fedzk:secure_password@localhost:5432/fedzk
redis:
  url: redis://localhost:6379
server:
  host: 127.0.0.1
  port: 8000
logging:
  level: INFO
  file: /home/fedzk/logs/fedzk.log
EOF

# Start FEDZK service
sudo -u fedzk nohup python3 -m fedzk.server > /home/fedzk/logs/fedzk.out 2>&1 &
```

### Using systemd

```ini
# /etc/systemd/system/fedzk.service
[Unit]
Description=FEDZK Federated Learning Server
After=network.target postgresql.service redis-server.service

[Service]
Type=simple
User=fedzk
Group=fedzk
WorkingDirectory=/home/fedzk
Environment=PATH=/home/fedzk/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/fedzk/.local/bin/python3 -m fedzk.server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable fedzk
sudo systemctl start fedzk
sudo systemctl status fedzk
```

## High Availability Setup

### Load Balancing with HAProxy

```cfg
# /etc/haproxy/haproxy.cfg
global
    log /dev/log local0
    log /dev/log local1 notice
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin expose-fd listeners
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

defaults
    log global
    mode http
    option httplog
    option dontlognull
    timeout connect 5000
    timeout client 50000
    timeout server 50000

frontend fedzk_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/fedzk.pem
    redirect scheme https if !{ ssl_fc }
    default_backend fedzk_backend

backend fedzk_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    server fedzk1 192.168.1.10:8000 check
    server fedzk2 192.168.1.11:8000 check
    server fedzk3 192.168.1.12:8000 check
```

### Database Replication

```sql
-- Enable PostgreSQL replication
-- On primary server
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET max_wal_senders = 3;
ALTER SYSTEM SET wal_keep_size = '64MB';

-- Create replication user
CREATE USER replicator REPLICATION LOGIN PASSWORD 'replication_password';

-- On standby servers
pg_basebackup -h primary_host -U replicator -D /var/lib/postgresql/data -P
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fedzk'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'fedzk-workers'
    static_configs:
      - targets:
        - 'worker1:8001'
        - 'worker2:8001'
        - 'worker3:8001'
    metrics_path: '/metrics'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "FEDZK Monitoring",
    "tags": ["fedzk", "federated-learning"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Active Federations",
        "type": "stat",
        "targets": [
          {
            "expr": "fedzk_federations_active",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Training Sessions",
        "type": "graph",
        "targets": [
          {
            "expr": "fedzk_training_sessions_total",
            "refId": "A"
          }
        ]
      },
      {
        "title": "ZK Proof Generation Time",
        "type": "heatmap",
        "targets": [
          {
            "expr": "fedzk_zk_proof_generation_duration_seconds",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```yaml
# logging.yml
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(name)s %(levelname)s %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    formatter: json
    level: INFO

  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/fedzk/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: json
    level: DEBUG

root:
  level: INFO
  handlers: [console, file]

loggers:
  fedzk:
    level: DEBUG
    handlers: [console, file]
    propagate: false
```

## Next Steps

- 📖 Read the [API Documentation](./api/)
- 🔧 Check [Configuration Guide](./configuration.md)
- 🚨 Review [Security Best Practices](./security.md)
- 📊 Explore [Monitoring Guide](./monitoring.md)
- 🆘 Check [Troubleshooting Guide](./troubleshooting.md)

---

For support and questions, visit our [Community Forum](https://community.fedzk.io) or [GitHub Issues](https://github.com/fedzk/fedzk/issues).
